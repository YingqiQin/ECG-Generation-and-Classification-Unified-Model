#!/usr/bin/env python
"""
Convert user train.csv + npy slices into SIGMA-PPG tokenizer H5 format.

Output H5 fields:
- signals:   [N, window_size]
- feat_amp:  [N, num_patches]
- feat_skew: [N, num_patches]
- feat_avg:  [N, num_patches]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.stats import skew


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build H5 files for SIGMA-PPG codebook training from train.csv")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated H5 files")

    parser.add_argument("--id_col", type=str, default="id_clean", help="Patient id column name")
    parser.add_argument("--ppg_path_col", type=str, default="ppg_path", help="Numpy path column name")
    parser.add_argument("--i0_col", type=str, default="i0", help="Slice start index column name")
    parser.add_argument("--i1_col", type=str, default="i1", help="Slice end index column name")
    parser.add_argument("--n_ppg_col", type=str, default="n_ppg", help="Optional valid length column name")

    parser.add_argument("--window_size", type=int, default=12000, help="Final segment length for tokenizer input")
    parser.add_argument("--patch_size", type=int, default=100, help="Patch size used for feature vectors")
    parser.add_argument("--channel_index", type=int, default=0, help="PPG channel index to use from npy")
    parser.add_argument(
        "--channel_axis",
        type=str,
        default="auto",
        choices=["auto", "0", "1"],
        help="Which axis is channel for 2D npy. auto supports (T,C) or (C,T).",
    )

    parser.add_argument("--segments_per_file", type=int, default=5000, help="How many segments per output H5 file")
    parser.add_argument("--output_prefix", type=str, default="custom", help="Output H5 filename prefix")
    parser.add_argument("--normalize", action="store_true", help="Min-max normalize each segment to [-1, 1]")
    parser.add_argument("--max_rows", type=int, default=0, help="If > 0, only process first N rows (debug)")
    parser.add_argument("--log_every", type=int, default=10000, help="Progress log interval in processed rows")
    return parser.parse_args()


def resolve_path(csv_path: Path, raw_path: str) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute():
        return p
    return (csv_path.parent / p).resolve()


def fill_nan_linear(sig: np.ndarray) -> np.ndarray:
    if not np.isnan(sig).any():
        return sig
    x = np.arange(sig.shape[0])
    mask = np.isnan(sig)
    if mask.all():
        return np.zeros_like(sig)
    sig = sig.copy()
    sig[mask] = np.interp(x[mask], x[~mask], sig[~mask])
    return sig


def normalize_minus_one_to_one(sig: np.ndarray) -> np.ndarray:
    mn = float(np.min(sig))
    mx = float(np.max(sig))
    if mx - mn < 1e-9:
        return np.zeros_like(sig, dtype=np.float32)
    return (2.0 * (sig - mn) / (mx - mn) - 1.0).astype(np.float32)


def choose_channel_1d(arr: np.ndarray, channel_index: int, channel_axis: str) -> np.ndarray:
    if arr.ndim == 1:
        return arr.astype(np.float32)

    if arr.ndim != 2:
        raise ValueError(f"Unsupported npy shape {arr.shape}. Expected 1D or 2D.")

    if channel_axis == "0":
        if channel_index >= arr.shape[0]:
            raise IndexError(f"channel_index={channel_index} out of range for shape {arr.shape} on axis 0")
        return arr[channel_index, :].astype(np.float32)

    if channel_axis == "1":
        if channel_index >= arr.shape[1]:
            raise IndexError(f"channel_index={channel_index} out of range for shape {arr.shape} on axis 1")
        return arr[:, channel_index].astype(np.float32)

    # auto mode
    if arr.shape[1] <= 64 and arr.shape[0] > arr.shape[1]:
        if channel_index >= arr.shape[1]:
            raise IndexError(f"channel_index={channel_index} out of range for shape {arr.shape} inferred as (T,C)")
        return arr[:, channel_index].astype(np.float32)

    if arr.shape[0] <= 64 and arr.shape[1] > arr.shape[0]:
        if channel_index >= arr.shape[0]:
            raise IndexError(f"channel_index={channel_index} out of range for shape {arr.shape} inferred as (C,T)")
        return arr[channel_index, :].astype(np.float32)

    raise ValueError(
        f"Cannot infer channel axis for shape {arr.shape}. Use --channel_axis 0 or 1 explicitly."
    )


def to_fixed_window(sig: np.ndarray, window_size: int) -> np.ndarray:
    out = np.zeros(window_size, dtype=np.float32)
    valid = min(window_size, sig.shape[0])
    if valid > 0:
        out[:valid] = sig[:valid].astype(np.float32)
    return out


def calculate_amplitude_stability_sqi(
    signal: np.ndarray,
    patch_size: int,
    min_valid_std: float = 0.05,
    max_valid_std: float = 2.0,
) -> np.ndarray:
    num_patches = signal.shape[0] // patch_size
    if num_patches <= 0:
        return np.zeros(0, dtype=np.float32)

    patch_stds = np.zeros(num_patches, dtype=np.float32)
    for i in range(num_patches):
        s = i * patch_size
        e = s + patch_size
        patch_stds[i] = np.std(signal[s:e])

    median_std = np.median(patch_stds)
    mad = np.median(np.abs(patch_stds - median_std))

    if mad < 1e-9:
        rel_scores = np.ones(num_patches, dtype=np.float32)
    else:
        modified_z = 0.6745 * (patch_stds - median_std) / mad
        rel_scores = np.exp(-0.2 * (modified_z ** 2)).astype(np.float32)

    rise_k = 50.0
    low_gate = 1.0 / (1.0 + np.exp(-rise_k * (patch_stds - min_valid_std)))

    fall_k = 5.0
    high_gate = 1.0 / (1.0 + np.exp(fall_k * (patch_stds - max_valid_std)))

    return (rel_scores * low_gate * high_gate).astype(np.float32)


def calculate_patch_features(segment: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_patches = segment.shape[0] // patch_size
    if num_patches <= 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z

    trunc_len = num_patches * patch_size
    patches = segment[:trunc_len].reshape(num_patches, patch_size)

    norm_amp = calculate_amplitude_stability_sqi(segment[:trunc_len], patch_size=patch_size)
    patch_skew = skew(patches, axis=1, bias=False)
    patch_skew = np.nan_to_num(patch_skew, nan=0.0)
    norm_skew = np.tanh(np.abs(patch_skew)).astype(np.float32)
    feat_avg = (0.5 * norm_amp + 0.5 * norm_skew).astype(np.float32)

    return norm_amp.astype(np.float32), norm_skew, feat_avg


def flush_h5(
    output_dir: Path,
    output_prefix: str,
    file_index: int,
    signals_buf: List[np.ndarray],
    feat_amp_buf: List[np.ndarray],
    feat_skew_buf: List[np.ndarray],
    feat_avg_buf: List[np.ndarray],
) -> Path:
    out_path = output_dir / f"{output_prefix}_segments_part_{file_index:04d}.h5"

    signals_np = np.stack(signals_buf).astype(np.float32)
    feat_amp_np = np.stack(feat_amp_buf).astype(np.float32)
    feat_skew_np = np.stack(feat_skew_buf).astype(np.float32)
    feat_avg_np = np.stack(feat_avg_buf).astype(np.float32)

    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("signals", data=signals_np, compression="gzip")
        hf.create_dataset("feat_amp", data=feat_amp_np, compression="gzip")
        hf.create_dataset("feat_skew", data=feat_skew_np, compression="gzip")
        hf.create_dataset("feat_avg", data=feat_avg_np, compression="gzip")

    return out_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    csv_path = Path(args.csv_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.window_size % args.patch_size != 0:
        raise ValueError("window_size must be divisible by patch_size")

    required_cols = [args.ppg_path_col, args.i0_col, args.i1_col]
    df = pd.read_csv(csv_path)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    if args.max_rows > 0:
        df = df.head(args.max_rows)

    # keep deterministic order
    df = df.reset_index(drop=True)

    num_rows = len(df)
    logging.info("Loaded CSV rows: %d", num_rows)
    logging.info("Output dir: %s", output_dir)

    # Buffers for chunked writing
    signals_buf: List[np.ndarray] = []
    feat_amp_buf: List[np.ndarray] = []
    feat_skew_buf: List[np.ndarray] = []
    feat_avg_buf: List[np.ndarray] = []

    file_index = 0
    output_files = 0
    saved_segments = 0
    skipped_rows = 0

    grouped = df.groupby(args.ppg_path_col, sort=False)

    for group_idx, (ppg_path_raw, group_df) in enumerate(grouped, start=1):
        npy_path = resolve_path(csv_path, ppg_path_raw)
        if not npy_path.exists():
            logging.warning("Skip missing npy: %s", npy_path)
            skipped_rows += len(group_df)
            continue

        try:
            arr = np.load(npy_path)
            ppg_1d = choose_channel_1d(arr, args.channel_index, args.channel_axis)
            ppg_1d = fill_nan_linear(ppg_1d)
        except Exception as exc:
            logging.warning("Skip npy load/parse failed: %s (%s)", npy_path, exc)
            skipped_rows += len(group_df)
            continue

        for _, row in group_df.iterrows():
            try:
                i0 = int(row[args.i0_col])
                i1 = int(row[args.i1_col])
            except Exception:
                skipped_rows += 1
                continue

            if i1 <= i0:
                skipped_rows += 1
                continue

            # clamp to range
            s = max(0, i0)
            e = min(i1, ppg_1d.shape[0])
            if e <= s:
                skipped_rows += 1
                continue

            seg = ppg_1d[s:e]

            # Optional n_ppg consistency handling
            if args.n_ppg_col in group_df.columns:
                try:
                    n_ppg = int(row[args.n_ppg_col])
                    if n_ppg > 0:
                        seg = seg[:n_ppg]
                except Exception:
                    pass

            seg_fixed = to_fixed_window(seg, args.window_size)
            if args.normalize:
                seg_fixed = normalize_minus_one_to_one(seg_fixed)

            feat_amp, feat_skew, feat_avg = calculate_patch_features(seg_fixed, args.patch_size)

            signals_buf.append(seg_fixed)
            feat_amp_buf.append(feat_amp)
            feat_skew_buf.append(feat_skew)
            feat_avg_buf.append(feat_avg)

            if len(signals_buf) >= args.segments_per_file:
                out_file = flush_h5(
                    output_dir,
                    args.output_prefix,
                    file_index,
                    signals_buf,
                    feat_amp_buf,
                    feat_skew_buf,
                    feat_avg_buf,
                )
                saved_segments += len(signals_buf)
                logging.info("Saved %d segments -> %s", len(signals_buf), out_file)

                file_index += 1
                output_files += 1
                signals_buf.clear()
                feat_amp_buf.clear()
                feat_skew_buf.clear()
                feat_avg_buf.clear()

            if (saved_segments + len(signals_buf) + skipped_rows) % args.log_every == 0:
                logging.info(
                    "Progress: processed=%d, saved=%d, skipped=%d, files=%d",
                    saved_segments + len(signals_buf) + skipped_rows,
                    saved_segments + len(signals_buf),
                    skipped_rows,
                    file_index,
                )

        if group_idx % 100 == 0:
            logging.info("Processed %d unique npy files", group_idx)

    if signals_buf:
        out_file = flush_h5(
            output_dir,
            args.output_prefix,
            file_index,
            signals_buf,
            feat_amp_buf,
            feat_skew_buf,
            feat_avg_buf,
        )
        saved_segments += len(signals_buf)
        logging.info("Saved %d segments -> %s", len(signals_buf), out_file)
        output_files += 1

    logging.info(
        "Done. Total rows=%d, saved=%d, skipped=%d, output_files=%d",
        num_rows,
        saved_segments,
        skipped_rows,
        output_files,
    )


if __name__ == "__main__":
    main()
