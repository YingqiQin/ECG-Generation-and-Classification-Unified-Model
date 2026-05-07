#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX FP32/FP16 PPG-BP product evaluation helpers.

Goal
----
Your existing FP16 conversion code can compare PyTorch vs ONNX BP regression output.
This file focuses on the product pipeline:

    ONNX model inference -> raw_predictions_*.csv
    -> reuse your existing calibration / sensitivity / classification scripts

The generated raw CSV is compatible with:
    - run_calib_sensitivity_affine_repro.py
    - run_1p0_sleep_shrinkage_experiment.py
    - bp_htn_classification_multi_run_custom_thresholds.py
    - bp_htn_threshold_grid_search_abpm.py

Expected test_loader batch:
    (x, y, meta) or (x, y)

where:
    x: [B, K, 1, L]
    y: [B, 2] = calibration/reference target, usually y_true_sbp/y_true_dbp
    meta: dict or list[dict], ideally containing id_clean, t_bp_ms, sleep,
          and optionally ABPM_SBP / ABPM_DBP.

Typical usage in your eval script
---------------------------------
from onnx_fp16_product_eval import (
    convert_onnx_to_fp16_lazy,
    evaluate_torch_to_raw_csv,
    evaluate_onnx_to_raw_csv,
    compare_raw_prediction_csvs,
)

# 1) Convert FP32 ONNX to FP16 ONNX if needed.
convert_onnx_to_fp16_lazy("model_fp32.onnx", "model_fp16.onnx", force=True)

# 2) Save compatible raw CSVs.
evaluate_torch_to_raw_csv(model, test_loader, "raw_predictions_torch_fp32.csv", device="cuda")
evaluate_onnx_to_raw_csv(test_loader, "model_fp32.onnx", "raw_predictions_onnx_fp32.csv")
evaluate_onnx_to_raw_csv(test_loader, "model_fp16.onnx", "raw_predictions_onnx_fp16.csv")

# 3) Check numeric drop.
compare_raw_prediction_csvs("raw_predictions_torch_fp32.csv", "raw_predictions_onnx_fp16.csv")

# 4) Then run your existing product scripts, e.g.
# python run_calib_sensitivity_affine_repro.py --raw_csv raw_predictions_onnx_fp16.csv ...
# python bp_htn_classification_multi_run_custom_thresholds.py --csv-glob "..." ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# =============================================================================
# Basic utilities
# =============================================================================

def model_size_mb(path: str | Path) -> float:
    path = Path(path)
    return path.stat().st_size / (1024.0 * 1024.0)


def to_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _one_meta_value(v: Any, i: int) -> Any:
    if torch is not None and isinstance(v, torch.Tensor):
        vv = v.detach().cpu()
        if vv.ndim == 0:
            return vv.item()
        item = vv[i]
        if item.numel() == 1:
            return item.item()
        return item.numpy()

    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        item = v[i]
        if isinstance(item, np.generic):
            return item.item()
        return item

    if isinstance(v, (list, tuple)):
        return v[i]

    return v


def meta_to_rows(meta: Any, batch_size: int, start_index: int) -> List[Dict[str, Any]]:
    """Convert meta into list of per-sample rows."""
    if isinstance(meta, dict):
        rows: List[Dict[str, Any]] = []
        for i in range(batch_size):
            row = {}
            for k, v in meta.items():
                row[k] = _one_meta_value(v, i)
            rows.append(row)
        return rows

    if isinstance(meta, (list, tuple)) and len(meta) == batch_size and isinstance(meta[0], dict):
        return [dict(m) for m in meta]

    return [
        {
            "row_id": start_index + i,
            "id_clean": "unknown",
            "t_bp_ms": start_index + i,
            "sleep": 0,
        }
        for i in range(batch_size)
    ]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "id_clean" not in out.columns:
        for c in ["id", "subject", "subject_id", "pid", "patient_id"]:
            if c in out.columns:
                out["id_clean"] = out[c].astype(str)
                break
    if "id_clean" not in out.columns:
        out["id_clean"] = "unknown"

    if "t_bp_ms" not in out.columns:
        for c in ["time_ms", "timestamp_ms", "bp_time_ms", "t_ms", "timestamp"]:
            if c in out.columns:
                out["t_bp_ms"] = out[c]
                break
    if "t_bp_ms" not in out.columns:
        out["t_bp_ms"] = np.arange(len(out), dtype=np.int64)

    if "sleep" not in out.columns:
        out["sleep"] = 0

    out["id_clean"] = out["id_clean"].astype(str)
    out["t_bp_ms"] = pd.to_numeric(out["t_bp_ms"], errors="coerce").fillna(0).astype(np.int64)
    out["sleep"] = pd.to_numeric(out["sleep"], errors="coerce").fillna(0).astype(int)

    return out


def me_std_mae(y_pred: Sequence[float], y_true: Sequence[float]) -> Dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    err = y_pred - y_true
    if len(err) == 0:
        return {"ME": np.nan, "STD": np.nan, "MAE": np.nan, "N": 0}
    return {
        "ME": float(np.mean(err)),
        "STD": float(np.std(err, ddof=1)) if len(err) > 1 else np.nan,
        "MAE": float(np.mean(np.abs(err))),
        "N": int(len(err)),
    }


def bp_metrics_from_df(
    df: pd.DataFrame,
    pred_sbp_col: str = "y_pred_sbp_raw",
    pred_dbp_col: str = "y_pred_dbp_raw",
    true_sbp_col: str = "y_true_sbp",
    true_dbp_col: str = "y_true_dbp",
) -> Dict[str, Dict[str, float]]:
    return {
        "sbp": me_std_mae(df[pred_sbp_col], df[true_sbp_col]),
        "dbp": me_std_mae(df[pred_dbp_col], df[true_dbp_col]),
    }


def print_bp_metrics(report: Dict[str, Dict[str, float]], title: str) -> None:
    def fmt(m: Dict[str, float]) -> str:
        return f"ME={m['ME']:+.4f}, STD={m['STD']:.4f}, MAE={m['MAE']:.4f}, N={m['N']}"

    print(f"\n===== {title} =====")
    print("SBP:", fmt(report["sbp"]))
    print("DBP:", fmt(report["dbp"]))


# =============================================================================
# FP16 ONNX conversion
# =============================================================================

def convert_onnx_to_fp16_lazy(
    fp32_onnx_path: str | Path,
    fp16_onnx_path: str | Path,
    *,
    force: bool = False,
    keep_io_types: bool = True,
    disable_shape_infer: bool = False,
    min_positive_val: float = 1e-7,
    max_finite_val: float = 1e4,
    op_block_list: Optional[List[str]] = None,
    node_block_list: Optional[List[str]] = None,
) -> Path:
    """
    Convert FP32 ONNX to FP16 ONNX.

    keep_io_types=True keeps input/output as float32, which is convenient for
    your existing dataloader and CSV-generation pipeline.
    """
    try:
        import onnx
        from onnxconverter_common import float16
    except Exception as e:
        raise ImportError(
            "FP16 conversion requires onnx and onnxconverter-common. "
            "Install with: pip install onnx onnxconverter-common"
        ) from e

    fp32_onnx_path = Path(fp32_onnx_path)
    fp16_onnx_path = Path(fp16_onnx_path)
    fp16_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not fp32_onnx_path.exists():
        raise FileNotFoundError(f"FP32 ONNX does not exist: {fp32_onnx_path}")

    if fp16_onnx_path.exists() and not force:
        print(f"[SKIP] FP16 ONNX exists: {fp16_onnx_path} ({model_size_mb(fp16_onnx_path):.3f} MB)")
        return fp16_onnx_path

    print("[FP16] Converting FP32 ONNX -> FP16")
    print(f"[FP16] Input : {fp32_onnx_path} ({model_size_mb(fp32_onnx_path):.3f} MB)")
    print(f"[FP16] Output: {fp16_onnx_path}")
    print(f"[FP16] keep_io_types={keep_io_types}")

    model = onnx.load(str(fp32_onnx_path))
    model_fp16 = float16.convert_float_to_float16(
        model,
        min_positive_val=min_positive_val,
        max_finite_val=max_finite_val,
        keep_io_types=keep_io_types,
        disable_shape_infer=disable_shape_infer,
        op_block_list=op_block_list,
        node_block_list=node_block_list,
    )
    onnx.save(model_fp16, str(fp16_onnx_path))
    print(f"[DONE] FP16 ONNX: {fp16_onnx_path} ({model_size_mb(fp16_onnx_path):.3f} MB)")
    return fp16_onnx_path


# =============================================================================
# Torch / ONNX raw prediction CSV generation
# =============================================================================

def _parse_model_output(out: Any, output_index: int = 0) -> Any:
    if isinstance(out, dict):
        if "pred" in out:
            return out["pred"]
        if "y" in out:
            return out["y"]
        if "bp" in out:
            return out["bp"]
        # fallback to first value
        return list(out.values())[output_index]
    if isinstance(out, (tuple, list)):
        return out[output_index]
    return out


@torch.no_grad() if torch is not None else (lambda f: f)
def evaluate_torch_to_raw_csv(
    model: Any,
    test_loader: Iterable,
    out_csv: str | Path,
    *,
    device: str = "cuda",
    output_index: int = 0,
    include_alias_cols: bool = True,
) -> pd.DataFrame:
    """
    Run PyTorch model and save a compatible raw prediction CSV.
    """
    if torch is None:
        raise ImportError("PyTorch is required for evaluate_torch_to_raw_csv.")

    model.eval()
    model.to(device)

    all_rows: List[Dict[str, Any]] = []
    all_pred: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    start = 0

    for batch in test_loader:
        if len(batch) == 3:
            x, y, meta = batch
        elif len(batch) == 2:
            x, y = batch
            meta = None
        else:
            raise ValueError("Expected batch = (x, y, meta) or (x, y).")

        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        out = model(x)
        pred = _parse_model_output(out, output_index=output_index)

        pred_np = to_numpy(pred)
        y_np = to_numpy(y)
        if pred_np.ndim > 2:
            pred_np = pred_np.reshape(pred_np.shape[0], -1)
        if pred_np.shape[1] < 2:
            raise ValueError(f"Expected prediction shape [B,2], got {pred_np.shape}")

        bsz = int(pred_np.shape[0])
        all_rows.extend(meta_to_rows(meta, bsz, start))
        all_pred.append(pred_np[:, :2])
        all_true.append(y_np[:, :2])
        start += bsz

    pred_all = np.concatenate(all_pred, axis=0)
    true_all = np.concatenate(all_true, axis=0)

    df = pd.DataFrame(all_rows)
    df = standardize_columns(df)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df["y_true_sbp"] = true_all[:, 0].astype(float)
    df["y_true_dbp"] = true_all[:, 1].astype(float)
    df["y_pred_sbp_raw"] = pred_all[:, 0].astype(float)
    df["y_pred_dbp_raw"] = pred_all[:, 1].astype(float)

    # Keep old-code aliases to avoid mismatch with legacy affine scripts.
    if include_alias_cols:
        df["y_pred_sbp"] = df["y_pred_sbp_raw"]
        df["y_pred_dbp"] = df["y_pred_dbp_raw"]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    report = bp_metrics_from_df(df)
    print_bp_metrics(report, f"PyTorch raw CSV -> {out_csv}")
    return df


def make_ort_session(
    onnx_path: str | Path,
    providers: Optional[List[str]] = None,
    intra_op_num_threads: Optional[int] = None,
):
    try:
        import onnxruntime as ort
    except Exception as e:
        raise ImportError("Install onnxruntime or onnxruntime-gpu to evaluate ONNX models.") from e

    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    so = ort.SessionOptions()
    if intra_op_num_threads is not None:
        so.intra_op_num_threads = int(intra_op_num_threads)

    available = set(ort.get_available_providers())
    selected = [p for p in providers if p in available]
    if not selected:
        selected = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=selected)
    return sess


def evaluate_onnx_to_raw_csv(
    test_loader: Iterable,
    onnx_path: str | Path,
    out_csv: str | Path,
    *,
    input_name: Optional[str] = None,
    output_index: int = 0,
    providers: Optional[List[str]] = None,
    intra_op_num_threads: Optional[int] = None,
    include_alias_cols: bool = True,
    print_io: bool = True,
) -> pd.DataFrame:
    """
    Run ONNX model and save a compatible raw prediction CSV.

    This is the key function for checking FP16 classification drop.
    """
    sess = make_ort_session(onnx_path, providers=providers, intra_op_num_threads=intra_op_num_threads)

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    if input_name is None:
        input_name = inputs[0].name

    if print_io:
        print(f"\n[ONNX] {onnx_path}")
        print(f"[ONNX] Providers: {sess.get_providers()}")
        print("[ONNX] Inputs:", [(i.name, i.shape, i.type) for i in inputs])
        print("[ONNX] Outputs:", [(o.name, o.shape, o.type) for o in outputs])
        print(f"[ONNX] Using input_name={input_name}, output_index={output_index}")

    all_rows: List[Dict[str, Any]] = []
    all_pred: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    start = 0

    for batch in test_loader:
        if len(batch) == 3:
            x, y, meta = batch
        elif len(batch) == 2:
            x, y = batch
            meta = None
        else:
            raise ValueError("Expected batch = (x, y, meta) or (x, y).")

        x_np = to_numpy(x).astype(np.float32, copy=False)
        y_np = to_numpy(y).astype(np.float32, copy=False)

        ort_out = sess.run(None, {input_name: x_np})
        pred_np = np.asarray(ort_out[output_index])
        if pred_np.ndim > 2:
            pred_np = pred_np.reshape(pred_np.shape[0], -1)
        if pred_np.shape[1] < 2:
            raise ValueError(f"Expected ONNX prediction shape [B,2], got {pred_np.shape}")

        bsz = int(pred_np.shape[0])
        all_rows.extend(meta_to_rows(meta, bsz, start))
        all_pred.append(pred_np[:, :2].astype(np.float64))
        all_true.append(y_np[:, :2].astype(np.float64))
        start += bsz

    pred_all = np.concatenate(all_pred, axis=0)
    true_all = np.concatenate(all_true, axis=0)

    df = pd.DataFrame(all_rows)
    df = standardize_columns(df)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df["y_true_sbp"] = true_all[:, 0].astype(float)
    df["y_true_dbp"] = true_all[:, 1].astype(float)
    df["y_pred_sbp_raw"] = pred_all[:, 0].astype(float)
    df["y_pred_dbp_raw"] = pred_all[:, 1].astype(float)

    if include_alias_cols:
        df["y_pred_sbp"] = df["y_pred_sbp_raw"]
        df["y_pred_dbp"] = df["y_pred_dbp_raw"]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    report = bp_metrics_from_df(df)
    print_bp_metrics(report, f"ONNX raw CSV -> {out_csv}")
    return df


# =============================================================================
# CSV comparison
# =============================================================================

def compare_raw_prediction_csvs(
    ref_csv: str | Path,
    test_csv: str | Path,
    *,
    ref_name: str = "ref",
    test_name: str = "test",
    pred_sbp_col: str = "y_pred_sbp_raw",
    pred_dbp_col: str = "y_pred_dbp_raw",
    out_json: Optional[str | Path] = None,
) -> Dict[str, Any]:
    ref = pd.read_csv(ref_csv)
    tst = pd.read_csv(test_csv)

    if len(ref) != len(tst):
        raise ValueError(f"CSV row mismatch: {len(ref)} vs {len(tst)}")

    out: Dict[str, Any] = {
        "ref_csv": str(ref_csv),
        "test_csv": str(test_csv),
        "n_rows": int(len(ref)),
    }

    for target, col in [("sbp", pred_sbp_col), ("dbp", pred_dbp_col)]:
        a = pd.to_numeric(ref[col], errors="coerce").to_numpy(dtype=np.float64)
        b = pd.to_numeric(tst[col], errors="coerce").to_numpy(dtype=np.float64)
        diff = b - a
        out[target] = {
            "mean_abs_diff": float(np.nanmean(np.abs(diff))),
            "max_abs_diff": float(np.nanmax(np.abs(diff))),
            "mean_diff": float(np.nanmean(diff)),
            "std_diff": float(np.nanstd(diff, ddof=1)) if np.sum(np.isfinite(diff)) > 1 else np.nan,
            "p95_abs_diff": float(np.nanpercentile(np.abs(diff), 95)),
            "p99_abs_diff": float(np.nanpercentile(np.abs(diff), 99)),
        }

    print(f"\n===== Raw prediction diff: {ref_name} vs {test_name} =====")
    for target in ["sbp", "dbp"]:
        m = out[target]
        print(
            f"{target.upper()}: mean_abs={m['mean_abs_diff']:.6f}, "
            f"max_abs={m['max_abs_diff']:.6f}, p95_abs={m['p95_abs_diff']:.6f}, "
            f"mean={m['mean_diff']:+.6f}, std={m['std_diff']:.6f}"
        )

    if out_json is not None:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    return out


# =============================================================================
# One-call helper
# =============================================================================

def create_fp32_fp16_raw_csvs(
    test_loader: Iterable,
    *,
    fp32_onnx_path: str | Path,
    fp16_onnx_path: str | Path,
    out_dir: str | Path = "onnx_fp16_product_eval",
    convert_fp16: bool = True,
    force_fp16: bool = False,
    providers: Optional[List[str]] = None,
    input_name: Optional[str] = None,
    output_index: int = 0,
    keep_io_types: bool = True,
) -> Dict[str, Any]:
    """
    If you already have FP32 ONNX, this creates:
        raw_predictions_onnx_fp32.csv
        raw_predictions_onnx_fp16.csv
        onnx_fp32_vs_fp16_diff.json
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_onnx_path = Path(fp32_onnx_path)
    fp16_onnx_path = Path(fp16_onnx_path)

    if convert_fp16:
        convert_onnx_to_fp16_lazy(
            fp32_onnx_path=fp32_onnx_path,
            fp16_onnx_path=fp16_onnx_path,
            force=force_fp16,
            keep_io_types=keep_io_types,
        )

    csv_fp32 = out_dir / "raw_predictions_onnx_fp32.csv"
    csv_fp16 = out_dir / "raw_predictions_onnx_fp16.csv"

    evaluate_onnx_to_raw_csv(
        test_loader=test_loader,
        onnx_path=fp32_onnx_path,
        out_csv=csv_fp32,
        input_name=input_name,
        output_index=output_index,
        providers=providers,
    )
    evaluate_onnx_to_raw_csv(
        test_loader=test_loader,
        onnx_path=fp16_onnx_path,
        out_csv=csv_fp16,
        input_name=input_name,
        output_index=output_index,
        providers=providers,
    )

    diff = compare_raw_prediction_csvs(
        csv_fp32,
        csv_fp16,
        ref_name="onnx_fp32",
        test_name="onnx_fp16",
        out_json=out_dir / "onnx_fp32_vs_fp16_diff.json",
    )

    summary = {
        "paths": {
            "out_dir": str(out_dir),
            "fp32_onnx": str(fp32_onnx_path),
            "fp16_onnx": str(fp16_onnx_path),
            "raw_csv_fp32": str(csv_fp32),
            "raw_csv_fp16": str(csv_fp16),
        },
        "sizes_mb": {
            "fp32_onnx": model_size_mb(fp32_onnx_path),
            "fp16_onnx": model_size_mb(fp16_onnx_path),
        },
        "diff": diff,
    }
    with open(out_dir / "product_fp16_csv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== Product FP16 CSV summary =====")
    print(json.dumps(summary["sizes_mb"], indent=2))
    print(f"Saved summary: {out_dir / 'product_fp16_csv_summary.json'}")
    return summary
