from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mcma_torch.unified.data import Sample


DEFAULT_TARGET_CHANNELS = [f"CH{i}" for i in range(1, 9)]


@dataclass
class PreparedUpperArmRecord:
    path: Path
    timestamps_ms: np.ndarray
    input_signal: np.ndarray
    target_signals: np.ndarray
    sampling_rate_hz: float


def _as_channel_list(value: Iterable[str] | str | None) -> list[str]:
    if value is None:
        return list(DEFAULT_TARGET_CHANNELS)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value]


def _is_generated_sidecar_csv(path: Path) -> bool:
    stem = path.stem
    return stem.endswith("_quality_report") or stem.endswith("_CH1-8_rpeaks") or stem.endswith("_CH20_rpeaks")


def _discover_files(csv_dir: str | Path, file_glob: str, max_files: int | None = None) -> list[Path]:
    root = Path(csv_dir)
    files = sorted(path for path in root.glob(file_glob) if not _is_generated_sidecar_csv(path))
    if max_files is not None:
        files = files[:max_files]
    return files


def _split_files(
    files: list[Path],
    split: str | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> list[Path]:
    if split is None:
        return files

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    shuffled = list(files)
    rng = random.Random(split_seed)
    rng.shuffle(shuffled)

    train_end = int(round(len(shuffled) * train_ratio))
    val_end = train_end + int(round(len(shuffled) * val_ratio))
    train_files = shuffled[:train_end]
    val_files = shuffled[train_end:val_end]
    test_files = shuffled[val_end:]

    selected = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }[split]
    return sorted(selected)


def _estimate_sampling_rate_hz(timestamps_ms: np.ndarray, fallback_fs: float) -> float:
    if timestamps_ms.size < 2:
        return fallback_fs
    diffs = np.diff(timestamps_ms.astype(np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return fallback_fs
    median_diff_ms = float(np.median(diffs))
    if median_diff_ms <= 0:
        return fallback_fs
    return 1000.0 / median_diff_ms


def _rolling_mean(signal: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return signal.astype(np.float32, copy=False)
    series = pd.Series(signal)
    return series.rolling(window=window_samples, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)


def _filter_signal(signal: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    if sampling_rate_hz <= 0:
        return (signal - signal.mean()).astype(np.float32, copy=False)

    highpass_window = max(3, int(round(sampling_rate_hz * 0.6)))
    lowpass_window = max(1, int(round(sampling_rate_hz * 0.03)))
    baseline = _rolling_mean(signal, highpass_window)
    highpassed = signal - baseline
    filtered = _rolling_mean(highpassed, lowpass_window)
    return filtered.astype(np.float32, copy=False)


def _normalize_signal(signal: np.ndarray, normalize_mode: str) -> np.ndarray:
    if normalize_mode == "none":
        return signal.astype(np.float32, copy=False)
    if normalize_mode != "zscore":
        raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")

    centered = signal - float(np.median(signal))
    scale = float(np.std(centered))
    if scale <= 1e-8:
        scale = float(np.quantile(np.abs(centered), 0.9))
    if scale <= 1e-8:
        scale = 1.0
    return (centered / scale).astype(np.float32, copy=False)


def _pad_1d(signal: np.ndarray, target_len: int, pad_mode: str) -> np.ndarray:
    if signal.shape[0] >= target_len:
        return signal[:target_len]
    pad_len = target_len - signal.shape[0]
    if pad_mode == "zero":
        pad = np.zeros((pad_len,), dtype=signal.dtype)
    elif pad_mode == "edge":
        fill_value = 0.0 if signal.shape[0] == 0 else float(signal[-1])
        pad = np.full((pad_len,), fill_value, dtype=signal.dtype)
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}")
    return np.concatenate([signal, pad], axis=0)


def _compute_window_starts(length: int, signal_length: int, stride: int, window_policy: str) -> list[int]:
    if length <= 0:
        return [0]
    if stride <= 0:
        raise ValueError("window_stride must be positive")
    if signal_length <= 0:
        raise ValueError("signal_length must be positive")

    if length <= signal_length:
        return [0]

    starts = list(range(0, length - signal_length + 1, stride))
    if window_policy == "drop":
        return starts
    if window_policy == "pad":
        tail_start = max(0, length - signal_length)
        if starts[-1] != tail_start:
            starts.append(tail_start)
        return starts
    raise ValueError(f"Unsupported window_policy: {window_policy}")


def prepare_upperarm_record(
    path: str | Path,
    input_channel: str,
    target_channels: Iterable[str] | str | None,
    apply_filter: bool,
    normalize_mode: str,
    fallback_fs: float,
) -> PreparedUpperArmRecord:
    path = Path(path)
    target_channels = _as_channel_list(target_channels)
    required_columns = ["timestamp_ms", input_channel, *target_channels]

    df = pd.read_csv(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    df = df[required_columns].copy()
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required_columns)
    if df.empty:
        raise ValueError(f"{path.name} has no valid numeric rows after cleaning")

    df = df.sort_values("timestamp_ms").drop_duplicates(subset=["timestamp_ms"], keep="first")
    timestamps_ms = df["timestamp_ms"].to_numpy(dtype=np.float64)
    sampling_rate_hz = _estimate_sampling_rate_hz(timestamps_ms, fallback_fs=fallback_fs)

    processed: dict[str, np.ndarray] = {}
    for channel in [input_channel, *target_channels]:
        signal = df[channel].to_numpy(dtype=np.float32)
        if apply_filter:
            signal = _filter_signal(signal, sampling_rate_hz=sampling_rate_hz)
        signal = _normalize_signal(signal, normalize_mode=normalize_mode)
        processed[channel] = signal

    input_signal = processed[input_channel]
    target_signals = np.stack([processed[channel] for channel in target_channels], axis=0)

    return PreparedUpperArmRecord(
        path=path,
        timestamps_ms=timestamps_ms,
        input_signal=input_signal,
        target_signals=target_signals,
        sampling_rate_hz=sampling_rate_hz,
    )


def load_upperarm_records(
    csv_dir: str | Path,
    file_glob: str = "emg_data_*.csv",
    input_channel: str = "CH20",
    target_channels: Iterable[str] | str | None = None,
    apply_filter: bool = True,
    normalize_mode: str = "zscore",
    fallback_fs: float = 250.0,
    split: str | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    split_seed: int = 42,
    max_files: int | None = None,
) -> list[PreparedUpperArmRecord]:
    files = _discover_files(csv_dir=csv_dir, file_glob=file_glob, max_files=max_files)
    if not files:
        raise RuntimeError(f"No CSV files matching {file_glob} found in {csv_dir}")
    files = _split_files(
        files,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
    )
    if not files:
        raise RuntimeError(f"No files assigned to split {split}")

    return [
        prepare_upperarm_record(
            path=path,
            input_channel=input_channel,
            target_channels=target_channels,
            apply_filter=apply_filter,
            normalize_mode=normalize_mode,
            fallback_fs=fallback_fs,
        )
        for path in files
    ]


class UpperArmCSVWindowDataset(Dataset):
    def __init__(
        self,
        csv_dir: str | Path,
        split: str | None,
        signal_length: int,
        window_stride: int,
        input_channel: str = "CH20",
        target_channels: Iterable[str] | str | None = None,
        file_glob: str = "emg_data_*.csv",
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        test_ratio: float = 0.0,
        split_seed: int = 42,
        apply_filter: bool = True,
        normalize_mode: str = "zscore",
        fallback_fs: float = 250.0,
        window_policy: str = "pad",
        pad_mode: str = "zero",
        max_files: int | None = None,
    ) -> None:
        self.signal_length = int(signal_length)
        self.window_stride = int(window_stride)
        self.window_policy = window_policy
        self.pad_mode = pad_mode
        self.input_channel = input_channel
        self.target_channels = _as_channel_list(target_channels)
        self.records = load_upperarm_records(
            csv_dir=csv_dir,
            file_glob=file_glob,
            input_channel=input_channel,
            target_channels=self.target_channels,
            apply_filter=apply_filter,
            normalize_mode=normalize_mode,
            fallback_fs=fallback_fs,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
            max_files=max_files,
        )

        self.items: list[tuple[int, int]] = []
        for record_idx, record in enumerate(self.records):
            starts = _compute_window_starts(
                length=record.input_signal.shape[0],
                signal_length=self.signal_length,
                stride=self.window_stride,
                window_policy=self.window_policy,
            )
            for start in starts:
                self.items.append((record_idx, start))

        if not self.items:
            raise RuntimeError("No windows generated from the selected upper-arm CSV files")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Sample:
        record_idx, start = self.items[index]
        record = self.records[record_idx]
        stop = start + self.signal_length

        x_single = _pad_1d(record.input_signal[start:stop], self.signal_length, self.pad_mode)
        targets = np.stack(
            [
                _pad_1d(record.target_signals[channel_idx, start:stop], self.signal_length, self.pad_mode)
                for channel_idx in range(record.target_signals.shape[0])
            ],
            axis=0,
        )

        return Sample(
            x_single=torch.from_numpy(x_single[None, :]),
            y_12=torch.from_numpy(targets),
            y_label=None,
            has_label=False,
        )

    @property
    def lead_names(self) -> list[str]:
        return list(self.target_channels)

    @property
    def num_target_channels(self) -> int:
        return len(self.target_channels)
