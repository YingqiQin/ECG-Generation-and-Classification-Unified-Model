from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mcma_torch.data.ui_beat_quality import ensure_ui_beat_quality_segments


DEFAULT_TARGET_CHANNELS = [f"CH{i}" for i in range(1, 9)]
UPPERARM_DATASET_TYPES = {"upperarm_csv", "upperarm_segmented_npz"}
DEFAULT_SEGMENT_GROUP_REGEX = r"^(?P<record>.+)_\d+s$"
DEFAULT_SEGMENT_OFFSET_REGEX = r"_(?P<offset_seconds>\d+)s$"


@dataclass
class PreparedUpperArmRecord:
    path: Path
    timestamps_ms: np.ndarray
    input_signal: np.ndarray
    target_signals: np.ndarray
    original_sampling_rate_hz: float
    sampling_rate_hz: float


@dataclass(frozen=True)
class UpperArmSourceUnit:
    path: Path
    source_paths: tuple[Path, ...]


def is_upperarm_dataset_type(dataset_type: str | None) -> bool:
    return str(dataset_type or "ptbxl") in UPPERARM_DATASET_TYPES


def _as_channel_list(value: Iterable[str] | str | None) -> list[str]:
    if value is None:
        return list(DEFAULT_TARGET_CHANNELS)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value]


def _normalize_name_list(value: Iterable[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _as_path_list(value: Iterable[str | Path] | str | Path | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


def _normalize_quality_preprocess_mode(value: str | None) -> str:
    return str(value or "none").strip().lower()


def _is_generated_sidecar_csv(path: Path) -> bool:
    stem = path.stem
    return stem.endswith("_quality_report") or stem.endswith("_CH1-8_rpeaks") or stem.endswith("_CH20_rpeaks")


def _discover_files(csv_dir: str | Path, file_glob: str, max_files: int | None = None) -> list[Path]:
    files = sorted(
        path
        for path in Path(csv_dir).glob(file_glob)
        if not _is_generated_sidecar_csv(path)
    )
    if max_files is not None:
        files = files[:max_files]
    return files


def _logical_record_name_from_segment_path(path: Path, segment_group_regex: str) -> str:
    match = re.match(segment_group_regex, path.stem)
    if match and match.groupdict().get("record"):
        return str(match.group("record"))
    return path.stem


def _segment_offset_seconds(path: Path, segment_offset_regex: str) -> float:
    match = re.search(segment_offset_regex, path.stem)
    if not match:
        return 0.0
    group_name = "offset_seconds" if "offset_seconds" in match.groupdict() else None
    raw = match.group(group_name) if group_name else match.group(1)
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _group_segmented_npz_files(
    files: list[Path],
    segment_group_regex: str,
    segment_offset_regex: str,
) -> list[UpperArmSourceUnit]:
    grouped: dict[str, list[Path]] = {}
    for path in files:
        grouped.setdefault(_logical_record_name_from_segment_path(path, segment_group_regex), []).append(path)

    units: list[UpperArmSourceUnit] = []
    for group_name, members in sorted(grouped.items()):
        ordered = tuple(sorted(members, key=lambda item: (_segment_offset_seconds(item, segment_offset_regex), item.name)))
        logical_path = ordered[0].parent / f"{group_name}__grouped.npz"
        units.append(UpperArmSourceUnit(path=logical_path, source_paths=ordered))
    return units


def discover_upperarm_source_units(
    csv_dir: str | Path,
    file_glob: str,
    dataset_type: str = "upperarm_csv",
    max_files: int | None = None,
    segment_group_regex: str = DEFAULT_SEGMENT_GROUP_REGEX,
    segment_offset_regex: str = DEFAULT_SEGMENT_OFFSET_REGEX,
    quality_preprocess_mode: str = "none",
    quality_preprocess_config: dict[str, Any] | None = None,
) -> list[UpperArmSourceUnit]:
    files = _discover_files(csv_dir=csv_dir, file_glob=file_glob, max_files=max_files)
    if not files:
        return []
    preprocess_mode = _normalize_quality_preprocess_mode(quality_preprocess_mode)
    if dataset_type == "upperarm_csv" and preprocess_mode == "ui_beat":
        cfg = quality_preprocess_config or {}
        units: list[UpperArmSourceUnit] = []
        for path in files:
            logical_path, segment_paths = ensure_ui_beat_quality_segments(csv_path=path, cfg=cfg)
            if not segment_paths:
                print(f"Skipping {path.name}: UI_Beat produced no quality segments")
                continue
            units.append(UpperArmSourceUnit(path=logical_path, source_paths=segment_paths))
        return units
    if dataset_type == "upperarm_csv":
        return [UpperArmSourceUnit(path=path, source_paths=(path,)) for path in files]
    if dataset_type == "upperarm_segmented_npz":
        return _group_segmented_npz_files(
            files=files,
            segment_group_regex=segment_group_regex,
            segment_offset_regex=segment_offset_regex,
        )
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def discover_upperarm_source_units_from_dirs(
    csv_dirs: Iterable[str | Path] | str | Path,
    file_glob: str,
    dataset_type: str = "upperarm_csv",
    max_files: int | None = None,
    segment_group_regex: str = DEFAULT_SEGMENT_GROUP_REGEX,
    segment_offset_regex: str = DEFAULT_SEGMENT_OFFSET_REGEX,
    quality_preprocess_mode: str = "none",
    quality_preprocess_config: dict[str, Any] | None = None,
) -> list[UpperArmSourceUnit]:
    units: list[UpperArmSourceUnit] = []
    for root in _as_path_list(csv_dirs):
        root_units = discover_upperarm_source_units(
            csv_dir=root,
            file_glob=file_glob,
            dataset_type=dataset_type,
            max_files=None,
            segment_group_regex=segment_group_regex,
            segment_offset_regex=segment_offset_regex,
            quality_preprocess_mode=quality_preprocess_mode,
            quality_preprocess_config=quality_preprocess_config,
        )
        units.extend(root_units)
    units = sorted(units, key=lambda unit: str(unit.path))
    if max_files is not None:
        units = units[:max_files]
    return units


def _split_units(
    units: list[UpperArmSourceUnit],
    split: str | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    split_files: dict[str, Iterable[str] | str] | None = None,
) -> list[UpperArmSourceUnit]:
    if split_files:
        if split is None:
            return units
        requested = set(_normalize_name_list(split_files.get(split)))
        if not requested:
            raise RuntimeError(f"No files specified for split {split}")
        unit_keys = {
            unit: {
                unit.path.name,
                str(unit.path),
                *[source.name for source in unit.source_paths],
                *[str(source) for source in unit.source_paths],
            }
            for unit in units
        }
        selected = [unit for unit, keys in unit_keys.items() if keys & requested]
        matched = {key for keys in unit_keys.values() for key in keys & requested}
        missing = sorted(requested - matched)
        if missing:
            raise RuntimeError(f"Requested files for split {split} not found: {missing}")
        return sorted(selected, key=lambda unit: unit.path.name)

    if split is None:
        return units
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("At least one split ratio must be positive")
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    shuffled = list(units)
    random.Random(split_seed).shuffle(shuffled)

    train_end = int(round(len(shuffled) * train_ratio))
    val_end = train_end + int(round(len(shuffled) * val_ratio))
    parts = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }
    return sorted(parts[split], key=lambda unit: unit.path.name)


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


def _resample_signal(
    signal: np.ndarray,
    old_time_s: np.ndarray,
    new_time_s: np.ndarray,
) -> np.ndarray:
    return np.interp(new_time_s, old_time_s, signal).astype(np.float32, copy=False)


def _resample_record(
    timestamps_ms: np.ndarray,
    signal_map: dict[str, np.ndarray],
    target_fs: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if timestamps_ms.size < 2 or target_fs <= 0:
        return timestamps_ms.astype(np.float64, copy=False), {
            key: value.astype(np.float32, copy=False) for key, value in signal_map.items()
        }

    old_time_s = (timestamps_ms.astype(np.float64) - float(timestamps_ms[0])) / 1000.0
    duration_s = float(old_time_s[-1])
    if duration_s <= 0:
        return timestamps_ms.astype(np.float64, copy=False), {
            key: value.astype(np.float32, copy=False) for key, value in signal_map.items()
        }

    step_s = 1.0 / float(target_fs)
    new_time_s = np.arange(0.0, duration_s + 0.5 * step_s, step_s, dtype=np.float64)
    if new_time_s.size < 2:
        return timestamps_ms.astype(np.float64, copy=False), {
            key: value.astype(np.float32, copy=False) for key, value in signal_map.items()
        }

    new_timestamps_ms = float(timestamps_ms[0]) + new_time_s * 1000.0
    resampled = {
        key: _resample_signal(signal=value, old_time_s=old_time_s, new_time_s=new_time_s)
        for key, value in signal_map.items()
    }
    return new_timestamps_ms, resampled


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


def compute_window_starts(length: int, signal_length: int, stride: int, window_policy: str) -> list[int]:
    if length <= 0:
        return [0]
    if stride <= 0:
        raise ValueError("segment_stride must be positive")
    if signal_length <= 0:
        raise ValueError("segment_length must be positive")
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


def aggregate_window_predictions(
    predictions: np.ndarray,
    starts: list[int],
    total_length: int,
) -> np.ndarray:
    if predictions.ndim != 3:
        raise ValueError("predictions must have shape (num_windows, num_channels, signal_length)")
    num_channels = predictions.shape[1]
    aggregated = np.zeros((num_channels, total_length), dtype=np.float32)
    counts = np.zeros((total_length,), dtype=np.float32)

    for pred, start in zip(predictions, starts):
        valid_len = min(pred.shape[1], total_length - start)
        if valid_len <= 0:
            continue
        aggregated[:, start:start + valid_len] += pred[:, :valid_len]
        counts[start:start + valid_len] += 1.0

    counts[counts == 0] = 1.0
    aggregated /= counts[None, :]
    return aggregated


def _coerce_npz_channel_names(raw_value: np.ndarray) -> list[str]:
    values = np.asarray(raw_value)
    if values.ndim == 0:
        return [str(values.item())]
    return [str(item.decode("utf-8") if isinstance(item, bytes) else item) for item in values.tolist()]


def _build_timestamps_from_fs(length: int, sampling_rate_hz: float, start_time_ms: float = 0.0) -> np.ndarray:
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive when timestamps are missing")
    return start_time_ms + np.arange(length, dtype=np.float64) * (1000.0 / float(sampling_rate_hz))


def _extract_npz_signal_map(
    npz_data: np.lib.npyio.NpzFile,
    input_channel: str,
    target_channels: list[str],
    signal_matrix_key: str | None,
    channel_names_key: str | None,
) -> dict[str, np.ndarray]:
    required_channels = [input_channel, *target_channels]
    if all(channel in npz_data.files for channel in required_channels):
        return {
            channel: np.asarray(npz_data[channel], dtype=np.float32).reshape(-1)
            for channel in required_channels
        }

    matrix_key = signal_matrix_key or ("signals" if "signals" in npz_data.files else None)
    names_key = channel_names_key or ("channel_names" if "channel_names" in npz_data.files else None)
    if matrix_key is None or names_key is None:
        missing = [channel for channel in required_channels if channel not in npz_data.files]
        raise ValueError(
            "NPZ segment is missing direct per-channel arrays and no matrix/channel_names keys were provided. "
            f"Missing channels: {missing}"
        )

    matrix = np.asarray(npz_data[matrix_key], dtype=np.float32)
    channel_names = _coerce_npz_channel_names(npz_data[names_key])
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D signal matrix in {matrix_key}, got shape {matrix.shape}")

    if matrix.shape[0] == len(channel_names) and matrix.shape[1] != len(channel_names):
        matrix = matrix.T
    if matrix.shape[1] != len(channel_names):
        raise ValueError(
            f"Signal matrix shape {matrix.shape} is incompatible with {len(channel_names)} channel names"
        )

    index = {name: idx for idx, name in enumerate(channel_names)}
    missing = [channel for channel in required_channels if channel not in index]
    if missing:
        raise ValueError(f"NPZ segment is missing requested channels: {missing}")

    return {
        channel: matrix[:, index[channel]].astype(np.float32, copy=False).reshape(-1)
        for channel in required_channels
    }


def _load_npz_segment(
    path: Path,
    input_channel: str,
    target_channels: list[str],
    fallback_fs: float,
    timestamp_key: str,
    sampling_rate_key: str | None,
    start_time_key: str | None,
    start_time_scale: float,
    signal_matrix_key: str | None,
    channel_names_key: str | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=True) as npz_data:
        signal_map = _extract_npz_signal_map(
            npz_data=npz_data,
            input_channel=input_channel,
            target_channels=target_channels,
            signal_matrix_key=signal_matrix_key,
            channel_names_key=channel_names_key,
        )

        if timestamp_key in npz_data.files:
            timestamps_ms = np.asarray(npz_data[timestamp_key], dtype=np.float64).reshape(-1)
        else:
            sampling_rate_hz = fallback_fs
            if sampling_rate_key and sampling_rate_key in npz_data.files:
                sampling_rate_hz = float(np.asarray(npz_data[sampling_rate_key]).reshape(-1)[0])
            start_time_ms = 0.0
            if start_time_key and start_time_key in npz_data.files:
                start_time_ms = float(np.asarray(npz_data[start_time_key]).reshape(-1)[0]) * float(start_time_scale)
            length = next(iter(signal_map.values())).shape[0]
            timestamps_ms = _build_timestamps_from_fs(
                length=length,
                sampling_rate_hz=sampling_rate_hz,
                start_time_ms=start_time_ms,
            )

    expected_len = timestamps_ms.shape[0]
    for channel_name, signal in signal_map.items():
        if signal.shape[0] != expected_len:
            raise ValueError(
                f"{path.name} has inconsistent lengths: timestamps={expected_len}, {channel_name}={signal.shape[0]}"
            )
    return timestamps_ms, signal_map


def _prepare_record_from_raw_signals(
    path: Path,
    timestamps_ms: np.ndarray,
    raw_signals: dict[str, np.ndarray],
    input_channel: str,
    target_channels: list[str],
    apply_filter: bool,
    normalize_mode: str,
    fallback_fs: float,
    target_fs: float | None,
) -> PreparedUpperArmRecord:
    timestamps_ms = np.asarray(timestamps_ms, dtype=np.float64).reshape(-1)
    if timestamps_ms.size == 0:
        raise ValueError(f"{path.name} has no timestamp samples")

    original_sampling_rate_hz = _estimate_sampling_rate_hz(timestamps_ms, fallback_fs=fallback_fs)
    if target_fs is not None and target_fs > 0:
        timestamps_ms, raw_signals = _resample_record(
            timestamps_ms=timestamps_ms,
            signal_map=raw_signals,
            target_fs=float(target_fs),
        )
        sampling_rate_hz = float(target_fs)
    else:
        sampling_rate_hz = original_sampling_rate_hz

    processed: dict[str, np.ndarray] = {}
    for channel in [input_channel, *target_channels]:
        signal = raw_signals[channel]
        if apply_filter:
            signal = _filter_signal(signal, sampling_rate_hz=sampling_rate_hz)
        signal = _normalize_signal(signal, normalize_mode=normalize_mode)
        processed[channel] = signal

    return PreparedUpperArmRecord(
        path=path,
        timestamps_ms=timestamps_ms,
        input_signal=processed[input_channel],
        target_signals=np.stack([processed[channel] for channel in target_channels], axis=0),
        original_sampling_rate_hz=original_sampling_rate_hz,
        sampling_rate_hz=sampling_rate_hz,
    )


def prepare_upperarm_record(
    path: str | Path,
    input_channel: str,
    target_channels: Iterable[str] | str | None,
    apply_filter: bool,
    normalize_mode: str,
    fallback_fs: float,
    target_fs: float | None = None,
) -> PreparedUpperArmRecord:
    path = Path(path)
    target_channels_list = _as_channel_list(target_channels)
    required_columns = ["timestamp_ms", input_channel, *target_channels_list]

    df = pd.read_csv(path)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    df = df[required_columns].copy()
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required_columns)
    if df.empty:
        raise ValueError(f"{path.name} has no valid rows after cleaning")

    df = df.sort_values("timestamp_ms").drop_duplicates(subset=["timestamp_ms"], keep="first")
    timestamps_ms = df["timestamp_ms"].to_numpy(dtype=np.float64)
    raw_signals = {
        channel: df[channel].to_numpy(dtype=np.float32)
        for channel in [input_channel, *target_channels_list]
    }
    return _prepare_record_from_raw_signals(
        path=path,
        timestamps_ms=timestamps_ms,
        raw_signals=raw_signals,
        input_channel=input_channel,
        target_channels=target_channels_list,
        apply_filter=apply_filter,
        normalize_mode=normalize_mode,
        fallback_fs=fallback_fs,
        target_fs=target_fs,
    )


def prepare_upperarm_record_from_unit(
    unit: UpperArmSourceUnit,
    dataset_type: str,
    input_channel: str,
    target_channels: Iterable[str] | str | None,
    apply_filter: bool,
    normalize_mode: str,
    fallback_fs: float,
    target_fs: float | None = None,
    npz_timestamp_key: str = "timestamp_ms",
    npz_sampling_rate_key: str | None = "sampling_rate_hz",
    npz_start_time_key: str | None = "start_time_ms",
    npz_start_time_scale: float = 1.0,
    npz_signal_matrix_key: str | None = None,
    npz_channel_names_key: str | None = None,
    quality_preprocess_mode: str = "none",
) -> PreparedUpperArmRecord:
    target_channels_list = _as_channel_list(target_channels)
    preprocess_mode = _normalize_quality_preprocess_mode(quality_preprocess_mode)
    effective_dataset_type = dataset_type
    if (
        dataset_type == "upperarm_csv"
        and preprocess_mode == "ui_beat"
        and unit.source_paths
        and all(path.suffix.lower() == ".npz" for path in unit.source_paths)
    ):
        effective_dataset_type = "upperarm_segmented_npz"
        npz_sampling_rate_key = "fs"
        npz_start_time_key = "start_s"
        npz_start_time_scale = 1000.0

    if effective_dataset_type == "upperarm_csv":
        return prepare_upperarm_record(
            path=unit.source_paths[0],
            input_channel=input_channel,
            target_channels=target_channels_list,
            apply_filter=apply_filter,
            normalize_mode=normalize_mode,
            fallback_fs=fallback_fs,
            target_fs=target_fs,
        )
    if effective_dataset_type != "upperarm_segmented_npz":
        raise ValueError(f"Unsupported dataset_type: {effective_dataset_type}")

    timestamp_parts: list[np.ndarray] = []
    signal_parts: dict[str, list[np.ndarray]] = {
        channel: [] for channel in [input_channel, *target_channels_list]
    }
    for source_path in unit.source_paths:
        timestamps_ms, signal_map = _load_npz_segment(
            path=source_path,
            input_channel=input_channel,
            target_channels=target_channels_list,
            fallback_fs=fallback_fs,
            timestamp_key=npz_timestamp_key,
            sampling_rate_key=npz_sampling_rate_key,
            start_time_key=npz_start_time_key,
            start_time_scale=npz_start_time_scale,
            signal_matrix_key=npz_signal_matrix_key,
            channel_names_key=npz_channel_names_key,
        )
        timestamp_parts.append(timestamps_ms)
        for channel in signal_parts:
            signal_parts[channel].append(signal_map[channel])

    timestamps_ms = np.concatenate(timestamp_parts, axis=0)
    raw_signals = {
        channel: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        for channel, parts in signal_parts.items()
    }
    order = np.argsort(timestamps_ms, kind="mergesort")
    timestamps_ms = timestamps_ms[order]
    raw_signals = {channel: values[order] for channel, values in raw_signals.items()}
    if timestamps_ms.size == 0:
        raise ValueError(f"{unit.path.name} has no samples after concatenating segments")
    dedup_mask = np.ones(timestamps_ms.shape[0], dtype=bool)
    dedup_mask[1:] = np.diff(timestamps_ms) > 0
    timestamps_ms = timestamps_ms[dedup_mask]
    raw_signals = {channel: values[dedup_mask] for channel, values in raw_signals.items()}

    return _prepare_record_from_raw_signals(
        path=unit.path,
        timestamps_ms=timestamps_ms,
        raw_signals=raw_signals,
        input_channel=input_channel,
        target_channels=target_channels_list,
        apply_filter=apply_filter,
        normalize_mode=normalize_mode,
        fallback_fs=fallback_fs,
        target_fs=target_fs,
    )


def load_upperarm_records(
    csv_dir: str | Path,
    file_glob: str = "emg_data_*.csv",
    input_channel: str = "CH20",
    target_channels: Iterable[str] | str | None = None,
    apply_filter: bool = True,
    normalize_mode: str = "zscore",
    fallback_fs: float = 250.0,
    target_fs: float | None = None,
    split: str | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    split_seed: int = 42,
    max_files: int | None = None,
    split_files: dict[str, Iterable[str] | str] | None = None,
    dataset_type: str = "upperarm_csv",
    segment_group_regex: str = DEFAULT_SEGMENT_GROUP_REGEX,
    segment_offset_regex: str = DEFAULT_SEGMENT_OFFSET_REGEX,
    npz_timestamp_key: str = "timestamp_ms",
    npz_sampling_rate_key: str | None = "sampling_rate_hz",
    npz_start_time_key: str | None = "start_time_ms",
    npz_start_time_scale: float = 1.0,
    npz_signal_matrix_key: str | None = None,
    npz_channel_names_key: str | None = None,
    csv_dirs: Iterable[str | Path] | str | Path | None = None,
    quality_preprocess_mode: str = "none",
    quality_preprocess_config: dict[str, Any] | None = None,
) -> list[PreparedUpperArmRecord]:
    roots = csv_dirs or csv_dir
    units = discover_upperarm_source_units_from_dirs(
        csv_dirs=roots,
        file_glob=file_glob,
        dataset_type=dataset_type,
        max_files=max_files,
        segment_group_regex=segment_group_regex,
        segment_offset_regex=segment_offset_regex,
        quality_preprocess_mode=quality_preprocess_mode,
        quality_preprocess_config=quality_preprocess_config,
    )
    if not units:
        raise RuntimeError(f"No usable files matching {file_glob} found in {csv_dir}")
    units = _split_units(
        units,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        split_files=split_files,
    )
    if not units:
        raise RuntimeError(f"No files assigned to split {split}")

    return [
        prepare_upperarm_record_from_unit(
            unit=unit,
            dataset_type=dataset_type,
            input_channel=input_channel,
            target_channels=target_channels,
            apply_filter=apply_filter,
            normalize_mode=normalize_mode,
            fallback_fs=fallback_fs,
            target_fs=target_fs,
            npz_timestamp_key=npz_timestamp_key,
            npz_sampling_rate_key=npz_sampling_rate_key,
            npz_start_time_key=npz_start_time_key,
            npz_start_time_scale=npz_start_time_scale,
            npz_signal_matrix_key=npz_signal_matrix_key,
            npz_channel_names_key=npz_channel_names_key,
            quality_preprocess_mode=quality_preprocess_mode,
        )
        for unit in units
    ]


class UpperArmCSVWindowsDataset(Dataset):
    def __init__(
        self,
        csv_dir: str | Path,
        split: str | None,
        segment_length: int,
        segment_stride: int,
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
        target_fs: float | None = None,
        segment_policy: str = "pad",
        padding_mode: str = "zero",
        max_files: int | None = None,
        split_files: dict[str, Iterable[str] | str] | None = None,
        dataset_type: str = "upperarm_csv",
        segment_group_regex: str = DEFAULT_SEGMENT_GROUP_REGEX,
        segment_offset_regex: str = DEFAULT_SEGMENT_OFFSET_REGEX,
        npz_timestamp_key: str = "timestamp_ms",
        npz_sampling_rate_key: str | None = "sampling_rate_hz",
        npz_start_time_key: str | None = "start_time_ms",
        npz_start_time_scale: float = 1.0,
        npz_signal_matrix_key: str | None = None,
        npz_channel_names_key: str | None = None,
        csv_dirs: Iterable[str | Path] | str | Path | None = None,
        quality_preprocess_mode: str = "none",
        quality_preprocess_config: dict[str, Any] | None = None,
    ) -> None:
        self.segment_length = int(segment_length)
        self.segment_stride = int(segment_stride)
        self.segment_policy = segment_policy
        self.padding_mode = padding_mode
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
            target_fs=target_fs,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
            max_files=max_files,
            split_files=split_files,
            dataset_type=dataset_type,
            segment_group_regex=segment_group_regex,
            segment_offset_regex=segment_offset_regex,
            npz_timestamp_key=npz_timestamp_key,
            npz_sampling_rate_key=npz_sampling_rate_key,
            npz_start_time_key=npz_start_time_key,
            npz_start_time_scale=npz_start_time_scale,
            npz_signal_matrix_key=npz_signal_matrix_key,
            npz_channel_names_key=npz_channel_names_key,
            csv_dirs=csv_dirs,
            quality_preprocess_mode=quality_preprocess_mode,
            quality_preprocess_config=quality_preprocess_config,
        )

        self.items: list[tuple[int, int]] = []
        for record_idx, record in enumerate(self.records):
            starts = compute_window_starts(
                length=record.input_signal.shape[0],
                signal_length=self.segment_length,
                stride=self.segment_stride,
                window_policy=self.segment_policy,
            )
            for start in starts:
                self.items.append((record_idx, start))

        if not self.items:
            raise RuntimeError("No windows generated from upper-arm dataset")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        record_idx, start = self.items[index]
        record = self.records[record_idx]
        stop = start + self.segment_length
        x = _pad_1d(record.input_signal[start:stop], self.segment_length, self.padding_mode)[None, :]
        y = np.stack(
            [
                _pad_1d(record.target_signals[channel_idx, start:stop], self.segment_length, self.padding_mode)
                for channel_idx in range(record.target_signals.shape[0])
            ],
            axis=0,
        )
        return torch.from_numpy(x), torch.from_numpy(y), 0

    @property
    def lead_names(self) -> list[str]:
        return list(self.target_channels)

    @property
    def input_channels(self) -> int:
        return 1

    @property
    def output_channels(self) -> int:
        return len(self.target_channels)
