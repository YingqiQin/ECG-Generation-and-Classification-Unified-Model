from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from mcma_torch.data.upperarm_csv import (
    PreparedUpperArmRecord,
    aggregate_window_predictions,
    compute_window_starts,
    load_upperarm_records,
)
from mcma_torch.models.mcma import MCMA
from mcma_torch.utils.checkpoint import load_shape_matched_checkpoint
from mcma_torch.utils.config import load_config
from mcma_torch.utils.io import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct target leads from upper-arm ECG CSV files.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--ckpt", required=True, help="Path to a checkpoint produced by mcma_torch.train.fit.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides like data.csv_dir=/path model.filters=16,32,64,128,256,512",
    )
    return parser


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if "," in raw and not (raw.startswith("[") and raw.endswith("]")):
        return [_parse_value(chunk.strip()) for chunk in raw.split(",")]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _set_nested(config: dict, keys: list[str], value: Any) -> None:
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _apply_overrides(config: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item}")
        key, raw_value = item.split("=", 1)
        _set_nested(config, key.split("."), _parse_value(raw_value))
    return config


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
        raise ValueError(f"Unsupported padding_mode: {pad_mode}")
    return np.concatenate([signal, pad], axis=0)


def _corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    if x.shape != y.shape:
        raise ValueError("Correlation inputs must have identical shapes")
    x_series = pd.Series(x.astype(np.float64, copy=False))
    y_series = pd.Series(y.astype(np.float64, copy=False))
    if float(x_series.std(ddof=0)) <= 1e-12 or float(y_series.std(ddof=0)) <= 1e-12:
        return float("nan")
    return float(x_series.corr(y_series, method=method))


def _write_reconstruction_csv(
    output_path: Path,
    timestamps_ms: np.ndarray,
    target_channels: list[str],
    reconstructed: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_ms", *target_channels])
        for idx in range(timestamps_ms.shape[0]):
            timestamp_value = float(timestamps_ms[idx])
            if abs(timestamp_value - round(timestamp_value)) <= 1e-9:
                timestamp_cell: int | float = int(round(timestamp_value))
            else:
                timestamp_cell = timestamp_value
            writer.writerow(
                [timestamp_cell, *[float(reconstructed[channel_idx, idx]) for channel_idx in range(reconstructed.shape[0])]]
            )


def _append_metrics(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _select_evenly_spaced(values: list[int], max_count: int) -> list[int]:
    if max_count <= 0 or len(values) <= max_count:
        return list(values)
    positions = np.linspace(0, len(values) - 1, num=max_count)
    indices = np.unique(np.round(positions).astype(np.int64))
    return [values[int(idx)] for idx in indices]


def _encode_latent_features(
    model: MCMA,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch_start in range(0, windows.shape[0], batch_size):
            batch = windows[batch_start:batch_start + batch_size]
            x = torch.from_numpy(batch[:, None, :]).to(device=device, dtype=torch.float32)
            e1 = model.e1(x)
            e2 = model.e2(e1)
            e3 = model.e3(e2)
            e4 = model.e4(e3)
            e5 = model.e5(e4)
            e6 = model.e6(e5)
            pooled_mean = e6.mean(dim=2)
            pooled_std = e6.std(dim=2, unbiased=False)
            features = torch.cat([pooled_mean, pooled_std], dim=1)
            outputs.append(features.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0), dtype=np.float32)


def _project_pca(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if features.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((2,), dtype=np.float32)
    centered = features.astype(np.float64, copy=False) - features.mean(axis=0, keepdims=True)
    if centered.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32), np.zeros((2,), dtype=np.float32)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    projection = centered @ components.T
    variance = singular_values ** 2
    denom = float(variance.sum()) if variance.size > 0 else 0.0
    explained = variance[:2] / denom if denom > 0 else np.zeros((min(2, variance.shape[0]),), dtype=np.float64)
    if explained.shape[0] < 2:
        explained = np.pad(explained, (0, 2 - explained.shape[0]), constant_values=0.0)
    if projection.shape[1] < 2:
        projection = np.pad(projection, ((0, 0), (0, 2 - projection.shape[1])), constant_values=0.0)
    return projection.astype(np.float32, copy=False), explained.astype(np.float32, copy=False)


def save_latent_space_plot(
    output_path: Path,
    model: MCMA,
    record: PreparedUpperArmRecord,
    target_channels: list[str],
    reconstructed: np.ndarray,
    segment_length: int,
    segment_stride: int,
    segment_policy: str,
    padding_mode: str,
    batch_size: int,
    device: torch.device,
    max_windows_per_signal: int = 24,
    dpi: int = 180,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    starts = compute_window_starts(
        length=record.input_signal.shape[0],
        signal_length=segment_length,
        stride=segment_stride,
        window_policy=segment_policy,
    )
    starts = _select_evenly_spaced(starts, max_windows_per_signal)
    if not starts:
        starts = [0]

    group_specs: list[tuple[str, str, np.ndarray]] = [("upperarm", "CH20", record.input_signal)]
    for lead_idx, lead_name in enumerate(target_channels):
        group_specs.append(("target", lead_name, record.target_signals[lead_idx]))
    for lead_idx, lead_name in enumerate(target_channels):
        group_specs.append(("reconstructed", lead_name, reconstructed[lead_idx]))

    feature_blocks: list[np.ndarray] = []
    group_meta: list[tuple[str, str, int, int]] = []
    for family, lead_name, signal in group_specs:
        windows = np.stack(
            [
                _pad_1d(signal[start:start + segment_length], segment_length, padding_mode)
                for start in starts
            ],
            axis=0,
        )
        features = _encode_latent_features(model=model, windows=windows, batch_size=batch_size, device=device)
        start_idx = sum(block.shape[0] for block in feature_blocks)
        feature_blocks.append(features)
        group_meta.append((family, lead_name, start_idx, start_idx + features.shape[0]))

    feature_matrix = np.concatenate(feature_blocks, axis=0)
    projection, explained = _project_pca(feature_matrix)

    lead_colors: dict[str, Any] = {"CH20": "black"}
    cmap = plt.get_cmap("tab10")
    for idx, lead_name in enumerate(target_channels):
        lead_colors[lead_name] = cmap(idx % 10)
    family_markers = {
        "upperarm": "X",
        "target": "o",
        "reconstructed": "^",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    scatter_ax, centroid_ax = axes
    centroid_features: dict[tuple[str, str], np.ndarray] = {}
    centroid_proj: dict[tuple[str, str], np.ndarray] = {}

    for family, lead_name, start_idx, stop_idx in group_meta:
        points = projection[start_idx:stop_idx]
        if points.shape[0] == 0:
            continue
        color = lead_colors.get(lead_name, "gray")
        scatter_ax.scatter(
            points[:, 0],
            points[:, 1],
            s=22 if family != "upperarm" else 45,
            alpha=0.45 if family != "upperarm" else 0.95,
            c=[color],
            marker=family_markers[family],
            edgecolors="none",
        )
        centroid_features[(family, lead_name)] = feature_matrix[start_idx:stop_idx].mean(axis=0)
        centroid_proj[(family, lead_name)] = points.mean(axis=0)

    scatter_ax.set_title(
        f"Encoder latent PCA | PC1={explained[0] * 100:.1f}% PC2={explained[1] * 100:.1f}%",
        fontsize=11,
    )
    scatter_ax.set_xlabel("PC1")
    scatter_ax.set_ylabel("PC2")
    scatter_ax.grid(alpha=0.2)

    family_handles = [
        Line2D([0], [0], marker=family_markers["upperarm"], color="black", linestyle="none", markersize=8, label="upperarm"),
        Line2D([0], [0], marker=family_markers["target"], color="black", linestyle="none", markersize=8, label="target"),
        Line2D([0], [0], marker=family_markers["reconstructed"], color="black", linestyle="none", markersize=8, label="reconstructed"),
    ]
    lead_handles = [
        Line2D([0], [0], marker="o", color=lead_colors["CH20"], linestyle="none", markersize=7, label="CH20")
    ]
    lead_handles.extend(
        [
            Line2D([0], [0], marker="o", color=lead_colors[lead_name], linestyle="none", markersize=7, label=lead_name)
            for lead_name in target_channels
        ]
    )
    legend_family = scatter_ax.legend(handles=family_handles, title="family", loc="upper right")
    scatter_ax.add_artist(legend_family)
    scatter_ax.legend(handles=lead_handles, title="lead color", loc="lower right", ncol=2, fontsize=8)

    centroid_ax.set_title("Centroid Alignment", fontsize=11)
    centroid_ax.set_xlabel("PC1")
    centroid_ax.set_ylabel("PC2")
    centroid_ax.grid(alpha=0.2)

    upperarm_key = ("upperarm", "CH20")
    if upperarm_key in centroid_proj:
        point = centroid_proj[upperarm_key]
        centroid_ax.scatter(point[0], point[1], c=[lead_colors["CH20"]], marker=family_markers["upperarm"], s=120)
        centroid_ax.annotate("CH20-U", (point[0], point[1]), textcoords="offset points", xytext=(6, 6), fontsize=9)

    mean_cosine_values: list[float] = []
    mean_l2_values: list[float] = []
    for lead_name in target_channels:
        target_key = ("target", lead_name)
        recon_key = ("reconstructed", lead_name)
        if target_key not in centroid_proj or recon_key not in centroid_proj:
            continue
        target_point = centroid_proj[target_key]
        recon_point = centroid_proj[recon_key]
        centroid_ax.plot(
            [target_point[0], recon_point[0]],
            [target_point[1], recon_point[1]],
            color=lead_colors[lead_name],
            linewidth=1.2,
            alpha=0.8,
        )
        centroid_ax.scatter(target_point[0], target_point[1], c=[lead_colors[lead_name]], marker=family_markers["target"], s=80)
        centroid_ax.scatter(recon_point[0], recon_point[1], c=[lead_colors[lead_name]], marker=family_markers["reconstructed"], s=80)
        centroid_ax.annotate(f"{lead_name}-T", (target_point[0], target_point[1]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        centroid_ax.annotate(f"{lead_name}-R", (recon_point[0], recon_point[1]), textcoords="offset points", xytext=(4, -10), fontsize=8)

        target_feature = centroid_features[target_key]
        recon_feature = centroid_features[recon_key]
        l2_value = float(np.linalg.norm(target_feature - recon_feature))
        denom = float(np.linalg.norm(target_feature) * np.linalg.norm(recon_feature))
        cosine_value = float(np.dot(target_feature, recon_feature) / denom) if denom > 1e-12 else float("nan")
        mean_l2_values.append(l2_value)
        mean_cosine_values.append(cosine_value)

    finite_cosine_values = [value for value in mean_cosine_values if not math.isnan(value)]
    mean_l2 = float(np.mean(mean_l2_values)) if mean_l2_values else float("nan")
    mean_cos = float(np.mean(finite_cosine_values)) if finite_cosine_values else float("nan")
    fig.suptitle(
        (
            f"{record.path.name} | encoder feature space | windows/group={len(starts)} | "
            f"mean target-recon centroid l2={mean_l2:.3f} | mean cosine={mean_cos:.3f}"
        ),
        fontsize=12,
        y=1.02,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _resolve_focus_lead(
    target_channels: list[str],
    metrics_row: dict[str, object],
    corr_method: str,
    requested_lead: str | None,
) -> tuple[int, str]:
    if requested_lead:
        if requested_lead not in target_channels:
            raise ValueError(f"Requested focus lead {requested_lead} not found in {target_channels}")
        idx = target_channels.index(requested_lead)
        return idx, requested_lead

    best_idx = 0
    best_score = -float("inf")
    for idx, lead_name in enumerate(target_channels):
        score = float(metrics_row.get(f"{lead_name}_{corr_method}", float("nan")))
        if math.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, target_channels[best_idx]


def _find_peak_indices(signal: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    if signal.size < 3:
        return np.array([], dtype=np.int64)
    magnitude = np.abs(signal.astype(np.float64, copy=False))
    threshold = max(float(np.quantile(magnitude, 0.92)), float(np.mean(magnitude) + 0.5 * np.std(magnitude)))
    candidate_mask = (magnitude[1:-1] >= magnitude[:-2]) & (magnitude[1:-1] >= magnitude[2:]) & (magnitude[1:-1] >= threshold)
    candidates = np.where(candidate_mask)[0] + 1
    if candidates.size == 0:
        return np.array([], dtype=np.int64)

    refractory = max(1, int(round(max(sampling_rate_hz, 1.0) * 0.35)))
    ranked = candidates[np.argsort(magnitude[candidates])[::-1]]
    selected: list[int] = []
    for idx in ranked:
        if all(abs(int(idx) - prev) > refractory for prev in selected):
            selected.append(int(idx))
    return np.asarray(sorted(selected), dtype=np.int64)


def _select_focus_centers(
    signal: np.ndarray,
    sampling_rate_hz: float,
    num_beats: int,
    half_window_samples: int,
) -> np.ndarray:
    if num_beats <= 0:
        return np.array([], dtype=np.int64)

    peaks = _find_peak_indices(signal=signal, sampling_rate_hz=sampling_rate_hz)
    valid = peaks[(peaks >= half_window_samples) & (peaks < signal.shape[0] - half_window_samples)]
    if valid.size > 0:
        if valid.size <= num_beats:
            return valid
        positions = np.linspace(0, valid.size - 1, num=num_beats)
        indices = np.unique(np.round(positions).astype(np.int64))
        return valid[indices]

    available_start = half_window_samples
    available_stop = max(available_start + 1, signal.shape[0] - half_window_samples)
    positions = np.linspace(available_start, available_stop - 1, num=num_beats)
    return np.unique(np.round(positions).astype(np.int64))


def save_reconstruction_comparison_plot(
    output_path: Path,
    record: PreparedUpperArmRecord,
    target_channels: list[str],
    reconstructed: np.ndarray,
    metrics_row: dict[str, object],
    max_plot_samples: int = 4000,
    dpi: int = 150,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    num_leads = len(target_channels)
    cols = 2 if num_leads > 1 else 1
    rows = int(math.ceil(num_leads / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(3.5, rows * 2.8)), sharex=True)
    axes_array = np.atleast_1d(axes).reshape(-1)

    time_s = (record.timestamps_ms.astype(np.float64) - float(record.timestamps_ms[0])) / 1000.0
    if max_plot_samples > 0 and time_s.shape[0] > max_plot_samples:
        stride = int(math.ceil(time_s.shape[0] / max_plot_samples))
    else:
        stride = 1
    time_plot = time_s[::stride]

    corr_method = str(metrics_row.get("corr_method", "pearson"))
    for lead_idx, lead_name in enumerate(target_channels):
        ax = axes_array[lead_idx]
        target = record.target_signals[lead_idx][::stride]
        pred = reconstructed[lead_idx][::stride]
        ax.plot(time_plot, target, color="black", linewidth=1.0, label="target (eval-domain)")
        ax.plot(time_plot, pred, color="red", linewidth=1.0, alpha=0.8, label="reconstructed")
        corr_value = float(metrics_row.get(f"{lead_name}_{corr_method}", float("nan")))
        rmse_value = float(metrics_row.get(f"{lead_name}_rmse", float("nan")))
        mae_value = float(metrics_row.get(f"{lead_name}_mae", float("nan")))
        ax.set_title(
            f"{lead_name} | {corr_method}={corr_value:.3f} | rmse={rmse_value:.3f} | mae={mae_value:.3f}",
            fontsize=10,
        )
        ax.grid(alpha=0.2)
        if lead_idx % cols == 0:
            ax.set_ylabel("normalized amp")

    for ax in axes_array[num_leads:]:
        ax.axis("off")

    axes_array[min(num_leads - 1, len(axes_array) - 1)].set_xlabel("time (s)")
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(
        (
            f"{record.path.name} | original_fs={record.original_sampling_rate_hz:.1f} Hz "
            f"-> eval_fs={record.sampling_rate_hz:.1f} Hz | "
            f"mean_{corr_method}={float(metrics_row[f'mean_{corr_method}']):.3f} | "
            f"mean_rmse={float(metrics_row['mean_rmse']):.3f} | eval-domain comparison"
        ),
        fontsize=12,
        y=1.01,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_focus_lead_plot(
    output_path: Path,
    record: PreparedUpperArmRecord,
    target_channels: list[str],
    reconstructed: np.ndarray,
    metrics_row: dict[str, object],
    focus_lead: str | None = None,
    num_beats: int = 4,
    window_ms: float = 900.0,
    max_plot_samples: int = 4000,
    dpi: int = 180,
) -> tuple[str, int]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    corr_method = str(metrics_row.get("corr_method", "pearson"))
    lead_idx, lead_name = _resolve_focus_lead(
        target_channels=target_channels,
        metrics_row=metrics_row,
        corr_method=corr_method,
        requested_lead=focus_lead,
    )

    time_s = (record.timestamps_ms.astype(np.float64) - float(record.timestamps_ms[0])) / 1000.0
    target = record.target_signals[lead_idx]
    pred = reconstructed[lead_idx]
    sampling_rate_hz = max(float(record.sampling_rate_hz), 1.0)
    half_window_samples = max(8, int(round((window_ms / 1000.0) * sampling_rate_hz / 2.0)))
    centers = _select_focus_centers(
        signal=target,
        sampling_rate_hz=sampling_rate_hz,
        num_beats=num_beats,
        half_window_samples=half_window_samples,
    )

    if max_plot_samples > 0 and time_s.shape[0] > max_plot_samples:
        overview_stride = int(math.ceil(time_s.shape[0] / max_plot_samples))
    else:
        overview_stride = 1

    total_rows = 1 + max(1, centers.shape[0])
    height_ratios = [1.6] + [1.0] * max(1, centers.shape[0])
    fig, axes = plt.subplots(
        total_rows,
        1,
        figsize=(14, 2.8 + 2.2 * total_rows),
        sharex=False,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes_array = np.atleast_1d(axes).reshape(-1)

    overview_ax = axes_array[0]
    overview_ax.plot(time_s[::overview_stride], target[::overview_stride], color="black", linewidth=1.0, label="target (eval-domain)")
    overview_ax.plot(time_s[::overview_stride], pred[::overview_stride], color="red", linewidth=1.0, alpha=0.8, label="reconstructed")
    for center in centers:
        left = max(0, int(center) - half_window_samples)
        right = min(target.shape[0], int(center) + half_window_samples)
        overview_ax.axvspan(time_s[left], time_s[right - 1], color="gold", alpha=0.18)
    overview_ax.set_ylabel("normalized amp")
    overview_ax.grid(alpha=0.2)
    overview_ax.set_title(
        (
            f"{lead_name} full trace | {corr_method}={float(metrics_row[f'{lead_name}_{corr_method}']):.3f} | "
            f"rmse={float(metrics_row[f'{lead_name}_rmse']):.3f} | mae={float(metrics_row[f'{lead_name}_mae']):.3f}"
        ),
        fontsize=11,
    )

    if centers.shape[0] == 0:
        centers = np.array([target.shape[0] // 2], dtype=np.int64)

    for zoom_idx, center in enumerate(centers, start=1):
        ax = axes_array[zoom_idx]
        left = max(0, int(center) - half_window_samples)
        right = min(target.shape[0], int(center) + half_window_samples)
        segment_time_ms = (time_s[left:right] - time_s[int(center)]) * 1000.0
        ax.plot(segment_time_ms, target[left:right], color="black", linewidth=1.2, label="target (eval-domain)")
        ax.plot(segment_time_ms, pred[left:right], color="red", linewidth=1.2, alpha=0.85, label="reconstructed")
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.set_ylabel("amp")
        ax.grid(alpha=0.25)
        ax.set_title(f"Zoom {zoom_idx} centered at {time_s[int(center)]:.3f}s", fontsize=10)

    axes_array[-1].set_xlabel("time relative to center (ms)")
    handles, labels = overview_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(
        (
            f"{record.path.name} | focus_lead={lead_name} | original_fs={record.original_sampling_rate_hz:.1f} Hz "
            f"-> eval_fs={record.sampling_rate_hz:.1f} Hz | beat-level zoom"
        ),
        fontsize=12,
        y=1.01,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return lead_name, lead_idx


def build_upperarm_model(
    model_cfg: dict,
    target_channels: list[str],
    device: torch.device,
) -> MCMA:
    return MCMA(
        in_channels=1,
        out_channels=len(target_channels),
        kernel_size=int(model_cfg.get("kernel_size", 13)),
        window_size=int(model_cfg.get("window_size", 2)),
        filters=model_cfg.get("filters", [16, 32, 64, 128, 256, 512]),
    ).to(device)


def reconstruct_record(
    model: MCMA,
    record: PreparedUpperArmRecord,
    segment_length: int,
    segment_stride: int,
    segment_policy: str,
    padding_mode: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    starts = compute_window_starts(
        length=record.input_signal.shape[0],
        signal_length=segment_length,
        stride=segment_stride,
        window_policy=segment_policy,
    )

    windows = np.stack(
        [
            _pad_1d(record.input_signal[start:start + segment_length], segment_length, padding_mode)
            for start in starts
        ],
        axis=0,
    )

    predictions = []
    with torch.no_grad():
        for batch_start in range(0, windows.shape[0], batch_size):
            batch = windows[batch_start:batch_start + batch_size]
            x = torch.from_numpy(batch[:, None, :]).to(device=device, dtype=torch.float32)
            y_hat = model(x).detach().cpu().numpy().astype(np.float32, copy=False)
            predictions.append(y_hat)

    return aggregate_window_predictions(
        predictions=np.concatenate(predictions, axis=0),
        starts=starts,
        total_length=record.input_signal.shape[0],
    )


def build_reconstruction_metrics_row(
    record: PreparedUpperArmRecord,
    target_channels: list[str],
    reconstructed: np.ndarray,
    corr_method: str = "pearson",
) -> dict[str, object]:
    row: dict[str, object] = {
        "file_name": record.path.name,
        "original_sampling_rate_hz": record.original_sampling_rate_hz,
        "effective_sampling_rate_hz": record.sampling_rate_hz,
        "corr_method": corr_method,
    }
    lead_corr_values = []
    lead_mse_values = []
    lead_mae_values = []
    lead_rmse_values = []
    for lead_idx, lead_name in enumerate(target_channels):
        target = record.target_signals[lead_idx]
        pred = reconstructed[lead_idx]
        mse = float(np.mean((pred - target) ** 2))
        mae = float(np.mean(np.abs(pred - target)))
        rmse = float(np.sqrt(mse))
        corr = _corr(pred, target, method=corr_method)
        row[f"{lead_name}_mse"] = mse
        row[f"{lead_name}_mae"] = mae
        row[f"{lead_name}_rmse"] = rmse
        row[f"{lead_name}_{corr_method}"] = corr
        lead_mse_values.append(mse)
        lead_mae_values.append(mae)
        lead_rmse_values.append(rmse)
        lead_corr_values.append(corr)
    row["mean_mse"] = float(np.mean(lead_mse_values))
    row["mean_mae"] = float(np.mean(lead_mae_values))
    row["mean_rmse"] = float(np.mean(lead_rmse_values))
    row[f"mean_{corr_method}"] = float(np.nanmean(lead_corr_values))
    return row


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    ckpt_path = Path(args.ckpt)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    config = load_config(config_path)
    if not isinstance(config, dict):
        raise ValueError("Config must parse into a dict")
    config = _apply_overrides(config, args.overrides)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    reconstruct_cfg = config.get("reconstruct", {})

    if data_cfg.get("dataset_type", "ptbxl") != "upperarm_csv":
        raise ValueError("reconstruct_upperarm requires data.dataset_type=upperarm_csv")

    device_name = reconstruct_cfg.get("device", config.get("trainer", {}).get("device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(reconstruct_cfg.get("output_dir", "artifacts/upperarm_reconstruction"))
    corr_method = reconstruct_cfg.get("corr_method", "pearson")
    save_plots = bool(reconstruct_cfg.get("save_plots", True))
    plot_dir_value = reconstruct_cfg.get("plot_dir")
    plot_dir = ensure_dir(plot_dir_value) if plot_dir_value else output_dir / "plots"
    max_plot_samples = int(reconstruct_cfg.get("max_plot_samples", 4000))
    plot_dpi = int(reconstruct_cfg.get("plot_dpi", 150))
    save_focus_plots = bool(reconstruct_cfg.get("save_focus_plots", True))
    focus_plot_dir_value = reconstruct_cfg.get("focus_plot_dir")
    focus_plot_dir = ensure_dir(focus_plot_dir_value) if focus_plot_dir_value else output_dir / "focus_plots"
    focus_lead = reconstruct_cfg.get("focus_lead")
    focus_num_beats = int(reconstruct_cfg.get("focus_num_beats", 4))
    focus_window_ms = float(reconstruct_cfg.get("focus_window_ms", 900.0))
    focus_plot_dpi = int(reconstruct_cfg.get("focus_plot_dpi", max(plot_dpi, 180)))
    save_latent_plots = bool(reconstruct_cfg.get("save_latent_plots", True))
    latent_plot_dir_value = reconstruct_cfg.get("latent_plot_dir")
    latent_plot_dir = ensure_dir(latent_plot_dir_value) if latent_plot_dir_value else output_dir / "latent_plots"
    latent_max_windows_per_signal = int(reconstruct_cfg.get("latent_max_windows_per_signal", 24))
    latent_plot_dpi = int(reconstruct_cfg.get("latent_plot_dpi", max(plot_dpi, 180)))

    target_channels = list(data_cfg.get("target_channels") or [f"CH{i}" for i in range(1, 9)])
    records = load_upperarm_records(
        csv_dir=data_cfg["csv_dir"],
        file_glob=data_cfg.get("file_glob", "emg_data_*.csv"),
        input_channel=data_cfg.get("input_channel", "CH20"),
        target_channels=target_channels,
        apply_filter=bool(data_cfg.get("apply_filter", True)),
        normalize_mode=data_cfg.get("normalize_mode", "zscore"),
        fallback_fs=float(data_cfg.get("fallback_fs", 250.0)),
        target_fs=data_cfg.get("target_fs"),
        split=reconstruct_cfg.get("split"),
        train_ratio=float(data_cfg.get("train_ratio", 0.8)),
        val_ratio=float(data_cfg.get("val_ratio", 0.2)),
        test_ratio=float(data_cfg.get("test_ratio", 0.0)),
        split_seed=int(data_cfg.get("split_seed", 42)),
        max_files=reconstruct_cfg.get("max_files"),
        split_files=data_cfg.get("split_files"),
    )

    model = build_upperarm_model(model_cfg=model_cfg, target_channels=target_channels, device=device)
    load_report = load_shape_matched_checkpoint(model, ckpt_path=ckpt_path, device=device)
    if load_report["missing_keys"]:
        print("Warning: missing keys:", load_report["missing_keys"])
    if load_report["unexpected_keys"]:
        print("Warning: unexpected keys:", load_report["unexpected_keys"])
    if load_report["skipped_shape_keys"]:
        print("Warning: skipped shape-mismatch keys:", load_report["skipped_shape_keys"])
    model.eval()

    segment_length = int(data_cfg.get("segment_length", 1024))
    segment_stride = int(data_cfg.get("segment_stride", segment_length))
    segment_policy = data_cfg.get("segment_policy", "pad")
    padding_mode = data_cfg.get("padding_mode", "zero")
    batch_size = int(reconstruct_cfg.get("batch_size", data_cfg.get("batch_size", 64)))

    metrics_rows: list[dict[str, object]] = []
    for record in records:
        y_hat_full = reconstruct_record(
            model=model,
            record=record,
            segment_length=segment_length,
            segment_stride=segment_stride,
            segment_policy=segment_policy,
            padding_mode=padding_mode,
            batch_size=batch_size,
            device=device,
        )

        output_path = output_dir / record.path.name
        _write_reconstruction_csv(
            output_path=output_path,
            timestamps_ms=record.timestamps_ms,
            target_channels=target_channels,
            reconstructed=y_hat_full,
        )

        row = build_reconstruction_metrics_row(
            record=record,
            target_channels=target_channels,
            reconstructed=y_hat_full,
            corr_method=corr_method,
        )
        if save_plots:
            plot_path = plot_dir / f"{record.path.stem}_comparison.png"
            save_reconstruction_comparison_plot(
                output_path=plot_path,
                record=record,
                target_channels=target_channels,
                reconstructed=y_hat_full,
                metrics_row=row,
                max_plot_samples=max_plot_samples,
                dpi=plot_dpi,
            )
            row["plot_path"] = str(plot_path)
        if save_focus_plots:
            focus_plot_path = focus_plot_dir / f"{record.path.stem}_focus.png"
            selected_focus_lead, _ = save_focus_lead_plot(
                output_path=focus_plot_path,
                record=record,
                target_channels=target_channels,
                reconstructed=y_hat_full,
                metrics_row=row,
                focus_lead=focus_lead,
                num_beats=focus_num_beats,
                window_ms=focus_window_ms,
                max_plot_samples=max_plot_samples,
                dpi=focus_plot_dpi,
            )
            row["focus_lead"] = selected_focus_lead
            row["focus_plot_path"] = str(focus_plot_path)
        if save_latent_plots:
            latent_plot_path = latent_plot_dir / f"{record.path.stem}_latent.png"
            save_latent_space_plot(
                output_path=latent_plot_path,
                model=model,
                record=record,
                target_channels=target_channels,
                reconstructed=y_hat_full,
                segment_length=segment_length,
                segment_stride=segment_stride,
                segment_policy=segment_policy,
                padding_mode=padding_mode,
                batch_size=batch_size,
                device=device,
                max_windows_per_signal=latent_max_windows_per_signal,
                dpi=latent_plot_dpi,
            )
            row["latent_plot_path"] = str(latent_plot_path)
        metrics_rows.append(row)

        print(
            f"Reconstructed {record.path.name} | mean_{corr_method}={row[f'mean_{corr_method}']:.4f} | "
            f"mean_rmse={row['mean_rmse']:.6f}"
        )

    metrics_path = output_dir / "reconstruction_metrics.csv"
    fieldnames = [
        "file_name",
        "original_sampling_rate_hz",
        "effective_sampling_rate_hz",
        "corr_method",
        "plot_path",
        "focus_lead",
        "focus_plot_path",
        "latent_plot_path",
        "mean_mse",
        "mean_mae",
        "mean_rmse",
        f"mean_{corr_method}",
    ]
    for lead_name in target_channels:
        fieldnames.extend(
            [
                f"{lead_name}_mse",
                f"{lead_name}_mae",
                f"{lead_name}_rmse",
                f"{lead_name}_{corr_method}",
            ]
        )
    _append_metrics(metrics_path, metrics_rows, fieldnames=fieldnames)
    print(f"Saved reconstruction CSVs and metrics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
