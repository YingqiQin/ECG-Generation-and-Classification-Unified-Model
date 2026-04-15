from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import distance
import torch

from mcma_torch.data.upperarm_csv import (
    PreparedUpperArmRecord,
    _filter_signal,
    aggregate_window_predictions,
    compute_window_starts,
    is_upperarm_dataset_type,
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


def _overlap_with_lag(target: np.ndarray, pred: np.ndarray, lag_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if target.shape != pred.shape:
        raise ValueError("Lag alignment requires equal-length arrays")
    if lag_samples > 0:
        return target[:-lag_samples], pred[lag_samples:]
    if lag_samples < 0:
        return target[-lag_samples:], pred[:lag_samples]
    return target, pred


def _shift_for_plot(signal: np.ndarray, lag_samples: int) -> np.ndarray:
    shifted = np.full(signal.shape, np.nan, dtype=np.float32)
    if lag_samples > 0:
        shifted[:-lag_samples] = signal[lag_samples:]
        return shifted
    if lag_samples < 0:
        shifted[-lag_samples:] = signal[:lag_samples]
        return shifted
    return signal.astype(np.float32, copy=False)


def _prepare_display_signals(
    target: np.ndarray,
    pred: np.ndarray,
    sampling_rate_hz: float,
    visual_filter_mode: str,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    mode = str(visual_filter_mode).strip().lower()
    target_display = target.astype(np.float32, copy=False)
    pred_display = pred.astype(np.float32, copy=False)
    target_label = "target (eval-domain)"
    pred_label = "reconstructed"

    if mode in {"none", "raw"}:
        return target_display, pred_display, target_label, pred_label
    if mode == "recon_only":
        pred_display = _filter_signal(pred_display, sampling_rate_hz=sampling_rate_hz)
        pred_label = "reconstructed (display-filtered)"
        return target_display, pred_display, target_label, pred_label
    if mode == "both":
        target_display = _filter_signal(target_display, sampling_rate_hz=sampling_rate_hz)
        pred_display = _filter_signal(pred_display, sampling_rate_hz=sampling_rate_hz)
        target_label = "target (display-filtered eval-domain)"
        pred_label = "reconstructed (display-filtered)"
        return target_display, pred_display, target_label, pred_label
    raise ValueError(f"Unsupported visual_filter_mode: {visual_filter_mode}")


def _bounded_xcorr_metrics(
    target: np.ndarray,
    pred: np.ndarray,
    sampling_rate_hz: float,
    corr_method: str,
    max_lag_ms: float,
) -> dict[str, float]:
    max_lag_samples = max(0, int(round(max_lag_ms * sampling_rate_hz / 1000.0)))
    best_corr = -float("inf")
    best_lag = 0
    for lag_samples in range(-max_lag_samples, max_lag_samples + 1):
        target_seg, pred_seg = _overlap_with_lag(target=target, pred=pred, lag_samples=lag_samples)
        if target_seg.size < 8:
            continue
        corr_value = _corr(pred_seg, target_seg, method=corr_method)
        if math.isnan(corr_value):
            continue
        if corr_value > best_corr:
            best_corr = corr_value
            best_lag = lag_samples

    if best_corr == -float("inf"):
        best_corr = float("nan")
        best_lag = 0
    target_aligned, pred_aligned = _overlap_with_lag(target=target, pred=pred, lag_samples=best_lag)
    if target_aligned.size == 0:
        lag_rmse = float("nan")
        lag_mae = float("nan")
        lag_corr = float("nan")
    else:
        lag_mse = float(np.mean((pred_aligned - target_aligned) ** 2))
        lag_rmse = float(np.sqrt(lag_mse))
        lag_mae = float(np.mean(np.abs(pred_aligned - target_aligned)))
        lag_corr = _corr(pred_aligned, target_aligned, method=corr_method)
    return {
        "best_lag_samples": float(best_lag),
        "best_lag_ms": float(best_lag) * 1000.0 / max(sampling_rate_hz, 1e-6),
        "max_xcorr": float(best_corr),
        f"lag_corrected_{corr_method}": float(lag_corr),
        "lag_corrected_rmse": float(lag_rmse),
        "lag_corrected_mae": float(lag_mae),
    }


def _downsample_for_dtw(signal: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or signal.size <= max_points:
        return signal.astype(np.float32, copy=False)
    stride = int(math.ceil(signal.size / max_points))
    return signal[::stride].astype(np.float32, copy=False)


def _banded_dtw_distance(target: np.ndarray, pred: np.ndarray, band_radius: int) -> float:
    if target.size == 0 or pred.size == 0:
        return float("nan")
    n = int(target.size)
    m = int(pred.size)
    radius = max(int(band_radius), abs(n - m))
    previous = np.full((m + 1,), np.inf, dtype=np.float64)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = np.full((m + 1,), np.inf, dtype=np.float64)
        j_start = max(1, i - radius)
        j_stop = min(m, i + radius)
        for j in range(j_start, j_stop + 1):
            cost = abs(float(target[i - 1]) - float(pred[j - 1]))
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    if not np.isfinite(previous[m]):
        return float("nan")
    return float(previous[m] / max(n, m))


def _match_rpeak_mae_ms(
    target: np.ndarray,
    pred: np.ndarray,
    sampling_rate_hz: float,
    tolerance_ms: float,
) -> tuple[float, float]:
    target_peaks = _find_peak_indices(target, sampling_rate_hz=sampling_rate_hz)
    pred_peaks = _find_peak_indices(pred, sampling_rate_hz=sampling_rate_hz)
    if target_peaks.size == 0 or pred_peaks.size == 0:
        return float("nan"), float("nan")

    tolerance_samples = max(1, int(round(tolerance_ms * sampling_rate_hz / 1000.0)))
    pred_cursor = 0
    errors_ms: list[float] = []
    for target_peak in target_peaks:
        while pred_cursor + 1 < pred_peaks.size and pred_peaks[pred_cursor + 1] <= target_peak:
            pred_cursor += 1
        candidate_indices = {pred_cursor}
        if pred_cursor + 1 < pred_peaks.size:
            candidate_indices.add(pred_cursor + 1)
        best_error = min(abs(int(target_peak) - int(pred_peaks[idx])) for idx in candidate_indices)
        if best_error <= tolerance_samples:
            errors_ms.append(best_error * 1000.0 / max(sampling_rate_hz, 1e-6))

    matched_fraction = float(len(errors_ms) / max(target_peaks.size, 1))
    if not errors_ms:
        return float("nan"), matched_fraction
    return float(np.mean(errors_ms)), matched_fraction


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


def _spectral_layout(affinity: sparse.spmatrix, random_seed: int) -> np.ndarray:
    num_points = int(affinity.shape[0])
    if num_points == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if num_points == 1:
        return np.zeros((1, 2), dtype=np.float32)

    degree = np.asarray(affinity.sum(axis=1)).reshape(-1)
    if float(degree.max(initial=0.0)) <= 1e-12:
        return np.zeros((num_points, 2), dtype=np.float32)

    laplacian = sparse.csgraph.laplacian(affinity, normed=True)
    max_components = min(num_points - 1, 4)
    if max_components <= 0:
        return np.zeros((num_points, 2), dtype=np.float32)
    try:
        eigenvalues, eigenvectors = sparse_linalg.eigsh(
            laplacian,
            k=max_components,
            which="SM",
        )
        order = np.argsort(np.real(eigenvalues))
        eigenvectors = np.real(eigenvectors[:, order])
    except Exception:
        lap_dense = laplacian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(lap_dense)
        order = np.argsort(np.real(eigenvalues))
        eigenvectors = np.real(eigenvectors[:, order[: max_components + 1]])

    usable = eigenvectors[:, 1:3]
    if usable.shape[1] < 2:
        usable = np.pad(usable, ((0, 0), (0, 2 - usable.shape[1])), constant_values=0.0)
    layout = usable.astype(np.float64, copy=False)
    layout -= layout.mean(axis=0, keepdims=True)
    scale = np.maximum(layout.std(axis=0, keepdims=True), 1e-6)
    layout = layout / scale

    rng = np.random.default_rng(random_seed)
    layout += rng.normal(scale=0.01, size=layout.shape)
    return layout.astype(np.float32, copy=False)


def _project_umap_like(
    features: np.ndarray,
    n_neighbors: int = 12,
    random_seed: int = 42,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    num_points = int(features.shape[0])
    if num_points == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if num_points == 1:
        return np.zeros((1, 2), dtype=np.float32)

    k = max(1, min(int(n_neighbors), num_points - 1))
    dense_features = features.astype(np.float64, copy=False)
    pairwise = distance.cdist(dense_features, dense_features, metric="euclidean")

    row_indices: list[int] = []
    col_indices: list[int] = []
    data_values: list[float] = []
    for idx in range(num_points):
        neighbor_order = np.argsort(pairwise[idx])[1 : k + 1]
        local_distances = pairwise[idx, neighbor_order]
        sigma = float(np.median(local_distances))
        if sigma <= 1e-12:
            sigma = float(np.max(local_distances))
        sigma = max(sigma, 1e-6)
        for neighbor, dist_value in zip(neighbor_order, local_distances, strict=False):
            weight = math.exp(-(float(dist_value) ** 2) / (2.0 * sigma * sigma))
            row_indices.append(idx)
            col_indices.append(int(neighbor))
            data_values.append(weight)

    affinity = sparse.csr_matrix((data_values, (row_indices, col_indices)), shape=(num_points, num_points))
    affinity = affinity.maximum(affinity.T)

    if affinity.nnz == 0:
        return _project_pca(features)[0]
    return _spectral_layout(affinity=affinity, random_seed=random_seed)


def _project_features(
    features: np.ndarray,
    method: str = "umap_like",
    n_neighbors: int = 12,
    random_seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    method_key = str(method).lower()
    effective_neighbors = max(1, min(int(n_neighbors), max(features.shape[0] - 1, 1)))
    if method_key == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=effective_neighbors,
                min_dist=0.15,
                metric="euclidean",
                random_state=random_seed,
            )
            projection = reducer.fit_transform(features.astype(np.float32, copy=False))
            return projection.astype(np.float32, copy=False), {
                "title": f"Encoder feature manifold | umap k={effective_neighbors}",
                "x_label": "UMAP-1",
                "y_label": "UMAP-2",
                "method_label": "umap",
            }
        except Exception:
            method_key = "umap_like"
    if method_key == "pca":
        projection, explained = _project_pca(features)
        return projection, {
            "title": f"Encoder latent PCA | PC1={explained[0] * 100:.1f}% PC2={explained[1] * 100:.1f}%",
            "x_label": "PC1",
            "y_label": "PC2",
            "method_label": "pca",
        }
    if method_key not in {"umap_like", "umap-like"}:
        raise ValueError(f"Unsupported latent projection method: {method}")
    projection = _project_umap_like(features, n_neighbors=n_neighbors, random_seed=random_seed)
    return projection, {
        "title": f"Encoder feature manifold | umap_like k={effective_neighbors}",
        "x_label": "Embed-1",
        "y_label": "Embed-2",
        "method_label": "umap_like",
    }


def _draw_density_cloud(ax: Any, points: np.ndarray, color: Any, alpha_fill: float = 0.10, alpha_line: float = 0.35) -> None:
    if points.shape[0] < 5:
        return
    spread_x = float(np.ptp(points[:, 0]))
    spread_y = float(np.ptp(points[:, 1]))
    if spread_x <= 1e-6 and spread_y <= 1e-6:
        return
    try:
        kde = stats.gaussian_kde(points.T)
    except Exception:
        return

    x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
    y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
    x_pad = max(0.15 * max(x_max - x_min, 1e-3), 1e-3)
    y_pad = max(0.15 * max(y_max - y_min, 1e-3), 1e-3)
    grid_x, grid_y = np.mgrid[(x_min - x_pad):(x_max + x_pad):60j, (y_min - y_pad):(y_max + y_pad):60j]
    grid = np.vstack([grid_x.ravel(), grid_y.ravel()])
    density = kde(grid).reshape(grid_x.shape)
    positive = density[density > 0]
    if positive.size == 0:
        return
    levels = np.quantile(positive, [0.60, 0.80, 0.92])
    levels = np.unique(levels)
    if levels.size < 2:
        return
    ax.contourf(grid_x, grid_y, density, levels=levels, colors=[color], alpha=alpha_fill, antialiased=True)
    ax.contour(grid_x, grid_y, density, levels=levels, colors=[color], linewidths=0.8, alpha=alpha_line)


def _style_latent_axis(ax: Any, title: str, x_label: str, y_label: str, x_limits: tuple[float, float], y_limits: tuple[float, float]) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15, linewidth=0.6)


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
    projection_method: str = "umap_like",
    projection_neighbors: int = 12,
    projection_seed: int = 42,
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
    projection, projection_meta = _project_features(
        feature_matrix,
        method=projection_method,
        n_neighbors=projection_neighbors,
        random_seed=projection_seed,
    )

    lead_colors: dict[str, Any] = {"CH20": "black"}
    cmap = plt.get_cmap("tab10")
    for idx, lead_name in enumerate(target_channels):
        lead_colors[lead_name] = cmap(idx % 10)
    family_markers = {"upperarm": "X", "target": "o", "reconstructed": "^"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    reference_ax, upperarm_ax, recon_ax = axes
    centroid_features: dict[tuple[str, str], np.ndarray] = {}
    centroid_proj: dict[tuple[str, str], np.ndarray] = {}
    group_points: dict[tuple[str, str], np.ndarray] = {}

    for family, lead_name, start_idx, stop_idx in group_meta:
        points = projection[start_idx:stop_idx]
        if points.shape[0] == 0:
            continue
        group_points[(family, lead_name)] = points
        centroid_features[(family, lead_name)] = feature_matrix[start_idx:stop_idx].mean(axis=0)
        centroid_proj[(family, lead_name)] = points.mean(axis=0)

    if projection.shape[0] == 0:
        x_limits = (-1.0, 1.0)
        y_limits = (-1.0, 1.0)
    else:
        x_min, x_max = float(projection[:, 0].min()), float(projection[:, 0].max())
        y_min, y_max = float(projection[:, 1].min()), float(projection[:, 1].max())
        x_pad = max(0.08 * max(x_max - x_min, 1e-3), 0.05)
        y_pad = max(0.08 * max(y_max - y_min, 1e-3), 0.05)
        x_limits = (x_min - x_pad, x_max + x_pad)
        y_limits = (y_min - y_pad, y_max + y_pad)

    for lead_name in target_channels:
        target_key = ("target", lead_name)
        if target_key not in group_points:
            continue
        points = group_points[target_key]
        color = lead_colors[lead_name]
        _draw_density_cloud(reference_ax, points, color=color, alpha_fill=0.10, alpha_line=0.40)
        reference_ax.scatter(points[:, 0], points[:, 1], s=15, c=[color], alpha=0.30, marker="o", edgecolors="none")
        centroid = centroid_proj[target_key]
        reference_ax.scatter(centroid[0], centroid[1], s=60, c=[color], marker="o", edgecolors="white", linewidths=0.5)
        reference_ax.annotate(lead_name, (centroid[0], centroid[1]), textcoords="offset points", xytext=(4, 4), fontsize=8, color=color)

    for lead_name in target_channels:
        target_key = ("target", lead_name)
        if target_key not in group_points:
            continue
        points = group_points[target_key]
        upperarm_ax.scatter(points[:, 0], points[:, 1], s=10, c=[lead_colors[lead_name]], alpha=0.10, marker="o", edgecolors="none")
        centroid = centroid_proj[target_key]
        upperarm_ax.scatter(centroid[0], centroid[1], s=36, c=[lead_colors[lead_name]], alpha=0.80, marker="o", edgecolors="white", linewidths=0.4)

    upperarm_key = ("upperarm", "CH20")
    if upperarm_key in centroid_proj:
        upperarm_points = group_points.get(upperarm_key)
        if upperarm_points is not None:
            _draw_density_cloud(upperarm_ax, upperarm_points, color="black", alpha_fill=0.12, alpha_line=0.30)
            upperarm_ax.scatter(upperarm_points[:, 0], upperarm_points[:, 1], s=24, c="black", alpha=0.75, marker="X", linewidths=0.0)
        upperarm_point = centroid_proj[upperarm_key]
        upperarm_ax.scatter(upperarm_point[0], upperarm_point[1], c=[lead_colors["CH20"]], marker="X", s=120)
        upperarm_ax.annotate("CH20", (upperarm_point[0], upperarm_point[1]), textcoords="offset points", xytext=(6, 6), fontsize=9, color="black")

    mean_cosine_values: list[float] = []
    mean_l2_values: list[float] = []
    upperarm_distances: list[tuple[float, str]] = []
    for lead_name in target_channels:
        target_key = ("target", lead_name)
        recon_key = ("reconstructed", lead_name)
        if target_key in group_points:
            target_points = group_points[target_key]
            recon_ax.scatter(target_points[:, 0], target_points[:, 1], s=10, c=[lead_colors[lead_name]], alpha=0.10, marker="o", edgecolors="none")
            _draw_density_cloud(recon_ax, target_points, color=lead_colors[lead_name], alpha_fill=0.05, alpha_line=0.20)
        if target_key not in centroid_proj:
            continue
        if upperarm_key in centroid_features:
            upperarm_feature = centroid_features[upperarm_key]
            target_feature = centroid_features[target_key]
            upperarm_distances.append((float(np.linalg.norm(upperarm_feature - target_feature)), lead_name))
        if recon_key not in centroid_proj:
            continue
        target_point = centroid_proj[target_key]
        recon_point = centroid_proj[recon_key]
        recon_points = group_points.get(recon_key)
        if recon_points is not None:
            recon_ax.scatter(
                recon_points[:, 0],
                recon_points[:, 1],
                s=24,
                facecolors="none",
                edgecolors=[lead_colors[lead_name]],
                alpha=0.75,
                marker="^",
                linewidths=0.9,
            )
        recon_ax.plot([target_point[0], recon_point[0]], [target_point[1], recon_point[1]], color=lead_colors[lead_name], linewidth=1.2, alpha=0.85)
        recon_ax.scatter(target_point[0], target_point[1], c=[lead_colors[lead_name]], marker="o", s=52, edgecolors="white", linewidths=0.4)
        recon_ax.scatter(recon_point[0], recon_point[1], facecolors="white", edgecolors=[lead_colors[lead_name]], marker="^", s=70, linewidths=1.1)

        target_feature = centroid_features[target_key]
        recon_feature = centroid_features[recon_key]
        l2_value = float(np.linalg.norm(target_feature - recon_feature))
        denom = float(np.linalg.norm(target_feature) * np.linalg.norm(recon_feature))
        cosine_value = float(np.dot(target_feature, recon_feature) / denom) if denom > 1e-12 else float("nan")
        mean_l2_values.append(l2_value)
        mean_cosine_values.append(cosine_value)

    if upperarm_key in centroid_proj and upperarm_distances:
        best_upperarm_distance, best_upperarm_lead = min(upperarm_distances, key=lambda item: item[0])
        best_target_point = centroid_proj[("target", best_upperarm_lead)]
        upperarm_point = centroid_proj[upperarm_key]
        upperarm_ax.plot(
            [upperarm_point[0], best_target_point[0]],
            [upperarm_point[1], best_target_point[1]],
            color=lead_colors[best_upperarm_lead],
            linestyle="--",
            linewidth=1.4,
            alpha=0.90,
        )
        upperarm_ax.annotate(
            f"nearest target: {best_upperarm_lead}\nlatent l2={best_upperarm_distance:.3f}",
            (upperarm_point[0], upperarm_point[1]),
            textcoords="offset points",
            xytext=(10, -30),
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )

    finite_cosine_values = [value for value in mean_cosine_values if not math.isnan(value)]
    mean_l2 = float(np.mean(mean_l2_values)) if mean_l2_values else float("nan")
    mean_cos = float(np.mean(finite_cosine_values)) if finite_cosine_values else float("nan")
    x_label = str(projection_meta["x_label"])
    y_label = str(projection_meta["y_label"])
    _style_latent_axis(reference_ax, f"Reference Leads | {projection_meta['method_label']}", x_label, y_label, x_limits, y_limits)
    _style_latent_axis(upperarm_ax, "Upper-Arm vs Reference", x_label, y_label, x_limits, y_limits)
    _style_latent_axis(recon_ax, "Reconstruction vs Reference", x_label, y_label, x_limits, y_limits)

    family_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="none", markersize=7, label="target centroid"),
        Line2D([0], [0], marker="X", color="black", linestyle="none", markersize=8, label="upper-arm windows"),
        Line2D([0], [0], marker="^", markerfacecolor="white", markeredgecolor="black", linestyle="none", markersize=8, label="reconstructed windows"),
    ]
    lead_handles = [Line2D([0], [0], marker="o", color=lead_colors[lead_name], linestyle="none", markersize=7, label=lead_name) for lead_name in target_channels]
    recon_ax.legend(handles=family_handles, loc="upper right", fontsize=8, frameon=True)
    fig.legend(handles=lead_handles, loc="lower center", ncol=min(4, len(target_channels)), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        (
            f"{record.path.name} | encoder feature map | {projection_meta['method_label']} | windows/group={len(starts)} | "
            f"mean target-recon centroid l2={mean_l2:.3f} | mean cosine={mean_cos:.3f}"
        ),
        fontsize=12,
        y=1.02,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.98))
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
    visual_filter_mode: str = "recon_only",
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
    lag_metrics_enabled = bool(metrics_row.get("lag_metrics_enabled", True))
    for lead_idx, lead_name in enumerate(target_channels):
        ax = axes_array[lead_idx]
        target_display, pred_display, target_label, pred_label = _prepare_display_signals(
            target=record.target_signals[lead_idx],
            pred=reconstructed[lead_idx],
            sampling_rate_hz=float(record.sampling_rate_hz),
            visual_filter_mode=visual_filter_mode,
        )
        target = target_display[::stride]
        pred = pred_display[::stride]
        lag_samples = int(round(float(metrics_row.get(f"{lead_name}_best_lag_samples", 0.0))))
        pred_shifted = _shift_for_plot(pred_display, lag_samples)[::stride]
        ax.plot(time_plot, target, color="black", linewidth=1.0, label=target_label)
        ax.plot(time_plot, pred, color="red", linewidth=1.0, alpha=0.8, label=pred_label)
        if lag_metrics_enabled:
            ax.plot(time_plot, pred_shifted, color="#1f77b4", linewidth=1.0, alpha=0.85, linestyle="--", label="lag-corrected recon")
        corr_value = float(metrics_row.get(f"{lead_name}_{corr_method}", float("nan")))
        rmse_value = float(metrics_row.get(f"{lead_name}_rmse", float("nan")))
        if lag_metrics_enabled:
            lag_corr_value = float(metrics_row.get(f"{lead_name}_lag_corrected_{corr_method}", float("nan")))
            lag_ms_value = float(metrics_row.get(f"{lead_name}_best_lag_ms", float("nan")))
            lag_rmse_value = float(metrics_row.get(f"{lead_name}_lag_corrected_rmse", float("nan")))
            title = (
                f"{lead_name} | raw {corr_method}={corr_value:.3f} | lag {lag_ms_value:.1f} ms | "
                f"lag-{corr_method}={lag_corr_value:.3f} | raw/lag-rmse={rmse_value:.3f}/{lag_rmse_value:.3f}"
            )
        else:
            title = f"{lead_name} | {corr_method}={corr_value:.3f} | rmse={rmse_value:.3f}"
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)
        if lead_idx % cols == 0:
            ax.set_ylabel("normalized amp")

    for ax in axes_array[num_leads:]:
        ax.axis("off")

    axes_array[min(num_leads - 1, len(axes_array) - 1)].set_xlabel("time (s)")
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    if lag_metrics_enabled:
        subtitle = (
            f"{record.path.name} | original_fs={record.original_sampling_rate_hz:.1f} Hz "
            f"-> eval_fs={record.sampling_rate_hz:.1f} Hz | "
            f"mean raw {corr_method}={float(metrics_row[f'mean_{corr_method}']):.3f} | "
            f"mean lag-{corr_method}={float(metrics_row[f'mean_lag_corrected_{corr_method}']):.3f} | "
            f"mean |lag|={float(metrics_row['mean_abs_best_lag_ms']):.1f} ms"
        )
    else:
        subtitle = (
            f"{record.path.name} | original_fs={record.original_sampling_rate_hz:.1f} Hz "
            f"-> eval_fs={record.sampling_rate_hz:.1f} Hz | "
            f"mean {corr_method}={float(metrics_row[f'mean_{corr_method}']):.3f} | "
            f"mean rmse={float(metrics_row['mean_rmse']):.3f}"
        )
    fig.suptitle(subtitle, fontsize=12, y=1.01)
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
    visual_filter_mode: str = "recon_only",
    num_beats: int = 4,
    window_ms: float = 900.0,
    max_plot_samples: int = 4000,
    dpi: int = 180,
) -> tuple[str, int]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    corr_method = str(metrics_row.get("corr_method", "pearson"))
    lag_metrics_enabled = bool(metrics_row.get("lag_metrics_enabled", True))
    lead_idx, lead_name = _resolve_focus_lead(
        target_channels=target_channels,
        metrics_row=metrics_row,
        corr_method=corr_method,
        requested_lead=focus_lead,
    )

    time_s = (record.timestamps_ms.astype(np.float64) - float(record.timestamps_ms[0])) / 1000.0
    target, pred, target_label, pred_label = _prepare_display_signals(
        target=record.target_signals[lead_idx],
        pred=reconstructed[lead_idx],
        sampling_rate_hz=float(record.sampling_rate_hz),
        visual_filter_mode=visual_filter_mode,
    )
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
    lag_samples = int(round(float(metrics_row.get(f"{lead_name}_best_lag_samples", 0.0))))
    pred_shifted = _shift_for_plot(pred, lag_samples)
    overview_ax.plot(time_s[::overview_stride], target[::overview_stride], color="black", linewidth=1.0, label=target_label)
    overview_ax.plot(time_s[::overview_stride], pred[::overview_stride], color="red", linewidth=1.0, alpha=0.8, label=pred_label)
    if lag_metrics_enabled:
        overview_ax.plot(time_s[::overview_stride], pred_shifted[::overview_stride], color="#1f77b4", linewidth=1.0, alpha=0.85, linestyle="--", label="lag-corrected recon")
    for center in centers:
        left = max(0, int(center) - half_window_samples)
        right = min(target.shape[0], int(center) + half_window_samples)
        overview_ax.axvspan(time_s[left], time_s[right - 1], color="gold", alpha=0.18)
    overview_ax.set_ylabel("normalized amp")
    overview_ax.grid(alpha=0.2)
    if lag_metrics_enabled:
        overview_title = (
            f"{lead_name} full trace | raw {corr_method}={float(metrics_row[f'{lead_name}_{corr_method}']):.3f} | "
            f"lag={float(metrics_row[f'{lead_name}_best_lag_ms']):.1f} ms | "
            f"lag-{corr_method}={float(metrics_row[f'{lead_name}_lag_corrected_{corr_method}']):.3f} | "
            f"rpeak-mae={float(metrics_row[f'{lead_name}_lag_corrected_rpeak_timing_mae_ms']):.1f} ms"
        )
    else:
        overview_title = (
            f"{lead_name} full trace | {corr_method}={float(metrics_row[f'{lead_name}_{corr_method}']):.3f} | "
            f"rmse={float(metrics_row[f'{lead_name}_rmse']):.3f}"
        )
    overview_ax.set_title(overview_title, fontsize=11)

    if centers.shape[0] == 0:
        centers = np.array([target.shape[0] // 2], dtype=np.int64)

    for zoom_idx, center in enumerate(centers, start=1):
        ax = axes_array[zoom_idx]
        left = max(0, int(center) - half_window_samples)
        right = min(target.shape[0], int(center) + half_window_samples)
        segment_time_ms = (time_s[left:right] - time_s[int(center)]) * 1000.0
        ax.plot(segment_time_ms, target[left:right], color="black", linewidth=1.2, label=target_label)
        ax.plot(segment_time_ms, pred[left:right], color="red", linewidth=1.2, alpha=0.85, label=pred_label)
        if lag_metrics_enabled:
            ax.plot(segment_time_ms, pred_shifted[left:right], color="#1f77b4", linewidth=1.1, alpha=0.85, linestyle="--", label="lag-corrected recon")
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
    enable_lag_metrics: bool = True,
    lag_search_window_ms: float = 150.0,
    dtw_max_points: int = 2000,
    rpeak_tolerance_ms: float = 120.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "file_name": record.path.name,
        "original_sampling_rate_hz": record.original_sampling_rate_hz,
        "effective_sampling_rate_hz": record.sampling_rate_hz,
        "corr_method": corr_method,
        "lag_metrics_enabled": enable_lag_metrics,
        "lag_search_window_ms": lag_search_window_ms,
        "rpeak_tolerance_ms": rpeak_tolerance_ms,
    }
    lead_corr_values = []
    lead_mse_values = []
    lead_mae_values = []
    lead_rmse_values = []
    lead_lag_corr_values = []
    lead_lag_rmse_values = []
    lead_lag_mae_values = []
    lead_max_xcorr_values = []
    lead_abs_lag_ms_values = []
    lead_dtw_values = []
    lead_rpeak_mae_values = []
    lead_rpeak_match_values = []
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
        if enable_lag_metrics:
            lag_metrics = _bounded_xcorr_metrics(
                target=target,
                pred=pred,
                sampling_rate_hz=float(record.sampling_rate_hz),
                corr_method=corr_method,
                max_lag_ms=lag_search_window_ms,
            )
            lag_samples = int(round(lag_metrics["best_lag_samples"]))
            target_aligned, pred_aligned = _overlap_with_lag(target=target, pred=pred, lag_samples=lag_samples)
            dtw_target = _downsample_for_dtw(target_aligned, max_points=dtw_max_points)
            dtw_pred = _downsample_for_dtw(pred_aligned, max_points=dtw_max_points)
            band_radius = max(2, int(round(abs(lag_metrics["best_lag_ms"]) * float(record.sampling_rate_hz) / 1000.0)))
            dtw_value = _banded_dtw_distance(dtw_target, dtw_pred, band_radius=band_radius)
            rpeak_mae_ms, rpeak_match_fraction = _match_rpeak_mae_ms(
                target=target_aligned,
                pred=pred_aligned,
                sampling_rate_hz=float(record.sampling_rate_hz),
                tolerance_ms=rpeak_tolerance_ms,
            )
        else:
            lag_metrics = {
                "max_xcorr": float("nan"),
                "best_lag_samples": 0,
                "best_lag_ms": 0.0,
                f"lag_corrected_{corr_method}": corr,
                "lag_corrected_rmse": rmse,
                "lag_corrected_mae": mae,
            }
            dtw_value = float("nan")
            rpeak_mae_ms = float("nan")
            rpeak_match_fraction = float("nan")
        row[f"{lead_name}_max_xcorr"] = lag_metrics["max_xcorr"]
        row[f"{lead_name}_best_lag_samples"] = lag_metrics["best_lag_samples"]
        row[f"{lead_name}_best_lag_ms"] = lag_metrics["best_lag_ms"]
        row[f"{lead_name}_lag_corrected_{corr_method}"] = lag_metrics[f"lag_corrected_{corr_method}"]
        row[f"{lead_name}_lag_corrected_rmse"] = lag_metrics["lag_corrected_rmse"]
        row[f"{lead_name}_lag_corrected_mae"] = lag_metrics["lag_corrected_mae"]
        row[f"{lead_name}_dtw_distance"] = dtw_value
        row[f"{lead_name}_lag_corrected_rpeak_timing_mae_ms"] = rpeak_mae_ms
        row[f"{lead_name}_lag_corrected_rpeak_match_fraction"] = rpeak_match_fraction
        lead_mse_values.append(mse)
        lead_mae_values.append(mae)
        lead_rmse_values.append(rmse)
        lead_corr_values.append(corr)
        lead_lag_corr_values.append(lag_metrics[f"lag_corrected_{corr_method}"])
        lead_lag_rmse_values.append(lag_metrics["lag_corrected_rmse"])
        lead_lag_mae_values.append(lag_metrics["lag_corrected_mae"])
        lead_max_xcorr_values.append(lag_metrics["max_xcorr"])
        lead_abs_lag_ms_values.append(abs(lag_metrics["best_lag_ms"]))
        lead_dtw_values.append(dtw_value)
        lead_rpeak_mae_values.append(rpeak_mae_ms)
        lead_rpeak_match_values.append(rpeak_match_fraction)
    row["mean_mse"] = float(np.mean(lead_mse_values))
    row["mean_mae"] = float(np.mean(lead_mae_values))
    row["mean_rmse"] = float(np.mean(lead_rmse_values))
    row[f"mean_{corr_method}"] = float(np.nanmean(lead_corr_values))
    row["mean_max_xcorr"] = float(np.nanmean(lead_max_xcorr_values)) if enable_lag_metrics else float("nan")
    row["mean_abs_best_lag_ms"] = float(np.nanmean(lead_abs_lag_ms_values))
    row[f"mean_lag_corrected_{corr_method}"] = float(np.nanmean(lead_lag_corr_values))
    row["mean_lag_corrected_rmse"] = float(np.nanmean(lead_lag_rmse_values))
    row["mean_lag_corrected_mae"] = float(np.nanmean(lead_lag_mae_values))
    row["mean_dtw_distance"] = float(np.nanmean(lead_dtw_values)) if enable_lag_metrics else float("nan")
    row["mean_lag_corrected_rpeak_timing_mae_ms"] = (
        float(np.nanmean(lead_rpeak_mae_values)) if enable_lag_metrics else float("nan")
    )
    row["mean_lag_corrected_rpeak_match_fraction"] = (
        float(np.nanmean(lead_rpeak_match_values)) if enable_lag_metrics else float("nan")
    )
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

    dataset_type = str(data_cfg.get("dataset_type", "ptbxl"))
    if not is_upperarm_dataset_type(dataset_type):
        raise ValueError("reconstruct_upperarm requires an upper-arm dataset type")

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
    visual_filter_mode = str(reconstruct_cfg.get("visual_filter_mode", "recon_only"))
    save_latent_plots = bool(reconstruct_cfg.get("save_latent_plots", True))
    latent_plot_dir_value = reconstruct_cfg.get("latent_plot_dir")
    latent_plot_dir = ensure_dir(latent_plot_dir_value) if latent_plot_dir_value else output_dir / "latent_plots"
    latent_max_windows_per_signal = int(reconstruct_cfg.get("latent_max_windows_per_signal", 24))
    latent_plot_dpi = int(reconstruct_cfg.get("latent_plot_dpi", max(plot_dpi, 180)))
    latent_projection_method = str(reconstruct_cfg.get("latent_projection_method", "umap_like"))
    latent_projection_neighbors = int(reconstruct_cfg.get("latent_projection_neighbors", 12))
    latent_projection_seed = int(reconstruct_cfg.get("latent_projection_seed", config.get("seed", 42)))
    enable_lag_metrics = bool(reconstruct_cfg.get("enable_lag_metrics", True))
    lag_search_window_ms = float(reconstruct_cfg.get("lag_search_window_ms", 150.0))
    dtw_max_points = int(reconstruct_cfg.get("dtw_max_points", 2000))
    rpeak_tolerance_ms = float(reconstruct_cfg.get("rpeak_tolerance_ms", 120.0))

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
        dataset_type=dataset_type,
        segment_group_regex=data_cfg.get("segment_group_regex", r"^(?P<record>.+)_\d+s$"),
        segment_offset_regex=data_cfg.get("segment_offset_regex", r"_(?P<offset_seconds>\d+)s$"),
        npz_timestamp_key=data_cfg.get("npz_timestamp_key", "timestamp_ms"),
        npz_sampling_rate_key=data_cfg.get("npz_sampling_rate_key", "sampling_rate_hz"),
        npz_start_time_key=data_cfg.get("npz_start_time_key", "start_time_ms"),
        npz_signal_matrix_key=data_cfg.get("npz_signal_matrix_key"),
        npz_channel_names_key=data_cfg.get("npz_channel_names_key"),
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
            enable_lag_metrics=enable_lag_metrics,
            lag_search_window_ms=lag_search_window_ms,
            dtw_max_points=dtw_max_points,
            rpeak_tolerance_ms=rpeak_tolerance_ms,
        )
        if save_plots:
            plot_path = plot_dir / f"{record.path.stem}_comparison.png"
            save_reconstruction_comparison_plot(
                output_path=plot_path,
                record=record,
                target_channels=target_channels,
                reconstructed=y_hat_full,
                metrics_row=row,
                visual_filter_mode=visual_filter_mode,
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
                visual_filter_mode=visual_filter_mode,
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
                projection_method=latent_projection_method,
                projection_neighbors=latent_projection_neighbors,
                projection_seed=latent_projection_seed,
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
        "lag_metrics_enabled",
        "lag_search_window_ms",
        "rpeak_tolerance_ms",
        "plot_path",
        "focus_lead",
        "focus_plot_path",
        "latent_plot_path",
        "mean_mse",
        "mean_mae",
        "mean_rmse",
        f"mean_{corr_method}",
        "mean_max_xcorr",
        "mean_abs_best_lag_ms",
        f"mean_lag_corrected_{corr_method}",
        "mean_lag_corrected_rmse",
        "mean_lag_corrected_mae",
        "mean_dtw_distance",
        "mean_lag_corrected_rpeak_timing_mae_ms",
        "mean_lag_corrected_rpeak_match_fraction",
    ]
    for lead_name in target_channels:
        fieldnames.extend(
            [
                f"{lead_name}_mse",
                f"{lead_name}_mae",
                f"{lead_name}_rmse",
                f"{lead_name}_{corr_method}",
                f"{lead_name}_max_xcorr",
                f"{lead_name}_best_lag_samples",
                f"{lead_name}_best_lag_ms",
                f"{lead_name}_lag_corrected_{corr_method}",
                f"{lead_name}_lag_corrected_rmse",
                f"{lead_name}_lag_corrected_mae",
                f"{lead_name}_dtw_distance",
                f"{lead_name}_lag_corrected_rpeak_timing_mae_ms",
                f"{lead_name}_lag_corrected_rpeak_match_fraction",
            ]
        )
    _append_metrics(metrics_path, metrics_rows, fieldnames=fieldnames)
    print(f"Saved reconstruction CSVs and metrics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
