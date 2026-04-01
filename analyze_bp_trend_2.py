#!/usr/bin/env python3
"""
Analyze subject-level BP trend metrics from calibrated event-level predictions.

Supports:
  - event-level analysis: use each BP event / scatter point directly
  - hourly-level analysis: aggregate each subject's points within each hour by mean,
    then compute per-subject trend metrics on the hourly curves

Primary metric:
  - per-subject correlation using --corr_method (pearson or spearman)

Optional extra metrics (individually selectable):
  - spearman: rank correlation on aligned points
  - lagged_corr: max lag-tolerant correlation within a small lag window
  - direction_acc: direction-of-change agreement based on first differences
  - dtw: z-normalized DTW distance (lower is better)

Typical usage:
    python analyze_bp_trend.py \
        --input_csv your_calibrated_test_results.csv \
        --output_dir bp_trend_analysis \
        --analysis_mode both \
        --id_col id_upper \
        --time_col t_bp_ms \
        --sbp_true_col y_true_sbp \
        --sbp_pred_col y_pred_sbp_aff \
        --dbp_true_col y_true_dbp \
        --dbp_pred_col y_pred_dbp_aff \
        --extra_metrics spearman lagged_corr direction_acc dtw \
        --lag_max 1 \
        --plot_all_subjects

Main outputs (for each analysis level):
    - per_subject_trend_correlations.csv
    - summary.json
    - summary.txt
    - primary_correlation_histograms.png
    - optional histogram pngs for each extra metric
    - trend_plots_selected/*.png
    - trend_plots_all/*.png  (if --plot_all_subjects is set)

Notes:
    - By default, only non-calibration points are used (is_calib == False).
    - Metrics are computed per subject after sorting by time.
    - Cohort metrics are means / medians across per-subject metrics.
    - For DTW, lower distance indicates better trend-shape similarity.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# =========================
# Basic helpers
# =========================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def safe_bool_mask(series: pd.Series) -> pd.Series:
    """Convert common truthy/falsy representations into a boolean mask."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    lowered = series.astype(str).str.strip().str.lower()
    true_set = {"true", "1", "yes", "y", "t"}
    false_set = {"false", "0", "no", "n", "f", "nan", "none", ""}

    if lowered.isin(true_set | false_set).all():
        return lowered.isin(true_set)

    return series.fillna(False).astype(bool)



def validate_required_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")



def sanitize_subject_id(subject_id: object) -> str:
    text = str(subject_id)
    for old, new in [("/", "_"), ("\\", "_"), (" ", "_"), (":", "_"), ("*", "_")]:
        text = text.replace(old, new)
    return text



def _nan_if_not_finite(x: float) -> float:
    return float(x) if np.isfinite(x) else np.nan



def z_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-8:
        return arr - mean
    return (arr - mean) / std


# =========================
# Pre-filtering and aggregation
# =========================


def prefilter_analysis_df(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    is_calib_col: str,
    use_non_calib_only: bool,
) -> pd.DataFrame:
    data = df.copy()
    validate_required_columns(data, [id_col, time_col])

    if use_non_calib_only and is_calib_col in data.columns:
        data = data.loc[~safe_bool_mask(data[is_calib_col])].copy()

    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data = data.dropna(subset=[id_col, time_col]).copy()
    data = data.sort_values([id_col, time_col]).reset_index(drop=True)
    return data



def aggregate_to_hourly_bp(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    sbp_true_col: str,
    sbp_pred_col: str,
    dbp_true_col: str,
    dbp_pred_col: str,
) -> pd.DataFrame:
    """Aggregate event-level BP points to hourly means for each subject."""
    required = [id_col, time_col, sbp_true_col, sbp_pred_col, dbp_true_col, dbp_pred_col]
    validate_required_columns(df, required)

    data = df.copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data = data.dropna(subset=[id_col, time_col]).copy()

    dt = pd.to_datetime(data[time_col], unit="ms", errors="coerce")
    data = data.loc[dt.notna()].copy()
    data["_time_dt"] = dt.loc[dt.notna()].values
    data["hour_start_dt"] = data["_time_dt"].dt.floor("h")
    data["hour_start_ms"] = (data["hour_start_dt"].astype("int64") // 10**6).astype("int64")

    agg_df = (
        data.groupby([id_col, "hour_start_ms", "hour_start_dt"], as_index=False)
        .agg(
            n_events=(time_col, "size"),
            **{
                sbp_true_col: (sbp_true_col, "mean"),
                sbp_pred_col: (sbp_pred_col, "mean"),
                dbp_true_col: (dbp_true_col, "mean"),
                dbp_pred_col: (dbp_pred_col, "mean"),
            },
        )
        .sort_values([id_col, "hour_start_ms"])
        .reset_index(drop=True)
    )
    return agg_df


# =========================
# Metric computations
# =========================


def compute_corr_value(x: np.ndarray, y: np.ndarray, method: str) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        return np.nan
    if np.unique(x).size <= 1 or np.unique(y).size <= 1:
        return np.nan

    if method == "pearson":
        r, _ = pearsonr(x, y)
        return _nan_if_not_finite(r)
    if method == "spearman":
        r, _ = spearmanr(x, y)
        return _nan_if_not_finite(r)
    raise ValueError(f"Unsupported correlation method: {method}")



def compute_corr_with_pvalue(x: np.ndarray, y: np.ndarray, method: str) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2 or np.unique(x).size <= 1 or np.unique(y).size <= 1:
        return np.nan, np.nan

    if method == "pearson":
        r, p = pearsonr(x, y)
        return _nan_if_not_finite(r), _nan_if_not_finite(p)
    if method == "spearman":
        r, p = spearmanr(x, y)
        return _nan_if_not_finite(r), _nan_if_not_finite(p)
    raise ValueError(f"Unsupported correlation method: {method}")



def compute_lagged_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    max_lag: int,
    method: str,
    min_points: int,
) -> Tuple[float, float]:
    """
    Return (best_corr, best_lag).

    Lag definition:
      positive lag  -> prediction lags behind truth by `lag` points
      negative lag  -> prediction leads truth by `abs(lag)` points
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) < min_points:
        return np.nan, np.nan

    best_r = np.nan
    best_lag = np.nan
    best_abs_r = -np.inf

    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            xt = y_true
            yp = y_pred
        elif lag > 0:
            xt = y_true[:-lag]
            yp = y_pred[lag:]
        else:
            xt = y_true[-lag:]
            yp = y_pred[:lag]

        if len(xt) < min_points:
            continue

        r = compute_corr_value(xt, yp, method=method)
        if not np.isfinite(r):
            continue

        abs_r = abs(r)
        if abs_r > best_abs_r:
            best_abs_r = abs_r
            best_r = float(r)
            best_lag = float(lag)

    return best_r, best_lag



def compute_direction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    zero_policy: str = "ignore",
) -> Tuple[float, int]:
    """
    Compare the signs of first differences.

    zero_policy:
      - ignore: ignore steps where either diff is 0
      - keep: keep zeros; equality of signs counts as correct
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) < 2:
        return np.nan, 0

    dt = np.diff(y_true)
    dp = np.diff(y_pred)
    st = np.sign(dt)
    sp = np.sign(dp)

    if zero_policy == "ignore":
        mask = (st != 0) & (sp != 0)
    elif zero_policy == "keep":
        mask = np.ones_like(st, dtype=bool)
    else:
        raise ValueError(f"Unsupported zero_policy: {zero_policy}")

    if mask.sum() == 0:
        return np.nan, 0

    acc = np.mean(st[mask] == sp[mask])
    return float(acc), int(mask.sum())



def compute_dtw_distance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    z_norm: bool = True,
    window: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Standard DTW distance normalized by warping path length.
    Lower is better.
    """
    x = np.asarray(y_true, dtype=np.float64)
    y = np.asarray(y_pred, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return np.nan, 0

    if z_norm:
        x = z_normalize(x)
        y = z_normalize(y)

    n, m = len(x), len(y)
    if window is None:
        window = max(n, m)
    window = max(window, abs(n - m))

    inf = np.inf
    cost = np.full((n + 1, m + 1), inf, dtype=np.float64)
    steps = np.full((n + 1, m + 1), 0, dtype=np.int32)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            dist = abs(x[i - 1] - y[j - 1])
            prev_costs = (cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            prev_idx = int(np.argmin(prev_costs))
            if prev_idx == 0:
                prev_cost = cost[i - 1, j]
                prev_steps = steps[i - 1, j]
            elif prev_idx == 1:
                prev_cost = cost[i, j - 1]
                prev_steps = steps[i, j - 1]
            else:
                prev_cost = cost[i - 1, j - 1]
                prev_steps = steps[i - 1, j - 1]

            cost[i, j] = dist + prev_cost
            steps[i, j] = prev_steps + 1

    final_cost = cost[n, m]
    final_steps = int(steps[n, m])
    if not np.isfinite(final_cost) or final_steps <= 0:
        return np.nan, 0
    return float(final_cost / final_steps), final_steps


# =========================
# Subject-level analysis
# =========================


def compute_subject_trend_correlations(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    sbp_true_col: str,
    sbp_pred_col: str,
    dbp_true_col: str,
    dbp_pred_col: str,
    corr_method: str,
    min_points: int,
    extra_metrics: Sequence[str],
    lag_max: int,
    lag_corr_method: str,
    direction_zero_policy: str,
    dtw_z_normalize: bool,
    dtw_window: Optional[int],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute per-subject BP trend metrics."""
    required = [id_col, time_col, sbp_true_col, sbp_pred_col, dbp_true_col, dbp_pred_col]
    validate_required_columns(df, required)

    data = df.copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data = data.dropna(subset=[id_col, time_col]).copy()
    data = data.sort_values([id_col, time_col]).reset_index(drop=True)

    extra_metrics = tuple(extra_metrics)
    rows: List[Dict[str, float]] = []

    for subject_id, g in data.groupby(id_col, sort=False):
        g = g.sort_values(time_col).copy()

        sbp_g = g[[time_col, sbp_true_col, sbp_pred_col]].dropna().copy()
        dbp_g = g[[time_col, dbp_true_col, dbp_pred_col]].dropna().copy()

        row: Dict[str, float] = {
            id_col: subject_id,
            "n_points_total": int(len(g)),
            "n_points_sbp": int(len(sbp_g)),
            "n_points_dbp": int(len(dbp_g)),
        }
        if "n_events" in g.columns:
            row["n_events_total"] = int(g["n_events"].sum())
            row["mean_events_per_point"] = float(g["n_events"].mean())

        # Primary metric with p-value
        if len(sbp_g) >= min_points:
            r_sbp, p_sbp = compute_corr_with_pvalue(
                sbp_g[sbp_true_col].to_numpy(),
                sbp_g[sbp_pred_col].to_numpy(),
                method=corr_method,
            )
        else:
            r_sbp, p_sbp = np.nan, np.nan

        if len(dbp_g) >= min_points:
            r_dbp, p_dbp = compute_corr_with_pvalue(
                dbp_g[dbp_true_col].to_numpy(),
                dbp_g[dbp_pred_col].to_numpy(),
                method=corr_method,
            )
        else:
            r_dbp, p_dbp = np.nan, np.nan

        row["r_sbp"] = r_sbp
        row["p_sbp"] = p_sbp
        row["r_dbp"] = r_dbp
        row["p_dbp"] = p_dbp

        # Extra metrics
        if "spearman" in extra_metrics:
            row["spearman_r_sbp"] = (
                compute_corr_value(sbp_g[sbp_true_col].to_numpy(), sbp_g[sbp_pred_col].to_numpy(), method="spearman")
                if len(sbp_g) >= min_points else np.nan
            )
            row["spearman_r_dbp"] = (
                compute_corr_value(dbp_g[dbp_true_col].to_numpy(), dbp_g[dbp_pred_col].to_numpy(), method="spearman")
                if len(dbp_g) >= min_points else np.nan
            )

        if "lagged_corr" in extra_metrics:
            if len(sbp_g) >= min_points:
                lag_r_sbp, best_lag_sbp = compute_lagged_correlation(
                    sbp_g[sbp_true_col].to_numpy(),
                    sbp_g[sbp_pred_col].to_numpy(),
                    max_lag=lag_max,
                    method=lag_corr_method,
                    min_points=min_points,
                )
            else:
                lag_r_sbp, best_lag_sbp = np.nan, np.nan
            if len(dbp_g) >= min_points:
                lag_r_dbp, best_lag_dbp = compute_lagged_correlation(
                    dbp_g[dbp_true_col].to_numpy(),
                    dbp_g[dbp_pred_col].to_numpy(),
                    max_lag=lag_max,
                    method=lag_corr_method,
                    min_points=min_points,
                )
            else:
                lag_r_dbp, best_lag_dbp = np.nan, np.nan

            row["lagged_r_sbp"] = lag_r_sbp
            row["best_lag_sbp"] = best_lag_sbp
            row["lagged_r_dbp"] = lag_r_dbp
            row["best_lag_dbp"] = best_lag_dbp

        if "direction_acc" in extra_metrics:
            if len(sbp_g) >= min_points:
                dir_acc_sbp, dir_n_sbp = compute_direction_accuracy(
                    sbp_g[sbp_true_col].to_numpy(),
                    sbp_g[sbp_pred_col].to_numpy(),
                    zero_policy=direction_zero_policy,
                )
            else:
                dir_acc_sbp, dir_n_sbp = np.nan, 0
            if len(dbp_g) >= min_points:
                dir_acc_dbp, dir_n_dbp = compute_direction_accuracy(
                    dbp_g[dbp_true_col].to_numpy(),
                    dbp_g[dbp_pred_col].to_numpy(),
                    zero_policy=direction_zero_policy,
                )
            else:
                dir_acc_dbp, dir_n_dbp = np.nan, 0

            row["direction_acc_sbp"] = dir_acc_sbp
            row["direction_n_sbp"] = int(dir_n_sbp)
            row["direction_acc_dbp"] = dir_acc_dbp
            row["direction_n_dbp"] = int(dir_n_dbp)

        if "dtw" in extra_metrics:
            if len(sbp_g) >= min_points:
                dtw_sbp, dtw_steps_sbp = compute_dtw_distance(
                    sbp_g[sbp_true_col].to_numpy(),
                    sbp_g[sbp_pred_col].to_numpy(),
                    z_norm=dtw_z_normalize,
                    window=dtw_window,
                )
            else:
                dtw_sbp, dtw_steps_sbp = np.nan, 0
            if len(dbp_g) >= min_points:
                dtw_dbp, dtw_steps_dbp = compute_dtw_distance(
                    dbp_g[dbp_true_col].to_numpy(),
                    dbp_g[dbp_pred_col].to_numpy(),
                    z_norm=dtw_z_normalize,
                    window=dtw_window,
                )
            else:
                dtw_dbp, dtw_steps_dbp = np.nan, 0

            row["dtw_dist_sbp"] = dtw_sbp
            row["dtw_steps_sbp"] = int(dtw_steps_sbp)
            row["dtw_dist_dbp"] = dtw_dbp
            row["dtw_steps_dbp"] = int(dtw_steps_dbp)

        rows.append(row)

    per_subject_df = pd.DataFrame(rows)

    summary: Dict[str, float] = {
        "corr_method": corr_method,
        "min_points": int(min_points),
        "extra_metrics": list(extra_metrics),
        "lag_max": int(lag_max),
        "lag_corr_method": lag_corr_method,
        "direction_zero_policy": direction_zero_policy,
        "dtw_z_normalize": bool(dtw_z_normalize),
        "dtw_window": None if dtw_window is None else int(dtw_window),
        "n_rows_after_filter": int(len(data)),
        "n_subjects_total": int(len(per_subject_df)),
    }

    if len(per_subject_df) == 0:
        return per_subject_df, summary

    # Primary metric summary
    summary.update({
        "n_subjects_valid_sbp": int(per_subject_df["r_sbp"].notna().sum()),
        "n_subjects_valid_dbp": int(per_subject_df["r_dbp"].notna().sum()),
        "mean_r_sbp": float(per_subject_df["r_sbp"].mean(skipna=True)),
        "std_r_sbp": float(per_subject_df["r_sbp"].std(skipna=True)),
        "median_r_sbp": float(per_subject_df["r_sbp"].median(skipna=True)),
        "mean_r_dbp": float(per_subject_df["r_dbp"].mean(skipna=True)),
        "std_r_dbp": float(per_subject_df["r_dbp"].std(skipna=True)),
        "median_r_dbp": float(per_subject_df["r_dbp"].median(skipna=True)),
    })

    def add_metric_summary(prefix: str, sbp_col: str, dbp_col: str) -> None:
        if sbp_col not in per_subject_df.columns or dbp_col not in per_subject_df.columns:
            return
        summary[f"n_subjects_valid_{prefix}_sbp"] = int(per_subject_df[sbp_col].notna().sum())
        summary[f"n_subjects_valid_{prefix}_dbp"] = int(per_subject_df[dbp_col].notna().sum())
        summary[f"mean_{prefix}_sbp"] = float(per_subject_df[sbp_col].mean(skipna=True))
        summary[f"std_{prefix}_sbp"] = float(per_subject_df[sbp_col].std(skipna=True))
        summary[f"median_{prefix}_sbp"] = float(per_subject_df[sbp_col].median(skipna=True))
        summary[f"mean_{prefix}_dbp"] = float(per_subject_df[dbp_col].mean(skipna=True))
        summary[f"std_{prefix}_dbp"] = float(per_subject_df[dbp_col].std(skipna=True))
        summary[f"median_{prefix}_dbp"] = float(per_subject_df[dbp_col].median(skipna=True))

    add_metric_summary("spearman_r", "spearman_r_sbp", "spearman_r_dbp")
    add_metric_summary("lagged_r", "lagged_r_sbp", "lagged_r_dbp")
    add_metric_summary("direction_acc", "direction_acc_sbp", "direction_acc_dbp")
    add_metric_summary("dtw_dist", "dtw_dist_sbp", "dtw_dist_dbp")

    if "best_lag_sbp" in per_subject_df.columns:
        summary["mean_best_lag_sbp"] = float(per_subject_df["best_lag_sbp"].mean(skipna=True))
        summary["median_best_lag_sbp"] = float(per_subject_df["best_lag_sbp"].median(skipna=True))
    if "best_lag_dbp" in per_subject_df.columns:
        summary["mean_best_lag_dbp"] = float(per_subject_df["best_lag_dbp"].mean(skipna=True))
        summary["median_best_lag_dbp"] = float(per_subject_df["best_lag_dbp"].median(skipna=True))

    return per_subject_df, summary


# =========================
# Plotting
# =========================


def _prepare_subject_data(
    df: pd.DataFrame,
    *,
    subject_id: object,
    id_col: str,
    time_col: str,
    sbp_true_col: str,
    sbp_pred_col: str,
    dbp_true_col: str,
    dbp_pred_col: str,
) -> pd.DataFrame:
    data = df.copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data = data.dropna(subset=[id_col, time_col]).copy()
    g = data.loc[data[id_col] == subject_id].copy()
    if len(g) == 0:
        raise ValueError(f"Subject not found after filtering: {subject_id}")

    g = g.sort_values(time_col).copy()
    time_dt = pd.to_datetime(g[time_col], unit="ms", errors="coerce")
    if time_dt.notna().all():
        g["time_dt"] = time_dt.values
    else:
        g["time_dt"] = np.arange(len(g))

    keep_cols = [
        id_col,
        time_col,
        "time_dt",
        sbp_true_col,
        sbp_pred_col,
        dbp_true_col,
        dbp_pred_col,
    ]
    if "n_events" in g.columns:
        keep_cols.append("n_events")
    keep_cols = [c for c in keep_cols if c in g.columns]
    return g[keep_cols].copy()



def _format_metric_for_title(value: float, name: str) -> str:
    if not np.isfinite(value):
        return f"{name}=NA"
    return f"{name}={value:.3f}"



def plot_subject_bp_trend(
    df: pd.DataFrame,
    *,
    subject_id: object,
    id_col: str,
    time_col: str,
    sbp_true_col: str,
    sbp_pred_col: str,
    dbp_true_col: str,
    dbp_pred_col: str,
    corr_method: str = "pearson",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Path] = None,
) -> None:
    g = _prepare_subject_data(
        df,
        subject_id=subject_id,
        id_col=id_col,
        time_col=time_col,
        sbp_true_col=sbp_true_col,
        sbp_pred_col=sbp_pred_col,
        dbp_true_col=dbp_true_col,
        dbp_pred_col=dbp_pred_col,
    )

    sbp_valid = g[[sbp_true_col, sbp_pred_col]].dropna()
    dbp_valid = g[[dbp_true_col, dbp_pred_col]].dropna()

    r_sbp = compute_corr_value(sbp_valid[sbp_true_col].to_numpy(), sbp_valid[sbp_pred_col].to_numpy(), method=corr_method)
    r_dbp = compute_corr_value(dbp_valid[dbp_true_col].to_numpy(), dbp_valid[dbp_pred_col].to_numpy(), method=corr_method)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(g["time_dt"], g[sbp_true_col], marker="o", label="True SBP")
    axes[0].plot(g["time_dt"], g[sbp_pred_col], marker="o", label="Pred SBP")
    axes[0].set_ylabel("SBP (mmHg)")
    axes[0].set_title(f"Subject {subject_id} | SBP trend | {_format_metric_for_title(r_sbp, corr_method + ' r')}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(g["time_dt"], g[dbp_true_col], marker="o", label="True DBP")
    axes[1].plot(g["time_dt"], g[dbp_pred_col], marker="o", label="Pred DBP")
    axes[1].set_ylabel("DBP (mmHg)")
    axes[1].set_xlabel("Time")
    axes[1].set_title(f"Subject {subject_id} | DBP trend | {_format_metric_for_title(r_dbp, corr_method + ' r')}")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()



def plot_metric_histograms(
    per_subject_df: pd.DataFrame,
    *,
    sbp_col: str,
    dbp_col: str,
    title_prefix: str,
    xlabel: str,
    save_path: Path,
) -> None:
    if sbp_col not in per_subject_df.columns or dbp_col not in per_subject_df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(per_subject_df[sbp_col].dropna(), bins=20)
    axes[0].set_title(f"SBP {title_prefix}")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(per_subject_df[dbp_col].dropna(), bins=20)
    axes[1].set_title(f"DBP {title_prefix}")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================
# Reporting helpers
# =========================


def save_summary(summary: Dict[str, float], output_dir: Path) -> None:
    json_path = output_dir / "summary.json"
    txt_path = output_dir / "summary.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = ["=== BP Trend Correlation Summary ==="]
    for k, v in summary.items():
        lines.append(f"{k}: {v}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def print_summary(summary: Dict[str, float], prefix: str = "") -> None:
    print(f"{prefix}=== BP Trend Correlation Summary ===")
    for k, v in summary.items():
        print(f"{prefix}{k}: {v}")


# =========================
# Subject selection
# =========================


def select_subjects_for_plot(
    per_subject_df: pd.DataFrame,
    *,
    id_col: str,
    metric_col: str,
    n_best: int,
    n_worst: int,
    manual_subjects: Optional[Sequence[str]] = None,
) -> List[str]:
    selected: List[str] = []

    if manual_subjects:
        selected.extend([str(x) for x in manual_subjects])

    if metric_col not in per_subject_df.columns:
        fallback_cols = [c for c in ["r_sbp", "r_dbp"] if c in per_subject_df.columns]
        if fallback_cols:
            print(f"[WARN] plot_metric '{metric_col}' not found. Falling back to '{fallback_cols[0]}'.")
            metric_col = fallback_cols[0]
        else:
            return list(dict.fromkeys(selected))

    valid = per_subject_df.dropna(subset=[metric_col]).copy()
    higher_is_better = not metric_col.startswith("dtw_")

    if len(valid) > 0:
        if n_best > 0:
            selected.extend(
                valid.sort_values(metric_col, ascending=not higher_is_better)
                .head(n_best)[id_col].astype(str).tolist()
            )
        if n_worst > 0:
            selected.extend(
                valid.sort_values(metric_col, ascending=higher_is_better)
                .head(n_worst)[id_col].astype(str).tolist()
            )

    # Deduplicate while preserving order.
    seen = set()
    deduped: List[str] = []
    for sid in selected:
        if sid not in seen:
            seen.add(sid)
            deduped.append(sid)
    return deduped


# =========================
# Analysis runner
# =========================


def run_single_analysis(
    raw_df: pd.DataFrame,
    *,
    output_dir: Path,
    analysis_name: str,
    level_mode: str,
    id_col: str,
    time_col: str,
    is_calib_col: str,
    sbp_true_col: str,
    sbp_pred_col: str,
    dbp_true_col: str,
    dbp_pred_col: str,
    corr_method: str,
    min_points: int,
    include_calib: bool,
    extra_metrics: Sequence[str],
    lag_max: int,
    lag_corr_method: str,
    direction_zero_policy: str,
    dtw_z_normalize: bool,
    dtw_window: Optional[int],
    plot_metric: str,
    plot_best_n: int,
    plot_worst_n: int,
    plot_subjects: Optional[Sequence[str]],
    no_plots: bool,
    plot_all_subjects: bool,
) -> None:
    ensure_dir(output_dir)

    data = prefilter_analysis_df(
        raw_df,
        id_col=id_col,
        time_col=time_col,
        is_calib_col=is_calib_col,
        use_non_calib_only=not include_calib,
    )

    if level_mode == "event":
        analysis_df = data.copy()
        analysis_time_col = time_col
    elif level_mode == "hourly":
        analysis_df = aggregate_to_hourly_bp(
            data,
            id_col=id_col,
            time_col=time_col,
            sbp_true_col=sbp_true_col,
            sbp_pred_col=sbp_pred_col,
            dbp_true_col=dbp_true_col,
            dbp_pred_col=dbp_pred_col,
        )
        analysis_time_col = "hour_start_ms"
        analysis_df.to_csv(output_dir / "hourly_aggregated_points.csv", index=False)
    else:
        raise ValueError(f"Unsupported level_mode: {level_mode}")

    per_subject_df, summary = compute_subject_trend_correlations(
        analysis_df,
        id_col=id_col,
        time_col=analysis_time_col,
        sbp_true_col=sbp_true_col,
        sbp_pred_col=sbp_pred_col,
        dbp_true_col=dbp_true_col,
        dbp_pred_col=dbp_pred_col,
        corr_method=corr_method,
        min_points=min_points,
        extra_metrics=extra_metrics,
        lag_max=lag_max,
        lag_corr_method=lag_corr_method,
        direction_zero_policy=direction_zero_policy,
        dtw_z_normalize=dtw_z_normalize,
        dtw_window=dtw_window,
    )

    summary["analysis_name"] = analysis_name
    summary["level_mode"] = level_mode
    summary["include_calib"] = bool(include_calib)
    summary["n_subjects_plotted_selected"] = 0
    summary["n_subjects_plotted_all"] = 0

    per_subject_path = output_dir / "per_subject_trend_correlations.csv"
    per_subject_df.to_csv(per_subject_path, index=False)
    save_summary(summary, output_dir)
    print_summary(summary, prefix=f"{analysis_name} | ")
    print(f"Saved per-subject metrics to: {per_subject_path}")

    if no_plots:
        return

    if len(per_subject_df) > 0:
        plot_metric_histograms(
            per_subject_df,
            sbp_col="r_sbp",
            dbp_col="r_dbp",
            title_prefix=f"primary {corr_method} correlation",
            xlabel="Correlation",
            save_path=output_dir / "primary_correlation_histograms.png",
        )
        if "spearman" in extra_metrics:
            plot_metric_histograms(
                per_subject_df,
                sbp_col="spearman_r_sbp",
                dbp_col="spearman_r_dbp",
                title_prefix="Spearman correlation",
                xlabel="Spearman r",
                save_path=output_dir / "spearman_histograms.png",
            )
        if "lagged_corr" in extra_metrics:
            plot_metric_histograms(
                per_subject_df,
                sbp_col="lagged_r_sbp",
                dbp_col="lagged_r_dbp",
                title_prefix=f"lagged {lag_corr_method} correlation",
                xlabel="Max lagged correlation",
                save_path=output_dir / "lagged_correlation_histograms.png",
            )
        if "direction_acc" in extra_metrics:
            plot_metric_histograms(
                per_subject_df,
                sbp_col="direction_acc_sbp",
                dbp_col="direction_acc_dbp",
                title_prefix="direction agreement",
                xlabel="Direction accuracy",
                save_path=output_dir / "direction_accuracy_histograms.png",
            )
        if "dtw" in extra_metrics:
            plot_metric_histograms(
                per_subject_df,
                sbp_col="dtw_dist_sbp",
                dbp_col="dtw_dist_dbp",
                title_prefix="DTW distance",
                xlabel="DTW distance (lower is better)",
                save_path=output_dir / "dtw_distance_histograms.png",
            )

    plot_ids = select_subjects_for_plot(
        per_subject_df,
        id_col=id_col,
        metric_col=plot_metric,
        n_best=plot_best_n,
        n_worst=plot_worst_n,
        manual_subjects=plot_subjects,
    )

    selected_dir = output_dir / "trend_plots_selected"
    ensure_dir(selected_dir)
    n_selected_saved = 0

    for sid in plot_ids:
        save_path = selected_dir / f"subject_{sanitize_subject_id(sid)}.png"
        try:
            plot_subject_bp_trend(
                analysis_df,
                subject_id=sid,
                id_col=id_col,
                time_col=analysis_time_col,
                sbp_true_col=sbp_true_col,
                sbp_pred_col=sbp_pred_col,
                dbp_true_col=dbp_true_col,
                dbp_pred_col=dbp_pred_col,
                corr_method=corr_method,
                save_path=save_path,
            )
            n_selected_saved += 1
        except Exception as exc:
            print(f"[WARN] Failed to plot selected subject {sid}: {exc}")

    print(f"Saved {n_selected_saved} selected trend plots to: {selected_dir}")

    n_all_saved = 0
    if plot_all_subjects and len(per_subject_df) > 0:
        all_dir = output_dir / "trend_plots_all"
        ensure_dir(all_dir)
        all_subject_ids: Iterable[object] = per_subject_df[id_col].tolist()
        for sid in all_subject_ids:
            save_path = all_dir / f"subject_{sanitize_subject_id(sid)}.png"
            try:
                plot_subject_bp_trend(
                    analysis_df,
                    subject_id=sid,
                    id_col=id_col,
                    time_col=analysis_time_col,
                    sbp_true_col=sbp_true_col,
                    sbp_pred_col=sbp_pred_col,
                    dbp_true_col=dbp_true_col,
                    dbp_pred_col=dbp_pred_col,
                    corr_method=corr_method,
                    save_path=save_path,
                )
                n_all_saved += 1
            except Exception as exc:
                print(f"[WARN] Failed to plot subject {sid}: {exc}")
        print(f"Saved {n_all_saved} all-subject trend plots to: {all_dir}")

    summary["n_subjects_plotted_selected"] = n_selected_saved
    summary["n_subjects_plotted_all"] = n_all_saved
    save_summary(summary, output_dir)


# =========================
# CLI
# =========================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze subject-level BP trend metrics.")

    parser.add_argument("--input_csv", type=str, required=True, help="Input calibrated event-level CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")

    parser.add_argument("--id_col", type=str, default="id_upper")
    parser.add_argument("--time_col", type=str, default="t_bp_ms")
    parser.add_argument("--is_calib_col", type=str, default="is_calib")

    parser.add_argument("--sbp_true_col", type=str, default="y_true_sbp")
    parser.add_argument("--sbp_pred_col", type=str, default="y_pred_sbp_aff")
    parser.add_argument("--dbp_true_col", type=str, default="y_true_dbp")
    parser.add_argument("--dbp_pred_col", type=str, default="y_pred_dbp_aff")

    parser.add_argument(
        "--analysis_mode",
        type=str,
        default="both",
        choices=["event", "hourly", "both"],
        help="Analyze event-level, hourly-level, or both.",
    )
    parser.add_argument(
        "--corr_method",
        type=str,
        default="pearson",
        choices=["pearson", "spearman"],
        help="Primary correlation method used per subject.",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=3,
        help="Minimum number of valid points per subject required to compute subject-level metrics.",
    )

    parser.add_argument(
        "--extra_metrics",
        type=str,
        nargs="*",
        default=[],
        choices=["spearman", "lagged_corr", "direction_acc", "dtw"],
        help="Optional extra trend metrics to compute.",
    )
    parser.add_argument(
        "--enable_all_extra_metrics",
        action="store_true",
        help="Enable spearman, lagged_corr, direction_acc, and dtw all at once.",
    )
    parser.add_argument(
        "--lag_max",
        type=int,
        default=1,
        help="Maximum lag in number of points for lagged correlation.",
    )
    parser.add_argument(
        "--lag_corr_method",
        type=str,
        default="pearson",
        choices=["pearson", "spearman"],
        help="Correlation method used inside lagged correlation.",
    )
    parser.add_argument(
        "--direction_zero_policy",
        type=str,
        default="ignore",
        choices=["ignore", "keep"],
        help="How to handle zero first-difference steps in direction accuracy.",
    )
    parser.add_argument(
        "--dtw_window",
        type=int,
        default=None,
        help="Optional Sakoe-Chiba half-window (in points) for DTW. Default: unrestricted.",
    )
    parser.add_argument(
        "--no_dtw_z_normalize",
        action="store_true",
        help="Disable z-normalization before DTW distance computation.",
    )

    parser.add_argument(
        "--include_calib",
        action="store_true",
        help="If set, include calibration points; otherwise use only is_calib == False.",
    )

    parser.add_argument(
        "--plot_metric",
        type=str,
        default="r_sbp",
        help=(
            "Metric column used to choose best/worst subjects for selected plotting. "
            "Examples: r_sbp, r_dbp, spearman_r_sbp, lagged_r_sbp, direction_acc_sbp, dtw_dist_sbp"
        ),
    )
    parser.add_argument("--plot_best_n", type=int, default=2, help="Number of best subjects to plot.")
    parser.add_argument("--plot_worst_n", type=int, default=2, help="Number of worst subjects to plot.")
    parser.add_argument(
        "--plot_subjects",
        type=str,
        nargs="*",
        default=None,
        help="Additional manual subject IDs to plot in selected plots.",
    )
    parser.add_argument(
        "--plot_all_subjects",
        action="store_true",
        help="If set, save trend plots for all subjects into trend_plots_all/.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="If set, skip saving histogram and trend plots.",
    )

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    raw_df = pd.read_csv(input_csv)

    extra_metrics = list(args.extra_metrics)
    if args.enable_all_extra_metrics:
        extra_metrics = ["spearman", "lagged_corr", "direction_acc", "dtw"]

    common_kwargs = dict(
        raw_df=raw_df,
        id_col=args.id_col,
        time_col=args.time_col,
        is_calib_col=args.is_calib_col,
        sbp_true_col=args.sbp_true_col,
        sbp_pred_col=args.sbp_pred_col,
        dbp_true_col=args.dbp_true_col,
        dbp_pred_col=args.dbp_pred_col,
        corr_method=args.corr_method,
        min_points=args.min_points,
        include_calib=args.include_calib,
        extra_metrics=extra_metrics,
        lag_max=args.lag_max,
        lag_corr_method=args.lag_corr_method,
        direction_zero_policy=args.direction_zero_policy,
        dtw_z_normalize=not args.no_dtw_z_normalize,
        dtw_window=args.dtw_window,
        plot_metric=args.plot_metric,
        plot_best_n=args.plot_best_n,
        plot_worst_n=args.plot_worst_n,
        plot_subjects=args.plot_subjects,
        no_plots=args.no_plots,
        plot_all_subjects=args.plot_all_subjects,
    )

    if args.analysis_mode in {"event", "both"}:
        event_dir = output_dir / "event_level"
        run_single_analysis(
            output_dir=event_dir,
            analysis_name="event_level",
            level_mode="event",
            **common_kwargs,
        )

    if args.analysis_mode in {"hourly", "both"}:
        hourly_dir = output_dir / "hourly_level"
        run_single_analysis(
            output_dir=hourly_dir,
            analysis_name="hourly_level",
            level_mode="hourly",
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
