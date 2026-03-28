#!/usr/bin/env python3
"""
Analyze subject-level BP trend correlation from calibrated event-level predictions.

Supports:
  - event-level analysis: use each BP event / scatter point directly
  - hourly-level analysis: aggregate each subject's points within each hour by mean,
    then compute per-subject trend correlation on the hourly curves

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
        --plot_all_subjects

Main outputs (for each analysis level):
    - per_subject_trend_correlations.csv
    - summary.json
    - summary.txt
    - correlation_histograms.png
    - trend_plots_selected/*.png
    - trend_plots_all/*.png  (if --plot_all_subjects is set)

Notes:
    - By default, only non-calibration points are used (is_calib == False).
    - Correlation is computed per subject after sorting by time.
    - The reported cohort metric is the mean of per-subject correlations.
"""

from __future__ import annotations

import argparse
import json
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



def compute_corr(x: np.ndarray, y: np.ndarray, method: str) -> Tuple[float, float]:
    if method == "pearson":
        return pearsonr(x, y)
    if method == "spearman":
        return spearmanr(x, y)
    raise ValueError(f"Unsupported correlation method: {method}")



def sanitize_subject_id(subject_id: object) -> str:
    text = str(subject_id)
    for old, new in [("/", "_"), ("\\", "_"), (" ", "_"), (":", "_"), ("*", "_")]:
        text = text.replace(old, new)
    return text


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
# Correlation analysis
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
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute per-subject SBP/DBP trend correlations."""
    required = [id_col, time_col, sbp_true_col, sbp_pred_col, dbp_true_col, dbp_pred_col]
    validate_required_columns(df, required)

    data = df.copy()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data = data.dropna(subset=[id_col, time_col]).copy()
    data = data.sort_values([id_col, time_col]).reset_index(drop=True)

    rows: List[Dict[str, float]] = []

    for subject_id, g in data.groupby(id_col, sort=False):
        g = g.sort_values(time_col).copy()

        sbp_g = g[[time_col, sbp_true_col, sbp_pred_col]].dropna().copy()
        n_sbp = len(sbp_g)
        if n_sbp >= min_points and sbp_g[sbp_true_col].nunique() > 1 and sbp_g[sbp_pred_col].nunique() > 1:
            r_sbp, p_sbp = compute_corr(
                sbp_g[sbp_true_col].to_numpy(),
                sbp_g[sbp_pred_col].to_numpy(),
                method=corr_method,
            )
        else:
            r_sbp, p_sbp = np.nan, np.nan

        dbp_g = g[[time_col, dbp_true_col, dbp_pred_col]].dropna().copy()
        n_dbp = len(dbp_g)
        if n_dbp >= min_points and dbp_g[dbp_true_col].nunique() > 1 and dbp_g[dbp_pred_col].nunique() > 1:
            r_dbp, p_dbp = compute_corr(
                dbp_g[dbp_true_col].to_numpy(),
                dbp_g[dbp_pred_col].to_numpy(),
                method=corr_method,
            )
        else:
            r_dbp, p_dbp = np.nan, np.nan

        row = {
            id_col: subject_id,
            "n_points_total": int(len(g)),
            "n_points_sbp": int(n_sbp),
            "n_points_dbp": int(n_dbp),
            "r_sbp": float(r_sbp) if pd.notna(r_sbp) else np.nan,
            "p_sbp": float(p_sbp) if pd.notna(p_sbp) else np.nan,
            "r_dbp": float(r_dbp) if pd.notna(r_dbp) else np.nan,
            "p_dbp": float(p_dbp) if pd.notna(p_dbp) else np.nan,
        }
        if "n_events" in g.columns:
            row["n_events_total"] = int(g["n_events"].sum())
            row["mean_events_per_point"] = float(g["n_events"].mean())

        rows.append(row)

    per_subject_df = pd.DataFrame(rows)

    if len(per_subject_df) == 0:
        summary = {
            "corr_method": corr_method,
            "min_points": int(min_points),
            "n_rows_after_filter": 0,
            "n_subjects_total": 0,
            "n_subjects_valid_sbp": 0,
            "n_subjects_valid_dbp": 0,
            "mean_r_sbp": np.nan,
            "std_r_sbp": np.nan,
            "median_r_sbp": np.nan,
            "mean_r_dbp": np.nan,
            "std_r_dbp": np.nan,
            "median_r_dbp": np.nan,
        }
        return per_subject_df, summary

    summary = {
        "corr_method": corr_method,
        "min_points": int(min_points),
        "n_rows_after_filter": int(len(data)),
        "n_subjects_total": int(len(per_subject_df)),
        "n_subjects_valid_sbp": int(per_subject_df["r_sbp"].notna().sum()),
        "n_subjects_valid_dbp": int(per_subject_df["r_dbp"].notna().sum()),
        "mean_r_sbp": float(per_subject_df["r_sbp"].mean(skipna=True)),
        "std_r_sbp": float(per_subject_df["r_sbp"].std(skipna=True)),
        "median_r_sbp": float(per_subject_df["r_sbp"].median(skipna=True)),
        "mean_r_dbp": float(per_subject_df["r_dbp"].mean(skipna=True)),
        "std_r_dbp": float(per_subject_df["r_dbp"].std(skipna=True)),
        "median_r_dbp": float(per_subject_df["r_dbp"].median(skipna=True)),
    }
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



def _subject_corr_from_curve(
    g: pd.DataFrame,
    *,
    sbp_true_col: str,
    sbp_pred_col: str,
    dbp_true_col: str,
    dbp_pred_col: str,
    corr_method: str,
) -> Tuple[float, float]:
    sbp_valid = g[[sbp_true_col, sbp_pred_col]].dropna()
    dbp_valid = g[[dbp_true_col, dbp_pred_col]].dropna()

    r_sbp = np.nan
    r_dbp = np.nan

    if len(sbp_valid) >= 2 and sbp_valid[sbp_true_col].nunique() > 1 and sbp_valid[sbp_pred_col].nunique() > 1:
        r_sbp, _ = compute_corr(
            sbp_valid[sbp_true_col].to_numpy(),
            sbp_valid[sbp_pred_col].to_numpy(),
            method=corr_method,
        )

    if len(dbp_valid) >= 2 and dbp_valid[dbp_true_col].nunique() > 1 and dbp_valid[dbp_pred_col].nunique() > 1:
        r_dbp, _ = compute_corr(
            dbp_valid[dbp_true_col].to_numpy(),
            dbp_valid[dbp_pred_col].to_numpy(),
            method=corr_method,
        )

    return r_sbp, r_dbp



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
    corr_method: str,
    title_prefix: str = "",
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

    r_sbp, r_dbp = _subject_corr_from_curve(
        g,
        sbp_true_col=sbp_true_col,
        sbp_pred_col=sbp_pred_col,
        dbp_true_col=dbp_true_col,
        dbp_pred_col=dbp_pred_col,
        corr_method=corr_method,
    )

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(g["time_dt"], g[sbp_true_col], marker="o", label="True SBP")
    axes[0].plot(g["time_dt"], g[sbp_pred_col], marker="o", label="Pred SBP")
    axes[0].set_ylabel("SBP (mmHg)")
    if pd.notna(r_sbp):
        axes[0].set_title(f"{title_prefix}Subject {subject_id} | SBP trend | {corr_method} r = {r_sbp:.3f}")
    else:
        axes[0].set_title(f"{title_prefix}Subject {subject_id} | SBP trend")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(g["time_dt"], g[dbp_true_col], marker="o", label="True DBP")
    axes[1].plot(g["time_dt"], g[dbp_pred_col], marker="o", label="Pred DBP")
    axes[1].set_ylabel("DBP (mmHg)")
    axes[1].set_xlabel("Time")
    if pd.notna(r_dbp):
        axes[1].set_title(f"{title_prefix}Subject {subject_id} | DBP trend | {corr_method} r = {r_dbp:.3f}")
    else:
        axes[1].set_title(f"{title_prefix}Subject {subject_id} | DBP trend")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    if save_path is not None:
        ensure_dir(save_path.parent)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()



def plot_correlation_histograms(
    per_subject_df: pd.DataFrame,
    *,
    corr_method: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(per_subject_df["r_sbp"].dropna(), bins=20)
    axes[0].set_title(f"Per-subject SBP {corr_method} correlation")
    axes[0].set_xlabel("Correlation")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(per_subject_df["r_dbp"].dropna(), bins=20)
    axes[1].set_title(f"Per-subject DBP {corr_method} correlation")
    axes[1].set_xlabel("Correlation")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================
# Reporting and subject selection
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



def print_summary(summary: Dict[str, float], *, prefix: str = "") -> None:
    print(f"=== {prefix}BP Trend Correlation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")



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

    valid = per_subject_df.dropna(subset=[metric_col]).copy()
    if len(valid) > 0:
        if n_best > 0:
            selected.extend(valid.sort_values(metric_col, ascending=False).head(n_best)[id_col].astype(str).tolist())
        if n_worst > 0:
            selected.extend(valid.sort_values(metric_col, ascending=True).head(n_worst)[id_col].astype(str).tolist())

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
    print(f"Saved per-subject correlations to: {per_subject_path}")

    if no_plots:
        return

    if len(per_subject_df) > 0:
        hist_path = output_dir / "correlation_histograms.png"
        plot_correlation_histograms(
            per_subject_df,
            corr_method=corr_method,
            save_path=hist_path,
        )
        print(f"Saved histogram to: {hist_path}")

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
                title_prefix=f"[{analysis_name}] ",
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
                    title_prefix=f"[{analysis_name}] ",
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
    parser = argparse.ArgumentParser(description="Analyze subject-level BP trend correlation.")

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
        help="Correlation method used per subject.",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=3,
        help="Minimum number of valid points per subject required to compute correlation.",
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
        choices=["r_sbp", "r_dbp"],
        help="Metric used to choose best/worst subjects for selected plotting.",
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

    if args.analysis_mode in {"event", "both"}:
        event_dir = output_dir / "event_level"
        run_single_analysis(
            raw_df,
            output_dir=event_dir,
            analysis_name="event_level",
            level_mode="event",
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
            plot_metric=args.plot_metric,
            plot_best_n=args.plot_best_n,
            plot_worst_n=args.plot_worst_n,
            plot_subjects=args.plot_subjects,
            no_plots=args.no_plots,
            plot_all_subjects=args.plot_all_subjects,
        )

    if args.analysis_mode in {"hourly", "both"}:
        hourly_dir = output_dir / "hourly_level"
        run_single_analysis(
            raw_df,
            output_dir=hourly_dir,
            analysis_name="hourly_level",
            level_mode="hourly",
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
            plot_metric=args.plot_metric,
            plot_best_n=args.plot_best_n,
            plot_worst_n=args.plot_worst_n,
            plot_subjects=args.plot_subjects,
            no_plots=args.no_plots,
            plot_all_subjects=args.plot_all_subjects,
        )


if __name__ == "__main__":
    main()
