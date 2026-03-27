#!/usr/bin/env python3
"""
Analyze subject-level BP trend correlation from calibrated event-level predictions.

Typical usage:
    python analyze_bp_trend.py \
        --input_csv your_calibrated_test_results.csv \
        --output_dir bp_trend_analysis \
        --id_col id_upper \
        --time_col t_bp_ms \
        --sbp_true_col y_true_sbp \
        --sbp_pred_col y_pred_sbp_aff \
        --dbp_true_col y_true_dbp \
        --dbp_pred_col y_pred_dbp_aff

Main outputs:
    - per_subject_trend_correlations.csv
    - summary.json
    - summary.txt
    - correlation_histograms.png
    - trend plots for selected subjects

Notes:
    - By default, only non-calibration points are used (is_calib == False), which is
      consistent with the event-level calibration / non-calibration evaluation logic.
    - Correlation is computed per subject after sorting by time.
    - The reported cohort metric is the mean of per-subject correlations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# =========================
# Core utilities
# =========================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_bool_mask(series: pd.Series) -> pd.Series:
    """Convert common truthy/falsy representations to boolean mask when possible."""
    if series.dtype == bool:
        return series

    lowered = series.astype(str).str.strip().str.lower()
    true_set = {"true", "1", "yes", "y"}
    false_set = {"false", "0", "no", "n", "nan", "none", ""}

    if lowered.isin(true_set | false_set).all():
        return lowered.isin(true_set)

    # Fallback: pandas truthiness conversion is too permissive, so preserve NaN as False.
    return series.fillna(False).astype(bool)


def compute_corr(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> Tuple[float, float]:
    """Return (r, pvalue)."""
    if method == "pearson":
        return pearsonr(x, y)
    if method == "spearman":
        return spearmanr(x, y)
    raise ValueError(f"Unsupported method: {method}")


# =========================
# Analysis
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
    is_calib_col: str = "is_calib",
    use_non_calib_only: bool = True,
    corr_method: str = "pearson",
    min_points: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute per-subject BP trend correlations.

    Returns
    -------
    per_subject_df : pd.DataFrame
        One row per subject, containing SBP/DBP correlation and counts.
    summary : dict
        Summary metrics over the cohort.
    """
    data = df.copy()

    required = [
        id_col,
        time_col,
        sbp_true_col,
        sbp_pred_col,
        dbp_true_col,
        dbp_pred_col,
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if use_non_calib_only and is_calib_col in data.columns:
        calib_mask = safe_bool_mask(data[is_calib_col])
        data = data.loc[~calib_mask].copy()

    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data = data.dropna(subset=[id_col, time_col]).copy()
    data = data.sort_values([id_col, time_col]).reset_index(drop=True)

    rows: List[Dict[str, float]] = []

    for subject_id, g in data.groupby(id_col, sort=False):
        g = g.sort_values(time_col).copy()

        # SBP
        sbp_g = g[[time_col, sbp_true_col, sbp_pred_col]].dropna().copy()
        n_sbp = len(sbp_g)
        if (
            n_sbp >= min_points
            and sbp_g[sbp_true_col].nunique() > 1
            and sbp_g[sbp_pred_col].nunique() > 1
        ):
            r_sbp, p_sbp = compute_corr(
                sbp_g[sbp_true_col].to_numpy(),
                sbp_g[sbp_pred_col].to_numpy(),
                method=corr_method,
            )
        else:
            r_sbp, p_sbp = np.nan, np.nan

        # DBP
        dbp_g = g[[time_col, dbp_true_col, dbp_pred_col]].dropna().copy()
        n_dbp = len(dbp_g)
        if (
            n_dbp >= min_points
            and dbp_g[dbp_true_col].nunique() > 1
            and dbp_g[dbp_pred_col].nunique() > 1
        ):
            r_dbp, p_dbp = compute_corr(
                dbp_g[dbp_true_col].to_numpy(),
                dbp_g[dbp_pred_col].to_numpy(),
                method=corr_method,
            )
        else:
            r_dbp, p_dbp = np.nan, np.nan

        rows.append(
            {
                id_col: subject_id,
                "n_points_total": int(len(g)),
                "n_points_sbp": int(n_sbp),
                "n_points_dbp": int(n_dbp),
                "r_sbp": float(r_sbp) if pd.notna(r_sbp) else np.nan,
                "p_sbp": float(p_sbp) if pd.notna(p_sbp) else np.nan,
                "r_dbp": float(r_dbp) if pd.notna(r_dbp) else np.nan,
                "p_dbp": float(p_dbp) if pd.notna(p_dbp) else np.nan,
            }
        )

    per_subject_df = pd.DataFrame(rows)

    summary: Dict[str, float] = {
        "corr_method": corr_method,
        "use_non_calib_only": bool(use_non_calib_only),
        "min_points": int(min_points),
        "n_rows_after_filter": int(len(data)),
        "n_subjects_total": int(len(per_subject_df)),
        "n_subjects_valid_sbp": int(per_subject_df["r_sbp"].notna().sum()) if len(per_subject_df) else 0,
        "n_subjects_valid_dbp": int(per_subject_df["r_dbp"].notna().sum()) if len(per_subject_df) else 0,
        "mean_r_sbp": float(per_subject_df["r_sbp"].mean(skipna=True)) if len(per_subject_df) else np.nan,
        "std_r_sbp": float(per_subject_df["r_sbp"].std(skipna=True)) if len(per_subject_df) else np.nan,
        "median_r_sbp": float(per_subject_df["r_sbp"].median(skipna=True)) if len(per_subject_df) else np.nan,
        "mean_r_dbp": float(per_subject_df["r_dbp"].mean(skipna=True)) if len(per_subject_df) else np.nan,
        "std_r_dbp": float(per_subject_df["r_dbp"].std(skipna=True)) if len(per_subject_df) else np.nan,
        "median_r_dbp": float(per_subject_df["r_dbp"].median(skipna=True)) if len(per_subject_df) else np.nan,
    }

    return per_subject_df, summary


# =========================
# Plotting
# =========================

def _prepare_subject_data(
    df: pd.DataFrame,
    *,
    subject_id: str,
    id_col: str,
    time_col: str,
    sbp_true_col: str,
    sbp_pred_col: str,
    dbp_true_col: str,
    dbp_pred_col: str,
    is_calib_col: str,
    use_non_calib_only: bool,
) -> pd.DataFrame:
    data = df.copy()
    if use_non_calib_only and is_calib_col in data.columns:
        data = data.loc[~safe_bool_mask(data[is_calib_col])].copy()

    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data = data.dropna(subset=[id_col, time_col]).copy()
    g = data.loc[data[id_col] == subject_id].copy()
    if len(g) == 0:
        raise ValueError(f"Subject not found after filtering: {subject_id}")

    g = g.sort_values(time_col).copy()
    try:
        g["time_dt"] = pd.to_datetime(g[time_col], unit="ms")
    except Exception:
        # Fallback to integer index when timestamps are malformed.
        g["time_dt"] = np.arange(len(g))

    # Keep only relevant columns for readability.
    keep_cols = [
        id_col,
        time_col,
        "time_dt",
        sbp_true_col,
        sbp_pred_col,
        dbp_true_col,
        dbp_pred_col,
    ]
    keep_cols = [c for c in keep_cols if c in g.columns]
    return g[keep_cols].copy()


def plot_subject_bp_trend(
    df: pd.DataFrame,
    *,
    subject_id: str,
    id_col: str,
    time_col: str,
    sbp_true_col: str,
    sbp_pred_col: str,
    dbp_true_col: str,
    dbp_pred_col: str,
    is_calib_col: str = "is_calib",
    use_non_calib_only: bool = True,
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
        is_calib_col=is_calib_col,
        use_non_calib_only=use_non_calib_only,
    )

    sbp_valid = g[[sbp_true_col, sbp_pred_col]].dropna()
    dbp_valid = g[[dbp_true_col, dbp_pred_col]].dropna()

    r_sbp = np.nan
    r_dbp = np.nan
    if len(sbp_valid) >= 2 and sbp_valid[sbp_true_col].nunique() > 1 and sbp_valid[sbp_pred_col].nunique() > 1:
        r_sbp, _ = compute_corr(sbp_valid[sbp_true_col].to_numpy(), sbp_valid[sbp_pred_col].to_numpy(), method=corr_method)
    if len(dbp_valid) >= 2 and dbp_valid[dbp_true_col].nunique() > 1 and dbp_valid[dbp_pred_col].nunique() > 1:
        r_dbp, _ = compute_corr(dbp_valid[dbp_true_col].to_numpy(), dbp_valid[dbp_pred_col].to_numpy(), method=corr_method)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(g["time_dt"], g[sbp_true_col], marker="o", label="True SBP")
    axes[0].plot(g["time_dt"], g[sbp_pred_col], marker="o", label="Pred SBP (aff)")
    axes[0].set_ylabel("SBP (mmHg)")
    axes[0].set_title(f"Subject {subject_id} | SBP trend | {corr_method} r = {r_sbp:.3f}" if pd.notna(r_sbp) else f"Subject {subject_id} | SBP trend")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(g["time_dt"], g[dbp_true_col], marker="o", label="True DBP")
    axes[1].plot(g["time_dt"], g[dbp_pred_col], marker="o", label="Pred DBP (aff)")
    axes[1].set_ylabel("DBP (mmHg)")
    axes[1].set_xlabel("Time")
    axes[1].set_title(f"Subject {subject_id} | DBP trend | {corr_method} r = {r_dbp:.3f}" if pd.notna(r_dbp) else f"Subject {subject_id} | DBP trend")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
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


def print_summary(summary: Dict[str, float]) -> None:
    print("=== BP Trend Correlation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


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

    valid = per_subject_df.dropna(subset=[metric_col]).copy()
    if len(valid) > 0:
        if n_best > 0:
            selected.extend(valid.sort_values(metric_col, ascending=False).head(n_best)[id_col].astype(str).tolist())
        if n_worst > 0:
            selected.extend(valid.sort_values(metric_col, ascending=True).head(n_worst)[id_col].astype(str).tolist())

    # Stable dedup while preserving order.
    seen = set()
    deduped: List[str] = []
    for sid in selected:
        if sid not in seen:
            seen.add(sid)
            deduped.append(sid)
    return deduped


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
        help="Metric used to choose best/worst subjects for plotting.",
    )
    parser.add_argument("--plot_best_n", type=int, default=2, help="Number of best subjects to plot.")
    parser.add_argument("--plot_worst_n", type=int, default=2, help="Number of worst subjects to plot.")
    parser.add_argument(
        "--plot_subjects",
        type=str,
        nargs="*",
        default=None,
        help="Additional manual subject IDs to plot.",
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

    df = pd.read_csv(input_csv)

    per_subject_df, summary = compute_subject_trend_correlations(
        df,
        id_col=args.id_col,
        time_col=args.time_col,
        sbp_true_col=args.sbp_true_col,
        sbp_pred_col=args.sbp_pred_col,
        dbp_true_col=args.dbp_true_col,
        dbp_pred_col=args.dbp_pred_col,
        is_calib_col=args.is_calib_col,
        use_non_calib_only=not args.include_calib,
        corr_method=args.corr_method,
        min_points=args.min_points,
    )

    per_subject_csv = output_dir / "per_subject_trend_correlations.csv"
    per_subject_df.to_csv(per_subject_csv, index=False)
    save_summary(summary, output_dir)
    print_summary(summary)
    print(f"Saved per-subject correlations to: {per_subject_csv}")

    if not args.no_plots and len(per_subject_df) > 0:
        plot_correlation_histograms(
            per_subject_df,
            corr_method=args.corr_method,
            save_path=output_dir / "correlation_histograms.png",
        )
        print(f"Saved histogram to: {output_dir / 'correlation_histograms.png'}")

        plot_ids = select_subjects_for_plot(
            per_subject_df,
            id_col=args.id_col,
            metric_col=args.plot_metric,
            n_best=args.plot_best_n,
            n_worst=args.plot_worst_n,
            manual_subjects=args.plot_subjects,
        )

        trends_dir = output_dir / "trend_plots"
        ensure_dir(trends_dir)

        for sid in plot_ids:
            safe_sid = str(sid).replace("/", "_").replace("\\", "_").replace(" ", "_")
            save_path = trends_dir / f"subject_{safe_sid}.png"
            try:
                plot_subject_bp_trend(
                    df,
                    subject_id=sid,
                    id_col=args.id_col,
                    time_col=args.time_col,
                    sbp_true_col=args.sbp_true_col,
                    sbp_pred_col=args.sbp_pred_col,
                    dbp_true_col=args.dbp_true_col,
                    dbp_pred_col=args.dbp_pred_col,
                    is_calib_col=args.is_calib_col,
                    use_non_calib_only=not args.include_calib,
                    corr_method=args.corr_method,
                    save_path=save_path,
                )
                print(f"Saved trend plot: {save_path}")
            except Exception as e:
                print(f"[WARN] Failed to plot subject {sid}: {e}")


if __name__ == "__main__":
    main()
