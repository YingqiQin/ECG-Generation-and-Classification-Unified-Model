#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute correlation between predicted BP and ABPM gold BP.

Supports:
1) Overall/test-level correlation across all rows.
2) Per-subject correlation, then mean correlation across subjects.
3) Fisher-z averaged per-subject correlation, recommended.
4) Optional all/day/night split using sleep column.
5) Optional exclusion of calibration rows.

Example:
python compute_bp_abpm_correlation.py \
  --csv predictions.csv \
  --pred-sbp-col y_pred_sbp_bias \
  --pred-dbp-col y_pred_dbp_bias \
  --true-sbp-col ABPM_SBP \
  --true-dbp-col ABPM_DBP \
  --out-prefix corr_abpm_bias

If using raw predictions:
python compute_bp_abpm_correlation.py \
  --csv predictions.csv \
  --pred-sbp-col y_pred_sbp_raw \
  --pred-dbp-col y_pred_dbp_raw \
  --out-prefix corr_abpm_raw
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRUEY = {"1", "true", "yes", "y", "t"}


def parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(TRUEY)


def safe_corr(x, y, method: str = "pearson") -> float:
    d = pd.DataFrame({"x": x, "y": y})
    d["x"] = pd.to_numeric(d["x"], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d = d.dropna()
    if len(d) < 2:
        return np.nan
    if d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return np.nan
    return float(d["x"].corr(d["y"], method=method))


def fisher_z(r: float) -> float:
    if not np.isfinite(r):
        return np.nan
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def inv_fisher_z(z: float) -> float:
    if not np.isfinite(z):
        return np.nan
    return float(np.tanh(z))


def summarize_per_subject_corr(per_subject: pd.DataFrame, prefix: str) -> Dict[str, float]:
    r = pd.to_numeric(per_subject["r"], errors="coerce").dropna().to_numpy(dtype=float)
    n = pd.to_numeric(per_subject["n"], errors="coerce").to_numpy(dtype=float)
    valid_n = n[np.isfinite(pd.to_numeric(per_subject["r"], errors="coerce").to_numpy(dtype=float))]

    out: Dict[str, float] = {
        f"{prefix}_n_subjects_valid": int(len(r)),
        f"{prefix}_r_mean_arithmetic": float(np.mean(r)) if len(r) else np.nan,
        f"{prefix}_r_median": float(np.median(r)) if len(r) else np.nan,
        f"{prefix}_r_std_across_subjects": float(np.std(r, ddof=1)) if len(r) > 1 else np.nan,
    }

    if len(r):
        z = np.array([fisher_z(v) for v in r], dtype=float)
        out[f"{prefix}_r_mean_fisher_z"] = inv_fisher_z(float(np.nanmean(z)))

        # Optional weighted Fisher-z mean. For correlation, approximate weight is n-3.
        weights = np.maximum(valid_n - 3.0, 1.0)
        if len(weights) == len(z) and np.sum(weights) > 0:
            out[f"{prefix}_r_mean_fisher_z_weighted_by_n_minus_3"] = inv_fisher_z(
                float(np.nansum(z * weights) / np.nansum(weights))
            )
        else:
            out[f"{prefix}_r_mean_fisher_z_weighted_by_n_minus_3"] = np.nan
    else:
        out[f"{prefix}_r_mean_fisher_z"] = np.nan
        out[f"{prefix}_r_mean_fisher_z_weighted_by_n_minus_3"] = np.nan

    return out


def filter_split(df: pd.DataFrame, split: str, sleep_col: str) -> pd.DataFrame:
    if split == "all":
        return df.copy()
    if split == "day":
        return df[pd.to_numeric(df[sleep_col], errors="coerce").fillna(0).astype(int) == 0].copy()
    if split in ["night", "sleep"]:
        return df[pd.to_numeric(df[sleep_col], errors="coerce").fillna(0).astype(int) == 1].copy()
    raise ValueError(f"Unknown split: {split}")


def compute_corr_for_split(
    df: pd.DataFrame,
    split: str,
    target_name: str,
    true_col: str,
    pred_col: str,
    subject_col: str,
    sleep_col: str,
    method: str,
    min_points_per_subject: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    d = filter_split(df, split, sleep_col)
    need = [subject_col, true_col, pred_col]
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise ValueError(f"Missing columns for {split}/{target_name}: {missing}")

    d[true_col] = pd.to_numeric(d[true_col], errors="coerce")
    d[pred_col] = pd.to_numeric(d[pred_col], errors="coerce")
    d = d.dropna(subset=[subject_col, true_col, pred_col])

    # Overall/test-level correlation across all rows.
    overall_r = safe_corr(d[pred_col], d[true_col], method=method)

    rows: List[Dict] = []
    for sid, g in d.groupby(subject_col, sort=False):
        gg = g.dropna(subset=[true_col, pred_col])
        n = int(len(gg))
        r = np.nan
        if n >= min_points_per_subject:
            r = safe_corr(gg[pred_col], gg[true_col], method=method)
        rows.append({
            "split": split,
            "target": target_name,
            "subject_id": str(sid),
            "n": n,
            "r": r,
            "method": method,
            "true_col": true_col,
            "pred_col": pred_col,
        })

    per_subj = pd.DataFrame(rows)

    prefix = f"{split}_{target_name}_{method}"
    summary = {
        f"{prefix}_overall_row_level_r": overall_r,
        f"{prefix}_n_rows": int(len(d)),
        f"{prefix}_n_subjects_total": int(d[subject_col].nunique()),
        f"{prefix}_min_points_per_subject": int(min_points_per_subject),
    }
    summary.update(summarize_per_subject_corr(per_subj, prefix=prefix))

    return per_subj, summary


def compute_all_correlations(
    df: pd.DataFrame,
    true_sbp_col: str,
    true_dbp_col: str,
    pred_sbp_col: str,
    pred_dbp_col: str,
    subject_col: str = "id_clean",
    sleep_col: str = "sleep",
    is_calib_col: str = "is_calib",
    include_calib: bool = False,
    method: str = "pearson",
    min_points_per_subject: int = 3,
    splits: Tuple[str, ...] = ("all", "day", "night"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    if not include_calib and is_calib_col in d.columns:
        d = d[~parse_bool_series(d[is_calib_col])].copy()

    if sleep_col not in d.columns:
        d[sleep_col] = 0

    per_subject_tables = []
    summary: Dict[str, float] = {}

    for split in splits:
        for target_name, true_col, pred_col in [
            ("SBP", true_sbp_col, pred_sbp_col),
            ("DBP", true_dbp_col, pred_dbp_col),
        ]:
            per_subj, summ = compute_corr_for_split(
                d,
                split=split,
                target_name=target_name,
                true_col=true_col,
                pred_col=pred_col,
                subject_col=subject_col,
                sleep_col=sleep_col,
                method=method,
                min_points_per_subject=min_points_per_subject,
            )
            per_subject_tables.append(per_subj)
            summary.update(summ)

    per_subject_df = pd.concat(per_subject_tables, axis=0, ignore_index=True)
    summary_df = pd.DataFrame([summary])
    return per_subject_df, summary_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute predicted BP vs ABPM correlation.")
    p.add_argument("--csv", type=str, required=True, help="Prediction CSV file.")
    p.add_argument("--out-prefix", type=str, default="bp_abpm_corr")

    p.add_argument("--true-sbp-col", type=str, default="ABPM_SBP")
    p.add_argument("--true-dbp-col", type=str, default="ABPM_DBP")
    p.add_argument("--pred-sbp-col", type=str, default="y_pred_sbp_bias")
    p.add_argument("--pred-dbp-col", type=str, default="y_pred_dbp_bias")

    p.add_argument("--subject-col", type=str, default="id_clean")
    p.add_argument("--sleep-col", type=str, default="sleep")
    p.add_argument("--is-calib-col", type=str, default="is_calib")
    p.add_argument("--include-calib", action="store_true")

    p.add_argument("--method", type=str, default="pearson", choices=["pearson", "spearman"])
    p.add_argument("--min-points-per-subject", type=int, default=3)
    p.add_argument("--splits", type=str, default="all,day,night")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    per_subject_df, summary_df = compute_all_correlations(
        df,
        true_sbp_col=args.true_sbp_col,
        true_dbp_col=args.true_dbp_col,
        pred_sbp_col=args.pred_sbp_col,
        pred_dbp_col=args.pred_dbp_col,
        subject_col=args.subject_col,
        sleep_col=args.sleep_col,
        is_calib_col=args.is_calib_col,
        include_calib=args.include_calib,
        method=args.method,
        min_points_per_subject=args.min_points_per_subject,
        splits=splits,
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    per_subject_path = out_prefix.with_name(out_prefix.name + "_per_subject.csv")
    summary_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    json_path = out_prefix.with_name(out_prefix.name + "_summary.json")

    per_subject_df.to_csv(per_subject_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_df.iloc[0].to_dict(), f, indent=2, ensure_ascii=False)

    print("\n=== Correlation summary ===")
    # Compact print.
    for split in splits:
        for target in ["SBP", "DBP"]:
            prefix = f"{split}_{target}_{args.method}"
            print(
                f"{split:>5} {target}: "
                f"overall_r={summary_df.iloc[0].get(prefix + '_overall_row_level_r', np.nan):+.4f}, "
                f"mean_subject_r={summary_df.iloc[0].get(prefix + '_r_mean_arithmetic', np.nan):+.4f}, "
                f"fisher_mean_r={summary_df.iloc[0].get(prefix + '_r_mean_fisher_z', np.nan):+.4f}, "
                f"weighted_fisher_mean_r={summary_df.iloc[0].get(prefix + '_r_mean_fisher_z_weighted_by_n_minus_3', np.nan):+.4f}, "
                f"valid_subjects={summary_df.iloc[0].get(prefix + '_n_subjects_valid', 0)}"
            )

    print(f"\nSaved:")
    print(f"  {per_subject_path}")
    print(f"  {summary_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
