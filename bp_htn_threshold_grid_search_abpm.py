#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hypertension classification threshold grid search with separate calibration truth
and evaluation gold-standard columns.

Main change for the current PPG-BP project:
    - Calibration can still be done upstream using y_true_sbp / y_true_dbp.
    - This script uses ABPM_SBP / ABPM_DBP by default as the gold standard for
      BP accuracy and hypertension classification.

It also supports split-specific aggregation strategies:
    - All   default: mean
    - Day   default: mix_mean_p75, alpha=0.7 -> 0.7*mean + 0.3*p75
    - Night default: median

True labels are still thresholded using fixed clinical thresholds:
    all   true: 130/80 on ABPM columns, aggregated by mean
    day   true: 135/85 on ABPM columns, aggregated by mean, sleep == 0
    night true: 120/70 on ABPM columns, aggregated by mean, sleep == 1

Predicted labels use searched thresholds and split-specific prediction aggregation.

Typical usage:
    python bp_htn_threshold_grid_search_abpm.py \
        --csv-glob "eval_1p0_sensitivity/predictions_*.csv" \
        --methods bias \
        --out-dir eval_1p0_sensitivity/threshold_search_abpm_bias

Compare bias / aff / bank:
    python bp_htn_threshold_grid_search_abpm.py \
        --csv-glob "eval_1p0_sensitivity/predictions_*.csv" \
        --methods bias,aff,bank \
        --out-dir eval_1p0_sensitivity/threshold_search_abpm_compare
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TRUEY = {"1", "true", "yes", "y", "t"}


@dataclass(frozen=True)
class Threshold:
    sbp: float
    dbp: float


@dataclass(frozen=True)
class Metrics:
    tp: int
    tn: int
    fp: int
    fn: int
    sensitivity: float
    specificity: float
    precision: float
    accuracy: float
    f1: float
    n: int


DEFAULT_METHOD_COLS: Dict[str, Tuple[str, str]] = {
    "raw": ("y_pred_sbp_raw", "y_pred_dbp_raw"),
    "bias": ("y_pred_sbp_bias", "y_pred_dbp_bias"),
    "aff": ("y_pred_sbp_aff", "y_pred_dbp_aff"),
    "bank": ("y_pred_sbp_bank", "y_pred_dbp_bank"),
    "cal": ("y_pred_sbp_cal", "y_pred_dbp_cal"),
}


def parse_bool(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in TRUEY


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num / den)


def is_positive(sbp: float, dbp: float, thr: Threshold) -> bool:
    return (float(sbp) >= float(thr.sbp)) or (float(dbp) >= float(thr.dbp))


def compute_binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Metrics:
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true={len(yt)}, y_pred={len(yp)}")

    tp = int(np.sum(yt & yp))
    tn = int(np.sum((~yt) & (~yp)))
    fp = int(np.sum((~yt) & yp))
    fn = int(np.sum(yt & (~yp)))

    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    accuracy = safe_div(tp + tn, len(yt))
    f1 = safe_div(2 * precision * sensitivity, precision + sensitivity)

    return Metrics(tp, tn, fp, fn, sensitivity, specificity, precision, accuracy, f1, int(len(yt)))


def metrics_to_dict(m: Metrics) -> Dict[str, float | int]:
    return {
        "tp": m.tp,
        "tn": m.tn,
        "fp": m.fp,
        "fn": m.fn,
        "sensitivity": m.sensitivity,
        "specificity": m.specificity,
        "precision": m.precision,
        "accuracy": m.accuracy,
        "f1": m.f1,
        "n": m.n,
    }


def me_std_mae(y_pred: Sequence[float], y_true: Sequence[float]) -> Dict[str, float | int]:
    yp = np.asarray(y_pred, dtype=np.float64)
    yt = np.asarray(y_true, dtype=np.float64)
    mask = np.isfinite(yp) & np.isfinite(yt)
    yp = yp[mask]
    yt = yt[mask]
    if len(yp) == 0:
        return {"ME": np.nan, "STD": np.nan, "MAE": np.nan, "N": 0}
    err = yp - yt
    return {
        "ME": float(np.mean(err)),
        "STD": float(np.std(err, ddof=1)) if len(err) > 1 else np.nan,
        "MAE": float(np.mean(np.abs(err))),
        "N": int(len(err)),
    }


def resolve_csv_paths(csv_glob: Optional[str], csv_paths: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    if csv_glob:
        paths.extend(Path(p) for p in sorted(glob.glob(csv_glob)))
    if csv_paths:
        for p in csv_paths.split(","):
            p = p.strip()
            if p:
                paths.append(Path(p))

    seen = set()
    unique: List[Path] = []
    for p in paths:
        rp = str(p)
        if rp not in seen:
            unique.append(p)
            seen.add(rp)

    if not unique:
        raise ValueError("No CSV files found. Provide --csv-glob or --csv-paths.")
    return unique


def load_one_csv(
    path: Path,
    include_calib: bool,
    true_sbp_col: str,
    true_dbp_col: str,
    fallback_true_sbp_col: Optional[str] = None,
    fallback_true_dbp_col: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Allow fallback only when explicitly provided.
    if true_sbp_col not in df.columns and fallback_true_sbp_col and fallback_true_sbp_col in df.columns:
        df[true_sbp_col] = df[fallback_true_sbp_col]
    if true_dbp_col not in df.columns and fallback_true_dbp_col and fallback_true_dbp_col in df.columns:
        df[true_dbp_col] = df[fallback_true_dbp_col]

    required = ["id_clean", "sleep", true_sbp_col, true_dbp_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} missing required columns: {missing}. "
            f"Current gold columns are --true-sbp-col {true_sbp_col} and --true-dbp-col {true_dbp_col}."
        )

    if not include_calib and "is_calib" in df.columns:
        df = df[~df["is_calib"].map(parse_bool)].copy()

    df["id_clean"] = df["id_clean"].astype(str)
    df["sleep"] = pd.to_numeric(df["sleep"], errors="coerce").fillna(0).astype(int)
    for c in [true_sbp_col, true_dbp_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def method_columns(method: str, custom: Optional[str] = None) -> Tuple[str, str]:
    if custom:
        # format: method:sbp_col:dbp_col
        # Example: --custom-cols my:y_pred_sbp_x:y_pred_dbp_x --methods my
        for spec in custom.split(","):
            parts = spec.split(":")
            if len(parts) != 3:
                raise ValueError("--custom-cols must be formatted as method:sbp_col:dbp_col[,method2:sbp:dbp]")
            name, sbp, dbp = [x.strip() for x in parts]
            if name == method:
                return sbp, dbp
    if method not in DEFAULT_METHOD_COLS:
        raise ValueError(f"Unknown method={method}. Use one of {list(DEFAULT_METHOD_COLS)} or --custom-cols.")
    return DEFAULT_METHOD_COLS[method]


def split_filter(df: pd.DataFrame, split: str) -> pd.DataFrame:
    split = split.lower()
    if split == "all":
        return df
    if split == "day":
        return df[df["sleep"].astype(int) == 0].copy()
    if split in ["night", "sleep"]:
        return df[df["sleep"].astype(int) == 1].copy()
    raise ValueError(f"Unknown split={split}")


def aggregate_vector(values: Sequence[float], mode: str, mix_alpha: float = 0.7, trim_ratio: float = 0.1) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan

    mode = str(mode).lower()
    if mode == "mean":
        return float(np.mean(arr))
    if mode == "median":
        return float(np.median(arr))
    if mode in ["p60", "q60"]:
        return float(np.percentile(arr, 60))
    if mode in ["p70", "q70"]:
        return float(np.percentile(arr, 70))
    if mode in ["p75", "q75"]:
        return float(np.percentile(arr, 75))
    if mode in ["p80", "q80"]:
        return float(np.percentile(arr, 80))
    if mode in ["trimmed_mean", "trimmean"]:
        if len(arr) <= 2:
            return float(np.mean(arr))
        x = np.sort(arr)
        k = int(math.floor(len(x) * float(trim_ratio)))
        if 2 * k >= len(x):
            return float(np.mean(x))
        return float(np.mean(x[k: len(x) - k]))
    if mode in ["remove_top1_mean", "drop_top1_mean"]:
        if len(arr) <= 1:
            return float(np.mean(arr))
        x = np.sort(arr)
        return float(np.mean(x[:-1]))
    if mode in ["mix_mean_p75", "mean_p75_mix"]:
        return float(float(mix_alpha) * np.mean(arr) + (1.0 - float(mix_alpha)) * np.percentile(arr, 75))
    if mode in ["mix_mean_p80", "mean_p80_mix"]:
        return float(float(mix_alpha) * np.mean(arr) + (1.0 - float(mix_alpha)) * np.percentile(arr, 80))

    raise ValueError(
        f"Unknown aggregation mode={mode}. Use mean/median/p60/p75/p80/trimmed_mean/"
        "remove_top1_mean/mix_mean_p75/mix_mean_p80."
    )


def split_pred_agg_mode(split: str, all_agg: str, day_agg: str, night_agg: str) -> str:
    split = split.lower()
    if split == "all":
        return all_agg
    if split == "day":
        return day_agg
    if split in ["night", "sleep"]:
        return night_agg
    raise ValueError(split)


def aggregate_subject_table(
    df: pd.DataFrame,
    method: str,
    split: str,
    subject_col: str,
    true_sbp_col: str,
    true_dbp_col: str,
    pred_agg: str,
    pred_mix_alpha: float,
    pred_trim_ratio: float,
    custom_cols: Optional[str] = None,
) -> pd.DataFrame:
    """
    One row per subject for one split.

    True gold uses mean of ABPM columns by default.
    Predicted BP uses configurable aggregation, e.g. day mix_mean_p75 or night median.
    """
    sbp_col, dbp_col = method_columns(method, custom_cols)
    if sbp_col not in df.columns or dbp_col not in df.columns:
        raise ValueError(f"Prediction columns for method={method} missing: {sbp_col}, {dbp_col}")

    d = split_filter(df, split)
    if len(d) == 0:
        return pd.DataFrame(columns=[subject_col, "true_sbp", "true_dbp", "pred_sbp", "pred_dbp", "n_rows"])

    needed = [true_sbp_col, true_dbp_col, sbp_col, dbp_col]
    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    rows = []
    for sid, g in d.groupby(subject_col, sort=False):
        true_sbp = aggregate_vector(g[true_sbp_col], mode="mean")
        true_dbp = aggregate_vector(g[true_dbp_col], mode="mean")
        pred_sbp = aggregate_vector(g[sbp_col], mode=pred_agg, mix_alpha=pred_mix_alpha, trim_ratio=pred_trim_ratio)
        pred_dbp = aggregate_vector(g[dbp_col], mode=pred_agg, mix_alpha=pred_mix_alpha, trim_ratio=pred_trim_ratio)
        rows.append({
            subject_col: str(sid),
            "true_sbp": true_sbp,
            "true_dbp": true_dbp,
            "pred_sbp": pred_sbp,
            "pred_dbp": pred_dbp,
            "n_rows": int(len(g)),
        })

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["true_sbp", "true_dbp", "pred_sbp", "pred_dbp"])
    return out


def evaluate_subject_split(
    df: pd.DataFrame,
    method: str,
    split: str,
    true_thr: Threshold,
    pred_thr: Threshold,
    subject_col: str,
    true_sbp_col: str,
    true_dbp_col: str,
    pred_agg: str,
    pred_mix_alpha: float,
    pred_trim_ratio: float,
    custom_cols: Optional[str] = None,
) -> Metrics:
    table = aggregate_subject_table(
        df=df,
        method=method,
        split=split,
        subject_col=subject_col,
        true_sbp_col=true_sbp_col,
        true_dbp_col=true_dbp_col,
        pred_agg=pred_agg,
        pred_mix_alpha=pred_mix_alpha,
        pred_trim_ratio=pred_trim_ratio,
        custom_cols=custom_cols,
    )
    if len(table) == 0:
        return compute_binary_metrics([], [])

    y_true = [is_positive(r.true_sbp, r.true_dbp, true_thr) for r in table.itertuples(index=False)]
    y_pred = [is_positive(r.pred_sbp, r.pred_dbp, pred_thr) for r in table.itertuples(index=False)]
    return compute_binary_metrics(y_true, y_pred)


def evaluate_bp_error_split(
    df: pd.DataFrame,
    method: str,
    split: str,
    subject_col: str,
    true_sbp_col: str,
    true_dbp_col: str,
    pred_agg: str,
    pred_mix_alpha: float,
    pred_trim_ratio: float,
    custom_cols: Optional[str] = None,
) -> Dict[str, object]:
    table = aggregate_subject_table(
        df=df,
        method=method,
        split=split,
        subject_col=subject_col,
        true_sbp_col=true_sbp_col,
        true_dbp_col=true_dbp_col,
        pred_agg=pred_agg,
        pred_mix_alpha=pred_mix_alpha,
        pred_trim_ratio=pred_trim_ratio,
        custom_cols=custom_cols,
    )
    sbp = me_std_mae(table["pred_sbp"], table["true_sbp"]) if len(table) else {"ME": np.nan, "STD": np.nan, "MAE": np.nan, "N": 0}
    dbp = me_std_mae(table["pred_dbp"], table["true_dbp"]) if len(table) else {"ME": np.nan, "STD": np.nan, "MAE": np.nan, "N": 0}
    out = {}
    for k, v in sbp.items():
        out[f"sbp_{k}"] = v
    for k, v in dbp.items():
        out[f"dbp_{k}"] = v
    return out


def evaluate_combined_or(
    df: pd.DataFrame,
    method: str,
    true_thresholds: Dict[str, Threshold],
    pred_thresholds: Dict[str, Threshold],
    pred_aggs: Dict[str, str],
    subject_col: str,
    true_sbp_col: str,
    true_dbp_col: str,
    pred_mix_alpha: float,
    pred_trim_ratio: float,
    custom_cols: Optional[str] = None,
) -> Metrics:
    """
    Subject-level OR rule:
        combined positive if all/day/night any one is positive.
    """
    subjects = set()
    split_tables: Dict[str, pd.DataFrame] = {}

    for split in ["all", "day", "night"]:
        tab = aggregate_subject_table(
            df=df,
            method=method,
            split=split,
            subject_col=subject_col,
            true_sbp_col=true_sbp_col,
            true_dbp_col=true_dbp_col,
            pred_agg=pred_aggs[split],
            pred_mix_alpha=pred_mix_alpha,
            pred_trim_ratio=pred_trim_ratio,
            custom_cols=custom_cols,
        )
        split_tables[split] = tab
        subjects.update(tab[subject_col].astype(str).tolist())

    y_true: List[bool] = []
    y_pred: List[bool] = []

    for sid in sorted(subjects):
        true_any = False
        pred_any = False
        has_any = False

        for split, tab in split_tables.items():
            row = tab[tab[subject_col].astype(str) == sid]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            has_any = True
            true_any = true_any or is_positive(r["true_sbp"], r["true_dbp"], true_thresholds[split])
            pred_any = pred_any or is_positive(r["pred_sbp"], r["pred_dbp"], pred_thresholds[split])

        if has_any:
            y_true.append(true_any)
            y_pred.append(pred_any)

    return compute_binary_metrics(y_true, y_pred)


def make_threshold_grid(base_thr: Threshold, delta_min: float, delta_max: float, step: float) -> List[Threshold]:
    deltas = np.arange(delta_min, delta_max + 1e-9, step, dtype=float)
    return [Threshold(float(base_thr.sbp + ds), float(base_thr.dbp + dd)) for ds in deltas for dd in deltas]


def summarize_across_runs(rows: List[Dict[str, object]], group_cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    metric_cols = ["sensitivity", "specificity", "precision", "accuracy", "f1", "tp", "tn", "fp", "fn", "n"]
    agg_spec = {}
    for c in metric_cols:
        if c in df.columns:
            agg_spec[f"{c}_mean"] = (c, "mean")
            agg_spec[f"{c}_std_across_runs"] = (c, "std")
            agg_spec[f"{c}_min"] = (c, "min")
            agg_spec[f"{c}_max"] = (c, "max")
    return df.groupby(group_cols, dropna=False, sort=False).agg(**agg_spec).reset_index()


def summarize_bp_error_across_runs(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    metric_cols = [c for c in df.columns if c.startswith("sbp_") or c.startswith("dbp_")]
    agg_spec = {}
    for c in metric_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            agg_spec[f"{c}_mean"] = (c, "mean")
            agg_spec[f"{c}_std_across_runs"] = (c, "std")
            agg_spec[f"{c}_min"] = (c, "min")
            agg_spec[f"{c}_max"] = (c, "max")
    return df.groupby(["method", "split", "pred_agg"], dropna=False, sort=False).agg(**agg_spec).reset_index()


def score_candidate(sensitivity: float, specificity: float, min_sensitivity: float, min_specificity: float, prefer: str) -> Tuple[float, float, float, float]:
    sens = float(sensitivity)
    spec = float(specificity)
    ok = 1.0 if (sens >= min_sensitivity and spec >= min_specificity) else 0.0
    violation = (max(0.0, min_sensitivity - sens) ** 2 + max(0.0, min_specificity - spec) ** 2)
    bal_acc = 0.5 * (sens + spec)

    if prefer == "balanced_accuracy":
        pref = bal_acc
    elif prefer == "sensitivity":
        pref = sens
    elif prefer == "specificity":
        pref = spec
    elif prefer == "f1_proxy":
        pref = 2 * sens * spec / max(sens + spec, 1e-12)
    else:
        raise ValueError(f"Unknown prefer={prefer}")
    return (ok, -violation, pref, bal_acc)


def grid_search_one_split(
    dfs: List[pd.DataFrame],
    csv_paths: List[Path],
    method: str,
    split: str,
    true_thr: Threshold,
    pred_grid: List[Threshold],
    pred_agg: str,
    subject_col: str,
    true_sbp_col: str,
    true_dbp_col: str,
    pred_mix_alpha: float,
    pred_trim_ratio: float,
    min_sensitivity: float,
    min_specificity: float,
    prefer: str,
    custom_cols: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    candidate_rows: List[Dict[str, object]] = []

    for pred_thr in pred_grid:
        per_run_rows: List[Dict[str, object]] = []
        for run_id, (df, path) in enumerate(zip(dfs, csv_paths)):
            m = evaluate_subject_split(
                df=df,
                method=method,
                split=split,
                true_thr=true_thr,
                pred_thr=pred_thr,
                subject_col=subject_col,
                true_sbp_col=true_sbp_col,
                true_dbp_col=true_dbp_col,
                pred_agg=pred_agg,
                pred_mix_alpha=pred_mix_alpha,
                pred_trim_ratio=pred_trim_ratio,
                custom_cols=custom_cols,
            )
            row = {
                "run_id": run_id,
                "csv_path": str(path),
                "method": method,
                "split": split,
                "pred_agg": pred_agg,
                "true_sbp_thr": true_thr.sbp,
                "true_dbp_thr": true_thr.dbp,
                "pred_sbp_thr": pred_thr.sbp,
                "pred_dbp_thr": pred_thr.dbp,
            }
            row.update(metrics_to_dict(m))
            per_run_rows.append(row)

        summary = summarize_across_runs(
            per_run_rows,
            group_cols=["method", "split", "pred_agg", "true_sbp_thr", "true_dbp_thr", "pred_sbp_thr", "pred_dbp_thr"],
        )
        if len(summary) != 1:
            raise RuntimeError("Unexpected grid candidate summary shape.")
        candidate_rows.append(summary.iloc[0].to_dict())

    cand_df = pd.DataFrame(candidate_rows)

    best_idx = None
    best_score = None
    for i, r in cand_df.iterrows():
        sc = score_candidate(
            sensitivity=r["sensitivity_mean"],
            specificity=r["specificity_mean"],
            min_sensitivity=min_sensitivity,
            min_specificity=min_specificity,
            prefer=prefer,
        )
        if best_score is None or sc > best_score:
            best_score = sc
            best_idx = i

    best = cand_df.loc[best_idx].to_dict()
    best["meets_target"] = bool(best["sensitivity_mean"] >= min_sensitivity and best["specificity_mean"] >= min_specificity)
    best["target_min_sensitivity"] = min_sensitivity
    best["target_min_specificity"] = min_specificity
    best["selection_prefer"] = prefer
    return cand_df, best


def evaluate_with_best_thresholds(
    dfs: List[pd.DataFrame],
    csv_paths: List[Path],
    methods: List[str],
    best_thresholds: Dict[Tuple[str, str], Threshold],
    true_thresholds: Dict[str, Threshold],
    pred_aggs: Dict[str, str],
    subject_col: str,
    true_sbp_col: str,
    true_dbp_col: str,
    pred_mix_alpha: float,
    pred_trim_ratio: float,
    include_combined: bool,
    custom_cols: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_run: List[Dict[str, object]] = []
    bp_error_rows: List[Dict[str, object]] = []

    for run_id, (df, path) in enumerate(zip(dfs, csv_paths)):
        for method in methods:
            pred_thr_map = {split: best_thresholds[(method, split)] for split in ["all", "day", "night"]}

            for split in ["all", "day", "night"]:
                m = evaluate_subject_split(
                    df=df,
                    method=method,
                    split=split,
                    true_thr=true_thresholds[split],
                    pred_thr=pred_thr_map[split],
                    subject_col=subject_col,
                    true_sbp_col=true_sbp_col,
                    true_dbp_col=true_dbp_col,
                    pred_agg=pred_aggs[split],
                    pred_mix_alpha=pred_mix_alpha,
                    pred_trim_ratio=pred_trim_ratio,
                    custom_cols=custom_cols,
                )
                row = {
                    "run_id": run_id,
                    "csv_path": str(path),
                    "method": method,
                    "split": split,
                    "pred_agg": pred_aggs[split],
                    "true_sbp_thr": true_thresholds[split].sbp,
                    "true_dbp_thr": true_thresholds[split].dbp,
                    "pred_sbp_thr": pred_thr_map[split].sbp,
                    "pred_dbp_thr": pred_thr_map[split].dbp,
                }
                row.update(metrics_to_dict(m))
                per_run.append(row)

                e = evaluate_bp_error_split(
                    df=df,
                    method=method,
                    split=split,
                    subject_col=subject_col,
                    true_sbp_col=true_sbp_col,
                    true_dbp_col=true_dbp_col,
                    pred_agg=pred_aggs[split],
                    pred_mix_alpha=pred_mix_alpha,
                    pred_trim_ratio=pred_trim_ratio,
                    custom_cols=custom_cols,
                )
                erow = {
                    "run_id": run_id,
                    "csv_path": str(path),
                    "method": method,
                    "split": split,
                    "pred_agg": pred_aggs[split],
                }
                erow.update(e)
                bp_error_rows.append(erow)

            if include_combined:
                m = evaluate_combined_or(
                    df=df,
                    method=method,
                    true_thresholds=true_thresholds,
                    pred_thresholds=pred_thr_map,
                    pred_aggs=pred_aggs,
                    subject_col=subject_col,
                    true_sbp_col=true_sbp_col,
                    true_dbp_col=true_dbp_col,
                    pred_mix_alpha=pred_mix_alpha,
                    pred_trim_ratio=pred_trim_ratio,
                    custom_cols=custom_cols,
                )
                row = {
                    "run_id": run_id,
                    "csv_path": str(path),
                    "method": method,
                    "split": "combined",
                    "pred_agg": "all/day/night OR",
                    "true_sbp_thr": np.nan,
                    "true_dbp_thr": np.nan,
                    "pred_sbp_thr": np.nan,
                    "pred_dbp_thr": np.nan,
                    "combined_rule": "all OR day OR night",
                }
                row.update(metrics_to_dict(m))
                per_run.append(row)

    per_run_df = pd.DataFrame(per_run)
    mean_std_df = summarize_across_runs(per_run, group_cols=["method", "split", "pred_agg"])
    bp_err_df = pd.DataFrame(bp_error_rows)
    bp_err_summary_df = summarize_bp_error_across_runs(bp_error_rows)
    return per_run_df, mean_std_df, bp_err_df, bp_err_summary_df


def parse_methods(s: str) -> List[str]:
    out = [x.strip() for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("No methods specified.")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-search predicted BP thresholds using ABPM gold columns.")

    p.add_argument("--csv-glob", type=str, default=None)
    p.add_argument("--csv-paths", type=str, default=None)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--methods", type=str, default="bias", help="Comma-separated methods: raw,bias,aff,bank,cal.")
    p.add_argument("--custom-cols", type=str, default=None, help="Optional custom mapping: method:sbp_col:dbp_col[,method2:sbp:dbp]")
    p.add_argument("--subject-col", type=str, default="id_clean")
    p.add_argument("--include-calib", action="store_true", help="Include calibration rows. Default excludes is_calib=True rows.")

    # Gold-standard columns for evaluation/classification.
    p.add_argument("--true-sbp-col", type=str, default="ABPM_SBP", help="Gold SBP column for evaluation/classification. Default: ABPM_SBP.")
    p.add_argument("--true-dbp-col", type=str, default="ABPM_DBP", help="Gold DBP column for evaluation/classification. Default: ABPM_DBP.")
    p.add_argument("--fallback-true-sbp-col", type=str, default=None, help="Optional fallback if --true-sbp-col is missing, e.g. y_true_sbp.")
    p.add_argument("--fallback-true-dbp-col", type=str, default=None, help="Optional fallback if --true-dbp-col is missing, e.g. y_true_dbp.")

    # True thresholds.
    p.add_argument("--all-sbp", type=float, default=130.0)
    p.add_argument("--all-dbp", type=float, default=80.0)
    p.add_argument("--day-sbp", type=float, default=135.0)
    p.add_argument("--day-dbp", type=float, default=85.0)
    p.add_argument("--night-sbp", type=float, default=120.0)
    p.add_argument("--night-dbp", type=float, default=70.0)

    # Predicted threshold grid: true threshold + delta.
    p.add_argument("--delta-min", type=float, default=-8.0)
    p.add_argument("--delta-max", type=float, default=8.0)
    p.add_argument("--delta-step", type=float, default=1.0)

    # Split-specific prediction aggregation.
    p.add_argument("--all-pred-agg", type=str, default="mean")
    p.add_argument("--day-pred-agg", type=str, default="mix_mean_p75")
    p.add_argument("--night-pred-agg", type=str, default="median")
    p.add_argument("--pred-mix-alpha", type=float, default=0.70, help="For mix_mean_p75: alpha*mean + (1-alpha)*p75. Default 0.70.")
    p.add_argument("--pred-trim-ratio", type=float, default=0.10)

    # Target and selection.
    p.add_argument("--min-sensitivity", type=float, default=0.75)
    p.add_argument("--min-specificity", type=float, default=0.90)
    p.add_argument(
        "--prefer",
        type=str,
        default="balanced_accuracy",
        choices=["balanced_accuracy", "sensitivity", "specificity", "f1_proxy"],
    )
    p.add_argument("--no-combined", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = resolve_csv_paths(args.csv_glob, args.csv_paths)
    dfs = [
        load_one_csv(
            p,
            include_calib=args.include_calib,
            true_sbp_col=args.true_sbp_col,
            true_dbp_col=args.true_dbp_col,
            fallback_true_sbp_col=args.fallback_true_sbp_col,
            fallback_true_dbp_col=args.fallback_true_dbp_col,
        )
        for p in csv_paths
    ]
    methods = parse_methods(args.methods)

    true_thresholds = {
        "all": Threshold(args.all_sbp, args.all_dbp),
        "day": Threshold(args.day_sbp, args.day_dbp),
        "night": Threshold(args.night_sbp, args.night_dbp),
    }
    pred_aggs = {
        "all": args.all_pred_agg,
        "day": args.day_pred_agg,
        "night": args.night_pred_agg,
    }

    all_candidate_frames: List[pd.DataFrame] = []
    best_rows: List[Dict[str, object]] = []
    best_thresholds: Dict[Tuple[str, str], Threshold] = {}

    for method in methods:
        sbp_col, dbp_col = method_columns(method, args.custom_cols)
        for pth, df in zip(csv_paths, dfs):
            if sbp_col not in df.columns or dbp_col not in df.columns:
                raise ValueError(f"{pth} missing method={method} columns: {sbp_col}, {dbp_col}")

        for split in ["all", "day", "night"]:
            pred_grid = make_threshold_grid(true_thresholds[split], args.delta_min, args.delta_max, args.delta_step)
            cand_df, best = grid_search_one_split(
                dfs=dfs,
                csv_paths=csv_paths,
                method=method,
                split=split,
                true_thr=true_thresholds[split],
                pred_grid=pred_grid,
                pred_agg=pred_aggs[split],
                subject_col=args.subject_col,
                true_sbp_col=args.true_sbp_col,
                true_dbp_col=args.true_dbp_col,
                pred_mix_alpha=args.pred_mix_alpha,
                pred_trim_ratio=args.pred_trim_ratio,
                min_sensitivity=args.min_sensitivity,
                min_specificity=args.min_specificity,
                prefer=args.prefer,
                custom_cols=args.custom_cols,
            )
            all_candidate_frames.append(cand_df)
            best_rows.append(best)
            best_thresholds[(method, split)] = Threshold(float(best["pred_sbp_thr"]), float(best["pred_dbp_thr"]))

    candidates = pd.concat(all_candidate_frames, axis=0, ignore_index=True)
    best_df = pd.DataFrame(best_rows)

    per_run_tuned, mean_std_tuned, bp_err_per_run, bp_err_summary = evaluate_with_best_thresholds(
        dfs=dfs,
        csv_paths=csv_paths,
        methods=methods,
        best_thresholds=best_thresholds,
        true_thresholds=true_thresholds,
        pred_aggs=pred_aggs,
        subject_col=args.subject_col,
        true_sbp_col=args.true_sbp_col,
        true_dbp_col=args.true_dbp_col,
        pred_mix_alpha=args.pred_mix_alpha,
        pred_trim_ratio=args.pred_trim_ratio,
        include_combined=not args.no_combined,
        custom_cols=args.custom_cols,
    )

    candidates_path = out_dir / "threshold_grid_candidates.csv"
    best_path = out_dir / "threshold_grid_best_by_split.csv"
    per_run_path = out_dir / "tuned_classification_per_run.csv"
    mean_std_path = out_dir / "tuned_classification_mean_std_across_runs.csv"
    bp_err_path = out_dir / "bp_error_per_run_abpm_gold.csv"
    bp_err_summary_path = out_dir / "bp_error_mean_std_abpm_gold.csv"
    manifest_path = out_dir / "threshold_grid_manifest.json"

    candidates.to_csv(candidates_path, index=False)
    best_df.to_csv(best_path, index=False)
    per_run_tuned.to_csv(per_run_path, index=False)
    mean_std_tuned.to_csv(mean_std_path, index=False)
    bp_err_per_run.to_csv(bp_err_path, index=False)
    bp_err_summary.to_csv(bp_err_summary_path, index=False)

    manifest = {
        "csv_paths": [str(p) for p in csv_paths],
        "out_dir": str(out_dir),
        "methods": methods,
        "include_calib": bool(args.include_calib),
        "gold_columns_for_eval_and_classification": {"sbp": args.true_sbp_col, "dbp": args.true_dbp_col},
        "calibration_note": "This script does not recalibrate. Upstream calibration may still use y_true_sbp/y_true_dbp.",
        "true_thresholds": {k: {"sbp": v.sbp, "dbp": v.dbp} for k, v in true_thresholds.items()},
        "pred_aggs": pred_aggs,
        "pred_mix_alpha": args.pred_mix_alpha,
        "pred_trim_ratio": args.pred_trim_ratio,
        "delta_min": args.delta_min,
        "delta_max": args.delta_max,
        "delta_step": args.delta_step,
        "min_sensitivity": args.min_sensitivity,
        "min_specificity": args.min_specificity,
        "prefer": args.prefer,
        "outputs": {
            "threshold_grid_candidates": str(candidates_path),
            "threshold_grid_best_by_split": str(best_path),
            "tuned_classification_per_run": str(per_run_path),
            "tuned_classification_mean_std_across_runs": str(mean_std_path),
            "bp_error_per_run_abpm_gold": str(bp_err_path),
            "bp_error_mean_std_abpm_gold": str(bp_err_summary_path),
        },
        "warning": "For formal reporting, tune thresholds on validation data and apply fixed thresholds to held-out test data.",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== ABPM-gold threshold grid search finished ===")
    print(f"CSV count:                  {len(csv_paths)}")
    print(f"Methods:                    {methods}")
    print(f"Gold columns:               {args.true_sbp_col}, {args.true_dbp_col}")
    print(f"Pred aggregation:           all={args.all_pred_agg}, day={args.day_pred_agg}, night={args.night_pred_agg}")
    print(f"Best thresholds:            {best_path}")
    print(f"Tuned mean/std summary:     {mean_std_path}")
    print(f"BP error summary:           {bp_err_summary_path}")
    print("\nBest thresholds preview:")
    cols = [
        "method", "split", "pred_agg", "pred_sbp_thr", "pred_dbp_thr",
        "sensitivity_mean", "specificity_mean", "accuracy_mean", "f1_mean", "meets_target",
    ]
    existing = [c for c in cols if c in best_df.columns]
    print(best_df[existing].to_string(index=False))


if __name__ == "__main__":
    main()
