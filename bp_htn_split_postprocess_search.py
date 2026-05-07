#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split-specific hypertension classification search for 1p0 / low-calibration PPG-BP outputs.

This script is designed for the current 1p0 situation:
    - one calibration point, usually daytime
    - bias calibration applied to all events may hurt night specificity
    - bank behaves almost like bias
    - affine is not reliable for 1p0

What this script adds beyond plain threshold grid search:
    1) sleep-aware bias transfer shrinkage:
        If query sleep state matches calibration sleep state:
            pred = raw + residual
        If query sleep state differs:
            pred = raw + gamma_cross * residual
       This directly tests whether a daytime calibration bias should be fully transferred to night.

    2) split-specific prediction aggregation:
        All   can use mean/median/etc.
        Day   can use p75 or mean+p75 to improve sensitivity.
        Night can use median/trimmed/remove_top1 to improve specificity.

    3) split-specific predicted threshold grid search:
        True labels keep clinical thresholds:
            All   130/80
            Day   135/85
            Night 120/70
        Predicted thresholds are tuned separately per split.

    4) combined OR metric:
        A subject is positive if All OR Day OR Night is positive.

Typical usage for 1p0 daytime calibration outputs:
    python bp_htn_split_postprocess_search.py \
        --csv-glob "eval_1p0_sensitivity/predictions_*.csv" \
        --methods xstate_bias,bias \
        --gamma-grid 0,0.3,0.5,0.7,1.0 \
        --all-aggs mean,median \
        --day-aggs mean,p75,mean_p75_0.3,mean_p75_0.5 \
        --night-aggs mean,median,trimmed_mean,remove_top1_mean,p60 \
        --out-dir eval_1p0_sensitivity/split_postprocess_search

Required input columns:
    id_clean, sleep, is_calib
    y_true_sbp, y_true_dbp
    y_pred_sbp_raw, y_pred_dbp_raw

Optional existing prediction columns:
    y_pred_sbp_bias / y_pred_dbp_bias
    y_pred_sbp_bank / y_pred_dbp_bank
    y_pred_sbp_aff  / y_pred_dbp_aff
    y_pred_sbp_cal  / y_pred_dbp_cal
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
    # xstate_bias is computed dynamically from raw + calibration residual.
    "xstate_bias": ("__xstate_sbp__", "__xstate_dbp__"),
}


# =============================================================================
# Basic utilities
# =============================================================================


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

    if len(yt) == 0:
        return Metrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

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


def parse_csv_list(csv_glob: Optional[str], csv_paths: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    if csv_glob:
        paths.extend(Path(p) for p in sorted(glob.glob(csv_glob)))
    if csv_paths:
        for p in csv_paths.split(","):
            p = p.strip()
            if p:
                paths.append(Path(p))

    seen = set()
    out: List[Path] = []
    for p in paths:
        rp = str(p)
        if rp not in seen:
            out.append(p)
            seen.add(rp)

    if not out:
        raise ValueError("No CSV files found. Provide --csv-glob or --csv-paths.")
    return out


def parse_float_list(s: str) -> List[float]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    if not out:
        raise ValueError(f"Empty float list: {s}")
    return out


def parse_str_list(s: str) -> List[str]:
    out = [x.strip() for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError(f"Empty string list: {s}")
    return out


# =============================================================================
# Loading and prediction-column handling
# =============================================================================


def load_one_csv(path: Path, include_calib_for_eval: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["id_clean", "sleep", "y_true_sbp", "y_true_dbp", "y_pred_sbp_raw", "y_pred_dbp_raw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    df = df.copy()
    df["id_clean"] = df["id_clean"].astype(str)
    df["sleep"] = pd.to_numeric(df["sleep"], errors="coerce").fillna(0).astype(int)
    if "is_calib" not in df.columns:
        df["is_calib"] = False
    else:
        df["is_calib"] = df["is_calib"].map(parse_bool).astype(bool)

    if "t_bp_ms" in df.columns:
        df["t_bp_ms"] = pd.to_numeric(df["t_bp_ms"], errors="coerce").fillna(0).astype(np.int64)

    numeric_cols = [
        "y_true_sbp", "y_true_dbp", "y_pred_sbp_raw", "y_pred_dbp_raw",
        "y_pred_sbp_bias", "y_pred_dbp_bias", "y_pred_sbp_bank", "y_pred_dbp_bank",
        "y_pred_sbp_aff", "y_pred_dbp_aff", "y_pred_sbp_cal", "y_pred_dbp_cal",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Do not drop calib rows here. Calibration rows may be needed to recompute xstate_bias.
    df["__include_eval__"] = True
    if not include_calib_for_eval:
        df.loc[df["is_calib"].astype(bool), "__include_eval__"] = False

    return df


def add_xstate_bias_predictions(
    df: pd.DataFrame,
    gamma_cross: float,
    subject_col: str = "id_clean",
    sleep_col: str = "sleep",
) -> pd.DataFrame:
    """
    Recompute sleep-aware 1p0 / low-calibration bias from raw predictions.

    For each subject:
        calib residual = mean(y_true - y_raw) from calibration rows.

    For each query row:
        if same-sleep calibration rows exist:
            use same-sleep residual with full weight.
        else:
            use all calibration residuals scaled by gamma_cross.

    This covers the current case:
        daytime calibration point -> day rows full correction, night rows gamma correction.
    """
    out = df.copy()
    out["__xstate_sbp__"] = out["y_pred_sbp_raw"].astype(float)
    out["__xstate_dbp__"] = out["y_pred_dbp_raw"].astype(float)
    out["__xstate_gamma__"] = float(gamma_cross)
    out["__xstate_same_sleep_used__"] = False

    for sid, g in out.groupby(subject_col, sort=False):
        idx_all = g.index.to_numpy()
        g_cal = g[g["is_calib"].astype(bool)].copy()
        if len(g_cal) == 0:
            continue

        # All-calib residual fallback.
        all_res_sbp = float((g_cal["y_true_sbp"] - g_cal["y_pred_sbp_raw"]).mean())
        all_res_dbp = float((g_cal["y_true_dbp"] - g_cal["y_pred_dbp_raw"]).mean())

        # Same-sleep residuals when available.
        same_sleep_res: Dict[int, Tuple[float, float]] = {}
        for slp, gs in g_cal.groupby(sleep_col, sort=False):
            same_sleep_res[int(slp)] = (
                float((gs["y_true_sbp"] - gs["y_pred_sbp_raw"]).mean()),
                float((gs["y_true_dbp"] - gs["y_pred_dbp_raw"]).mean()),
            )

        for idx in idx_all:
            slp = int(out.loc[idx, sleep_col])
            if slp in same_sleep_res:
                res_sbp, res_dbp = same_sleep_res[slp]
                scale = 1.0
                out.loc[idx, "__xstate_same_sleep_used__"] = True
            else:
                res_sbp, res_dbp = all_res_sbp, all_res_dbp
                scale = float(gamma_cross)
                out.loc[idx, "__xstate_same_sleep_used__"] = False

            out.loc[idx, "__xstate_sbp__"] = float(out.loc[idx, "y_pred_sbp_raw"]) + scale * res_sbp
            out.loc[idx, "__xstate_dbp__"] = float(out.loc[idx, "y_pred_dbp_raw"]) + scale * res_dbp

    return out


def method_columns(method: str) -> Tuple[str, str]:
    if method not in DEFAULT_METHOD_COLS:
        raise ValueError(f"Unknown method={method}. Choose from {list(DEFAULT_METHOD_COLS)}")
    return DEFAULT_METHOD_COLS[method]


# =============================================================================
# Aggregation
# =============================================================================


def aggregate_values(values: Sequence[float], mode: str, trim_ratio: float = 0.10) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan

    mode = mode.lower()

    if mode == "mean":
        return float(np.mean(arr))
    if mode == "median":
        return float(np.median(arr))
    if mode == "p60":
        return float(np.percentile(arr, 60))
    if mode == "p70":
        return float(np.percentile(arr, 70))
    if mode == "p75":
        return float(np.percentile(arr, 75))
    if mode == "p80":
        return float(np.percentile(arr, 80))
    if mode == "p90":
        return float(np.percentile(arr, 90))
    if mode == "trimmed_mean":
        if len(arr) < 3:
            return float(np.mean(arr))
        arr = np.sort(arr)
        k = int(math.floor(len(arr) * trim_ratio))
        if 2 * k >= len(arr):
            return float(np.mean(arr))
        return float(np.mean(arr[k: len(arr) - k]))
    if mode == "remove_top1_mean":
        if len(arr) <= 2:
            return float(np.mean(arr))
        arr = np.sort(arr)
        return float(np.mean(arr[:-1]))
    if mode == "remove_top2_mean":
        if len(arr) <= 4:
            return float(np.mean(arr))
        arr = np.sort(arr)
        return float(np.mean(arr[:-2]))

    # Hybrid modes: mean_p75_0.3 means 0.7*mean + 0.3*p75.
    if mode.startswith("mean_p75_"):
        alpha = float(mode.split("_")[-1])
        return float((1.0 - alpha) * np.mean(arr) + alpha * np.percentile(arr, 75))

    if mode.startswith("mean_p80_"):
        alpha = float(mode.split("_")[-1])
        return float((1.0 - alpha) * np.mean(arr) + alpha * np.percentile(arr, 80))

    raise ValueError(f"Unknown aggregation mode={mode}")


def split_filter(df: pd.DataFrame, split: str) -> pd.DataFrame:
    split = split.lower()
    if split == "all":
        return df.copy()
    if split == "day":
        return df[df["sleep"].astype(int) == 0].copy()
    if split in ["night", "sleep"]:
        return df[df["sleep"].astype(int) == 1].copy()
    raise ValueError(f"Unknown split={split}")


def subject_table(
    df: pd.DataFrame,
    method: str,
    split: str,
    pred_agg: str,
    true_agg: str,
    subject_col: str,
    include_eval_col: str = "__include_eval__",
) -> pd.DataFrame:
    sbp_col, dbp_col = method_columns(method)
    if sbp_col not in df.columns or dbp_col not in df.columns:
        raise ValueError(f"Prediction columns missing for method={method}: {sbp_col}, {dbp_col}")

    d = df[df[include_eval_col].astype(bool)].copy()
    d = split_filter(d, split)
    if len(d) == 0:
        return pd.DataFrame(columns=[subject_col, "true_sbp", "true_dbp", "pred_sbp", "pred_dbp", "n_rows"])

    rows: List[Dict[str, object]] = []
    for sid, g in d.groupby(subject_col, sort=False):
        rows.append({
            subject_col: str(sid),
            "true_sbp": aggregate_values(g["y_true_sbp"], true_agg),
            "true_dbp": aggregate_values(g["y_true_dbp"], true_agg),
            "pred_sbp": aggregate_values(g[sbp_col], pred_agg),
            "pred_dbp": aggregate_values(g[dbp_col], pred_agg),
            "n_rows": int(len(g)),
        })

    tab = pd.DataFrame(rows)
    return tab.dropna(subset=["true_sbp", "true_dbp", "pred_sbp", "pred_dbp"])


def evaluate_subject_split(
    df: pd.DataFrame,
    method: str,
    split: str,
    pred_agg: str,
    true_agg: str,
    true_thr: Threshold,
    pred_thr: Threshold,
    subject_col: str,
) -> Metrics:
    tab = subject_table(df, method, split, pred_agg, true_agg, subject_col)
    y_true = [is_positive(r.true_sbp, r.true_dbp, true_thr) for r in tab.itertuples(index=False)]
    y_pred = [is_positive(r.pred_sbp, r.pred_dbp, pred_thr) for r in tab.itertuples(index=False)]
    return compute_binary_metrics(y_true, y_pred)


# =============================================================================
# Grid search
# =============================================================================


def make_threshold_grid(base_thr: Threshold, delta_min: float, delta_max: float, step: float) -> List[Threshold]:
    deltas = np.arange(float(delta_min), float(delta_max) + 1e-9, float(step), dtype=float)
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


def score_candidate(sens: float, spec: float, min_sens: float, min_spec: float, prefer: str) -> Tuple[float, float, float, float]:
    ok = 1.0 if (sens >= min_sens and spec >= min_spec) else 0.0
    violation = (max(0.0, min_sens - sens) ** 2 + max(0.0, min_spec - spec) ** 2)
    bal = 0.5 * (sens + spec)
    if prefer == "balanced_accuracy":
        pref = bal
    elif prefer == "sensitivity":
        pref = sens
    elif prefer == "specificity":
        pref = spec
    elif prefer == "f1_proxy":
        pref = 2 * sens * spec / max(sens + spec, 1e-12)
    else:
        raise ValueError(f"Unknown prefer={prefer}")
    return (ok, -violation, pref, bal)


def grid_search_split(
    dfs_by_gamma: Dict[Tuple[str, float], List[pd.DataFrame]],
    csv_paths: List[Path],
    method: str,
    split: str,
    agg_list: List[str],
    true_agg: str,
    true_thr: Threshold,
    pred_grid: List[Threshold],
    subject_col: str,
    min_sens: float,
    min_spec: float,
    prefer: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    candidate_rows: List[Dict[str, object]] = []

    # For normal methods, gamma key is nan only. For xstate_bias, multiple gamma values exist.
    keys = [k for k in dfs_by_gamma.keys() if k[0] == method]
    if not keys:
        raise ValueError(f"No prepared dataframes for method={method}")

    for key_method, gamma in keys:
        dfs = dfs_by_gamma[(key_method, gamma)]
        for pred_agg in agg_list:
            for pred_thr in pred_grid:
                per_run: List[Dict[str, object]] = []
                for run_id, (df, path) in enumerate(zip(dfs, csv_paths)):
                    m = evaluate_subject_split(
                        df=df,
                        method=method,
                        split=split,
                        pred_agg=pred_agg,
                        true_agg=true_agg,
                        true_thr=true_thr,
                        pred_thr=pred_thr,
                        subject_col=subject_col,
                    )
                    row = {
                        "run_id": run_id,
                        "csv_path": str(path),
                        "method": method,
                        "split": split,
                        "gamma_cross": gamma,
                        "true_agg": true_agg,
                        "pred_agg": pred_agg,
                        "true_sbp_thr": true_thr.sbp,
                        "true_dbp_thr": true_thr.dbp,
                        "pred_sbp_thr": pred_thr.sbp,
                        "pred_dbp_thr": pred_thr.dbp,
                    }
                    row.update(metrics_to_dict(m))
                    per_run.append(row)

                summary = summarize_across_runs(
                    per_run,
                    group_cols=[
                        "method", "split", "gamma_cross", "true_agg", "pred_agg",
                        "true_sbp_thr", "true_dbp_thr", "pred_sbp_thr", "pred_dbp_thr",
                    ],
                )
                if len(summary) != 1:
                    raise RuntimeError("Unexpected summary shape")
                candidate_rows.append(summary.iloc[0].to_dict())

    cand = pd.DataFrame(candidate_rows)

    best_idx = None
    best_score = None
    for i, r in cand.iterrows():
        sc = score_candidate(
            float(r["sensitivity_mean"]),
            float(r["specificity_mean"]),
            min_sens,
            min_spec,
            prefer,
        )
        if best_score is None or sc > best_score:
            best_score = sc
            best_idx = i

    best = cand.loc[best_idx].to_dict()
    best["meets_target"] = bool(best["sensitivity_mean"] >= min_sens and best["specificity_mean"] >= min_spec)
    best["target_min_sensitivity"] = float(min_sens)
    best["target_min_specificity"] = float(min_spec)
    best["selection_prefer"] = prefer
    return cand, best


# =============================================================================
# Tuned evaluation and combined OR
# =============================================================================


def evaluate_combined_or(
    df_by_key: Dict[Tuple[str, float], pd.DataFrame],
    method: str,
    selected: Dict[str, Dict[str, object]],
    true_thresholds: Dict[str, Threshold],
    subject_col: str,
) -> Metrics:
    subjects = set()
    split_tables: Dict[str, pd.DataFrame] = {}

    for split in ["all", "day", "night"]:
        b = selected[split]
        gamma = float(b["gamma_cross"])
        df = df_by_key[(method, gamma)]
        tab = subject_table(
            df=df,
            method=method,
            split=split,
            pred_agg=str(b["pred_agg"]),
            true_agg=str(b["true_agg"]),
            subject_col=subject_col,
        )
        split_tables[split] = tab
        if len(tab) > 0:
            subjects.update(tab[subject_col].astype(str).tolist())

    y_true: List[bool] = []
    y_pred: List[bool] = []

    for sid in sorted(subjects):
        t_any = False
        p_any = False
        has_any = False
        for split, tab in split_tables.items():
            r = tab[tab[subject_col].astype(str) == sid]
            if len(r) == 0:
                continue
            row = r.iloc[0]
            b = selected[split]
            true_thr = true_thresholds[split]
            pred_thr = Threshold(float(b["pred_sbp_thr"]), float(b["pred_dbp_thr"]))
            t_any = t_any or is_positive(row["true_sbp"], row["true_dbp"], true_thr)
            p_any = p_any or is_positive(row["pred_sbp"], row["pred_dbp"], pred_thr)
            has_any = True
        if has_any:
            y_true.append(t_any)
            y_pred.append(p_any)

    return compute_binary_metrics(y_true, y_pred)


def evaluate_tuned(
    dfs_by_gamma: Dict[Tuple[str, float], List[pd.DataFrame]],
    csv_paths: List[Path],
    methods: List[str],
    best_by_method_split: Dict[Tuple[str, str], Dict[str, object]],
    true_thresholds: Dict[str, Threshold],
    subject_col: str,
    include_combined: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_run: List[Dict[str, object]] = []

    for run_id, path in enumerate(csv_paths):
        for method in methods:
            selected: Dict[str, Dict[str, object]] = {
                split: best_by_method_split[(method, split)] for split in ["all", "day", "night"]
            }

            # Build per-run df map for this method.
            df_by_key_run: Dict[Tuple[str, float], pd.DataFrame] = {}
            for split, b in selected.items():
                gamma = float(b["gamma_cross"])
                key = (method, gamma)
                df_by_key_run[key] = dfs_by_gamma[key][run_id]

            for split in ["all", "day", "night"]:
                b = selected[split]
                gamma = float(b["gamma_cross"])
                df = dfs_by_gamma[(method, gamma)][run_id]
                pred_thr = Threshold(float(b["pred_sbp_thr"]), float(b["pred_dbp_thr"]))
                m = evaluate_subject_split(
                    df=df,
                    method=method,
                    split=split,
                    pred_agg=str(b["pred_agg"]),
                    true_agg=str(b["true_agg"]),
                    true_thr=true_thresholds[split],
                    pred_thr=pred_thr,
                    subject_col=subject_col,
                )
                row = {
                    "run_id": run_id,
                    "csv_path": str(path),
                    "method": method,
                    "split": split,
                    "gamma_cross": gamma,
                    "true_agg": str(b["true_agg"]),
                    "pred_agg": str(b["pred_agg"]),
                    "true_sbp_thr": true_thresholds[split].sbp,
                    "true_dbp_thr": true_thresholds[split].dbp,
                    "pred_sbp_thr": pred_thr.sbp,
                    "pred_dbp_thr": pred_thr.dbp,
                }
                row.update(metrics_to_dict(m))
                per_run.append(row)

            if include_combined:
                m = evaluate_combined_or(
                    df_by_key=df_by_key_run,
                    method=method,
                    selected=selected,
                    true_thresholds=true_thresholds,
                    subject_col=subject_col,
                )
                row = {
                    "run_id": run_id,
                    "csv_path": str(path),
                    "method": method,
                    "split": "combined",
                    "gamma_cross": np.nan,
                    "true_agg": "OR(all,day,night)",
                    "pred_agg": "OR(all,day,night)",
                    "true_sbp_thr": np.nan,
                    "true_dbp_thr": np.nan,
                    "pred_sbp_thr": np.nan,
                    "pred_dbp_thr": np.nan,
                    "combined_rule": "all OR day OR night",
                }
                row.update(metrics_to_dict(m))
                per_run.append(row)

    per_run_df = pd.DataFrame(per_run)
    mean_std = summarize_across_runs(per_run, group_cols=["method", "split"])
    return per_run_df, mean_std


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split-specific post-processing and threshold search for PPG-BP hypertension classification."
    )
    p.add_argument("--csv-glob", type=str, default=None)
    p.add_argument("--csv-paths", type=str, default=None)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--methods", type=str, default="xstate_bias,bias", help="Comma-separated: raw,bias,bank,aff,cal,xstate_bias")
    p.add_argument("--subject-col", type=str, default="id_clean")
    p.add_argument("--include-calib", action="store_true", help="Include calibration rows in evaluation. Default excludes them.")

    # True clinical thresholds.
    p.add_argument("--all-sbp", type=float, default=130.0)
    p.add_argument("--all-dbp", type=float, default=80.0)
    p.add_argument("--day-sbp", type=float, default=135.0)
    p.add_argument("--day-dbp", type=float, default=85.0)
    p.add_argument("--night-sbp", type=float, default=120.0)
    p.add_argument("--night-dbp", type=float, default=70.0)

    # Threshold search.
    p.add_argument("--delta-min", type=float, default=-8.0)
    p.add_argument("--delta-max", type=float, default=12.0)
    p.add_argument("--delta-step", type=float, default=1.0)
    p.add_argument("--min-sensitivity", type=float, default=0.75)
    p.add_argument("--min-specificity", type=float, default=0.90)
    p.add_argument("--prefer", type=str, default="balanced_accuracy", choices=["balanced_accuracy", "sensitivity", "specificity", "f1_proxy"])

    # Aggregation search.
    p.add_argument("--true-agg", type=str, default="mean", help="True label aggregation, usually mean.")
    p.add_argument("--all-aggs", type=str, default="mean,median")
    p.add_argument("--day-aggs", type=str, default="mean,p75,mean_p75_0.3,mean_p75_0.5")
    p.add_argument("--night-aggs", type=str, default="mean,median,trimmed_mean,remove_top1_mean,p60")

    # Cross-state shrinkage for xstate_bias.
    p.add_argument("--gamma-grid", type=str, default="0,0.3,0.5,0.7,1.0", help="Only used by xstate_bias.")

    p.add_argument("--no-combined", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = parse_csv_list(args.csv_glob, args.csv_paths)
    raw_dfs = [load_one_csv(p, include_calib_for_eval=args.include_calib) for p in csv_paths]
    methods = parse_str_list(args.methods)
    gammas = parse_float_list(args.gamma_grid)

    true_thresholds = {
        "all": Threshold(args.all_sbp, args.all_dbp),
        "day": Threshold(args.day_sbp, args.day_dbp),
        "night": Threshold(args.night_sbp, args.night_dbp),
    }
    split_aggs = {
        "all": parse_str_list(args.all_aggs),
        "day": parse_str_list(args.day_aggs),
        "night": parse_str_list(args.night_aggs),
    }

    # Prepare dataframes for each method/gamma.
    dfs_by_key: Dict[Tuple[str, float], List[pd.DataFrame]] = {}
    for method in methods:
        if method == "xstate_bias":
            for gamma in gammas:
                dfs_by_key[(method, float(gamma))] = [add_xstate_bias_predictions(df, gamma) for df in raw_dfs]
        else:
            sbp_col, dbp_col = method_columns(method)
            for pth, df in zip(csv_paths, raw_dfs):
                if sbp_col not in df.columns or dbp_col not in df.columns:
                    raise ValueError(f"{pth} missing columns for method={method}: {sbp_col}, {dbp_col}")
            dfs_by_key[(method, float("nan"))] = raw_dfs

    all_candidates: List[pd.DataFrame] = []
    best_rows: List[Dict[str, object]] = []
    best_by_method_split: Dict[Tuple[str, str], Dict[str, object]] = {}

    for method in methods:
        for split in ["all", "day", "night"]:
            grid = make_threshold_grid(true_thresholds[split], args.delta_min, args.delta_max, args.delta_step)
            cand, best = grid_search_split(
                dfs_by_gamma=dfs_by_key,
                csv_paths=csv_paths,
                method=method,
                split=split,
                agg_list=split_aggs[split],
                true_agg=args.true_agg,
                true_thr=true_thresholds[split],
                pred_grid=grid,
                subject_col=args.subject_col,
                min_sens=args.min_sensitivity,
                min_spec=args.min_specificity,
                prefer=args.prefer,
            )
            all_candidates.append(cand)
            best_rows.append(best)
            best_by_method_split[(method, split)] = best

    candidates_df = pd.concat(all_candidates, axis=0, ignore_index=True)
    best_df = pd.DataFrame(best_rows)

    per_run_df, mean_std_df = evaluate_tuned(
        dfs_by_gamma=dfs_by_key,
        csv_paths=csv_paths,
        methods=methods,
        best_by_method_split=best_by_method_split,
        true_thresholds=true_thresholds,
        subject_col=args.subject_col,
        include_combined=not args.no_combined,
    )

    candidates_path = out_dir / "split_postprocess_candidates.csv"
    best_path = out_dir / "split_postprocess_best_by_split.csv"
    per_run_path = out_dir / "split_postprocess_tuned_per_run.csv"
    mean_std_path = out_dir / "split_postprocess_tuned_mean_std_across_runs.csv"
    manifest_path = out_dir / "split_postprocess_manifest.json"

    candidates_df.to_csv(candidates_path, index=False)
    best_df.to_csv(best_path, index=False)
    per_run_df.to_csv(per_run_path, index=False)
    mean_std_df.to_csv(mean_std_path, index=False)

    manifest = {
        "csv_paths": [str(p) for p in csv_paths],
        "out_dir": str(out_dir),
        "methods": methods,
        "gamma_grid": gammas,
        "true_thresholds": {k: {"sbp": v.sbp, "dbp": v.dbp} for k, v in true_thresholds.items()},
        "true_agg": args.true_agg,
        "split_aggs": split_aggs,
        "delta_min": args.delta_min,
        "delta_max": args.delta_max,
        "delta_step": args.delta_step,
        "min_sensitivity": args.min_sensitivity,
        "min_specificity": args.min_specificity,
        "prefer": args.prefer,
        "outputs": {
            "candidates": str(candidates_path),
            "best_by_split": str(best_path),
            "per_run": str(per_run_path),
            "mean_std": str(mean_std_path),
        },
        "note": "Exploratory search. For formal report, tune on validation and apply fixed rules to held-out test.",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== Split-specific post-processing search finished ===")
    print(f"CSV count:      {len(csv_paths)}")
    print(f"Methods:        {methods}")
    print(f"Best rules:     {best_path}")
    print(f"Tuned summary:  {mean_std_path}")
    print("\nBest preview:")
    preview_cols = [
        "method", "split", "gamma_cross", "pred_agg", "pred_sbp_thr", "pred_dbp_thr",
        "sensitivity_mean", "specificity_mean", "accuracy_mean", "f1_mean", "meets_target",
    ]
    existing = [c for c in preview_cols if c in best_df.columns]
    print(best_df[existing].to_string(index=False))


if __name__ == "__main__":
    main()
