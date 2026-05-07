#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment A: 1p0 day-calibrated bias + sleep-aware cross-state shrinkage.

Purpose
-------
You currently use 1p0 calibration:
    - one calibration point, usually from daytime
    - bias residual is computed with y_true_sbp / y_true_dbp
    - same bias is applied to all time points

This script tests:
    if query_sleep == calib_sleep:
        pred = raw + 1.0 * residual
    else:
        pred = raw + gamma_cross * residual

It sweeps gamma_cross values, saves gamma-adjusted CSV files, and optionally
performs ABPM-based hypertension classification threshold grid search.

Gold label for classification/evaluation:
    ABPM_SBP / ABPM_DBP by default.

Calibration residual:
    y_true_sbp / y_true_dbp by default.

Recommended usage
-----------------
python run_1p0_sleep_shrinkage_experiment.py \
    --csv-glob "eval_1p0_sensitivity/predictions_*.csv" \
    --out-dir eval_1p0_sleep_shrinkage_exp \
    --gammas 0,0.25,0.5,0.75,1.0 \
    --delta-min -8 \
    --delta-max 8 \
    --delta-step 1

If your raw prediction columns are named y_pred_sbp/y_pred_dbp:
python run_1p0_sleep_shrinkage_experiment.py \
    --csv-glob "eval_1p0_sensitivity/predictions_*.csv" \
    --raw-sbp-col y_pred_sbp \
    --raw-dbp-col y_pred_dbp \
    --out-dir eval_1p0_sleep_shrinkage_exp

Outputs
-------
out_dir/
  gamma_0.00/
    predictions_*.csv
    threshold_grid_candidates.csv
    threshold_grid_best_by_split.csv
    tuned_classification_per_run.csv
    tuned_classification_mean_std_across_runs.csv
  gamma_0.25/
    ...
  experiment_gamma_summary.csv
  experiment_manifest.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Basic metric utilities
# =============================================================================

@dataclass
class Threshold:
    sbp: float
    dbp: float


def parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num) / float(den)


def is_positive(sbp: float, dbp: float, thr: Threshold) -> bool:
    return (float(sbp) >= thr.sbp) or (float(dbp) >= thr.dbp)


def confusion_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum((~y_true) & (~y_pred)))
    fp = int(np.sum((~y_true) & y_pred))
    fn = int(np.sum(y_true & (~y_pred)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    sensitivity = recall
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "n": int(len(y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
    }


def mean_std_min_max(values: Sequence[float]) -> Dict[str, float]:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return {"mean": np.nan, "std_across_runs": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(x)),
        "std_across_runs": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


# =============================================================================
# 1p0 sleep-aware shrinkage
# =============================================================================

def infer_raw_cols(df: pd.DataFrame, raw_sbp_col: Optional[str], raw_dbp_col: Optional[str]) -> Tuple[str, str]:
    if raw_sbp_col is not None and raw_dbp_col is not None:
        return raw_sbp_col, raw_dbp_col

    candidates = [
        ("y_pred_sbp_raw", "y_pred_dbp_raw"),
        ("y_pred_sbp", "y_pred_dbp"),
    ]
    for s_col, d_col in candidates:
        if s_col in df.columns and d_col in df.columns:
            return s_col, d_col

    raise ValueError(
        "Cannot infer raw prediction columns. Provide --raw-sbp-col and --raw-dbp-col."
    )


def apply_1p0_sleep_shrinkage(
    df: pd.DataFrame,
    gamma_cross: float,
    raw_sbp_col: Optional[str] = None,
    raw_dbp_col: Optional[str] = None,
    calib_true_sbp_col: str = "y_true_sbp",
    calib_true_dbp_col: str = "y_true_dbp",
    out_sbp_col: str = "y_pred_sbp_bias",
    out_dbp_col: str = "y_pred_dbp_bias",
    subject_col: str = "id_clean",
    sleep_col: str = "sleep",
    is_calib_col: str = "is_calib",
) -> pd.DataFrame:
    """
    Recompute bias prediction columns with sleep-aware cross-state shrinkage.

    Calibration residual uses calibration rows:
        residual = calib_true - raw_pred

    Same sleep state:
        scale = 1

    Cross sleep state:
        scale = gamma_cross
    """
    out = df.copy()
    raw_sbp_col, raw_dbp_col = infer_raw_cols(out, raw_sbp_col, raw_dbp_col)

    required = [
        subject_col,
        sleep_col,
        is_calib_col,
        raw_sbp_col,
        raw_dbp_col,
        calib_true_sbp_col,
        calib_true_dbp_col,
    ]
    miss = [c for c in required if c not in out.columns]
    if miss:
        raise ValueError(f"Missing columns for shrinkage: {miss}")

    out[sleep_col] = pd.to_numeric(out[sleep_col], errors="coerce").fillna(0).astype(int)
    out[is_calib_col] = parse_bool_series(out[is_calib_col])

    out[out_sbp_col] = pd.to_numeric(out[raw_sbp_col], errors="coerce").astype(float)
    out[out_dbp_col] = pd.to_numeric(out[raw_dbp_col], errors="coerce").astype(float)

    out["bias_gamma_cross"] = float(gamma_cross)
    out["calib_sleep_state"] = np.nan
    out["bias_delta_sbp_1p0"] = np.nan
    out["bias_delta_dbp_1p0"] = np.nan
    out["bias_scale_applied"] = np.nan
    out["bias_raw_sbp_col"] = raw_sbp_col
    out["bias_raw_dbp_col"] = raw_dbp_col

    for pid, g in out.groupby(subject_col, sort=False):
        calib = g[g[is_calib_col].astype(bool)].copy()
        if len(calib) == 0:
            continue

        # In 1p0 there is usually one calibration point.
        # If more than one exists, averaging keeps the function robust.
        delta_sbp = float(
            (
                pd.to_numeric(calib[calib_true_sbp_col], errors="coerce")
                - pd.to_numeric(calib[raw_sbp_col], errors="coerce")
            ).mean()
        )
        delta_dbp = float(
            (
                pd.to_numeric(calib[calib_true_dbp_col], errors="coerce")
                - pd.to_numeric(calib[raw_dbp_col], errors="coerce")
            ).mean()
        )

        if not np.isfinite(delta_sbp) or not np.isfinite(delta_dbp):
            continue

        calib_sleep = int(round(float(calib[sleep_col].astype(int).mean())))

        idx = g.index
        query_sleep = out.loc[idx, sleep_col].astype(int).to_numpy()
        scale = np.where(query_sleep == calib_sleep, 1.0, float(gamma_cross))

        raw_sbp = pd.to_numeric(out.loc[idx, raw_sbp_col], errors="coerce").to_numpy(dtype=float)
        raw_dbp = pd.to_numeric(out.loc[idx, raw_dbp_col], errors="coerce").to_numpy(dtype=float)

        out.loc[idx, out_sbp_col] = raw_sbp + scale * delta_sbp
        out.loc[idx, out_dbp_col] = raw_dbp + scale * delta_dbp
        out.loc[idx, "calib_sleep_state"] = calib_sleep
        out.loc[idx, "bias_delta_sbp_1p0"] = delta_sbp
        out.loc[idx, "bias_delta_dbp_1p0"] = delta_dbp
        out.loc[idx, "bias_scale_applied"] = scale

    return out


def batch_apply_shrinkage(
    csv_paths: List[Path],
    gamma: float,
    out_dir: Path,
    raw_sbp_col: Optional[str],
    raw_dbp_col: Optional[str],
    calib_true_sbp_col: str,
    calib_true_dbp_col: str,
    out_sbp_col: str,
    out_dbp_col: str,
    subject_col: str,
    sleep_col: str,
    is_calib_col: str,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[Path] = []

    for p in csv_paths:
        df = pd.read_csv(p)
        out = apply_1p0_sleep_shrinkage(
            df=df,
            gamma_cross=gamma,
            raw_sbp_col=raw_sbp_col,
            raw_dbp_col=raw_dbp_col,
            calib_true_sbp_col=calib_true_sbp_col,
            calib_true_dbp_col=calib_true_dbp_col,
            out_sbp_col=out_sbp_col,
            out_dbp_col=out_dbp_col,
            subject_col=subject_col,
            sleep_col=sleep_col,
            is_calib_col=is_calib_col,
        )
        out_path = out_dir / p.name
        out.to_csv(out_path, index=False)
        out_paths.append(out_path)

    return out_paths


# =============================================================================
# Subject-level classification with flexible aggregation
# =============================================================================

def aggregate_values(values: Sequence[float], mode: str, mix_alpha: float = 0.7, trim_ratio: float = 0.1) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan

    mode = str(mode).lower()

    if mode == "mean":
        return float(np.mean(x))

    if mode == "median":
        return float(np.median(x))

    if mode == "p75":
        return float(np.percentile(x, 75))

    if mode == "p60":
        return float(np.percentile(x, 60))

    if mode == "mix_mean_p75":
        return float(mix_alpha * np.mean(x) + (1.0 - mix_alpha) * np.percentile(x, 75))

    if mode == "trimmed_mean":
        if len(x) < 3:
            return float(np.mean(x))
        xs = np.sort(x)
        k = int(math.floor(len(xs) * trim_ratio))
        if 2 * k >= len(xs):
            return float(np.mean(xs))
        return float(np.mean(xs[k: len(xs) - k]))

    if mode == "remove_top1_mean":
        if len(x) <= 1:
            return float(np.mean(x))
        xs = np.sort(x)
        return float(np.mean(xs[:-1]))

    raise ValueError(f"Unknown aggregation mode: {mode}")


def build_subject_scores(
    df: pd.DataFrame,
    split: str,
    pred_sbp_col: str,
    pred_dbp_col: str,
    true_sbp_col: str,
    true_dbp_col: str,
    pred_agg: str,
    true_agg: str,
    subject_col: str = "id_clean",
    sleep_col: str = "sleep",
    exclude_calib: bool = True,
    is_calib_col: str = "is_calib",
    pred_mix_alpha: float = 0.7,
    true_mix_alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Build one row per subject for a split.
    split:
        all:   all rows
        day:   sleep == 0
        night: sleep == 1
    """
    d = df.copy()

    if exclude_calib and is_calib_col in d.columns:
        d[is_calib_col] = parse_bool_series(d[is_calib_col])
        d = d[~d[is_calib_col]].copy()

    if split == "day":
        d = d[pd.to_numeric(d[sleep_col], errors="coerce").fillna(0).astype(int) == 0].copy()
    elif split == "night":
        d = d[pd.to_numeric(d[sleep_col], errors="coerce").fillna(0).astype(int) == 1].copy()
    elif split == "all":
        pass
    else:
        raise ValueError(f"Unknown split: {split}")

    rows = []
    for pid, g in d.groupby(subject_col, sort=False):
        true_sbp = aggregate_values(g[true_sbp_col], true_agg, mix_alpha=true_mix_alpha)
        true_dbp = aggregate_values(g[true_dbp_col], true_agg, mix_alpha=true_mix_alpha)
        pred_sbp = aggregate_values(g[pred_sbp_col], pred_agg, mix_alpha=pred_mix_alpha)
        pred_dbp = aggregate_values(g[pred_dbp_col], pred_agg, mix_alpha=pred_mix_alpha)

        if not all(np.isfinite(v) for v in [true_sbp, true_dbp, pred_sbp, pred_dbp]):
            continue

        rows.append({
            "id_clean": str(pid),
            "split": split,
            "n_rows": int(len(g)),
            "true_sbp_score": true_sbp,
            "true_dbp_score": true_dbp,
            "pred_sbp_score": pred_sbp,
            "pred_dbp_score": pred_dbp,
        })

    return pd.DataFrame(rows)


def compute_subject_metrics_for_threshold(
    scores: pd.DataFrame,
    true_thr: Threshold,
    pred_thr: Threshold,
) -> Dict[str, Any]:
    if len(scores) == 0:
        return confusion_metrics([], [])

    y_true = [
        is_positive(r.true_sbp_score, r.true_dbp_score, true_thr)
        for r in scores.itertuples(index=False)
    ]
    y_pred = [
        is_positive(r.pred_sbp_score, r.pred_dbp_score, pred_thr)
        for r in scores.itertuples(index=False)
    ]
    return confusion_metrics(y_true, y_pred)


def summarize_across_runs(per_run: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    metric_cols = [
        "n", "tp", "tn", "fp", "fn",
        "precision", "recall", "sensitivity", "specificity", "accuracy", "f1",
    ]
    rows = []
    for keys, g in per_run.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        for m in metric_cols:
            stats = mean_std_min_max(g[m].to_numpy())
            for sk, sv in stats.items():
                row[f"{m}_{sk}"] = sv
        rows.append(row)
    return pd.DataFrame(rows)


def make_threshold_grid(base_thr: Threshold, delta_min: float, delta_max: float, delta_step: float) -> List[Threshold]:
    deltas = np.arange(delta_min, delta_max + 1e-9, delta_step, dtype=float)
    return [
        Threshold(sbp=base_thr.sbp + float(ds), dbp=base_thr.dbp + float(dd))
        for ds in deltas
        for dd in deltas
    ]


def select_best_threshold(
    candidates: pd.DataFrame,
    min_sensitivity: float,
    min_specificity: float,
) -> pd.Series:
    """
    Selection rule:
    1. Prefer candidates satisfying both sensitivity and specificity targets.
    2. Among feasible candidates, maximize balanced accuracy, then specificity, then sensitivity.
    3. If none feasible, minimize violation, then maximize balanced accuracy.
    """
    c = candidates.copy()
    c["balanced_accuracy_mean"] = (c["sensitivity_mean"] + c["specificity_mean"]) / 2.0
    c["sens_violation"] = np.maximum(0.0, min_sensitivity - c["sensitivity_mean"])
    c["spec_violation"] = np.maximum(0.0, min_specificity - c["specificity_mean"])
    c["total_violation"] = c["sens_violation"] + c["spec_violation"]
    c["meets_target"] = (c["sensitivity_mean"] >= min_sensitivity) & (c["specificity_mean"] >= min_specificity)

    feasible = c[c["meets_target"]].copy()
    if len(feasible) > 0:
        feasible = feasible.sort_values(
            ["balanced_accuracy_mean", "specificity_mean", "sensitivity_mean"],
            ascending=[False, False, False],
        )
        return feasible.iloc[0]

    c = c.sort_values(
        ["total_violation", "balanced_accuracy_mean", "specificity_mean", "sensitivity_mean"],
        ascending=[True, False, False, False],
    )
    return c.iloc[0]


def run_threshold_search_for_gamma(
    csv_paths: List[Path],
    out_dir: Path,
    method_name: str,
    pred_sbp_col: str,
    pred_dbp_col: str,
    true_sbp_col: str,
    true_dbp_col: str,
    subject_col: str,
    sleep_col: str,
    is_calib_col: str,
    exclude_calib: bool,
    delta_min: float,
    delta_max: float,
    delta_step: float,
    min_sensitivity: float,
    min_specificity: float,
    all_pred_agg: str,
    day_pred_agg: str,
    night_pred_agg: str,
    true_agg: str,
    pred_mix_alpha: float,
) -> Dict[str, pd.DataFrame]:
    """
    For one gamma directory:
        1. Compute best thresholds for all/day/night.
        2. Evaluate per-run with these tuned thresholds.
        3. Compute combined OR metrics per run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    true_thresholds = {
        "all": Threshold(130.0, 80.0),
        "day": Threshold(135.0, 85.0),
        "night": Threshold(120.0, 70.0),
    }
    pred_aggs = {
        "all": all_pred_agg,
        "day": day_pred_agg,
        "night": night_pred_agg,
    }

    split_candidate_rows = []
    best_rows = []
    tuned_per_run_rows = []

    # Build threshold candidates per split.
    for split in ["all", "day", "night"]:
        base_thr = true_thresholds[split]
        grid = make_threshold_grid(base_thr, delta_min, delta_max, delta_step)

        candidate_per_thr_rows = []
        for pred_thr in grid:
            per_run_rows = []
            for run_id, p in enumerate(csv_paths):
                df = pd.read_csv(p)
                scores = build_subject_scores(
                    df,
                    split=split,
                    pred_sbp_col=pred_sbp_col,
                    pred_dbp_col=pred_dbp_col,
                    true_sbp_col=true_sbp_col,
                    true_dbp_col=true_dbp_col,
                    pred_agg=pred_aggs[split],
                    true_agg=true_agg,
                    subject_col=subject_col,
                    sleep_col=sleep_col,
                    exclude_calib=exclude_calib,
                    is_calib_col=is_calib_col,
                    pred_mix_alpha=pred_mix_alpha,
                )
                m = compute_subject_metrics_for_threshold(scores, true_thr=base_thr, pred_thr=pred_thr)
                m.update({
                    "run_id": run_id,
                    "csv_path": str(p),
                    "method": method_name,
                    "split": split,
                    "pred_sbp_thr": pred_thr.sbp,
                    "pred_dbp_thr": pred_thr.dbp,
                    "true_sbp_thr": base_thr.sbp,
                    "true_dbp_thr": base_thr.dbp,
                    "pred_agg": pred_aggs[split],
                    "true_agg": true_agg,
                })
                per_run_rows.append(m)

            per_run_df = pd.DataFrame(per_run_rows)
            summ = summarize_across_runs(
                per_run_df,
                group_cols=[
                    "method", "split", "pred_sbp_thr", "pred_dbp_thr",
                    "true_sbp_thr", "true_dbp_thr", "pred_agg", "true_agg",
                ],
            )
            candidate_per_thr_rows.append(summ.iloc[0])

        candidates = pd.DataFrame(candidate_per_thr_rows)
        candidates["balanced_accuracy_mean"] = (
            candidates["sensitivity_mean"] + candidates["specificity_mean"]
        ) / 2.0
        candidates["meets_target"] = (
            (candidates["sensitivity_mean"] >= min_sensitivity)
            & (candidates["specificity_mean"] >= min_specificity)
        )
        split_candidate_rows.append(candidates)

        best = select_best_threshold(
            candidates,
            min_sensitivity=min_sensitivity,
            min_specificity=min_specificity,
        )
        best_rows.append(best.to_dict())

    all_candidates = pd.concat(split_candidate_rows, axis=0, ignore_index=True)
    best_df = pd.DataFrame(best_rows)

    all_candidates.to_csv(out_dir / "threshold_grid_candidates.csv", index=False)
    best_df.to_csv(out_dir / "threshold_grid_best_by_split.csv", index=False)

    # Evaluate tuned thresholds per run.
    best_by_split = {
        str(r["split"]): Threshold(float(r["pred_sbp_thr"]), float(r["pred_dbp_thr"]))
        for _, r in best_df.iterrows()
    }

    for run_id, p in enumerate(csv_paths):
        df = pd.read_csv(p)
        subject_labels: Dict[str, Dict[str, bool]] = {}

        for split in ["all", "day", "night"]:
            true_thr = true_thresholds[split]
            pred_thr = best_by_split[split]
            scores = build_subject_scores(
                df,
                split=split,
                pred_sbp_col=pred_sbp_col,
                pred_dbp_col=pred_dbp_col,
                true_sbp_col=true_sbp_col,
                true_dbp_col=true_dbp_col,
                pred_agg=pred_aggs[split],
                true_agg=true_agg,
                subject_col=subject_col,
                sleep_col=sleep_col,
                exclude_calib=exclude_calib,
                is_calib_col=is_calib_col,
                pred_mix_alpha=pred_mix_alpha,
            )
            y_true = []
            y_pred = []

            for r in scores.itertuples(index=False):
                t_pos = is_positive(r.true_sbp_score, r.true_dbp_score, true_thr)
                p_pos = is_positive(r.pred_sbp_score, r.pred_dbp_score, pred_thr)
                y_true.append(t_pos)
                y_pred.append(p_pos)

                sid = str(r.id_clean)
                subject_labels.setdefault(sid, {})
                subject_labels[sid][f"{split}_true"] = t_pos
                subject_labels[sid][f"{split}_pred"] = p_pos

            m = confusion_metrics(y_true, y_pred)
            m.update({
                "run_id": run_id,
                "csv_path": str(p),
                "method": method_name,
                "split": split,
                "pred_sbp_thr": pred_thr.sbp,
                "pred_dbp_thr": pred_thr.dbp,
                "true_sbp_thr": true_thr.sbp,
                "true_dbp_thr": true_thr.dbp,
                "pred_agg": pred_aggs[split],
                "true_agg": true_agg,
            })
            tuned_per_run_rows.append(m)

        # Combined OR rule.
        y_true_comb = []
        y_pred_comb = []
        for sid, lab in subject_labels.items():
            # Only use subjects with at least one available split.
            true_pos = bool(
                lab.get("all_true", False)
                or lab.get("day_true", False)
                or lab.get("night_true", False)
            )
            pred_pos = bool(
                lab.get("all_pred", False)
                or lab.get("day_pred", False)
                or lab.get("night_pred", False)
            )
            y_true_comb.append(true_pos)
            y_pred_comb.append(pred_pos)

        m = confusion_metrics(y_true_comb, y_pred_comb)
        m.update({
            "run_id": run_id,
            "csv_path": str(p),
            "method": method_name,
            "split": "combined",
            "pred_sbp_thr": np.nan,
            "pred_dbp_thr": np.nan,
            "true_sbp_thr": np.nan,
            "true_dbp_thr": np.nan,
            "pred_agg": "OR(all,day,night)",
            "true_agg": "OR(all,day,night)",
        })
        tuned_per_run_rows.append(m)

    tuned_per_run = pd.DataFrame(tuned_per_run_rows)
    tuned_summary = summarize_across_runs(
        tuned_per_run,
        group_cols=["method", "split", "pred_sbp_thr", "pred_dbp_thr", "pred_agg", "true_agg"],
    )

    tuned_per_run.to_csv(out_dir / "tuned_classification_per_run.csv", index=False)
    tuned_summary.to_csv(out_dir / "tuned_classification_mean_std_across_runs.csv", index=False)

    return {
        "candidates": all_candidates,
        "best": best_df,
        "tuned_per_run": tuned_per_run,
        "tuned_summary": tuned_summary,
    }


# =============================================================================
# Main experiment runner
# =============================================================================

def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def run_experiment(
    csv_glob: str,
    out_dir: str | Path,
    gammas: Sequence[float],
    raw_sbp_col: Optional[str],
    raw_dbp_col: Optional[str],
    calib_true_sbp_col: str,
    calib_true_dbp_col: str,
    eval_true_sbp_col: str,
    eval_true_dbp_col: str,
    out_sbp_col: str,
    out_dbp_col: str,
    subject_col: str,
    sleep_col: str,
    is_calib_col: str,
    exclude_calib: bool,
    delta_min: float,
    delta_max: float,
    delta_step: float,
    min_sensitivity: float,
    min_specificity: float,
    all_pred_agg: str,
    day_pred_agg: str,
    night_pred_agg: str,
    true_agg: str,
    pred_mix_alpha: float,
) -> None:
    paths = [Path(p) for p in sorted(glob.glob(csv_glob))]
    if not paths:
        raise FileNotFoundError(f"No CSV matched: {csv_glob}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    gamma_summary_rows = []

    for gamma in gammas:
        gamma_dir = out_root / f"gamma_{gamma:.2f}"
        pred_dir = gamma_dir / "predictions"

        adjusted_paths = batch_apply_shrinkage(
            csv_paths=paths,
            gamma=float(gamma),
            out_dir=pred_dir,
            raw_sbp_col=raw_sbp_col,
            raw_dbp_col=raw_dbp_col,
            calib_true_sbp_col=calib_true_sbp_col,
            calib_true_dbp_col=calib_true_dbp_col,
            out_sbp_col=out_sbp_col,
            out_dbp_col=out_dbp_col,
            subject_col=subject_col,
            sleep_col=sleep_col,
            is_calib_col=is_calib_col,
        )

        result = run_threshold_search_for_gamma(
            csv_paths=adjusted_paths,
            out_dir=gamma_dir,
            method_name="bias_sleep_shrink",
            pred_sbp_col=out_sbp_col,
            pred_dbp_col=out_dbp_col,
            true_sbp_col=eval_true_sbp_col,
            true_dbp_col=eval_true_dbp_col,
            subject_col=subject_col,
            sleep_col=sleep_col,
            is_calib_col=is_calib_col,
            exclude_calib=exclude_calib,
            delta_min=delta_min,
            delta_max=delta_max,
            delta_step=delta_step,
            min_sensitivity=min_sensitivity,
            min_specificity=min_specificity,
            all_pred_agg=all_pred_agg,
            day_pred_agg=day_pred_agg,
            night_pred_agg=night_pred_agg,
            true_agg=true_agg,
            pred_mix_alpha=pred_mix_alpha,
        )

        tuned_summary = result["tuned_summary"].copy()
        tuned_summary.insert(0, "gamma_cross", float(gamma))
        gamma_summary_rows.append(tuned_summary)

        print(f"[OK] gamma={gamma:.2f} -> {gamma_dir}")

    exp_summary = pd.concat(gamma_summary_rows, axis=0, ignore_index=True)
    exp_summary.to_csv(out_root / "experiment_gamma_summary.csv", index=False)

    # Helpful compact view.
    compact_cols = [
        "gamma_cross",
        "split",
        "sensitivity_mean",
        "specificity_mean",
        "accuracy_mean",
        "f1_mean",
        "sensitivity_std_across_runs",
        "specificity_std_across_runs",
        "pred_sbp_thr",
        "pred_dbp_thr",
        "pred_agg",
    ]
    existing = [c for c in compact_cols if c in exp_summary.columns]
    compact = exp_summary[existing].copy()
    compact.to_csv(out_root / "experiment_gamma_summary_compact.csv", index=False)

    manifest = {
        "csv_glob": csv_glob,
        "n_input_csv": len(paths),
        "gammas": [float(g) for g in gammas],
        "raw_sbp_col": raw_sbp_col,
        "raw_dbp_col": raw_dbp_col,
        "calib_true_sbp_col": calib_true_sbp_col,
        "calib_true_dbp_col": calib_true_dbp_col,
        "eval_true_sbp_col": eval_true_sbp_col,
        "eval_true_dbp_col": eval_true_dbp_col,
        "out_sbp_col": out_sbp_col,
        "out_dbp_col": out_dbp_col,
        "subject_col": subject_col,
        "sleep_col": sleep_col,
        "is_calib_col": is_calib_col,
        "exclude_calib": exclude_calib,
        "delta_min": delta_min,
        "delta_max": delta_max,
        "delta_step": delta_step,
        "min_sensitivity": min_sensitivity,
        "min_specificity": min_specificity,
        "all_pred_agg": all_pred_agg,
        "day_pred_agg": day_pred_agg,
        "night_pred_agg": night_pred_agg,
        "true_agg": true_agg,
        "pred_mix_alpha": pred_mix_alpha,
    }
    with open(out_root / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== Experiment A finished ===")
    print(f"Summary: {out_root / 'experiment_gamma_summary.csv'}")
    print(f"Compact: {out_root / 'experiment_gamma_summary_compact.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment A: 1p0 sleep-aware cross-state shrinkage + ABPM classification grid search."
    )

    p.add_argument("--csv-glob", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="eval_1p0_sleep_shrinkage_exp")
    p.add_argument("--gammas", type=str, default="0,0.25,0.5,0.75,1.0")

    # Column config.
    p.add_argument("--raw-sbp-col", type=str, default=None)
    p.add_argument("--raw-dbp-col", type=str, default=None)
    p.add_argument("--calib-true-sbp-col", type=str, default="y_true_sbp")
    p.add_argument("--calib-true-dbp-col", type=str, default="y_true_dbp")
    p.add_argument("--eval-true-sbp-col", type=str, default="ABPM_SBP")
    p.add_argument("--eval-true-dbp-col", type=str, default="ABPM_DBP")
    p.add_argument("--out-sbp-col", type=str, default="y_pred_sbp_bias")
    p.add_argument("--out-dbp-col", type=str, default="y_pred_dbp_bias")

    p.add_argument("--subject-col", type=str, default="id_clean")
    p.add_argument("--sleep-col", type=str, default="sleep")
    p.add_argument("--is-calib-col", type=str, default="is_calib")
    p.add_argument("--include-calib", action="store_true")

    # Threshold grid.
    p.add_argument("--delta-min", type=float, default=-8.0)
    p.add_argument("--delta-max", type=float, default=8.0)
    p.add_argument("--delta-step", type=float, default=1.0)
    p.add_argument("--min-sensitivity", type=float, default=0.75)
    p.add_argument("--min-specificity", type=float, default=0.90)

    # Aggregation.
    p.add_argument("--all-pred-agg", type=str, default="mean",
                   choices=["mean", "median", "p75", "p60", "mix_mean_p75", "trimmed_mean", "remove_top1_mean"])
    p.add_argument("--day-pred-agg", type=str, default="mix_mean_p75",
                   choices=["mean", "median", "p75", "p60", "mix_mean_p75", "trimmed_mean", "remove_top1_mean"])
    p.add_argument("--night-pred-agg", type=str, default="median",
                   choices=["mean", "median", "p75", "p60", "mix_mean_p75", "trimmed_mean", "remove_top1_mean"])
    p.add_argument("--true-agg", type=str, default="mean",
                   choices=["mean", "median", "p75", "p60", "mix_mean_p75", "trimmed_mean", "remove_top1_mean"])
    p.add_argument("--pred-mix-alpha", type=float, default=0.7,
                   help="For mix_mean_p75: score = alpha*mean + (1-alpha)*p75.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    run_experiment(
        csv_glob=args.csv_glob,
        out_dir=args.out_dir,
        gammas=parse_float_list(args.gammas),
        raw_sbp_col=args.raw_sbp_col,
        raw_dbp_col=args.raw_dbp_col,
        calib_true_sbp_col=args.calib_true_sbp_col,
        calib_true_dbp_col=args.calib_true_dbp_col,
        eval_true_sbp_col=args.eval_true_sbp_col,
        eval_true_dbp_col=args.eval_true_dbp_col,
        out_sbp_col=args.out_sbp_col,
        out_dbp_col=args.out_dbp_col,
        subject_col=args.subject_col,
        sleep_col=args.sleep_col,
        is_calib_col=args.is_calib_col,
        exclude_calib=not args.include_calib,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        delta_step=args.delta_step,
        min_sensitivity=args.min_sensitivity,
        min_specificity=args.min_specificity,
        all_pred_agg=args.all_pred_agg,
        day_pred_agg=args.day_pred_agg,
        night_pred_agg=args.night_pred_agg,
        true_agg=args.true_agg,
        pred_mix_alpha=args.pred_mix_alpha,
    )


if __name__ == "__main__":
    main()
