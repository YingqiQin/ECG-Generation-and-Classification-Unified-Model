#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subject-level hypertension classifier from raw All/Day/Night BP summaries.

This script builds subject-level features from prediction CSVs:
  - all/day/night raw BP summaries
  - day-night deltas
  - threshold margins
  - event counts / missing indicators

Then it trains Logistic Regression classifiers for:
  - all
  - day
  - night
  - combined = all OR day OR night

Gold label is generated from ABPM_SBP / ABPM_DBP by default.

Leakage control:
  If multiple CSVs are provided, the same subject may appear multiple times.
  GroupKFold(id_clean) is used so the same subject never appears in both
  train and test folds.

Example:
python bp_htn_logistic_from_summary.py \
  --csv-glob "eval_25hz_raw/predictions_*.csv" \
  --method raw \
  --true-sbp-col ABPM_SBP \
  --true-dbp-col ABPM_DBP \
  --out-dir eval_25hz_logistic_raw \
  --targets all,day,night,combined
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    raise ImportError("Install dependencies with: pip install scikit-learn joblib") from e


METHOD_COLS = {
    "raw": ("y_pred_sbp_raw", "y_pred_dbp_raw"),
    "bias": ("y_pred_sbp_bias", "y_pred_dbp_bias"),
    "aff": ("y_pred_sbp_aff", "y_pred_dbp_aff"),
    "bank": ("y_pred_sbp_bank", "y_pred_dbp_bank"),
    "cal": ("y_pred_sbp_cal", "y_pred_dbp_cal"),
}

TRUE_THRESHOLDS = {
    "all": (130.0, 80.0),
    "day": (135.0, 85.0),
    "night": (120.0, 70.0),
}


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num) / float(den)


def confusion_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum((~y_true) & (~y_pred)))
    fp = int(np.sum((~y_true) & y_pred))
    fn = int(np.sum(y_true & (~y_pred)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
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
        "sensitivity": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
    }


def add_auc_metrics(y_true: np.ndarray, prob: np.ndarray, out: Dict[str, Any]) -> None:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    if len(np.unique(y_true)) < 2:
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan
        return
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["average_precision"] = float(average_precision_score(y_true, prob))
    except Exception:
        out["average_precision"] = np.nan


def threshold_search(
    y_true: np.ndarray,
    prob: np.ndarray,
    min_sensitivity: float = 0.75,
    min_specificity: float = 0.90,
    step: float = 0.01,
) -> Dict[str, Any]:
    rows = []
    for thr in np.arange(0.01, 0.99 + 1e-9, step):
        pred = (prob >= thr).astype(int)
        m = confusion_metrics(y_true, pred)
        bal = 0.5 * (m["sensitivity"] + m["specificity"])
        sens_violation = max(0.0, min_sensitivity - m["sensitivity"])
        spec_violation = max(0.0, min_specificity - m["specificity"])
        rows.append({
            "threshold": float(thr),
            "balanced_accuracy": float(bal),
            "sens_violation": float(sens_violation),
            "spec_violation": float(spec_violation),
            "total_violation": float(sens_violation + spec_violation),
            "meets_target": bool(m["sensitivity"] >= min_sensitivity and m["specificity"] >= min_specificity),
            **m,
        })

    df = pd.DataFrame(rows)
    feasible = df[df["meets_target"]].copy()
    if len(feasible) > 0:
        feasible = feasible.sort_values(
            ["balanced_accuracy", "specificity", "sensitivity"],
            ascending=[False, False, False],
        )
        return feasible.iloc[0].to_dict()

    df = df.sort_values(
        ["total_violation", "balanced_accuracy", "specificity", "sensitivity"],
        ascending=[True, False, False, False],
    )
    return df.iloc[0].to_dict()


def parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def infer_prediction_cols(args: argparse.Namespace, sample_df: pd.DataFrame) -> Tuple[str, str, str]:
    if args.pred_sbp_col and args.pred_dbp_col:
        return args.pred_sbp_col, args.pred_dbp_col, args.method

    if args.method not in METHOD_COLS:
        raise ValueError(f"Unknown method={args.method}. Use one of {list(METHOD_COLS)} or pass custom columns.")

    sbp_col, dbp_col = METHOD_COLS[args.method]
    if sbp_col not in sample_df.columns or dbp_col not in sample_df.columns:
        raise ValueError(
            f"Prediction columns for method={args.method} not found: {sbp_col}, {dbp_col}. "
            "Use --pred-sbp-col and --pred-dbp-col."
        )
    return sbp_col, dbp_col, args.method


def agg_numeric(x: Sequence[float], mode: str) -> float:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan

    if mode == "mean":
        return float(np.mean(arr))
    if mode == "median":
        return float(np.median(arr))
    if mode == "std":
        return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    if mode == "min":
        return float(np.min(arr))
    if mode == "max":
        return float(np.max(arr))
    if mode == "p25":
        return float(np.percentile(arr, 25))
    if mode == "p60":
        return float(np.percentile(arr, 60))
    if mode == "p75":
        return float(np.percentile(arr, 75))
    if mode == "p90":
        return float(np.percentile(arr, 90))
    raise ValueError(f"Unknown aggregation mode={mode}")


def split_df(df: pd.DataFrame, split: str, sleep_col: str) -> pd.DataFrame:
    if split == "all":
        return df
    sleep = pd.to_numeric(df[sleep_col], errors="coerce").fillna(0).astype(int)
    if split == "day":
        return df[sleep == 0]
    if split == "night":
        return df[sleep == 1]
    raise ValueError(split)


def positive_label(sbp: float, dbp: float, split: str) -> int:
    thr_sbp, thr_dbp = TRUE_THRESHOLDS[split]
    if not np.isfinite(sbp) or not np.isfinite(dbp):
        return 0
    return int((sbp >= thr_sbp) or (dbp >= thr_dbp))


def build_one_subject_feature_row(
    g_subj: pd.DataFrame,
    run_id: int,
    csv_path: str,
    subject_col: str,
    sleep_col: str,
    pred_sbp_col: str,
    pred_dbp_col: str,
    true_sbp_col: str,
    true_dbp_col: str,
    agg_modes: Sequence[str],
) -> Dict[str, Any]:
    sid = str(g_subj[subject_col].iloc[0])
    row: Dict[str, Any] = {
        "id_clean": sid,
        "run_id": int(run_id),
        "csv_path": str(csv_path),
    }

    labels = {}

    for split in ["all", "day", "night"]:
        d = split_df(g_subj, split, sleep_col)
        row[f"n_{split}"] = int(len(d))
        row[f"has_{split}"] = int(len(d) > 0)

        if len(d) == 0:
            for bp in ["sbp", "dbp"]:
                for mode in agg_modes:
                    row[f"pred_{split}_{bp}_{mode}"] = np.nan
                    row[f"true_{split}_{bp}_{mode}"] = np.nan
            labels[f"label_{split}"] = np.nan
            row[f"pred_{split}_sbp_margin_mean"] = np.nan
            row[f"pred_{split}_dbp_margin_mean"] = np.nan
            row[f"pred_{split}_max_margin_mean"] = np.nan
            continue

        true_sbp_mean = agg_numeric(d[true_sbp_col], "mean")
        true_dbp_mean = agg_numeric(d[true_dbp_col], "mean")
        labels[f"label_{split}"] = positive_label(true_sbp_mean, true_dbp_mean, split)

        for mode in agg_modes:
            row[f"pred_{split}_sbp_{mode}"] = agg_numeric(d[pred_sbp_col], mode)
            row[f"pred_{split}_dbp_{mode}"] = agg_numeric(d[pred_dbp_col], mode)
            row[f"true_{split}_sbp_{mode}"] = agg_numeric(d[true_sbp_col], mode)
            row[f"true_{split}_dbp_{mode}"] = agg_numeric(d[true_dbp_col], mode)

        thr_sbp, thr_dbp = TRUE_THRESHOLDS[split]
        row[f"pred_{split}_sbp_margin_mean"] = row[f"pred_{split}_sbp_mean"] - thr_sbp
        row[f"pred_{split}_dbp_margin_mean"] = row[f"pred_{split}_dbp_mean"] - thr_dbp
        row[f"pred_{split}_max_margin_mean"] = max(
            row[f"pred_{split}_sbp_margin_mean"],
            row[f"pred_{split}_dbp_margin_mean"],
        )

    for split in ["all", "day", "night"]:
        row[f"label_{split}"] = labels.get(f"label_{split}", np.nan)

    available_labels = [row["label_all"], row["label_day"], row["label_night"]]
    available_labels = [x for x in available_labels if pd.notna(x)]
    row["label_combined"] = int(any(bool(x) for x in available_labels)) if available_labels else np.nan

    def _get(name: str) -> float:
        return float(row[name]) if name in row and pd.notna(row[name]) else np.nan

    for bp in ["sbp", "dbp"]:
        row[f"pred_day_minus_night_{bp}_mean"] = _get(f"pred_day_{bp}_mean") - _get(f"pred_night_{bp}_mean")
        row[f"pred_day_minus_all_{bp}_mean"] = _get(f"pred_day_{bp}_mean") - _get(f"pred_all_{bp}_mean")
        row[f"pred_night_minus_all_{bp}_mean"] = _get(f"pred_night_{bp}_mean") - _get(f"pred_all_{bp}_mean")
        row[f"pred_day_p75_minus_mean_{bp}"] = _get(f"pred_day_{bp}_p75") - _get(f"pred_day_{bp}_mean")
        row[f"pred_night_p75_minus_mean_{bp}"] = _get(f"pred_night_{bp}_p75") - _get(f"pred_night_{bp}_mean")

    n_all = max(float(row["n_all"]), 1.0)
    row["frac_day"] = float(row["n_day"]) / n_all
    row["frac_night"] = float(row["n_night"]) / n_all

    return row


def build_feature_table(
    csv_paths: List[Path],
    subject_col: str,
    sleep_col: str,
    is_calib_col: str,
    include_calib: bool,
    pred_sbp_col: str,
    pred_dbp_col: str,
    true_sbp_col: str,
    true_dbp_col: str,
    agg_modes: Sequence[str],
) -> pd.DataFrame:
    rows = []

    for run_id, path in enumerate(csv_paths):
        df = pd.read_csv(path)

        required = [subject_col, sleep_col, pred_sbp_col, pred_dbp_col, true_sbp_col, true_dbp_col]
        miss = [c for c in required if c not in df.columns]
        if miss:
            raise ValueError(f"{path} missing required columns: {miss}")

        if (not include_calib) and is_calib_col in df.columns:
            df = df[~parse_bool_series(df[is_calib_col])].copy()

        for c in [sleep_col, pred_sbp_col, pred_dbp_col, true_sbp_col, true_dbp_col]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=[subject_col, sleep_col, pred_sbp_col, pred_dbp_col, true_sbp_col, true_dbp_col]).copy()

        for _, g in df.groupby(subject_col, sort=False):
            rows.append(
                build_one_subject_feature_row(
                    g_subj=g,
                    run_id=run_id,
                    csv_path=str(path),
                    subject_col=subject_col,
                    sleep_col=sleep_col,
                    pred_sbp_col=pred_sbp_col,
                    pred_dbp_col=pred_dbp_col,
                    true_sbp_col=true_sbp_col,
                    true_dbp_col=true_dbp_col,
                    agg_modes=agg_modes,
                )
            )

    return pd.DataFrame(rows)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"id_clean", "csv_path", "run_id"}
    cols = []
    for c in df.columns:
        if c in exclude or c.startswith("label_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def make_model(C: float = 1.0, class_weight: str = "balanced", max_iter: int = 1000) -> Pipeline:
    cw = None if class_weight == "none" else class_weight
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=float(C),
            penalty="l2",
            solver="liblinear",
            class_weight=cw,
            max_iter=int(max_iter),
            random_state=42,
        )),
    ])


def choose_n_splits(y: np.ndarray, groups: np.ndarray, desired: int) -> int:
    n_groups = len(np.unique(groups))
    _, counts = np.unique(y, return_counts=True)
    min_class_count = int(np.min(counts)) if len(counts) > 1 else 0
    return max(2, min(int(desired), n_groups, min_class_count))


def evaluate_target_cv(
    feat: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    n_splits: int = 5,
    C: float = 1.0,
    class_weight: str = "balanced",
    min_sensitivity: float = 0.75,
    min_specificity: float = 0.90,
    prob_threshold_step: float = 0.01,
    max_iter: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Pipeline]:
    label_col = f"label_{target}"
    d = feat[pd.notna(feat[label_col])].copy()
    d[label_col] = d[label_col].astype(int)

    if d[label_col].nunique() < 2:
        raise ValueError(f"Target={target} has only one class.")

    X = d[feature_cols].to_numpy(dtype=float)
    y = d[label_col].to_numpy(dtype=int)
    groups = d["id_clean"].astype(str).to_numpy()

    k = choose_n_splits(y, groups, n_splits)
    cv = GroupKFold(n_splits=k)

    fold_rows = []
    oof_rows = []

    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups)):
        model = make_model(C=C, class_weight=class_weight, max_iter=max_iter)
        model.fit(X[tr], y[tr])

        prob_tr = model.predict_proba(X[tr])[:, 1]
        thr_info = threshold_search(
            y_true=y[tr],
            prob=prob_tr,
            min_sensitivity=min_sensitivity,
            min_specificity=min_specificity,
            step=prob_threshold_step,
        )
        thr = float(thr_info["threshold"])

        prob_te = model.predict_proba(X[te])[:, 1]
        pred_te = (prob_te >= thr).astype(int)

        m = confusion_metrics(y[te], pred_te)
        add_auc_metrics(y[te], prob_te, m)

        fold_rows.append({
            "target": target,
            "fold": fold,
            "threshold": thr,
            "train_threshold_sensitivity": thr_info["sensitivity"],
            "train_threshold_specificity": thr_info["specificity"],
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            **m,
        })

        for idx_local, prob, pred in zip(te, prob_te, pred_te):
            rr = d.iloc[idx_local]
            oof_rows.append({
                "target": target,
                "fold": fold,
                "id_clean": rr["id_clean"],
                "run_id": int(rr["run_id"]),
                "csv_path": rr["csv_path"],
                "y_true": int(rr[label_col]),
                "prob": float(prob),
                "pred": int(pred),
                "threshold": thr,
            })

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)

    metric_cols = [
        "n", "tp", "tn", "fp", "fn",
        "precision", "recall", "sensitivity", "specificity", "accuracy", "f1",
        "roc_auc", "average_precision",
    ]
    summary = {"target": target, "n_splits": int(k), "C": float(C), "class_weight": class_weight}
    for c in metric_cols:
        x = pd.to_numeric(fold_df[c], errors="coerce").dropna().to_numpy(dtype=float)
        summary[f"{c}_mean"] = float(np.mean(x)) if len(x) else np.nan
        summary[f"{c}_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
        summary[f"{c}_min"] = float(np.min(x)) if len(x) else np.nan
        summary[f"{c}_max"] = float(np.max(x)) if len(x) else np.nan

    pooled = confusion_metrics(oof_df["y_true"], oof_df["pred"])
    add_auc_metrics(oof_df["y_true"].to_numpy(), oof_df["prob"].to_numpy(), pooled)
    for k2, v in pooled.items():
        summary[f"oof_{k2}"] = v

    final_model = make_model(C=C, class_weight=class_weight, max_iter=max_iter)
    final_model.fit(X, y)
    prob_all = final_model.predict_proba(X)[:, 1]
    final_thr_info = threshold_search(
        y_true=y,
        prob=prob_all,
        min_sensitivity=min_sensitivity,
        min_specificity=min_specificity,
        step=prob_threshold_step,
    )
    summary["final_threshold_on_all_data"] = float(final_thr_info["threshold"])
    summary["final_train_sensitivity"] = float(final_thr_info["sensitivity"])
    summary["final_train_specificity"] = float(final_thr_info["specificity"])

    return fold_df, pd.DataFrame([summary]), oof_df, final_model


def extract_coefficients(model: Pipeline, feature_cols: List[str], target: str) -> pd.DataFrame:
    lr = model.named_steps["lr"]
    coef = lr.coef_.reshape(-1)
    return pd.DataFrame({
        "target": target,
        "feature": feature_cols,
        "coef": coef,
        "abs_coef": np.abs(coef),
    }).sort_values("abs_coef", ascending=False)


def parse_targets(s: str) -> List[str]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    for v in vals:
        if v not in ["all", "day", "night", "combined"]:
            raise ValueError(f"Unknown target={v}.")
    return vals


def parse_agg_modes(s: str) -> List[str]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    allowed = {"mean", "median", "std", "min", "max", "p25", "p60", "p75", "p90"}
    bad = [v for v in vals if v not in allowed]
    if bad:
        raise ValueError(f"Unknown agg modes: {bad}")
    return vals


def parse_csv_paths(csv_glob: Optional[str], csv_paths: Sequence[str]) -> List[Path]:
    paths = []
    if csv_glob:
        paths.extend(sorted(glob.glob(csv_glob)))
    paths.extend(csv_paths)

    unique = []
    seen = set()
    for p in paths:
        pp = Path(p)
        if str(pp) not in seen:
            unique.append(pp)
            seen.add(str(pp))

    if not unique:
        raise FileNotFoundError("No input CSV files provided or matched.")

    missing = [str(p) for p in unique if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing[:5]}")

    return unique


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train subject-level logistic regression classifier from All/Day/Night BP summaries.")

    p.add_argument("--csv-glob", type=str, default=None)
    p.add_argument("csv_paths", nargs="*")
    p.add_argument("--out-dir", type=str, default="bp_htn_logistic_summary")

    p.add_argument("--method", type=str, default="raw", choices=list(METHOD_COLS.keys()))
    p.add_argument("--pred-sbp-col", type=str, default=None)
    p.add_argument("--pred-dbp-col", type=str, default=None)

    p.add_argument("--true-sbp-col", type=str, default="ABPM_SBP")
    p.add_argument("--true-dbp-col", type=str, default="ABPM_DBP")
    p.add_argument("--subject-col", type=str, default="id_clean")
    p.add_argument("--sleep-col", type=str, default="sleep")
    p.add_argument("--is-calib-col", type=str, default="is_calib")
    p.add_argument("--include-calib", action="store_true")

    p.add_argument("--targets", type=str, default="all,day,night,combined")
    p.add_argument("--agg-modes", type=str, default="mean,median,std,min,max,p60,p75,p90")

    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--class-weight", type=str, default="balanced", choices=["balanced", "none"])
    p.add_argument("--max-iter", type=int, default=1000)

    p.add_argument("--min-sensitivity", type=float, default=0.75)
    p.add_argument("--min-specificity", type=float, default=0.90)
    p.add_argument("--prob-threshold-step", type=float, default=0.01)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = parse_csv_paths(args.csv_glob, args.csv_paths)
    sample_df = pd.read_csv(csv_paths[0], nrows=5)
    pred_sbp_col, pred_dbp_col, method_name = infer_prediction_cols(args, sample_df)

    agg_modes = parse_agg_modes(args.agg_modes)
    targets = parse_targets(args.targets)

    feat = build_feature_table(
        csv_paths=csv_paths,
        subject_col=args.subject_col,
        sleep_col=args.sleep_col,
        is_calib_col=args.is_calib_col,
        include_calib=args.include_calib,
        pred_sbp_col=pred_sbp_col,
        pred_dbp_col=pred_dbp_col,
        true_sbp_col=args.true_sbp_col,
        true_dbp_col=args.true_dbp_col,
        agg_modes=agg_modes,
    )
    feat_path = out_dir / "subject_summary_features.csv"
    feat.to_csv(feat_path, index=False)

    feature_cols = get_feature_columns(feat)
    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    all_fold = []
    all_summary = []
    all_oof = []
    all_coef = []

    for target in targets:
        fold_df, summary_df, oof_df, final_model = evaluate_target_cv(
            feat=feat,
            target=target,
            feature_cols=feature_cols,
            n_splits=args.n_splits,
            C=args.C,
            class_weight=args.class_weight,
            min_sensitivity=args.min_sensitivity,
            min_specificity=args.min_specificity,
            prob_threshold_step=args.prob_threshold_step,
            max_iter=args.max_iter,
        )

        fold_df["method"] = method_name
        summary_df["method"] = method_name
        oof_df["method"] = method_name

        all_fold.append(fold_df)
        all_summary.append(summary_df)
        all_oof.append(oof_df)
        all_coef.append(extract_coefficients(final_model, feature_cols, target))

        joblib.dump(
            {
                "model": final_model,
                "feature_cols": feature_cols,
                "target": target,
                "method": method_name,
                "pred_sbp_col": pred_sbp_col,
                "pred_dbp_col": pred_dbp_col,
                "true_sbp_col": args.true_sbp_col,
                "true_dbp_col": args.true_dbp_col,
            },
            out_dir / f"logistic_model_{target}.joblib",
        )

    fold_all = pd.concat(all_fold, axis=0, ignore_index=True)
    summary_all = pd.concat(all_summary, axis=0, ignore_index=True)
    oof_all = pd.concat(all_oof, axis=0, ignore_index=True)
    coef_all = pd.concat(all_coef, axis=0, ignore_index=True)

    fold_all.to_csv(out_dir / "logistic_cv_per_fold.csv", index=False)
    summary_all.to_csv(out_dir / "logistic_cv_summary.csv", index=False)
    oof_all.to_csv(out_dir / "logistic_oof_predictions.csv", index=False)
    coef_all.to_csv(out_dir / "logistic_final_coefficients.csv", index=False)

    manifest = {
        "csv_paths": [str(p) for p in csv_paths],
        "n_csv": len(csv_paths),
        "out_dir": str(out_dir),
        "method": method_name,
        "pred_sbp_col": pred_sbp_col,
        "pred_dbp_col": pred_dbp_col,
        "true_sbp_col": args.true_sbp_col,
        "true_dbp_col": args.true_dbp_col,
        "subject_col": args.subject_col,
        "sleep_col": args.sleep_col,
        "include_calib": bool(args.include_calib),
        "targets": targets,
        "agg_modes": agg_modes,
        "n_feature_cols": len(feature_cols),
        "n_rows_feature_table": int(len(feat)),
        "n_subjects": int(feat["id_clean"].nunique()),
        "n_splits": int(args.n_splits),
        "C": float(args.C),
        "class_weight": args.class_weight,
        "min_sensitivity": float(args.min_sensitivity),
        "min_specificity": float(args.min_specificity),
    }
    with open(out_dir / "logistic_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    compact_cols = [
        "method", "target",
        "sensitivity_mean", "specificity_mean", "accuracy_mean", "f1_mean",
        "roc_auc_mean", "average_precision_mean",
        "oof_sensitivity", "oof_specificity", "oof_accuracy", "oof_f1", "oof_roc_auc",
        "final_threshold_on_all_data",
    ]
    compact_cols = [c for c in compact_cols if c in summary_all.columns]

    print("\n=== Logistic CV summary ===")
    print(summary_all[compact_cols].to_string(index=False))
    print("\nSaved:")
    print(f"  {feat_path}")
    print(f"  {out_dir / 'logistic_cv_summary.csv'}")
    print(f"  {out_dir / 'logistic_oof_predictions.csv'}")
    print(f"  {out_dir / 'logistic_final_coefficients.csv'}")


if __name__ == "__main__":
    main()
