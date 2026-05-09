#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train/Val/Test subject-level hypertension Logistic Regression from BP summaries.

Purpose
-------
This is the clean version of:
    raw all/day/night summary + logistic regression classifier

Protocol
--------
1) Train subjects:
      fit LogisticRegression
2) Val subjects:
      select probability threshold for each target
      objective: satisfy sensitivity/specificity targets if possible
3) Test subjects:
      locked evaluation using the trained model and val-selected threshold

This avoids training on the test set.

Inputs
------
Option A: separate prediction CSVs for train/val/test
    --train-csv-glob "pred_train*.csv"
    --val-csv-glob   "pred_val*.csv"
    --test-csv-glob  "pred_test*.csv"

Option B: one or more all-subject CSVs plus subject list files
    --all-csv-glob "predictions_*.csv"
    --train-subjects train_ids.txt
    --val-subjects   val_ids.txt
    --test-subjects  test_ids.txt

Subject list file can be:
    - plain txt: one id per line
    - csv: contains id_clean column, or first column is used

Default:
    method=raw -> y_pred_sbp_raw / y_pred_dbp_raw
    gold label -> ABPM_SBP / ABPM_DBP
    targets -> all,day,night,combined

Outputs
-------
out_dir/
    features_train.csv
    features_val.csv
    features_test.csv
    metrics_train_val_test.csv
    coefficients_<target>.csv
    logistic_model_<target>.joblib
    manifest.json

Example
-------
python bp_htn_logistic_train_val_test.py \
  --train-csv-glob "pred_train*.csv" \
  --val-csv-glob "pred_val*.csv" \
  --test-csv-glob "pred_test*.csv" \
  --method raw \
  --true-sbp-col ABPM_SBP \
  --true-dbp-col ABPM_DBP \
  --out-dir eval_lr_train_val_test
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


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else float(a) / float(b)


def parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


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


def add_auc(y_true: Sequence[int], prob: Sequence[float], out: Dict[str, Any]) -> None:
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
        sens_v = max(0.0, min_sensitivity - m["sensitivity"])
        spec_v = max(0.0, min_specificity - m["specificity"])
        rows.append({
            "threshold": float(thr),
            "balanced_accuracy": float(bal),
            "sens_violation": float(sens_v),
            "spec_violation": float(spec_v),
            "total_violation": float(sens_v + spec_v),
            "meets_target": bool(m["sensitivity"] >= min_sensitivity and m["specificity"] >= min_specificity),
            **m,
        })

    df = pd.DataFrame(rows)
    feasible = df[df["meets_target"]].copy()
    if len(feasible):
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


def parse_list_arg(pattern: Optional[str], explicit: Sequence[str]) -> List[Path]:
    paths: List[str] = []
    if pattern:
        for ptn in pattern.split(","):
            ptn = ptn.strip()
            if ptn:
                paths.extend(sorted(glob.glob(ptn)))
    paths.extend(explicit or [])

    unique = []
    seen = set()
    for p in paths:
        pp = Path(p)
        key = str(pp)
        if key not in seen:
            unique.append(pp)
            seen.add(key)
    return unique


def read_many_csv(paths: List[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No input CSV files.")
    frames = []
    for i, p in enumerate(paths):
        if not p.exists():
            raise FileNotFoundError(p)
        d = pd.read_csv(p)
        d["_source_csv"] = str(p)
        d["_source_file_index"] = i
        frames.append(d)
    return pd.concat(frames, axis=0, ignore_index=True)


def read_subject_list(path: str, subject_col: str = "id_clean") -> set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in [".csv", ".tsv"]:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
        if subject_col in df.columns:
            ids = df[subject_col].astype(str).tolist()
        else:
            ids = df.iloc[:, 0].astype(str).tolist()
    else:
        ids = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return set(map(str, ids))


def infer_pred_cols(args: argparse.Namespace, df: pd.DataFrame) -> Tuple[str, str, str]:
    if args.pred_sbp_col and args.pred_dbp_col:
        return args.pred_sbp_col, args.pred_dbp_col, args.method

    if args.method not in METHOD_COLS:
        raise ValueError(f"Unknown method={args.method}.")
    sbp, dbp = METHOD_COLS[args.method]
    if sbp not in df.columns or dbp not in df.columns:
        raise ValueError(f"Missing prediction columns for method={args.method}: {sbp}, {dbp}")
    return sbp, dbp, args.method


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
    raise ValueError(f"Unknown agg mode: {mode}")


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


def build_subject_row(
    g: pd.DataFrame,
    split_name: str,
    subject_col: str,
    sleep_col: str,
    pred_sbp_col: str,
    pred_dbp_col: str,
    true_sbp_col: str,
    true_dbp_col: str,
    agg_modes: Sequence[str],
) -> Dict[str, Any]:
    sid = str(g[subject_col].iloc[0])
    row: Dict[str, Any] = {"id_clean": sid, "source_split": split_name}
    labels = {}

    for split in ["all", "day", "night"]:
        d = split_df(g, split, sleep_col)
        row[f"n_{split}"] = int(len(d))
        row[f"has_{split}"] = int(len(d) > 0)

        if len(d) == 0:
            for bp in ["sbp", "dbp"]:
                for mode in agg_modes:
                    row[f"pred_{split}_{bp}_{mode}"] = np.nan
                    row[f"true_{split}_{bp}_{mode}"] = np.nan
            row[f"pred_{split}_sbp_margin_mean"] = np.nan
            row[f"pred_{split}_dbp_margin_mean"] = np.nan
            row[f"pred_{split}_max_margin_mean"] = np.nan
            labels[f"label_{split}"] = np.nan
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

    available = [row["label_all"], row["label_day"], row["label_night"]]
    available = [x for x in available if pd.notna(x)]
    row["label_combined"] = int(any(bool(x) for x in available)) if available else np.nan

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


def build_features(
    df: pd.DataFrame,
    split_name: str,
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
    d = df.copy()

    if (not include_calib) and is_calib_col in d.columns:
        d = d[~parse_bool_series(d[is_calib_col])].copy()

    required = [subject_col, sleep_col, pred_sbp_col, pred_dbp_col, true_sbp_col, true_dbp_col]
    miss = [c for c in required if c not in d.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    for c in [sleep_col, pred_sbp_col, pred_dbp_col, true_sbp_col, true_dbp_col]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=required).copy()

    rows = []
    for _, g in d.groupby(subject_col, sort=False):
        rows.append(
            build_subject_row(
                g, split_name, subject_col, sleep_col, pred_sbp_col, pred_dbp_col,
                true_sbp_col, true_dbp_col, agg_modes
            )
        )
    return pd.DataFrame(rows)


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {"id_clean", "source_split"}
    cols = []
    for c in df.columns:
        if c in exclude or c.startswith("label_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def make_model(C: float, class_weight: str, max_iter: int) -> Pipeline:
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


def evaluate_probs(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    out = confusion_metrics(y_true, pred)
    add_auc(y_true, prob, out)
    return out


def train_val_test_one_target(
    train_feat: pd.DataFrame,
    val_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    C: float,
    class_weight: str,
    max_iter: int,
    min_sensitivity: float,
    min_specificity: float,
    threshold_step: float,
) -> Tuple[Dict[str, Any], Pipeline, pd.DataFrame, pd.DataFrame]:
    label_col = f"label_{target}"

    tr = train_feat[pd.notna(train_feat[label_col])].copy()
    va = val_feat[pd.notna(val_feat[label_col])].copy()
    te = test_feat[pd.notna(test_feat[label_col])].copy()

    if tr[label_col].nunique() < 2:
        raise ValueError(f"Train split target={target} has only one class.")
    if len(va) == 0:
        raise ValueError(f"Val split target={target} has no samples.")
    if len(te) == 0:
        raise ValueError(f"Test split target={target} has no samples.")

    X_train = tr[feature_cols].to_numpy(dtype=float)
    y_train = tr[label_col].astype(int).to_numpy()
    X_val = va[feature_cols].to_numpy(dtype=float)
    y_val = va[label_col].astype(int).to_numpy()
    X_test = te[feature_cols].to_numpy(dtype=float)
    y_test = te[label_col].astype(int).to_numpy()

    model = make_model(C=C, class_weight=class_weight, max_iter=max_iter)
    model.fit(X_train, y_train)

    prob_train = model.predict_proba(X_train)[:, 1]
    prob_val = model.predict_proba(X_val)[:, 1]
    prob_test = model.predict_proba(X_test)[:, 1]

    threshold_info = threshold_search(
        y_true=y_val,
        prob=prob_val,
        min_sensitivity=min_sensitivity,
        min_specificity=min_specificity,
        step=threshold_step,
    )
    threshold = float(threshold_info["threshold"])

    rows = []
    for split_name, y, prob in [
        ("train", y_train, prob_train),
        ("val", y_val, prob_val),
        ("test", y_test, prob_test),
    ]:
        m = evaluate_probs(y, prob, threshold)
        rows.append({
            "target": target,
            "eval_split": split_name,
            "threshold_selected_on_val": threshold,
            "val_threshold_sensitivity": threshold_info["sensitivity"],
            "val_threshold_specificity": threshold_info["specificity"],
            "C": float(C),
            "class_weight": class_weight,
            **m,
        })

    metrics_df = pd.DataFrame(rows)

    pred_rows = []
    for split_name, feat_df, y, prob in [
        ("train", tr, y_train, prob_train),
        ("val", va, y_val, prob_val),
        ("test", te, y_test, prob_test),
    ]:
        pred = (prob >= threshold).astype(int)
        for sid, yy, pp, pr in zip(feat_df["id_clean"], y, prob, pred):
            pred_rows.append({
                "target": target,
                "eval_split": split_name,
                "id_clean": sid,
                "y_true": int(yy),
                "prob": float(pp),
                "pred": int(pr),
                "threshold": threshold,
            })
    pred_df = pd.DataFrame(pred_rows)

    return threshold_info, model, metrics_df, pred_df


def parse_agg_modes(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_targets(s: str) -> List[str]:
    targets = [x.strip() for x in s.split(",") if x.strip()]
    for t in targets:
        if t not in ["all", "day", "night", "combined"]:
            raise ValueError(f"Unknown target={t}")
    return targets


def load_split_dfs(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.all_csv_glob:
        all_paths = parse_list_arg(args.all_csv_glob, [])
        all_df = read_many_csv(all_paths)

        if not (args.train_subjects and args.val_subjects and args.test_subjects):
            raise ValueError("--all-csv-glob requires --train-subjects, --val-subjects, --test-subjects")

        train_ids = read_subject_list(args.train_subjects, args.subject_col)
        val_ids = read_subject_list(args.val_subjects, args.subject_col)
        test_ids = read_subject_list(args.test_subjects, args.subject_col)

        train_df = all_df[all_df[args.subject_col].astype(str).isin(train_ids)].copy()
        val_df = all_df[all_df[args.subject_col].astype(str).isin(val_ids)].copy()
        test_df = all_df[all_df[args.subject_col].astype(str).isin(test_ids)].copy()
        return train_df, val_df, test_df

    train_paths = parse_list_arg(args.train_csv_glob, args.train_csv)
    val_paths = parse_list_arg(args.val_csv_glob, args.val_csv)
    test_paths = parse_list_arg(args.test_csv_glob, args.test_csv)

    return read_many_csv(train_paths), read_many_csv(val_paths), read_many_csv(test_paths)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LR on train subjects, tune threshold on val subjects, evaluate locked test.")

    # Separate CSV mode.
    p.add_argument("--train-csv-glob", type=str, default=None)
    p.add_argument("--val-csv-glob", type=str, default=None)
    p.add_argument("--test-csv-glob", type=str, default=None)
    p.add_argument("--train-csv", nargs="*", default=[])
    p.add_argument("--val-csv", nargs="*", default=[])
    p.add_argument("--test-csv", nargs="*", default=[])

    # One CSV + subject lists mode.
    p.add_argument("--all-csv-glob", type=str, default=None)
    p.add_argument("--train-subjects", type=str, default=None)
    p.add_argument("--val-subjects", type=str, default=None)
    p.add_argument("--test-subjects", type=str, default=None)

    p.add_argument("--out-dir", type=str, default="bp_htn_lr_train_val_test")

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

    train_df, val_df, test_df = load_split_dfs(args)

    sample_df = train_df.head(5)
    pred_sbp_col, pred_dbp_col, method_name = infer_pred_cols(args, sample_df)

    agg_modes = parse_agg_modes(args.agg_modes)
    targets = parse_targets(args.targets)

    train_feat = build_features(
        train_df, "train", args.subject_col, args.sleep_col, args.is_calib_col, args.include_calib,
        pred_sbp_col, pred_dbp_col, args.true_sbp_col, args.true_dbp_col, agg_modes
    )
    val_feat = build_features(
        val_df, "val", args.subject_col, args.sleep_col, args.is_calib_col, args.include_calib,
        pred_sbp_col, pred_dbp_col, args.true_sbp_col, args.true_dbp_col, agg_modes
    )
    test_feat = build_features(
        test_df, "test", args.subject_col, args.sleep_col, args.is_calib_col, args.include_calib,
        pred_sbp_col, pred_dbp_col, args.true_sbp_col, args.true_dbp_col, agg_modes
    )

    train_feat.to_csv(out_dir / "features_train.csv", index=False)
    val_feat.to_csv(out_dir / "features_val.csv", index=False)
    test_feat.to_csv(out_dir / "features_test.csv", index=False)

    all_feat_for_cols = pd.concat([train_feat, val_feat, test_feat], axis=0, ignore_index=True)
    feature_cols = get_feature_cols(all_feat_for_cols)
    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    all_metrics = []
    all_preds = []
    all_coef = []

    for target in targets:
        threshold_info, model, metrics_df, pred_df = train_val_test_one_target(
            train_feat=train_feat,
            val_feat=val_feat,
            test_feat=test_feat,
            target=target,
            feature_cols=feature_cols,
            C=args.C,
            class_weight=args.class_weight,
            max_iter=args.max_iter,
            min_sensitivity=args.min_sensitivity,
            min_specificity=args.min_specificity,
            threshold_step=args.prob_threshold_step,
        )
        metrics_df["method"] = method_name
        pred_df["method"] = method_name

        all_metrics.append(metrics_df)
        all_preds.append(pred_df)

        lr = model.named_steps["lr"]
        coef_df = pd.DataFrame({
            "target": target,
            "feature": feature_cols,
            "coef": lr.coef_.reshape(-1),
        })
        coef_df["abs_coef"] = coef_df["coef"].abs()
        coef_df = coef_df.sort_values("abs_coef", ascending=False)
        all_coef.append(coef_df)

        joblib.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "target": target,
                "threshold_selected_on_val": float(metrics_df["threshold_selected_on_val"].iloc[0]),
                "method": method_name,
                "pred_sbp_col": pred_sbp_col,
                "pred_dbp_col": pred_dbp_col,
                "true_sbp_col": args.true_sbp_col,
                "true_dbp_col": args.true_dbp_col,
            },
            out_dir / f"logistic_model_{target}.joblib",
        )

    metrics_all = pd.concat(all_metrics, axis=0, ignore_index=True)
    preds_all = pd.concat(all_preds, axis=0, ignore_index=True)
    coef_all = pd.concat(all_coef, axis=0, ignore_index=True)

    metrics_all.to_csv(out_dir / "metrics_train_val_test.csv", index=False)
    preds_all.to_csv(out_dir / "predictions_train_val_test.csv", index=False)
    coef_all.to_csv(out_dir / "coefficients_all_targets.csv", index=False)

    manifest = {
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
        "feature_count": len(feature_cols),
        "n_train_subjects": int(train_feat["id_clean"].nunique()),
        "n_val_subjects": int(val_feat["id_clean"].nunique()),
        "n_test_subjects": int(test_feat["id_clean"].nunique()),
        "C": float(args.C),
        "class_weight": args.class_weight,
        "min_sensitivity": float(args.min_sensitivity),
        "min_specificity": float(args.min_specificity),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    compact_cols = [
        "method", "target", "eval_split",
        "threshold_selected_on_val",
        "sensitivity", "specificity", "accuracy", "f1", "roc_auc", "average_precision",
        "tp", "tn", "fp", "fn",
    ]
    compact_cols = [c for c in compact_cols if c in metrics_all.columns]

    print("\n=== Train/Val/Test Logistic Summary ===")
    print(metrics_all[compact_cols].to_string(index=False))
    print("\nSaved:")
    print(f"  {out_dir / 'metrics_train_val_test.csv'}")
    print(f"  {out_dir / 'predictions_train_val_test.csv'}")
    print(f"  {out_dir / 'coefficients_all_targets.csv'}")


if __name__ == "__main__":
    main()
