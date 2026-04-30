#!/usr/bin/env python3
"""Compute hypertension classification metrics from one or many calibrated BP CSVs.

This is an extended version of a single-CSV hypertension classification script.

Main use case:
    You ran 1+1 calibration sensitivity 8 times and got:
        predictions_1p1_run00_seed2026.csv
        ...
        predictions_1p1_run07_seed2033.csv

    This script computes classification metrics for every CSV, then reports
    mean/std/min/max across the 8 runs, e.g. average sensitivity and specificity.

Default labels:
    true positive if SBP >= threshold_sbp OR DBP >= threshold_dbp

Default splits/thresholds:
    all:   130/80
    day:   135/85, rows with sleep == 0
    night: 120/70, rows with sleep == 1

Default evaluation mode:
    subject: average true/pred BP within each subject first, then classify one
             subject-level point per split.

Supported prediction methods / columns:
    raw:   y_pred_sbp_raw,  y_pred_dbp_raw
    bias:  y_pred_sbp_bias, y_pred_dbp_bias
    aff:   y_pred_sbp_aff,  y_pred_dbp_aff
    bank:  y_pred_sbp_bank, y_pred_dbp_bank
    cal:   y_pred_sbp_cal,  y_pred_dbp_cal

Examples:
    # 8-run sensitivity summary for bank method
    python bp_htn_classification_multi_run.py \
        --csv-glob 'eval_1p1_sensitivity/predictions_1p1_run*.csv' \
        --methods bank \
        --average subject \
        --out-dir eval_1p1_sensitivity/classification_bank

    # Compare bias/affine/bank across the same 8 CSVs
    python bp_htn_classification_multi_run.py \
        --csv-glob 'eval_1p1_sensitivity/predictions_1p1_run*.csv' \
        --methods bias,aff,bank \
        --average subject \
        --out-dir eval_1p1_sensitivity/classification_all_methods

    # Single CSV, backward-compatible style using y_pred_sbp_cal/y_pred_dbp_cal
    python bp_htn_classification_multi_run.py predictions.csv --methods cal
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
from glob import glob
import math
from pathlib import Path
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

TRUEY = {"1", "true", "yes", "y", "t"}


@dataclass
class Threshold:
    sbp: float
    dbp: float


@dataclass
class ValidRow:
    subject_id: str
    true_sbp: float
    true_dbp: float
    pred_sbp: float
    pred_dbp: float


@dataclass
class LabeledItem:
    true_positive: bool
    pred_positive: bool


@dataclass
class Metrics:
    run_id: int
    csv_path: str
    method: str
    split: str
    mode: str
    threshold_sbp: float
    threshold_dbp: float
    n_rows: int
    n_subjects: int
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    sensitivity: float
    specificity: float
    accuracy: float
    f1: float


METHOD_TO_COLS: Dict[str, Tuple[str, str]] = {
    "raw": ("y_pred_sbp_raw", "y_pred_dbp_raw"),
    "bias": ("y_pred_sbp_bias", "y_pred_dbp_bias"),
    "aff": ("y_pred_sbp_aff", "y_pred_dbp_aff"),
    "bank": ("y_pred_sbp_bank", "y_pred_dbp_bank"),
    "cal": ("y_pred_sbp_cal", "y_pred_dbp_cal"),
}


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------

def parse_bool(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in TRUEY


def parse_float(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key, "")
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def parse_int(row: Dict[str, str], key: str) -> Optional[int]:
    value = row.get(key, "")
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def is_positive(sbp: float, dbp: float, threshold: Threshold) -> bool:
    return (sbp >= threshold.sbp) or (dbp >= threshold.dbp)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def available_methods(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    header = set(rows[0].keys())
    out = []
    for method, (sbp_col, dbp_col) in METHOD_TO_COLS.items():
        if sbp_col in header and dbp_col in header:
            out.append(method)
    return out


def resolve_methods(method_arg: str, rows: List[Dict[str, str]]) -> List[str]:
    if method_arg.strip().lower() == "auto":
        methods = available_methods(rows)
        # Prefer calibrated methods before raw when all exist.
        preferred_order = ["bank", "aff", "bias", "cal", "raw"]
        methods = [m for m in preferred_order if m in methods]
        if not methods:
            raise ValueError("No supported prediction columns found in CSV.")
        return methods

    methods = [m.strip() for m in method_arg.split(",") if m.strip()]
    for m in methods:
        if m not in METHOD_TO_COLS:
            raise ValueError(f"Unknown method '{m}'. Available: {list(METHOD_TO_COLS)} or auto")
    return methods


# -----------------------------------------------------------------------------
# Core metrics
# -----------------------------------------------------------------------------

def collect_valid_rows(
    rows: Iterable[Dict[str, str]],
    row_filter: Callable[[Dict[str, str]], bool],
    subject_col: str,
    pred_sbp_col: str,
    pred_dbp_col: str,
    true_sbp_col: str = "y_true_sbp",
    true_dbp_col: str = "y_true_dbp",
) -> List[ValidRow]:
    valid_rows: List[ValidRow] = []

    for row in rows:
        if not row_filter(row):
            continue

        true_sbp = parse_float(row, true_sbp_col)
        true_dbp = parse_float(row, true_dbp_col)
        pred_sbp = parse_float(row, pred_sbp_col)
        pred_dbp = parse_float(row, pred_dbp_col)

        if None in (true_sbp, true_dbp, pred_sbp, pred_dbp):
            continue

        subject_id = str(row.get(subject_col, "")).strip()
        if not subject_id:
            subject_id = "unknown"

        valid_rows.append(
            ValidRow(
                subject_id=subject_id,
                true_sbp=float(true_sbp),
                true_dbp=float(true_dbp),
                pred_sbp=float(pred_sbp),
                pred_dbp=float(pred_dbp),
            )
        )

    return valid_rows


def accumulate_counts(items: Iterable[LabeledItem]) -> Dict[str, int]:
    tp = tn = fp = fn = 0

    for item in items:
        if item.true_positive and item.pred_positive:
            tp += 1
        elif (not item.true_positive) and (not item.pred_positive):
            tn += 1
        elif (not item.true_positive) and item.pred_positive:
            fp += 1
        else:
            fn += 1

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_counts(
    *,
    run_id: int,
    csv_path: str,
    method: str,
    split: str,
    mode: str,
    threshold: Threshold,
    n_rows: int,
    n_subjects: int,
    counts: Dict[str, int],
) -> Metrics:
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    sensitivity = recall
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return Metrics(
        run_id=run_id,
        csv_path=csv_path,
        method=method,
        split=split,
        mode=mode,
        threshold_sbp=float(threshold.sbp),
        threshold_dbp=float(threshold.dbp),
        n_rows=int(n_rows),
        n_subjects=int(n_subjects),
        tp=int(tp),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        precision=float(precision),
        recall=float(recall),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        accuracy=float(accuracy),
        f1=float(f1),
    )


def compute_subject_metrics(
    rows: Iterable[Dict[str, str]],
    threshold: Threshold,
    row_filter: Callable[[Dict[str, str]], bool],
    subject_col: str,
    pred_sbp_col: str,
    pred_dbp_col: str,
    *,
    run_id: int,
    csv_path: str,
    method: str,
    split: str,
) -> Metrics:
    valid_rows = collect_valid_rows(rows, row_filter, subject_col, pred_sbp_col, pred_dbp_col)
    grouped: DefaultDict[str, List[ValidRow]] = defaultdict(list)
    for row in valid_rows:
        grouped[row.subject_id].append(row)

    labeled_subjects: List[LabeledItem] = []
    for subject_rows in grouped.values():
        true_sbp_mean = mean([row.true_sbp for row in subject_rows])
        true_dbp_mean = mean([row.true_dbp for row in subject_rows])
        pred_sbp_mean = mean([row.pred_sbp for row in subject_rows])
        pred_dbp_mean = mean([row.pred_dbp for row in subject_rows])

        labeled_subjects.append(
            LabeledItem(
                true_positive=is_positive(true_sbp_mean, true_dbp_mean, threshold),
                pred_positive=is_positive(pred_sbp_mean, pred_dbp_mean, threshold),
            )
        )

    counts = accumulate_counts(labeled_subjects)
    return metrics_from_counts(
        run_id=run_id,
        csv_path=csv_path,
        method=method,
        split=split,
        mode="subject",
        threshold=threshold,
        n_rows=len(valid_rows),
        n_subjects=len(labeled_subjects),
        counts=counts,
    )


def compute_micro_metrics(
    rows: Iterable[Dict[str, str]],
    threshold: Threshold,
    row_filter: Callable[[Dict[str, str]], bool],
    subject_col: str,
    pred_sbp_col: str,
    pred_dbp_col: str,
    *,
    run_id: int,
    csv_path: str,
    method: str,
    split: str,
) -> Metrics:
    valid_rows = collect_valid_rows(rows, row_filter, subject_col, pred_sbp_col, pred_dbp_col)
    labeled_rows = [
        LabeledItem(
            true_positive=is_positive(row.true_sbp, row.true_dbp, threshold),
            pred_positive=is_positive(row.pred_sbp, row.pred_dbp, threshold),
        )
        for row in valid_rows
    ]
    counts = accumulate_counts(labeled_rows)
    return metrics_from_counts(
        run_id=run_id,
        csv_path=csv_path,
        method=method,
        split=split,
        mode="micro",
        threshold=threshold,
        n_rows=len(valid_rows),
        n_subjects=len({row.subject_id for row in valid_rows}),
        counts=counts,
    )


def compute_macro_metrics(
    rows: Iterable[Dict[str, str]],
    threshold: Threshold,
    row_filter: Callable[[Dict[str, str]], bool],
    subject_col: str,
    pred_sbp_col: str,
    pred_dbp_col: str,
    *,
    run_id: int,
    csv_path: str,
    method: str,
    split: str,
) -> Metrics:
    valid_rows = collect_valid_rows(rows, row_filter, subject_col, pred_sbp_col, pred_dbp_col)
    grouped: DefaultDict[str, List[ValidRow]] = defaultdict(list)
    for row in valid_rows:
        grouped[row.subject_id].append(row)

    # Total confusion is still row-level pooled; scalar metrics are averaged per subject.
    all_labeled_rows = [
        LabeledItem(
            true_positive=is_positive(row.true_sbp, row.true_dbp, threshold),
            pred_positive=is_positive(row.pred_sbp, row.pred_dbp, threshold),
        )
        for row in valid_rows
    ]
    total_counts = accumulate_counts(all_labeled_rows)

    if not grouped:
        return metrics_from_counts(
            run_id=run_id,
            csv_path=csv_path,
            method=method,
            split=split,
            mode="macro",
            threshold=threshold,
            n_rows=0,
            n_subjects=0,
            counts={"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        )

    subject_metrics: List[Dict[str, float]] = []
    for subject_rows in grouped.values():
        labeled_rows = [
            LabeledItem(
                true_positive=is_positive(row.true_sbp, row.true_dbp, threshold),
                pred_positive=is_positive(row.pred_sbp, row.pred_dbp, threshold),
            )
            for row in subject_rows
        ]
        c = accumulate_counts(labeled_rows)
        tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        accuracy = safe_div(tp + tn, tp + tn + fp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        subject_metrics.append(
            {
                "precision": precision,
                "recall": recall,
                "sensitivity": recall,
                "specificity": specificity,
                "accuracy": accuracy,
                "f1": f1,
            }
        )

    # Create metrics from total counts, then overwrite scalar metrics by macro averages.
    m = metrics_from_counts(
        run_id=run_id,
        csv_path=csv_path,
        method=method,
        split=split,
        mode="macro",
        threshold=threshold,
        n_rows=len(valid_rows),
        n_subjects=len(grouped),
        counts=total_counts,
    )
    for key in ["precision", "recall", "sensitivity", "specificity", "accuracy", "f1"]:
        setattr(m, key, float(sum(x[key] for x in subject_metrics) / len(subject_metrics)))
    return m


def compute_metrics_for_split(
    rows: List[Dict[str, str]],
    threshold: Threshold,
    row_filter: Callable[[Dict[str, str]], bool],
    subject_col: str,
    pred_sbp_col: str,
    pred_dbp_col: str,
    average: str,
    *,
    run_id: int,
    csv_path: str,
    method: str,
    split: str,
) -> Metrics:
    fn_map = {
        "subject": compute_subject_metrics,
        "micro": compute_micro_metrics,
        "macro": compute_macro_metrics,
    }
    return fn_map[average](
        rows,
        threshold,
        row_filter,
        subject_col,
        pred_sbp_col,
        pred_dbp_col,
        run_id=run_id,
        csv_path=csv_path,
        method=method,
        split=split,
    )


def compute_metrics_for_csv(
    csv_path: str | Path,
    run_id: int,
    methods: List[str],
    average: str,
    include_calib: bool,
    subject_col: str,
    all_threshold: Threshold,
    day_threshold: Threshold,
    night_threshold: Threshold,
) -> List[Metrics]:
    rows = read_csv_rows(csv_path)
    if not include_calib:
        rows = [row for row in rows if not parse_bool(row.get("is_calib"))]

    out: List[Metrics] = []
    split_specs = [
        ("all", all_threshold, lambda row: True),
        ("day", day_threshold, lambda row: parse_int(row, "sleep") == 0),
        ("night", night_threshold, lambda row: parse_int(row, "sleep") == 1),
    ]

    for method in methods:
        pred_sbp_col, pred_dbp_col = METHOD_TO_COLS[method]
        for split, threshold, row_filter in split_specs:
            out.append(
                compute_metrics_for_split(
                    rows=rows,
                    threshold=threshold,
                    row_filter=row_filter,
                    subject_col=subject_col,
                    pred_sbp_col=pred_sbp_col,
                    pred_dbp_col=pred_dbp_col,
                    average=average,
                    run_id=run_id,
                    csv_path=str(csv_path),
                    method=method,
                    split=split,
                )
            )

    return out


# -----------------------------------------------------------------------------
# Multi-run summary
# -----------------------------------------------------------------------------

def metrics_to_dataframe(metrics: List[Metrics]) -> pd.DataFrame:
    return pd.DataFrame([asdict(m) for m in metrics])


def summarize_across_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/min/max across CSV runs for each method/split/mode."""
    metric_cols = [
        "n_rows", "n_subjects", "tp", "tn", "fp", "fn",
        "precision", "recall", "sensitivity", "specificity", "accuracy", "f1",
    ]

    rows: List[Dict[str, object]] = []
    group_cols = ["method", "split", "mode", "threshold_sbp", "threshold_dbp"]
    for keys, g in df.groupby(group_cols, dropna=False, sort=False):
        base = dict(zip(group_cols, keys))
        base["n_runs"] = int(g["run_id"].nunique())
        for col in metric_cols:
            values = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(values) == 0:
                base[f"{col}_mean"] = math.nan
                base[f"{col}_std_across_runs"] = math.nan
                base[f"{col}_min"] = math.nan
                base[f"{col}_max"] = math.nan
            else:
                base[f"{col}_mean"] = float(values.mean())
                base[f"{col}_std_across_runs"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
                base[f"{col}_min"] = float(values.min())
                base[f"{col}_max"] = float(values.max())
        rows.append(base)

    return pd.DataFrame(rows)


def pooled_confusion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pool TP/TN/FP/FN across runs, then recompute metrics.

    This is different from averaging sensitivity/specificity across runs. It gives a
    pooled classification performance over all run-level evaluations.
    """
    rows: List[Dict[str, object]] = []
    group_cols = ["method", "split", "mode", "threshold_sbp", "threshold_dbp"]
    for keys, g in df.groupby(group_cols, dropna=False, sort=False):
        tp = int(pd.to_numeric(g["tp"], errors="coerce").fillna(0).sum())
        tn = int(pd.to_numeric(g["tn"], errors="coerce").fillna(0).sum())
        fp = int(pd.to_numeric(g["fp"], errors="coerce").fillna(0).sum())
        fn = int(pd.to_numeric(g["fn"], errors="coerce").fillna(0).sum())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        accuracy = safe_div(tp + tn, tp + tn + fp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_runs": int(g["run_id"].nunique()),
                "tp_sum": tp,
                "tn_sum": tn,
                "fp_sum": fp,
                "fn_sum": fn,
                "precision_pooled": precision,
                "recall_pooled": recall,
                "sensitivity_pooled": recall,
                "specificity_pooled": specificity,
                "accuracy_pooled": accuracy,
                "f1_pooled": f1,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def resolve_csv_paths(positional: List[str], csv_glob: Optional[str]) -> List[Path]:
    paths: List[Path] = []
    for p in positional:
        paths.append(Path(p))
    if csv_glob:
        paths.extend(Path(x) for x in sorted(glob(csv_glob)))
    # de-duplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in paths:
        sp = str(p)
        if sp not in seen:
            unique.append(p)
            seen.add(sp)
    if not unique:
        raise ValueError("No CSV files provided. Use positional paths or --csv-glob.")
    missing = [str(p) for p in unique if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CSV files: {missing[:5]}{'...' if len(missing)>5 else ''}")
    return unique


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute hypertension classification metrics from one or many BP prediction CSV files."
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        help="Input CSV file(s). For 8-run sensitivity, pass all 8 prediction CSVs or use --csv-glob.",
    )
    parser.add_argument(
        "--csv-glob",
        default=None,
        help="Glob pattern for multiple CSV files, e.g. 'eval_1p1/predictions_1p1_run*.csv'.",
    )
    parser.add_argument(
        "--methods",
        default="auto",
        help="Prediction methods to evaluate: auto, raw, bias, aff, bank, cal, or comma-separated list. Default: auto.",
    )
    parser.add_argument(
        "--include-calib",
        action="store_true",
        help="Include rows where is_calib is true. Default: exclude them.",
    )
    parser.add_argument(
        "--average",
        choices=["subject", "micro", "macro"],
        default="subject",
        help=(
            "Evaluation mode. 'subject' averages BP within each subject first and "
            "classifies one subject-level point per split. 'micro' pools all rows. "
            "'macro' computes row-level metrics per subject then averages them. Default: subject."
        ),
    )
    parser.add_argument("--subject-col", default="id_clean", help="Subject ID column. Default: id_clean.")

    parser.add_argument("--all-sbp", "--whole-sbp", dest="all_sbp", type=float, default=130.0)
    parser.add_argument("--all-dbp", "--whole-dbp", dest="all_dbp", type=float, default=80.0)
    parser.add_argument("--day-sbp", type=float, default=135.0)
    parser.add_argument("--day-dbp", type=float, default=85.0)
    parser.add_argument("--night-sbp", type=float, default=120.0)
    parser.add_argument("--night-dbp", type=float, default=70.0)

    parser.add_argument(
        "--out-dir",
        default="classification_metrics_multi_run",
        help="Output directory for per-run and summary CSV files.",
    )
    parser.add_argument(
        "--prefix",
        default="htn_classification",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = resolve_csv_paths(args.csv_paths, args.csv_glob)

    # Use first CSV to resolve auto methods.
    first_rows = read_csv_rows(csv_paths[0])
    methods = resolve_methods(args.methods, first_rows)

    thresholds = {
        "all": Threshold(args.all_sbp, args.all_dbp),
        "day": Threshold(args.day_sbp, args.day_dbp),
        "night": Threshold(args.night_sbp, args.night_dbp),
    }

    all_metrics: List[Metrics] = []
    for run_id, path in enumerate(csv_paths):
        # If method='auto', ensure each CSV has the same selected columns. If not, skip missing method with error.
        rows = read_csv_rows(path)
        header = set(rows[0].keys()) if rows else set()
        missing_methods = [
            m for m in methods
            if METHOD_TO_COLS[m][0] not in header or METHOD_TO_COLS[m][1] not in header
        ]
        if missing_methods:
            raise ValueError(f"{path} missing columns for methods {missing_methods}")

        all_metrics.extend(
            compute_metrics_for_csv(
                csv_path=path,
                run_id=run_id,
                methods=methods,
                average=args.average,
                include_calib=args.include_calib,
                subject_col=args.subject_col,
                all_threshold=thresholds["all"],
                day_threshold=thresholds["day"],
                night_threshold=thresholds["night"],
            )
        )

    per_run_df = metrics_to_dataframe(all_metrics)
    summary_df = summarize_across_runs(per_run_df)
    pooled_df = pooled_confusion_summary(per_run_df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_path = out_dir / f"{args.prefix}_per_run.csv"
    summary_path = out_dir / f"{args.prefix}_mean_std_across_runs.csv"
    pooled_path = out_dir / f"{args.prefix}_pooled_confusion.csv"

    per_run_df.to_csv(per_run_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pooled_df.to_csv(pooled_path, index=False)

    print("\n=== Hypertension classification multi-run evaluation ===")
    print(f"CSV files: {len(csv_paths)}")
    print(f"Methods: {methods}")
    print(f"Average mode: {args.average}")
    print(f"Calibration rows included: {'yes' if args.include_calib else 'no'}")
    print(f"Per-run metrics:     {per_run_path}")
    print(f"Across-run summary: {summary_path}")
    print(f"Pooled confusion:   {pooled_path}")

    # Compact console view for sensitivity/specificity.
    display_cols = [
        "method", "split", "mode", "n_runs",
        "sensitivity_mean", "sensitivity_std_across_runs", "sensitivity_min", "sensitivity_max",
        "specificity_mean", "specificity_std_across_runs", "specificity_min", "specificity_max",
        "accuracy_mean", "f1_mean",
    ]
    cols = [c for c in display_cols if c in summary_df.columns]
    print("\n--- Mean classification metrics across runs ---")
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
