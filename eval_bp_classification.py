#!/usr/bin/env python3
"""Compute hypertension classification metrics from calibrated BP CSV output.

Default logic:
- compare `y_true_sbp/y_true_dbp` vs `y_pred_sbp_cal/y_pred_dbp_cal`
- exclude calibration rows (`is_calib == True`) by default
- use subject-level macro averaging by default
- report metrics for:
  - whole: threshold 130/80
  - day:   threshold 135/85, rows with sleep == 0
  - night: threshold 120/70, rows with sleep == 1

Positive class:
- hypertension positive if `sbp >= threshold_sbp` OR `dbp >= threshold_dbp`
- negative otherwise
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional


TRUEY = {"1", "true", "yes", "y", "t"}


@dataclass
class Threshold:
    sbp: float
    dbp: float


@dataclass
class LabeledRow:
    subject_id: str
    true_positive: bool
    pred_positive: bool


@dataclass
class Metrics:
    average: str
    n_rows: int
    n_subjects: int
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    accuracy: float
    f1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute BP classification metrics from a calibrated CSV."
    )
    parser.add_argument("csv_path", help="Path to the input CSV file.")
    parser.add_argument(
        "--include-calib",
        action="store_true",
        help="Include rows where is_calib is true. Default: exclude them.",
    )
    parser.add_argument(
        "--average",
        choices=["macro", "micro"],
        default="macro",
        help=(
            "Averaging mode. 'macro' averages metrics across subjects equally; "
            "'micro' pools all rows together. Default: macro."
        ),
    )
    parser.add_argument(
        "--subject-col",
        default="id_clean",
        help="Subject ID column used for macro averaging. Default: id_clean.",
    )
    parser.add_argument(
        "--whole-sbp", type=float, default=130.0, help="Whole-day SBP threshold."
    )
    parser.add_argument(
        "--whole-dbp", type=float, default=80.0, help="Whole-day DBP threshold."
    )
    parser.add_argument(
        "--day-sbp", type=float, default=135.0, help="Daytime SBP threshold."
    )
    parser.add_argument(
        "--day-dbp", type=float, default=85.0, help="Daytime DBP threshold."
    )
    parser.add_argument(
        "--night-sbp", type=float, default=120.0, help="Nighttime SBP threshold."
    )
    parser.add_argument(
        "--night-dbp", type=float, default=70.0, help="Nighttime DBP threshold."
    )
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in TRUEY


def parse_float(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key, "")
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def parse_int(row: Dict[str, str], key: str) -> Optional[int]:
    value = row.get(key, "")
    if value is None:
        return None
    value = value.strip()
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


def collect_labeled_rows(
    rows: Iterable[Dict[str, str]],
    threshold: Threshold,
    row_filter: Callable[[Dict[str, str]], bool],
    subject_col: str,
) -> List[LabeledRow]:
    labeled_rows: List[LabeledRow] = []

    for row in rows:
        if not row_filter(row):
            continue

        true_sbp = parse_float(row, "y_true_sbp")
        true_dbp = parse_float(row, "y_true_dbp")
        pred_sbp = parse_float(row, "y_pred_sbp_cal")
        pred_dbp = parse_float(row, "y_pred_dbp_cal")

        if None in (true_sbp, true_dbp, pred_sbp, pred_dbp):
            continue

        labeled_rows.append(
            LabeledRow(
                subject_id=str(row.get(subject_col, "")).strip(),
                true_positive=is_positive(true_sbp, true_dbp, threshold),
                pred_positive=is_positive(pred_sbp, pred_dbp, threshold),
            )
        )

    return labeled_rows


def accumulate_counts(labeled_rows: Iterable[LabeledRow]) -> Dict[str, int]:
    tp = tn = fp = fn = 0

    for row in labeled_rows:
        if row.true_positive and row.pred_positive:
            tp += 1
        elif (not row.true_positive) and (not row.pred_positive):
            tn += 1
        elif (not row.true_positive) and row.pred_positive:
            fp += 1
        else:
            fn += 1

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def build_metrics(
    average: str,
    n_rows: int,
    n_subjects: int,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    precision: float,
    recall: float,
    accuracy: float,
    f1: float,
) -> Metrics:
    return Metrics(
        average=average,
        n_rows=n_rows,
        n_subjects=n_subjects,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        f1=f1,
    )


def compute_micro_metrics(
    rows: Iterable[Dict[str, str]],
    threshold: Threshold,
    row_filter: Callable[[Dict[str, str]], bool],
    subject_col: str,
) -> Metrics:
    labeled_rows = collect_labeled_rows(rows, threshold, row_filter, subject_col)
    counts = accumulate_counts(labeled_rows)
    n_rows = len(labeled_rows)
    n_subjects = len({row.subject_id for row in labeled_rows})

    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    accuracy = safe_div(tp + tn, n_rows)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return build_metrics(
        average="micro",
        n_rows=n_rows,
        n_subjects=n_subjects,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        f1=f1,
    )


def compute_macro_metrics(
    rows: Iterable[Dict[str, str]],
    threshold: Threshold,
    row_filter: Callable[[Dict[str, str]], bool],
    subject_col: str,
) -> Metrics:
    labeled_rows = collect_labeled_rows(rows, threshold, row_filter, subject_col)
    grouped: DefaultDict[str, List[LabeledRow]] = defaultdict(list)
    for row in labeled_rows:
        grouped[row.subject_id].append(row)

    if not grouped:
        return build_metrics(
            average="macro",
            n_rows=0,
            n_subjects=0,
            tp=0,
            tn=0,
            fp=0,
            fn=0,
            precision=0.0,
            recall=0.0,
            accuracy=0.0,
            f1=0.0,
        )

    total_counts = accumulate_counts(labeled_rows)
    subject_precisions: List[float] = []
    subject_recalls: List[float] = []
    subject_accuracies: List[float] = []
    subject_f1s: List[float] = []

    for subject_rows in grouped.values():
        counts = accumulate_counts(subject_rows)
        tp = counts["tp"]
        tn = counts["tn"]
        fp = counts["fp"]
        fn = counts["fn"]
        subject_n_rows = len(subject_rows)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        accuracy = safe_div(tp + tn, subject_n_rows)
        f1 = safe_div(2 * precision * recall, precision + recall)

        subject_precisions.append(precision)
        subject_recalls.append(recall)
        subject_accuracies.append(accuracy)
        subject_f1s.append(f1)

    return build_metrics(
        average="macro",
        n_rows=len(labeled_rows),
        n_subjects=len(grouped),
        tp=total_counts["tp"],
        tn=total_counts["tn"],
        fp=total_counts["fp"],
        fn=total_counts["fn"],
        precision=sum(subject_precisions) / len(subject_precisions),
        recall=sum(subject_recalls) / len(subject_recalls),
        accuracy=sum(subject_accuracies) / len(subject_accuracies),
        f1=sum(subject_f1s) / len(subject_f1s),
    )


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def print_metrics(name: str, metrics: Metrics, threshold: Threshold) -> None:
    if metrics.average == "macro":
        counts_str = (
            f"rows={metrics.n_rows} | subjects={metrics.n_subjects} | "
            f"tp_sum={metrics.tp} tn_sum={metrics.tn} fp_sum={metrics.fp} fn_sum={metrics.fn}"
        )
    else:
        counts_str = (
            f"rows={metrics.n_rows} | subjects={metrics.n_subjects} | "
            f"tp={metrics.tp} tn={metrics.tn} fp={metrics.fp} fn={metrics.fn}"
        )

    print(
        f"{name:>5} | threshold={threshold.sbp:.0f}/{threshold.dbp:.0f} | "
        f"avg={metrics.average} | {counts_str} | "
        f"precision={format_metric(metrics.precision)} "
        f"recall={format_metric(metrics.recall)} "
        f"accuracy={format_metric(metrics.accuracy)} "
        f"f1={format_metric(metrics.f1)}"
    )


def main() -> None:
    args = parse_args()

    with open(args.csv_path, "r", newline="") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))

    if not args.include_calib:
        rows = [row for row in rows if not parse_bool(row.get("is_calib"))]

    whole_threshold = Threshold(args.whole_sbp, args.whole_dbp)
    day_threshold = Threshold(args.day_sbp, args.day_dbp)
    night_threshold = Threshold(args.night_sbp, args.night_dbp)

    compute_metrics = (
        compute_macro_metrics if args.average == "macro" else compute_micro_metrics
    )

    whole_metrics = compute_metrics(rows, whole_threshold, lambda row: True, args.subject_col)
    day_metrics = compute_metrics(
        rows,
        day_threshold,
        lambda row: parse_int(row, "sleep") == 0,
        args.subject_col,
    )
    night_metrics = compute_metrics(
        rows,
        night_threshold,
        lambda row: parse_int(row, "sleep") == 1,
        args.subject_col,
    )

    print("Using columns: y_true_sbp, y_true_dbp, y_pred_sbp_cal, y_pred_dbp_cal")
    print(
        "Calibration rows included:"
        f" {'yes' if args.include_calib else 'no (default behavior)'}"
    )
    print(
        "Averaging mode:"
        f" {args.average} ({'per-subject mean metrics' if args.average == 'macro' else 'all rows pooled'})"
    )
    print(f"Subject column: {args.subject_col}")
    print_metrics("whole", whole_metrics, whole_threshold)
    print_metrics("day", day_metrics, day_threshold)
    print_metrics("night", night_metrics, night_threshold)


if __name__ == "__main__":
    main()
