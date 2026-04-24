#!/usr/bin/env python3
"""Compute hypertension classification metrics from calibrated BP CSV output.

Default logic:
- compare `y_true_sbp/y_true_dbp` vs `y_pred_sbp_cal/y_pred_dbp_cal`
- exclude calibration rows (`is_calib == True`) by default
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
import csv
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional


TRUEY = {"1", "true", "yes", "y", "t"}


@dataclass
class Threshold:
    sbp: float
    dbp: float


@dataclass
class Metrics:
    n: int
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


def compute_metrics(
    rows: Iterable[Dict[str, str]],
    threshold: Threshold,
    row_filter: Callable[[Dict[str, str]], bool],
) -> Metrics:
    tp = tn = fp = fn = 0
    n = 0

    for row in rows:
        if not row_filter(row):
            continue

        true_sbp = parse_float(row, "y_true_sbp")
        true_dbp = parse_float(row, "y_true_dbp")
        pred_sbp = parse_float(row, "y_pred_sbp_cal")
        pred_dbp = parse_float(row, "y_pred_dbp_cal")

        if None in (true_sbp, true_dbp, pred_sbp, pred_dbp):
            continue

        true_positive = is_positive(true_sbp, true_dbp, threshold)
        pred_positive = is_positive(pred_sbp, pred_dbp, threshold)
        n += 1

        if true_positive and pred_positive:
            tp += 1
        elif (not true_positive) and (not pred_positive):
            tn += 1
        elif (not true_positive) and pred_positive:
            fp += 1
        else:
            fn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    accuracy = safe_div(tp + tn, n)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return Metrics(
        n=n,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        f1=f1,
    )


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def print_metrics(name: str, metrics: Metrics, threshold: Threshold) -> None:
    print(
        f"{name:>5} | threshold={threshold.sbp:.0f}/{threshold.dbp:.0f} | "
        f"n={metrics.n} | tp={metrics.tp} tn={metrics.tn} fp={metrics.fp} fn={metrics.fn} | "
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

    whole_metrics = compute_metrics(
        rows,
        whole_threshold,
        row_filter=lambda row: True,
    )
    day_metrics = compute_metrics(
        rows,
        day_threshold,
        row_filter=lambda row: parse_int(row, "sleep") == 0,
    )
    night_metrics = compute_metrics(
        rows,
        night_threshold,
        row_filter=lambda row: parse_int(row, "sleep") == 1,
    )

    print("Using columns: y_true_sbp, y_true_dbp, y_pred_sbp_cal, y_pred_dbp_cal")
    print(
        "Calibration rows included:"
        f" {'yes' if args.include_calib else 'no (default behavior)'}"
    )
    print_metrics("whole", whole_metrics, whole_threshold)
    print_metrics("day", day_metrics, day_threshold)
    print_metrics("night", night_metrics, night_threshold)


if __name__ == "__main__":
    main()
