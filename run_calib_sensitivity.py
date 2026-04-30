#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General calibration-point sensitivity test for PPG-BP product evaluation.

This is the generalized version of run_1p1_sensitivity.py.

Purpose:
    Repeat a calibration protocol N times with different calibration-point
    selections, save one prediction CSV per run, and summarize performance
    variation across runs.

Supported examples:
    1+1 hourly:
        --support_n 1 --update_n 1 --eval_mode hourly

    2+2 macro24:
        --support_n 2 --update_n 2 --eval_mode macro24

    4+3 hourly:
        --support_n 4 --update_n 3 --eval_mode hourly

    4 day + 3 night hourly:
        --support_n 4 --update_n 3 --sleep_quota "0:4,1:3" --eval_mode hourly

Dependency:
    Put this file in the same folder as lazy_bp_product_eval.py, or make sure
    lazy_bp_product_eval.py is importable.

Inputs:
    raw_predictions.csv with at least:
        id_clean, t_bp_ms, sleep,
        y_true_sbp, y_true_dbp,
        y_pred_sbp_raw, y_pred_dbp_raw

Outputs:
    out_dir/
        predictions_<protocol>_runXX_seedYYYY.csv
        metrics_<protocol>_runXX_seedYYYY.json
        sensitivity_<protocol>_per_run_metrics.csv
        sensitivity_<protocol>_metric_mean_std_across_runs.csv
        sensitivity_<protocol>_pooled_event_metrics.csv
        sensitivity_<protocol>_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lazy_bp_product_eval import (
    ProductScenario,
    load_raw_predictions,
    run_one_scenario,
    me_std_mae,
)


# =============================================================================
# Utility functions
# =============================================================================

def parse_sleep_quota(value: Optional[str]) -> Optional[Dict[int, int]]:
    """
    Parse sleep quota string.

    Example:
        "0:4,1:3" -> {0: 4, 1: 3}

    sleep convention:
        0 = day / awake
        1 = night / sleep
    """
    if value is None:
        return None

    value = str(value).strip()
    if value == "" or value.lower() in {"none", "null", "no"}:
        return None

    quota: Dict[int, int] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"Invalid sleep_quota part '{part}'. Expected format like '0:4,1:3'."
            )
        k, v = part.split(":", 1)
        sleep_value = int(k.strip())
        n_points = int(v.strip())
        if sleep_value not in (0, 1):
            raise ValueError("sleep_quota keys must be 0 or 1.")
        if n_points < 0:
            raise ValueError("sleep_quota values must be non-negative.")
        quota[sleep_value] = n_points

    return quota if quota else None


def infer_calib_total(
    calib_total: Optional[int],
    support_n: int,
    update_n: int,
    sleep_quota: Optional[Dict[int, int]],
) -> int:
    """
    Infer calibration budget if --calib_total is not explicitly given.
    """
    if calib_total is not None:
        return int(calib_total)

    if sleep_quota is not None:
        return int(sum(sleep_quota.values()))

    total = int(support_n) + int(update_n)
    if total <= 0:
        raise ValueError("calib_total is not set and support_n + update_n <= 0.")
    return total


def protocol_name(
    calib_total: int,
    support_n: int,
    update_n: int,
    calib_sleep: str,
    eval_mode: str,
    sleep_quota: Optional[Dict[int, int]],
) -> str:
    if support_n > 0 or update_n > 0:
        name = f"{support_n}p{update_n}"
    else:
        name = f"{calib_total}calib"

    if sleep_quota is not None:
        quota_text = "_".join(f"s{k}-{v}" for k, v in sorted(sleep_quota.items()))
        name += f"_{quota_text}"
    elif calib_sleep != "all":
        name += f"_{calib_sleep}"

    name += f"_{eval_mode}"
    return name


def _safe_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize numeric metrics across repeated runs.

    Returns one row per metric:
        metric, mean, std_across_runs, min, max, n_runs_valid
    """
    numeric_cols = []
    for c in df.columns:
        if c in {"scenario"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    rows = []
    for c in numeric_cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) == 0:
            continue
        rows.append(
            {
                "metric": c,
                "mean": float(np.mean(x)),
                "std_across_runs": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "n_runs_valid": int(len(x)),
            }
        )

    return pd.DataFrame(rows)


def _method_pred_cols() -> Dict[str, Tuple[str, str]]:
    return {
        "raw": ("y_pred_sbp_raw", "y_pred_dbp_raw"),
        "bias": ("y_pred_sbp_bias", "y_pred_dbp_bias"),
        "aff": ("y_pred_sbp_aff", "y_pred_dbp_aff"),
        "bank": ("y_pred_sbp_bank", "y_pred_dbp_bank"),
    }


def compute_pooled_event_metrics(
    csv_paths: List[Path],
    eval_sleep: str = "all",
    exclude_calib: bool = True,
) -> pd.DataFrame:
    """
    Pool all non-calibration rows from prediction CSVs and compute event-level metrics.

    This gives a global pooled view. For calibration sensitivity, the more important
    file is usually metric_mean_std_across_runs.csv.
    """
    pooled = []
    for run_id, p in enumerate(csv_paths):
        d = pd.read_csv(p)
        d["sensitivity_run_id"] = run_id

        if exclude_calib and "is_calib" in d.columns:
            d = d[~d["is_calib"].astype(bool)].copy()

        if eval_sleep == "day":
            d = d[d["sleep"].astype(int) == 0].copy()
        elif eval_sleep in ["night", "sleep"]:
            d = d[d["sleep"].astype(int) == 1].copy()
        elif eval_sleep == "all":
            pass
        else:
            raise ValueError(f"Unknown eval_sleep={eval_sleep}")

        pooled.append(d)

    if len(pooled) == 0:
        return pd.DataFrame()

    all_df = pd.concat(pooled, axis=0, ignore_index=True)

    rows = []
    for method, (sbp_col, dbp_col) in _method_pred_cols().items():
        if sbp_col not in all_df.columns or dbp_col not in all_df.columns:
            continue

        sbp = me_std_mae(all_df[sbp_col], all_df["y_true_sbp"])
        dbp = me_std_mae(all_df[dbp_col], all_df["y_true_dbp"])

        rows.append(
            {
                "method": method,
                "target": "SBP",
                "ME": sbp["ME"],
                "STD": sbp["STD"],
                "MAE": sbp["MAE"],
                "N": sbp["N"],
            }
        )
        rows.append(
            {
                "method": method,
                "target": "DBP",
                "ME": dbp["ME"],
                "STD": dbp["STD"],
                "MAE": dbp["MAE"],
                "N": dbp["N"],
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Main API
# =============================================================================

def run_calib_sensitivity_from_raw(
    raw_csv: str | Path,
    emb_path: Optional[str | Path] = None,
    out_dir: str | Path = "eval_calib_sensitivity",
    n_runs: int = 8,
    base_seed: int = 2026,

    # Generic calibration protocol.
    calib_total: Optional[int] = None,
    support_n: int = 1,
    update_n: int = 1,
    sleep_quota: Optional[Dict[int, int]] = None,

    # Scenario config.
    calib_strategy: str = "random_min_gap",
    min_gap_minutes: float = 30.0,
    calib_sleep: str = "all",
    eval_sleep: str = "all",
    eval_mode: str = "hourly",
    min_events_per_hour: int = 2,
    macro_window_hours: int = 24,
    min_events_per_macro: int = 6,

    # Calibration method config.
    run_bias: bool = True,
    run_affine: bool = True,
    run_bank: bool = True,
    by_sleep_calibration: bool = False,
    affine_lam: float = 100.0,
    affine_min_points: int = 3,
    bank_temperature: float = 0.30,
    bank_time_weight_per_hour: float = 0.20,
    bank_sleep_bonus: float = 0.25,
    bank_uniform_mix: float = 0.20,
    bank_residual_clip: float = 35.0,
) -> Dict[str, Any]:
    """
    Run calibration sensitivity for arbitrary product calibration protocol.

    Examples:
        1+1: support_n=1, update_n=1
        2+2: support_n=2, update_n=2
        4+3: support_n=4, update_n=3
        4 day + 3 night: support_n=4, update_n=3, sleep_quota={0:4, 1:3}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_calib_total = infer_calib_total(calib_total, support_n, update_n, sleep_quota)

    if support_n + update_n > 0 and support_n + update_n != final_calib_total:
        print(
            "[WARN] support_n + update_n != calib_total. "
            f"support_n={support_n}, update_n={update_n}, calib_total={final_calib_total}. "
            "Extra selected calibration points, if any, will be labeled as calib."
        )

    if sleep_quota is not None and sum(sleep_quota.values()) != final_calib_total:
        print(
            "[WARN] sum(sleep_quota) != calib_total. "
            f"sleep_quota={sleep_quota}, calib_total={final_calib_total}. "
            "ProductScenario will cap selected points to calib_total."
        )

    p_name = protocol_name(
        calib_total=final_calib_total,
        support_n=support_n,
        update_n=update_n,
        calib_sleep=calib_sleep,
        eval_mode=eval_mode,
        sleep_quota=sleep_quota,
    )

    df_raw, emb = load_raw_predictions(raw_csv, emb_path)

    all_metrics: List[Dict[str, Any]] = []
    pred_csv_paths: List[Path] = []

    for run_id in range(int(n_runs)):
        seed = int(base_seed) + run_id

        scenario = ProductScenario(
            name=f"{p_name}_run{run_id:02d}_seed{seed}",
            calib_total=final_calib_total,
            support_n=support_n,
            update_n=update_n,
            calib_strategy=calib_strategy,
            min_gap_minutes=min_gap_minutes,
            seed=seed,
            calib_sleep=calib_sleep,
            eval_sleep=eval_sleep,
            sleep_quota=sleep_quota,
            eval_mode=eval_mode,
            min_events_per_hour=min_events_per_hour,
            macro_window_hours=macro_window_hours,
            min_events_per_macro=min_events_per_macro,
            exclude_calib_from_eval=True,
            separate_sleep_in_hourly=True,
        )

        df_run, metrics_run = run_one_scenario(
            df_raw=df_raw,
            scenario=scenario,
            emb=emb,
            run_bias=run_bias,
            run_affine=run_affine,
            run_bank=run_bank,
            by_sleep_calibration=by_sleep_calibration,
            affine_lam=affine_lam,
            affine_min_points=affine_min_points,
            bank_temperature=bank_temperature,
            bank_time_weight_per_hour=bank_time_weight_per_hour,
            bank_sleep_bonus=bank_sleep_bonus,
            bank_uniform_mix=bank_uniform_mix,
            bank_residual_clip=bank_residual_clip,
        )

        metrics_run["sensitivity_run_id"] = run_id
        metrics_run["sensitivity_seed"] = seed
        metrics_run["protocol_name"] = p_name
        metrics_run["calib_total"] = final_calib_total
        metrics_run["support_n"] = support_n
        metrics_run["update_n"] = update_n
        metrics_run["sleep_quota_json"] = json.dumps(sleep_quota, ensure_ascii=False) if sleep_quota else ""

        pred_path = out_dir / f"predictions_{p_name}_run{run_id:02d}_seed{seed}.csv"
        df_run.to_csv(pred_path, index=False)
        pred_csv_paths.append(pred_path)

        metrics_path = out_dir / f"metrics_{p_name}_run{run_id:02d}_seed{seed}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_run, f, indent=2, ensure_ascii=False)

        all_metrics.append(metrics_run)

    per_run_metrics = pd.DataFrame(all_metrics)
    per_run_path = out_dir / f"sensitivity_{p_name}_per_run_metrics.csv"
    per_run_metrics.to_csv(per_run_path, index=False)

    across_run_summary = _safe_numeric_summary(per_run_metrics)
    across_path = out_dir / f"sensitivity_{p_name}_metric_mean_std_across_runs.csv"
    across_run_summary.to_csv(across_path, index=False)

    pooled_event_metrics = compute_pooled_event_metrics(
        csv_paths=pred_csv_paths,
        eval_sleep=eval_sleep,
        exclude_calib=True,
    )
    pooled_path = out_dir / f"sensitivity_{p_name}_pooled_event_metrics.csv"
    pooled_event_metrics.to_csv(pooled_path, index=False)

    manifest = {
        "raw_csv": str(raw_csv),
        "emb_path": str(emb_path) if emb_path is not None else None,
        "out_dir": str(out_dir),
        "protocol_name": p_name,
        "n_runs": int(n_runs),
        "base_seed": int(base_seed),
        "calib_total": int(final_calib_total),
        "support_n": int(support_n),
        "update_n": int(update_n),
        "sleep_quota": sleep_quota,
        "calib_strategy": calib_strategy,
        "min_gap_minutes": float(min_gap_minutes),
        "calib_sleep": calib_sleep,
        "eval_sleep": eval_sleep,
        "eval_mode": eval_mode,
        "prediction_csvs": [str(p) for p in pred_csv_paths],
        "per_run_metrics_csv": str(per_run_path),
        "across_run_summary_csv": str(across_path),
        "pooled_event_metrics_csv": str(pooled_path),
    }
    manifest_path = out_dir / f"sensitivity_{p_name}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== calibration sensitivity finished ===")
    print(f"Protocol:             {p_name}")
    print(f"Per-run metrics:      {per_run_path}")
    print(f"Across-run summary:  {across_path}")
    print(f"Pooled event metrics: {pooled_path}")
    print(f"Prediction CSV folder:{out_dir}")

    return {
        "per_run_metrics": per_run_metrics,
        "across_run_summary": across_run_summary,
        "pooled_event_metrics": pooled_event_metrics,
        "prediction_csv_paths": pred_csv_paths,
        "manifest": manifest,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run generic calibration-point sensitivity test."
    )

    p.add_argument("--raw_csv", type=str, required=True)
    p.add_argument("--emb_path", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="eval_calib_sensitivity")

    p.add_argument("--n_runs", type=int, default=8)
    p.add_argument("--base_seed", type=int, default=2026)

    # Generic protocol arguments.
    p.add_argument(
        "--calib_total",
        type=int,
        default=None,
        help=(
            "Total calibration points. If omitted, inferred from support_n + update_n "
            "or from sleep_quota."
        ),
    )
    p.add_argument("--support_n", type=int, default=1)
    p.add_argument("--update_n", type=int, default=1)
    p.add_argument(
        "--sleep_quota",
        type=str,
        default=None,
        help='Optional day/night quota, e.g. "0:4,1:3" for 4 day + 3 night.',
    )

    p.add_argument(
        "--calib_strategy",
        type=str,
        default="random_min_gap",
        choices=["head", "tail", "quantile", "min_gap", "random", "random_min_gap"],
    )
    p.add_argument("--min_gap_minutes", type=float, default=30.0)

    p.add_argument("--calib_sleep", type=str, default="all", choices=["all", "day", "night", "sleep"])
    p.add_argument("--eval_sleep", type=str, default="all", choices=["all", "day", "night", "sleep"])
    p.add_argument("--eval_mode", type=str, default="hourly", choices=["event", "hourly", "macro24", "both", "all"])

    p.add_argument("--min_events_per_hour", type=int, default=2)
    p.add_argument("--macro_window_hours", type=int, default=24)
    p.add_argument("--min_events_per_macro", type=int, default=6)

    p.add_argument("--no_bias", action="store_true")
    p.add_argument("--no_affine", action="store_true")
    p.add_argument("--no_bank", action="store_true")
    p.add_argument("--by_sleep_calibration", action="store_true")

    p.add_argument("--affine_lam", type=float, default=100.0)
    p.add_argument("--affine_min_points", type=int, default=3)

    p.add_argument("--bank_temperature", type=float, default=0.30)
    p.add_argument("--bank_time_weight_per_hour", type=float, default=0.20)
    p.add_argument("--bank_sleep_bonus", type=float, default=0.25)
    p.add_argument("--bank_uniform_mix", type=float, default=0.20)
    p.add_argument("--bank_residual_clip", type=float, default=35.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    sleep_quota = parse_sleep_quota(args.sleep_quota)

    run_calib_sensitivity_from_raw(
        raw_csv=args.raw_csv,
        emb_path=args.emb_path,
        out_dir=args.out_dir,
        n_runs=args.n_runs,
        base_seed=args.base_seed,
        calib_total=args.calib_total,
        support_n=args.support_n,
        update_n=args.update_n,
        sleep_quota=sleep_quota,
        calib_strategy=args.calib_strategy,
        min_gap_minutes=args.min_gap_minutes,
        calib_sleep=args.calib_sleep,
        eval_sleep=args.eval_sleep,
        eval_mode=args.eval_mode,
        min_events_per_hour=args.min_events_per_hour,
        macro_window_hours=args.macro_window_hours,
        min_events_per_macro=args.min_events_per_macro,
        run_bias=not args.no_bias,
        run_affine=not args.no_affine,
        run_bank=not args.no_bank,
        by_sleep_calibration=args.by_sleep_calibration,
        affine_lam=args.affine_lam,
        affine_min_points=args.affine_min_points,
        bank_temperature=args.bank_temperature,
        bank_time_weight_per_hour=args.bank_time_weight_per_hour,
        bank_sleep_bonus=args.bank_sleep_bonus,
        bank_uniform_mix=args.bank_uniform_mix,
        bank_residual_clip=args.bank_residual_clip,
    )


if __name__ == "__main__":
    main()
