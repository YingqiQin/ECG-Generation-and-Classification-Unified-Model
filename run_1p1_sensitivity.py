#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run 1+1 calibration sensitivity test for PPG-BP product evaluation.

Purpose:
    Run 8 different 1+1 calibration selections:
        support_n = 1
        update_n  = 1
        calib_total = 2
    Each run uses a different random seed, saves one prediction CSV,
    then summarizes:
        1) per-run metrics
        2) mean/std/min/max of metrics across 8 runs
        3) pooled metrics across all 8 non-calibration predictions

Dependency:
    Put this file in the same folder as lazy_bp_product_eval.py, or make sure
    lazy_bp_product_eval.py is importable.

Typical usage:
    python run_1p1_sensitivity.py \
        --raw_csv raw_predictions.csv \
        --out_dir eval_1p1_sensitivity \
        --n_runs 8 \
        --base_seed 2026 \
        --eval_mode hourly \
        --calib_strategy random_min_gap \
        --min_gap_minutes 30

If you have embedding file:
    python run_1p1_sensitivity.py \
        --raw_csv raw_predictions.csv \
        --emb_path raw_embeddings.npy \
        --out_dir eval_1p1_sensitivity
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


def _safe_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize numeric metrics across repeated runs.
    Returns one row per metric:
        metric, mean, std, min, max, n
    """
    numeric_cols = []
    for c in df.columns:
        if c in ["scenario"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    rows = []
    for c in numeric_cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float)
        if len(x) == 0:
            continue
        rows.append({
            "metric": c,
            "mean": float(np.mean(x)),
            "std_across_runs": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "n_runs_valid": int(len(x)),
        })

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
    Pool all non-calibration rows from 8 prediction CSVs and compute event-level metrics.
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

        rows.append({
            "method": method,
            "target": "SBP",
            "ME": sbp["ME"],
            "STD": sbp["STD"],
            "MAE": sbp["MAE"],
            "N": sbp["N"],
        })
        rows.append({
            "method": method,
            "target": "DBP",
            "ME": dbp["ME"],
            "STD": dbp["STD"],
            "MAE": dbp["MAE"],
            "N": dbp["N"],
        })

    return pd.DataFrame(rows)


def run_1p1_sensitivity_from_raw(
    raw_csv: str | Path,
    emb_path: Optional[str | Path] = None,
    out_dir: str | Path = "eval_1p1_sensitivity",
    n_runs: int = 8,
    base_seed: int = 2026,

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
    Main API.

    Returns paths and DataFrames in a dictionary.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw, emb = load_raw_predictions(raw_csv, emb_path)

    all_metrics: List[Dict[str, Any]] = []
    pred_csv_paths: List[Path] = []

    for run_id in range(int(n_runs)):
        seed = int(base_seed) + run_id

        scenario = ProductScenario(
            name=f"one_plus_one_sensitivity_run{run_id:02d}_seed{seed}",
            calib_total=2,
            support_n=1,
            update_n=1,
            calib_strategy=calib_strategy,
            min_gap_minutes=min_gap_minutes,
            seed=seed,
            calib_sleep=calib_sleep,
            eval_sleep=eval_sleep,
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

        pred_path = out_dir / f"predictions_1p1_run{run_id:02d}_seed{seed}.csv"
        df_run.to_csv(pred_path, index=False)
        pred_csv_paths.append(pred_path)

        metrics_path = out_dir / f"metrics_1p1_run{run_id:02d}_seed{seed}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_run, f, indent=2, ensure_ascii=False)

        all_metrics.append(metrics_run)

    per_run_metrics = pd.DataFrame(all_metrics)
    per_run_path = out_dir / "sensitivity_1p1_per_run_metrics.csv"
    per_run_metrics.to_csv(per_run_path, index=False)

    across_run_summary = _safe_numeric_summary(per_run_metrics)
    across_path = out_dir / "sensitivity_1p1_metric_mean_std_across_runs.csv"
    across_run_summary.to_csv(across_path, index=False)

    pooled_event_metrics = compute_pooled_event_metrics(
        csv_paths=pred_csv_paths,
        eval_sleep=eval_sleep,
        exclude_calib=True,
    )
    pooled_path = out_dir / "sensitivity_1p1_pooled_event_metrics.csv"
    pooled_event_metrics.to_csv(pooled_path, index=False)

    # Small manifest for reproducibility.
    manifest = {
        "raw_csv": str(raw_csv),
        "emb_path": str(emb_path) if emb_path is not None else None,
        "out_dir": str(out_dir),
        "n_runs": int(n_runs),
        "base_seed": int(base_seed),
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
    manifest_path = out_dir / "sensitivity_1p1_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== 1+1 sensitivity finished ===")
    print(f"Per-run metrics:       {per_run_path}")
    print(f"Across-run summary:   {across_path}")
    print(f"Pooled event metrics:  {pooled_path}")
    print(f"Prediction CSV folder: {out_dir}")

    return {
        "per_run_metrics": per_run_metrics,
        "across_run_summary": across_run_summary,
        "pooled_event_metrics": pooled_event_metrics,
        "prediction_csv_paths": pred_csv_paths,
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 8x 1+1 calibration sensitivity test.")

    p.add_argument("--raw_csv", type=str, required=True)
    p.add_argument("--emb_path", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="eval_1p1_sensitivity")

    p.add_argument("--n_runs", type=int, default=8)
    p.add_argument("--base_seed", type=int, default=2026)

    p.add_argument("--calib_strategy", type=str, default="random_min_gap",
                   choices=["head", "tail", "quantile", "min_gap", "random", "random_min_gap"])
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

    run_1p1_sensitivity_from_raw(
        raw_csv=args.raw_csv,
        emb_path=args.emb_path,
        out_dir=args.out_dir,
        n_runs=args.n_runs,
        base_seed=args.base_seed,
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
