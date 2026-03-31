#!/usr/bin/env python3
"""
Build a new is_calib column from an external calibration-points CSV.

Workflow
--------
1) Read local event-level CSV (must contain subject id, event timestamp, sleep label).
2) Read external calibration CSV (must contain subject id and timestamp).
3) Match external points onto local events by (ID, timestamp).
4) Optionally downsample matched external points to a per-subject quota,
   e.g. day 2 + sleep 2.
5) Write a new CSV with a freshly generated is_calib column.

Typical use case
----------------
External team provides ~7 calibration points / subject.
You want to keep calibration consistent with them, but only use a capped subset
such as 2 daytime + 2 sleep points per subject, then use the new is_calib for
all downstream calibration / evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def _normalize_id(x: pd.Series) -> pd.Series:
    return x.astype(str).str.strip().str.upper()


def _normalize_time(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").astype("Int64")


def _select_indices_by_strategy(
    g: pd.DataFrame,
    n_select: int,
    *,
    time_col: str,
    strategy: str,
) -> List[int]:
    """Return row indices to keep from a time-sorted group."""
    if n_select <= 0 or len(g) == 0:
        return []

    g = g.sort_values(time_col).copy()
    if len(g) <= n_select:
        return g.index.tolist()

    if strategy == "earliest":
        return g.index[:n_select].tolist()

    if strategy == "latest":
        return g.index[-n_select:].tolist()

    if strategy == "evenly_spaced":
        # pick indices approximately uniformly over time span
        pos = np.linspace(0, len(g) - 1, n_select)
        chosen = np.round(pos).astype(int).tolist()
        # de-duplicate while preserving order
        uniq = []
        seen = set()
        for c in chosen:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        # back-fill if rounding caused duplicates
        if len(uniq) < n_select:
            for c in range(len(g)):
                if c not in seen:
                    uniq.append(c)
                    seen.add(c)
                if len(uniq) >= n_select:
                    break
        return g.index[uniq[:n_select]].tolist()

    raise ValueError(f"Unknown strategy: {strategy}")


# -----------------------------
# Core logic
# -----------------------------

def load_and_match_external_points(
    df: pd.DataFrame,
    ext_df: pd.DataFrame,
    *,
    df_id_col: str,
    df_time_col: str,
    ext_id_col: str,
    ext_time_col: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Match external points to local df by exact (normalized_id, normalized_time).

    Returns a copy of df with helper columns:
      _match_id, _match_t, _is_external_match
    """
    out = df.copy()
    ext = ext_df.copy()

    out["_match_id"] = _normalize_id(out[df_id_col])
    out["_match_t"] = _normalize_time(out[df_time_col])

    ext["_match_id"] = _normalize_id(ext[ext_id_col])
    ext["_match_t"] = _normalize_time(ext[ext_time_col])
    ext = ext.dropna(subset=["_match_id", "_match_t"]).copy()
    ext = ext.drop_duplicates(subset=["_match_id", "_match_t"]).copy()

    ext_keys = set(zip(ext["_match_id"].tolist(), ext["_match_t"].tolist()))
    out_keys = list(zip(out["_match_id"].tolist(), out["_match_t"].tolist()))
    out["_is_external_match"] = [k in ext_keys for k in out_keys]

    matched_keys = set(k for k in out_keys if k in ext_keys)
    unmatched_ext = ext_keys - matched_keys

    report = {
        "n_df_rows": int(len(out)),
        "n_external_unique_points": int(len(ext_keys)),
        "n_matched_df_rows": int(out["_is_external_match"].sum()),
        "n_matched_unique_points": int(len(matched_keys)),
        "n_unmatched_external_points": int(len(unmatched_ext)),
    }
    return out, report



def downsample_external_matches(
    matched_df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    sleep_col: str,
    mode: str,
    n_day: int,
    n_sleep: int,
    strategy: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    From rows already matched to external calibration points, select final is_calib rows.

    Modes
    -----
    - external_all: keep all matched external points
    - day_sleep_quota: keep up to n_day points with sleep==0 and up to n_sleep with sleep==1
    - total_quota: keep up to (n_day+n_sleep) points regardless of sleep state
    """
    if sleep_col not in matched_df.columns and mode == "day_sleep_quota":
        raise ValueError(f"sleep_col={sleep_col} not found, but mode=day_sleep_quota requires it")

    out = matched_df.copy()
    out["_selected_final_calib"] = False

    matched_only = out[out["_is_external_match"]].copy()

    selected_indices: List[int] = []

    for sid, g_sub in matched_only.groupby(id_col):
        g_sub = g_sub.sort_values(time_col).copy()

        if mode == "external_all":
            selected_indices.extend(g_sub.index.tolist())
            continue

        if mode == "total_quota":
            n_total = max(0, int(n_day) + int(n_sleep))
            selected_indices.extend(
                _select_indices_by_strategy(g_sub, n_total, time_col=time_col, strategy=strategy)
            )
            continue

        if mode == "day_sleep_quota":
            g_day = g_sub[g_sub[sleep_col] == 0].copy()
            g_sleep = g_sub[g_sub[sleep_col] == 1].copy()

            selected_indices.extend(
                _select_indices_by_strategy(g_day, n_day, time_col=time_col, strategy=strategy)
            )
            selected_indices.extend(
                _select_indices_by_strategy(g_sleep, n_sleep, time_col=time_col, strategy=strategy)
            )
            continue

        raise ValueError(f"Unknown mode: {mode}")

    out.loc[selected_indices, "_selected_final_calib"] = True

    per_subject = []
    for sid, g_sub in out.groupby(id_col):
        g_ext = g_sub[g_sub["_is_external_match"]]
        g_sel = g_sub[g_sub["_selected_final_calib"]]
        rec = {
            id_col: sid,
            "n_external_matched": int(len(g_ext)),
            "n_selected": int(len(g_sel)),
        }
        if sleep_col in out.columns:
            rec["n_selected_day"] = int((g_sel[sleep_col] == 0).sum())
            rec["n_selected_sleep"] = int((g_sel[sleep_col] == 1).sum())
        per_subject.append(rec)

    per_subject_df = pd.DataFrame(per_subject)

    summary = {
        "n_subjects_total": int(out[id_col].nunique()),
        "n_subjects_with_external_match": int((per_subject_df["n_external_matched"] > 0).sum()) if len(per_subject_df) else 0,
        "n_subjects_with_selected_calib": int((per_subject_df["n_selected"] > 0).sum()) if len(per_subject_df) else 0,
        "mean_external_matched_per_subject": float(per_subject_df["n_external_matched"].mean()) if len(per_subject_df) else float("nan"),
        "mean_selected_per_subject": float(per_subject_df["n_selected"].mean()) if len(per_subject_df) else float("nan"),
    }

    return out, {"summary": summary, "per_subject": per_subject_df}



def build_new_is_calib(
    df: pd.DataFrame,
    ext_df: pd.DataFrame,
    *,
    df_id_col: str = "id_upper",
    df_time_col: str = "t_bp_ms",
    sleep_col: str = "sleep",
    ext_id_col: str = "id",
    ext_time_col: str = "time_timestamp",
    mode: str = "day_sleep_quota",
    n_day: int = 2,
    n_sleep: int = 2,
    strategy: str = "evenly_spaced",
    is_calib_col: str = "is_calib",
    keep_old_is_calib: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    matched_df, match_report = load_and_match_external_points(
        df,
        ext_df,
        df_id_col=df_id_col,
        df_time_col=df_time_col,
        ext_id_col=ext_id_col,
        ext_time_col=ext_time_col,
    )

    selected_df, select_report = downsample_external_matches(
        matched_df,
        id_col=df_id_col,
        time_col=df_time_col,
        sleep_col=sleep_col,
        mode=mode,
        n_day=n_day,
        n_sleep=n_sleep,
        strategy=strategy,
    )

    out = selected_df.copy()
    if keep_old_is_calib and is_calib_col in out.columns:
        out[f"{is_calib_col}_old"] = out[is_calib_col]

    out[is_calib_col] = out["_selected_final_calib"].astype(bool)

    final_report = {
        "match_report": match_report,
        "selection_summary": select_report["summary"],
        "mode": mode,
        "n_day": int(n_day),
        "n_sleep": int(n_sleep),
        "strategy": strategy,
        "n_final_is_calib_rows": int(out[is_calib_col].sum()),
    }

    return out, {
        "report": final_report,
        "per_subject": select_report["per_subject"],
    }


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build new is_calib from external calibration points")
    parser.add_argument("--input_csv", required=True, help="Local event-level CSV")
    parser.add_argument("--external_csv", required=True, help="External calibration CSV")
    parser.add_argument("--output_csv", required=True, help="Output CSV with refreshed is_calib")
    parser.add_argument("--report_dir", required=True, help="Directory for reports")

    parser.add_argument("--df_id_col", default="id_upper")
    parser.add_argument("--df_time_col", default="t_bp_ms")
    parser.add_argument("--sleep_col", default="sleep")
    parser.add_argument("--ext_id_col", default="id")
    parser.add_argument("--ext_time_col", default="time_timestamp")
    parser.add_argument("--is_calib_col", default="is_calib")

    parser.add_argument(
        "--mode",
        default="day_sleep_quota",
        choices=["external_all", "day_sleep_quota", "total_quota"],
        help=(
            "external_all: keep all external matched points; "
            "day_sleep_quota: keep up to n_day daytime + n_sleep sleep points; "
            "total_quota: keep up to n_day+n_sleep points regardless of sleep"
        ),
    )
    parser.add_argument("--n_day", type=int, default=2)
    parser.add_argument("--n_sleep", type=int, default=2)
    parser.add_argument(
        "--strategy",
        default="evenly_spaced",
        choices=["evenly_spaced", "earliest", "latest"],
        help="How to choose kept points within each subject / state",
    )
    parser.add_argument(
        "--keep_old_is_calib",
        action="store_true",
        help="If set, preserve original is_calib into is_calib_old",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    external_csv = Path(args.external_csv)
    output_csv = Path(args.output_csv)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    ext_df = pd.read_csv(external_csv)

    out_df, artifacts = build_new_is_calib(
        df,
        ext_df,
        df_id_col=args.df_id_col,
        df_time_col=args.df_time_col,
        sleep_col=args.sleep_col,
        ext_id_col=args.ext_id_col,
        ext_time_col=args.ext_time_col,
        mode=args.mode,
        n_day=args.n_day,
        n_sleep=args.n_sleep,
        strategy=args.strategy,
        is_calib_col=args.is_calib_col,
        keep_old_is_calib=args.keep_old_is_calib,
    )

    # Clean helper columns before write
    helper_cols = [c for c in ["_match_id", "_match_t", "_is_external_match", "_selected_final_calib"] if c in out_df.columns]
    write_df = out_df.drop(columns=helper_cols)
    write_df.to_csv(output_csv, index=False)

    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(artifacts["report"], f, indent=2, ensure_ascii=False)

    artifacts["per_subject"].to_csv(report_dir / "per_subject_selection.csv", index=False)

    with open(report_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("=== build_external_calib_is_calib summary ===\n")
        for k, v in artifacts["report"].items():
            f.write(f"{k}: {v}\n")

    print("Saved:")
    print(f"  output_csv: {output_csv}")
    print(f"  summary.json: {report_dir / 'summary.json'}")
    print(f"  per_subject_selection.csv: {report_dir / 'per_subject_selection.csv'}")


if __name__ == "__main__":
    main()
