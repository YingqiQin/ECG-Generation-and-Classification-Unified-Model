import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _normalize_flag(value) -> int:
    if pd.isna(value):
        return 0
    s = str(value).strip()
    if s in {"1", "1.0", "True", "true"}:
        return 1
    return 0


def _value_distribution(series: pd.Series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in Counter(series.astype(str).tolist()).items()}


def _subtask_specs() -> Dict[str, Dict]:
    return {
        "no_problem": {
            "description": "Exclude rows flagged by the updated Shengrenyi is_problem marker only.",
            "exclude_if_any": [
                "is_problem",
            ],
        },
        "doc_error_reduced": {
            "description": "Exclude rows enriched in the doctor note's error analysis: surgery, pacing, severe left stenotic valve disease, MI history, bundle branch block, ventricular hypertrophy, bedside critical, and explicit problem flags.",
            "exclude_if_any": [
                "is_problem",
                "L_一_1_大血管_瓣膜_冠脉手术",
                "L_一_2_起搏大类",
                "L_二_2_左心重度狭窄瓣膜病",
                "L_三_1_梗死史",
                "L_四_1_束支阻滞",
                "L_二_6_心室肥厚",
                "L_四_4_床旁重症",
            ],
        },
        "low_interference": {
            "description": "Exclude the most obvious signal and clinical interference groups: surgery/intervention/device, bundle branch block, large pericardial effusion, bedside critical, and explicit problem flags.",
            "exclude_if_any": [
                "is_problem",
                "L_一_1_大血管_瓣膜_冠脉手术",
                "L_一_2_起搏大类",
                "L_一_3_消融术后",
                "L_一_4_左心耳干预",
                "L_四_1_束支阻滞",
                "L_四_3_大量心包积液",
                "L_四_4_床旁重症",
            ],
        },
        "cardiac_clean": {
            "description": "Further exclude strong right-heart-load and right-sided confounders after the low-interference filtering.",
            "exclude_if_any": [
                "is_problem",
                "L_一_1_大血管_瓣膜_冠脉手术",
                "L_一_2_起搏大类",
                "L_一_3_消融术后",
                "L_一_4_左心耳干预",
                "L_四_1_束支阻滞",
                "L_四_3_大量心包积液",
                "L_四_4_床旁重症",
                "L_二_3_右心大_肺动脉高压",
                "L_二_4_右心重度瓣膜病",
                "L_二_5_仅右房增大",
            ],
        },
        "left_ventricular_clean": {
            "description": "A stricter pilot subset that additionally removes MI history and ventricular hypertrophy, leaving a smaller but cleaner left-ventricular-focused task.",
            "exclude_if_any": [
                "is_problem",
                "L_一_1_大血管_瓣膜_冠脉手术",
                "L_一_2_起搏大类",
                "L_一_3_消融术后",
                "L_一_4_左心耳干预",
                "L_四_1_束支阻滞",
                "L_四_3_大量心包积液",
                "L_四_4_床旁重症",
                "L_二_3_右心大_肺动脉高压",
                "L_二_4_右心重度瓣膜病",
                "L_二_5_仅右房增大",
                "L_三_1_梗死史",
                "L_二_6_心室肥厚",
            ],
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create filtered HF subtask manifests from the refreshed 20260322 manifest."
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data/manifests/hf_manifest_combined_20260322_ascii.csv"),
        help="Base refreshed manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directory to write subtask manifest CSVs.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="hf_manifest_combined_20260322",
        help="Filename prefix for generated subtask manifests.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    df = pd.read_csv(args.manifest_csv)
    available_cols = set(df.columns)

    flag_cols = [col for col in df.columns if col == "is_problem" or col.startswith("L_")]
    for col in flag_cols:
        df[col] = df[col].map(_normalize_flag)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = _subtask_specs()
    summary = {
        "base_manifest": str(args.manifest_csv),
        "base_rows": int(len(df)),
        "base_class_distribution": _value_distribution(df["class"]),
        "subtasks": {},
    }

    for subtask_name, spec in specs.items():
        missing_cols = sorted(col for col in spec["exclude_if_any"] if col not in available_cols)
        if missing_cols:
            raise ValueError(
                "Subtask '{}' requires missing columns: {}".format(subtask_name, ", ".join(missing_cols))
            )

        exclude_mask = df[spec["exclude_if_any"]].sum(axis=1) > 0
        sub_df = df.loc[~exclude_mask].copy()
        out_path = args.output_dir / "{}_subtask_{}.csv".format(args.prefix, subtask_name)
        sub_df.to_csv(out_path, index=False)

        summary["subtasks"][subtask_name] = {
            "description": spec["description"],
            "exclude_if_any": list(spec["exclude_if_any"]),
            "rows": int(len(sub_df)),
            "excluded_rows": int(exclude_mask.sum()),
            "class_distribution": _value_distribution(sub_df["class"]),
            "output_csv": str(out_path),
        }

    summary_path = args.output_dir / "{}_subtasks_summary.json".format(args.prefix)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
