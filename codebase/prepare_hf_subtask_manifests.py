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


def _apply_spec_mask(df: pd.DataFrame, spec: Dict) -> pd.Series:
    exclude_mask = pd.Series(False, index=df.index)

    if spec.get("exclude_if_any"):
        exclude_mask = exclude_mask | (df[spec["exclude_if_any"]].sum(axis=1) > 0)

    bnp_col = spec.get("bnp_col")
    if bnp_col:
        if spec.get("negative_class_bnp_lt") is not None:
            threshold = float(spec["negative_class_bnp_lt"])
            class_value = int(spec.get("negative_class_value", 0))
            exclude_mask = exclude_mask | (
                (df["class"] == class_value) & ~(df[bnp_col] < threshold)
            )
        if spec.get("class_bnp_ge"):
            for class_key, threshold in spec["class_bnp_ge"].items():
                exclude_mask = exclude_mask | (
                    (df["class"] == int(class_key)) & ~(df[bnp_col] >= float(threshold))
                )
        if spec.get("exclude_if_bnp_gt") is not None:
            exclude_mask = exclude_mask | (df[bnp_col] > float(spec["exclude_if_bnp_gt"]))

    lvef_col = spec.get("lvef_col")
    if lvef_col and spec.get("class_lvef_ranges"):
        for class_key, bounds in spec["class_lvef_ranges"].items():
            lower = bounds.get("gt")
            upper = bounds.get("lt")
            upper_inclusive = bounds.get("le")
            cond = pd.Series(True, index=df.index)
            if lower is not None:
                cond = cond & (df[lvef_col] > float(lower))
            if upper is not None:
                cond = cond & (df[lvef_col] < float(upper))
            if upper_inclusive is not None:
                cond = cond & (df[lvef_col] <= float(upper_inclusive))
            exclude_mask = exclude_mask | ((df["class"] == int(class_key)) & ~cond)

    return exclude_mask


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
        "doctor_bnp_negative_pure": {
            "description": "Doctor-guided BNP purity subset: keep a row as class 0 only when BNP < 300 pg/mL. Class-0 rows with BNP >= 300 or missing BNP are excluded.",
            "exclude_if_any": [],
            "bnp_col": "bnp_value",
            "negative_class_value": 0,
            "negative_class_bnp_lt": 300.0,
        },
        "doctor_bnp_negative_pure_extreme_30000": {
            "description": "Doctor-guided BNP purity subset plus removal of very extreme BNP values above 30000 pg/mL.",
            "exclude_if_any": [],
            "bnp_col": "bnp_value",
            "negative_class_value": 0,
            "negative_class_bnp_lt": 300.0,
            "exclude_if_bnp_gt": 30000.0,
        },
        "doc_error_reduced_doctor_bnp_negative_pure_extreme_30000": {
            "description": "Combine doc_error_reduced with the practical doctor BNP rule: class 0 requires BNP < 300 and very extreme BNP > 30000 is excluded.",
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
            "bnp_col": "bnp_value",
            "negative_class_value": 0,
            "negative_class_bnp_lt": 300.0,
            "exclude_if_bnp_gt": 30000.0,
        },
        "low_interference_doctor_bnp_negative_pure_extreme_30000": {
            "description": "Combine low_interference with the practical doctor BNP rule: class 0 requires BNP < 300 and very extreme BNP > 30000 is excluded.",
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
            "bnp_col": "bnp_value",
            "negative_class_value": 0,
            "negative_class_bnp_lt": 300.0,
            "exclude_if_bnp_gt": 30000.0,
        },
        "cardiac_clean_doctor_bnp_negative_pure_extreme_30000": {
            "description": "Combine cardiac_clean with the practical doctor BNP rule: class 0 requires BNP < 300 and very extreme BNP > 30000 is excluded.",
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
            "bnp_col": "bnp_value",
            "negative_class_value": 0,
            "negative_class_bnp_lt": 300.0,
            "exclude_if_bnp_gt": 30000.0,
        },
        "left_ventricular_clean_doctor_bnp_negative_pure_extreme_30000": {
            "description": "Combine left_ventricular_clean with the practical doctor BNP rule: class 0 requires BNP < 300 and very extreme BNP > 30000 is excluded.",
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
            "bnp_col": "bnp_value",
            "negative_class_value": 0,
            "negative_class_bnp_lt": 300.0,
            "exclude_if_bnp_gt": 30000.0,
        },
        "doctor_bnp_rule_partial_extreme_30000": {
            "description": "Partial operationalization of the doctor BNP rule using available structured fields only: class 0 requires BNP < 300, class 1 requires BNP >= 900, class 2 requires 40 < LVEF < 50, class 3 requires LVEF <= 40, and BNP > 30000 is excluded. This is partial because the manifest does not contain structured echo wall-thickness/LAD measurements needed for the full HFpEF rule.",
            "exclude_if_any": [],
            "bnp_col": "bnp_value",
            "lvef_col": "LVEF",
            "negative_class_value": 0,
            "negative_class_bnp_lt": 300.0,
            "class_bnp_ge": {
                "1": 900.0,
            },
            "class_lvef_ranges": {
                "2": {"gt": 40.0, "lt": 50.0},
                "3": {"le": 40.0},
            },
            "exclude_if_bnp_gt": 30000.0,
        },
        "low_interference_doctor_bnp_rule_partial_extreme_30000": {
            "description": "Combine the existing low-interference filtering with the partial doctor BNP/LVEF rule and exclude very extreme BNP > 30000 pg/mL. This is the easiest BNP-aware subset and should be treated as a reduced-difficulty subtask.",
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
            "bnp_col": "bnp_value",
            "lvef_col": "LVEF",
            "negative_class_value": 0,
            "negative_class_bnp_lt": 300.0,
            "class_bnp_ge": {
                "1": 900.0,
            },
            "class_lvef_ranges": {
                "2": {"gt": 40.0, "lt": 50.0},
                "3": {"le": 40.0},
            },
            "exclude_if_bnp_gt": 30000.0,
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
    for col in ["class", "LVEF", "bnp_value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
        for optional_key in ["bnp_col", "lvef_col"]:
            col_name = spec.get(optional_key)
            if col_name is not None and col_name not in available_cols:
                missing_cols.append(col_name)
        missing_cols = sorted(set(missing_cols))
        if missing_cols:
            raise ValueError(
                "Subtask '{}' requires missing columns: {}".format(subtask_name, ", ".join(missing_cols))
            )

        exclude_mask = _apply_spec_mask(df, spec)
        sub_df = df.loc[~exclude_mask].copy()
        out_path = args.output_dir / "{}_subtask_{}.csv".format(args.prefix, subtask_name)
        sub_df.to_csv(out_path, index=False)

        summary["subtasks"][subtask_name] = {
            "description": spec["description"],
            "exclude_if_any": list(spec["exclude_if_any"]),
            "negative_class_bnp_lt": spec.get("negative_class_bnp_lt"),
            "class_bnp_ge": spec.get("class_bnp_ge"),
            "class_lvef_ranges": spec.get("class_lvef_ranges"),
            "exclude_if_bnp_gt": spec.get("exclude_if_bnp_gt"),
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
