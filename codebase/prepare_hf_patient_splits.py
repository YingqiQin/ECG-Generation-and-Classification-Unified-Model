import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _normalize_flag(value) -> int:
    if pd.isna(value):
        return 0
    s = str(value).strip()
    if s in {"1", "1.0", "True", "true"}:
        return 1
    return 0


def _allocate_counts(n: int, ratios: Sequence[float]) -> List[int]:
    raw = np.asarray(ratios, dtype=np.float64) * float(n)
    counts = np.floor(raw).astype(int)
    remainder = int(n - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts.tolist()


def _stratified_patient_split(
    patient_rows: pd.DataFrame,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    patient_to_split: Dict[str, str] = {}
    strata_to_patients: Dict[str, List[str]] = defaultdict(list)

    for _, row in patient_rows.iterrows():
        strata_to_patients[row["split_stratum"]].append(row["patient_id"])

    for stratum, patient_ids in sorted(strata_to_patients.items()):
        patient_ids = list(patient_ids)
        rng.shuffle(patient_ids)
        n = len(patient_ids)
        if n == 1:
            counts = [1, 0, 0]
        elif n == 2:
            counts = [1, 0, 1]
        else:
            counts = _allocate_counts(n, [train_ratio, valid_ratio, test_ratio])
            while counts[0] <= 0:
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[0] += 1
            if n >= 3 and counts[2] <= 0:
                donor = int(np.argmax(counts))
                counts[donor] -= 1
                counts[2] += 1

        start = 0
        for split_name, count in zip(["train", "valid", "test"], counts):
            for patient_id in patient_ids[start:start + count]:
                patient_to_split[patient_id] = split_name
            start += count

    split_map = {"train": [], "valid": [], "test": []}
    for patient_id, split_name in patient_to_split.items():
        split_map[split_name].append(patient_id)
    return split_map


def _value_distribution(series: pd.Series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in Counter(series.astype(str).tolist()).items()}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create reproducible patient-level train/valid/test splits from an HF manifest."
    )
    parser.add_argument("--manifest-csv", type=Path, required=True, help="Input HF manifest CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for split CSVs.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Patient-level train ratio.")
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Patient-level valid ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Patient-level test ratio.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--patient-col", type=str, default="patient_id", help="Patient ID column.")
    parser.add_argument("--label-col", type=str, default="class", help="HF class column.")
    parser.add_argument("--source-col", type=str, default="source", help="Source/site column.")
    parser.add_argument(
        "--problem-col",
        type=str,
        default="is_problem",
        help="Optional problem-flag column used only for reporting.",
    )
    parser.add_argument(
        "--exclude-problem",
        action="store_true",
        help="Drop rows where --problem-col indicates a flagged case before splitting.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    ratio_sum = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/valid/test ratios must sum to 1.0, got {}.".format(ratio_sum))

    df = pd.read_csv(args.manifest_csv)
    required_cols = {args.patient_col, args.label_col}
    missing = sorted(col for col in required_cols if col not in df.columns)
    if missing:
        raise ValueError("Manifest is missing required columns: {}.".format(", ".join(missing)))

    df = df.copy()
    df[args.patient_col] = df[args.patient_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(int)

    if args.exclude_problem and args.problem_col in df.columns:
        df = df[df[args.problem_col].map(_normalize_flag) == 0].reset_index(drop=True)

    patient_rows = []
    for patient_id, group in df.groupby(args.patient_col):
        max_class = int(group[args.label_col].max())
        source = "unknown"
        if args.source_col in group.columns:
            source_values = sorted({str(v) for v in group[args.source_col].dropna().tolist()})
            source = source_values[0] if len(source_values) == 1 else "multi"
        patient_rows.append(
            {
                "patient_id": str(patient_id),
                "patient_max_class": max_class,
                "source": source,
                "n_records": int(len(group)),
                "split_stratum": "{}__class{}".format(source, max_class),
            }
        )
    patient_df = pd.DataFrame(patient_rows)
    if patient_df.empty:
        raise ValueError("No rows available after preprocessing.")

    split_patients = _stratified_patient_split(
        patient_rows=patient_df,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_paths: Dict[str, str] = {}
    summary = {
        "input_manifest": str(args.manifest_csv),
        "seed": int(args.seed),
        "exclude_problem": bool(args.exclude_problem),
        "ratios": {
            "train": float(args.train_ratio),
            "valid": float(args.valid_ratio),
            "test": float(args.test_ratio),
        },
        "splits": {},
    }

    for split_name in ["train", "valid", "test"]:
        patient_ids = set(split_patients[split_name])
        split_df = df[df[args.patient_col].astype(str).isin(patient_ids)].copy()
        split_df = split_df.sort_values([args.patient_col, "ecg_time", "xml_file"], na_position="last")
        out_path = args.output_dir / "{}.csv".format(split_name)
        split_df.to_csv(out_path, index=False)
        split_paths[split_name] = str(out_path)

        patient_subset = patient_df[patient_df["patient_id"].isin(patient_ids)]
        split_info = {
            "rows": int(len(split_df)),
            "patients": int(len(patient_subset)),
            "class_distribution": _value_distribution(split_df[args.label_col]),
            "patient_max_class_distribution": _value_distribution(patient_subset["patient_max_class"]),
            "source_distribution": _value_distribution(split_df[args.source_col]) if args.source_col in split_df.columns else {},
        }
        if args.problem_col in split_df.columns:
            split_info["problem_distribution"] = _value_distribution(split_df[args.problem_col].map(_normalize_flag))
        summary["splits"][split_name] = split_info

    summary["output_files"] = split_paths
    summary_path = args.output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
