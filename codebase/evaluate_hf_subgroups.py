import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def _safe_ovr_auc(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true_bin).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true_bin, y_score))


def _safe_ovr_auprc(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true_bin).size < 2:
        return float("nan")
    return float(average_precision_score(y_true_bin, y_score))


def _json_ready(value):
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _normalize_flag(value) -> int:
    if pd.isna(value):
        return 0
    s = str(value).strip()
    if s in {"1", "1.0", "True", "true"}:
        return 1
    return 0


def _class_distribution(series: pd.Series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in Counter(series.astype(int).tolist()).items()}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
) -> Dict:
    num_classes = y_prob.shape[1]
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "class_distribution": {class_names[i]: int((y_true == i).sum()) for i in range(num_classes)},
    }

    per_class = []
    aurocs = []
    auprcs = []
    for class_idx, class_name in enumerate(class_names):
        y_bin = (y_true == class_idx).astype(np.int64)
        auc = _safe_ovr_auc(y_bin, y_prob[:, class_idx])
        auprc = _safe_ovr_auprc(y_bin, y_prob[:, class_idx])
        aurocs.append(auc)
        auprcs.append(auprc)
        per_class.append(
            {
                "class_idx": int(class_idx),
                "class_name": class_name,
                "n_true": int(y_bin.sum()),
                "auroc_ovr": auc,
                "auprc_ovr": auprc,
            }
        )
    metrics["macro_auroc"] = float(np.nanmean(aurocs)) if aurocs else float("nan")
    metrics["macro_auprc"] = float(np.nanmean(auprcs)) if auprcs else float("nan")
    metrics["per_class"] = per_class
    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate HF inference predictions on subgroup columns such as is_problem and L_* labels."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True, help="predictions.csv from MIL inference.")
    parser.add_argument("--manifest-csv", type=Path, required=True, help="Manifest CSV used for inference.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for subgroup reports.")
    parser.add_argument(
        "--merge-key",
        type=str,
        default="xml_file",
        help="Manifest/prediction join key. Default is xml_file.",
    )
    parser.add_argument(
        "--subgroup-columns",
        nargs="*",
        default=None,
        help="Explicit subgroup columns. Defaults to is_problem plus all L_* columns found in the manifest.",
    )
    parser.add_argument("--min-subgroup-size", type=int, default=20, help="Minimum rows required to score a subgroup.")
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional class names. Defaults to prob_* headers or class_0..class_C-1.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    pred_df = pd.read_csv(args.predictions_csv)
    manifest_df = pd.read_csv(args.manifest_csv)

    if args.merge_key not in pred_df.columns or args.merge_key not in manifest_df.columns:
        raise ValueError("merge key '{}' must exist in both CSVs.".format(args.merge_key))

    prob_cols = [col for col in pred_df.columns if col.startswith("prob_")]
    if not prob_cols:
        raise ValueError("predictions CSV must contain probability columns with prefix 'prob_'.")

    class_names = list(args.class_names) if args.class_names else [col[len("prob_"):] for col in prob_cols]
    if len(class_names) != len(prob_cols):
        raise ValueError("Number of class names must match probability columns.")

    keep_manifest_cols = [args.merge_key]
    if args.subgroup_columns is None:
        subgroup_columns = []
        if "is_problem" in manifest_df.columns:
            subgroup_columns.append("is_problem")
        subgroup_columns.extend(sorted(col for col in manifest_df.columns if col.startswith("L_")))
    else:
        subgroup_columns = list(args.subgroup_columns)
    for col in subgroup_columns:
        if col not in manifest_df.columns:
            raise ValueError("Manifest is missing subgroup column '{}'.".format(col))
        keep_manifest_cols.append(col)

    merged_df = pred_df.merge(
        manifest_df[keep_manifest_cols].drop_duplicates(subset=[args.merge_key]),
        on=args.merge_key,
        how="left",
        validate="many_to_one",
    )

    y_true = merged_df["y_true"].astype(int).to_numpy()
    y_pred = merged_df["y_pred"].astype(int).to_numpy()
    y_prob = merged_df[prob_cols].to_numpy(dtype=np.float64)

    overall_metrics = compute_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, class_names=class_names)

    subgroup_rows: List[Dict] = []
    subgroup_json: Dict[str, Dict] = {}
    for subgroup_col in subgroup_columns:
        subgroup_values = merged_df[subgroup_col].map(_normalize_flag)
        subgroup_mask = subgroup_values == 1
        subgroup_n = int(subgroup_mask.sum())
        if subgroup_n < args.min_subgroup_size:
            subgroup_json[subgroup_col] = {
                "n": subgroup_n,
                "skipped": True,
                "reason": "n < min_subgroup_size",
            }
            continue

        sub_metrics = compute_metrics(
            y_true=y_true[subgroup_mask.to_numpy()],
            y_pred=y_pred[subgroup_mask.to_numpy()],
            y_prob=y_prob[subgroup_mask.to_numpy()],
            class_names=class_names,
        )
        subgroup_json[subgroup_col] = sub_metrics
        subgroup_rows.append(
            {
                "subgroup": subgroup_col,
                "n": sub_metrics["n"],
                "accuracy": sub_metrics["accuracy"],
                "balanced_accuracy": sub_metrics["balanced_accuracy"],
                "macro_f1": sub_metrics["macro_f1"],
                "macro_auroc": sub_metrics["macro_auroc"],
                "macro_auprc": sub_metrics["macro_auprc"],
                "class_distribution": json.dumps(sub_metrics["class_distribution"], ensure_ascii=False),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "predictions_csv": str(args.predictions_csv),
        "manifest_csv": str(args.manifest_csv),
        "merge_key": args.merge_key,
        "overall": overall_metrics,
        "subgroups": subgroup_json,
        "available_subgroups": subgroup_columns,
        "min_subgroup_size": int(args.min_subgroup_size),
    }
    (args.out_dir / "subgroup_metrics.json").write_text(
        json.dumps(_json_ready(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(subgroup_rows).to_csv(args.out_dir / "subgroup_metrics.csv", index=False)
    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
