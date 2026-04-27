import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ecg_dataset import ECGMILDataset, lead_mode_num_channels
from net1d import build_net1d_backbone
from one_d_efficientnet import ECGModel_Attn


def _safe_divide(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


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
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _save_confusion_matrix_csv(cm: np.ndarray, class_names: Sequence[str], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *class_names])
        for idx, row in enumerate(cm):
            writer.writerow([class_names[idx], *row.tolist()])


def _save_per_class_metrics_csv(per_class: Sequence[Dict], out_path: Path) -> None:
    if not per_class:
        return
    fieldnames = list(per_class[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_class:
            writer.writerow(_json_ready(row))


def _save_predictions_csv(
    all_y: np.ndarray,
    all_pred: np.ndarray,
    all_prob: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
    all_lvef: Optional[Sequence[Optional[float]]] = None,
    all_xml: Optional[Sequence[str]] = None,
    all_patient_id: Optional[Sequence[str]] = None,
) -> None:
    prob_headers = [f"prob_{name}" for name in class_names]
    headers = ["index", "y_true", "y_pred", *prob_headers, "LVEF", "xml_file", "patient_id"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for idx in range(len(all_y)):
            lvef = "" if all_lvef is None else all_lvef[idx]
            xml_file = "" if all_xml is None else all_xml[idx]
            patient_id = "" if all_patient_id is None else all_patient_id[idx]
            writer.writerow(
                [
                    idx,
                    int(all_y[idx]),
                    int(all_pred[idx]),
                    *all_prob[idx].tolist(),
                    lvef,
                    xml_file,
                    patient_id,
                ]
            )


def compute_metrics(
    all_y,
    all_prob,
    all_pred,
    num_classes: Optional[int] = None,
    out_dir: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
):
    all_y = np.asarray(all_y, dtype=np.int64)
    all_prob = np.asarray(all_prob, dtype=np.float64)
    all_pred = np.asarray(all_pred, dtype=np.int64)

    if all_prob.ndim != 2:
        raise ValueError(f"Expected all_prob to have shape [N, C], got {all_prob.shape}")
    if all_y.ndim != 1 or all_pred.ndim != 1:
        raise ValueError(f"Expected all_y/all_pred to have shape [N], got {all_y.shape} / {all_pred.shape}")
    if not (len(all_y) == len(all_pred) == all_prob.shape[0]):
        raise ValueError("Length mismatch among all_y, all_pred, and all_prob.")

    inferred_num_classes = all_prob.shape[1]
    if num_classes is None:
        num_classes = inferred_num_classes
    if num_classes != inferred_num_classes:
        raise ValueError(f"num_classes={num_classes} does not match probability shape {all_prob.shape}.")

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError("Length of class_names must match num_classes.")

    cm = confusion_matrix(all_y, all_pred, labels=np.arange(num_classes))
    macro_f1 = float(f1_score(all_y, all_pred, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(all_y, all_pred))
    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        all_y,
        all_pred,
        labels=np.arange(num_classes),
        average="macro",
        zero_division=0,
    )

    per_class_metrics: List[Dict] = []
    per_class_aurocs: List[float] = []
    per_class_auprcs: List[float] = []
    specificities: List[float] = []

    for class_idx in range(num_classes):
        y_true_bin = (all_y == class_idx).astype(np.int64)
        y_pred_bin = (all_pred == class_idx).astype(np.int64)

        tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
        fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
        fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        f1_val = _safe_divide(2 * precision * recall, precision + recall)
        auroc = _safe_ovr_auc(y_true_bin, all_prob[:, class_idx])
        auprc = _safe_ovr_auprc(y_true_bin, all_prob[:, class_idx])

        per_class_aurocs.append(auroc)
        per_class_auprcs.append(auprc)
        specificities.append(specificity)

        per_class_metrics.append(
            {
                "class_idx": class_idx,
                "class_name": class_names[class_idx],
                "support": int(np.sum(y_true_bin)),
                "ovr_auroc": auroc,
                "ovr_auprc": auprc,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1_val,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )

    valid_aurocs = [v for v in per_class_aurocs if not np.isnan(v)]
    valid_auprcs = [v for v in per_class_auprcs if not np.isnan(v)]

    metrics = {
        "num_samples": int(len(all_y)),
        "num_classes": int(num_classes),
        "macro_auroc": float(np.mean(valid_aurocs)) if valid_aurocs else None,
        "macro_auprc": float(np.mean(valid_auprcs)) if valid_auprcs else None,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_specificity": float(np.mean(specificities)),
        "accuracy": float(np.mean(all_y == all_pred)),
        "confusion_matrix": cm,
        "per_class": per_class_metrics,
    }

    if out_dir is not None:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        summary = {
            "num_samples": metrics["num_samples"],
            "num_classes": metrics["num_classes"],
            "macro_auroc": metrics["macro_auroc"],
            "macro_auprc": metrics["macro_auprc"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_specificity": metrics["macro_specificity"],
            "accuracy": metrics["accuracy"],
        }
        (out_dir_path / "metrics_summary.json").write_text(
            json.dumps(_json_ready(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir_path / "metrics_full.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _save_confusion_matrix_csv(cm, class_names, out_dir_path / "confusion_matrix.csv")
        _save_per_class_metrics_csv(per_class_metrics, out_dir_path / "per_class_metrics.csv")

    return _json_ready(metrics)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "net", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if isinstance(checkpoint, dict) and checkpoint:
        first_key = next(iter(checkpoint.keys()))
        if isinstance(first_key, str) and first_key.startswith("module."):
            checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}

    return checkpoint


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def testing(args):
    device = _resolve_device(args.device)
    class_names = args.class_names or [f"class_{i}" for i in range(args.num_classes)]
    if len(class_names) != args.num_classes:
        raise ValueError("Length of --class-names must match --num-classes.")

    encoder = build_net1d_backbone(
        in_channels=args.in_channels,
        embedding_dim=512,
        preset="ecgfounder_large",
        num_classes=None,
        use_bn=None,
        use_do=None,
    )
    model = ECGModel_Attn(
        encoder,
        512,
        out_dim=args.num_classes,
        pool_type=args.pool_type,
        quality_dim=args.quality_dim,
        quality_alpha=args.quality_alpha,
        topk=args.topk,
        mix_beta=args.mix_beta,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=args.strict_ckpt)
    model.eval()

    test_dataset = ECGMILDataset(
        csv_path=args.csv_path,
        npy_root=args.root,
        base_seed=args.base_seed,
        K=args.K,
        seg_sec=args.seg_sec,
        deterministic=True,
        sample_mode="uniform",
        lead_mode=args.lead_mode,
        return_dict=True,
        return_seg_quality=args.pool_type != "attention",
        return_seg_starts=args.save_pool_outputs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    all_y: List[int] = []
    all_prob: List[np.ndarray] = []
    all_lvef: List[Optional[float]] = []
    all_xml: List[str] = []
    all_patient_id: List[str] = []
    all_weights: List[np.ndarray] = []
    all_seg_quality: List[np.ndarray] = []
    all_seg_starts: List[np.ndarray] = []
    all_evidence_logits: List[np.ndarray] = []
    all_quality_logits: List[np.ndarray] = []
    all_combined_logits: List[np.ndarray] = []
    all_topk_indices: List[np.ndarray] = []
    all_topk_valid: List[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            signal = batch["signal"]
            labels = batch["label"]
            seg_mask = batch.get("seg_mask")
            seg_quality = batch.get("seg_quality")

            if signal.ndim != 4:
                raise ValueError(
                    f"Expected MIL signal shape [B, K, C, L], got {tuple(signal.shape)}. "
                    "This script is inference-only for bag-level MIL inputs."
                )

            signal = signal.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if seg_mask is not None:
                seg_mask = seg_mask.to(device, non_blocking=True)
            if seg_quality is not None:
                seg_quality = seg_quality.to(device, non_blocking=True)

            if args.save_pool_outputs:
                pred, weights, aux = model(
                    signal,
                    mask=seg_mask,
                    seg_quality=seg_quality,
                    return_aux=True,
                )
                all_weights.extend(weights.cpu().numpy())
                all_evidence_logits.extend(aux["evidence_logits"].cpu().numpy())
                all_combined_logits.extend(aux["combined_logits"].cpu().numpy())
                if "quality_logits" in aux:
                    all_quality_logits.extend(aux["quality_logits"].cpu().numpy())
                if "topk_indices" in aux:
                    all_topk_indices.extend(aux["topk_indices"].cpu().numpy())
                    all_topk_valid.extend(aux["topk_valid"].cpu().numpy())
            else:
                pred, _ = model(signal, mask=seg_mask, seg_quality=seg_quality)
            y_prob = torch.softmax(pred, dim=1)

            all_y.extend(labels.cpu().numpy().tolist())
            all_prob.extend(y_prob.cpu().numpy())
            all_lvef.extend(batch["lvef"].cpu().numpy().tolist())
            all_xml.extend(list(batch["xml_file"]))
            all_patient_id.extend(list(batch["patient_id"]))
            if "seg_quality" in batch:
                all_seg_quality.extend(batch["seg_quality"].cpu().numpy())
            if "seg_starts" in batch:
                all_seg_starts.extend(batch["seg_starts"].cpu().numpy())

    all_prob = np.asarray(all_prob, dtype=np.float64)
    all_y = np.asarray(all_y, dtype=np.int64)
    all_pred = np.argmax(all_prob, axis=1)

    metrics = compute_metrics(
        all_y,
        all_prob,
        all_pred,
        num_classes=args.num_classes,
        out_dir=args.out_dir,
        class_names=class_names,
    )

    out_dir_path = Path(args.out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    _save_predictions_csv(
        all_y=all_y,
        all_pred=all_pred,
        all_prob=all_prob,
        class_names=class_names,
        out_path=out_dir_path / "predictions.csv",
        all_lvef=all_lvef if all_lvef else None,
        all_xml=all_xml if all_xml else None,
        all_patient_id=all_patient_id if all_patient_id else None,
    )
    np.save(out_dir_path / "probabilities.npy", all_prob)
    np.save(out_dir_path / "labels.npy", all_y)
    np.save(out_dir_path / "predictions.npy", all_pred)
    if all_weights:
        np.save(out_dir_path / "attention_weights.npy", np.asarray(all_weights, dtype=np.float32))
    if all_seg_quality:
        np.save(out_dir_path / "segment_quality.npy", np.asarray(all_seg_quality, dtype=np.float32))
    if all_seg_starts:
        np.save(out_dir_path / "segment_starts.npy", np.asarray(all_seg_starts, dtype=np.int64))
    if all_evidence_logits:
        np.save(out_dir_path / "evidence_logits.npy", np.asarray(all_evidence_logits, dtype=np.float32))
    if all_quality_logits:
        np.save(out_dir_path / "quality_logits.npy", np.asarray(all_quality_logits, dtype=np.float32))
    if all_combined_logits:
        np.save(out_dir_path / "combined_logits.npy", np.asarray(all_combined_logits, dtype=np.float32))
    if all_topk_indices:
        np.save(out_dir_path / "topk_indices.npy", np.asarray(all_topk_indices, dtype=np.int64))
    if all_topk_valid:
        np.save(out_dir_path / "topk_valid.npy", np.asarray(all_topk_valid, dtype=np.bool_))

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Inference-only MIL classification entrypoint for ECG heart failure classification."
    )
    parser.add_argument("--csv-path", type=Path, required=True, help="CSV manifest used to build the inference dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory that contains ECG .npy files.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Model checkpoint for the full MIL model.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save metrics and predictions.")

    parser.add_argument("--num-classes", type=int, required=True, help="Number of target classes.")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Optional class names in class-index order. Length must equal --num-classes.",
    )

    parser.add_argument("--in-channels", type=int, default=12, help="Encoder input channels.")
    parser.add_argument(
        "--lead-mode",
        type=str,
        default="12",
        help="Lead selection mode. Examples: 12, I, II, I_1ch, II_1ch, V2_1ch.",
    )
    parser.add_argument("--seg-sec", type=float, default=4.0, help="Segment length in seconds.")
    parser.add_argument("--K", type=int, default=16, help="Number of segments per bag.")
    parser.add_argument("--base-seed", type=int, default=1234, help="Seed used for deterministic inference sampling.")
    parser.add_argument(
        "--pool-type",
        type=str,
        default="attention",
        choices=["attention", "quality_attention", "hybrid"],
        help="Pooling head to use when building the MIL model.",
    )
    parser.add_argument("--quality-dim", type=int, default=4, help="Number of segment-quality features.")
    parser.add_argument("--quality-alpha", type=float, default=1.0, help="Scaling factor for quality logits.")
    parser.add_argument("--topk", type=int, default=4, help="Top-k used by hybrid pooling.")
    parser.add_argument("--mix-beta", type=float, default=0.5, help="Attention/top-k mixing ratio for hybrid pooling.")

    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--strict-ckpt",
        action="store_true",
        help="Require an exact checkpoint key match when loading the full MIL model.",
    )
    parser.add_argument(
        "--save-pool-outputs",
        action="store_true",
        help="Save attention weights and pooling diagnostics for later analysis.",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    expected_in_channels = lead_mode_num_channels(args.lead_mode)
    if args.in_channels != expected_in_channels:
        raise ValueError(
            "--in-channels={} does not match lead mode '{}' which produces {} channel(s).".format(
                args.in_channels,
                args.lead_mode,
                expected_in_channels,
            )
        )
    testing(args)


if __name__ == "__main__":
    main()
