import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from ecg_dataset import ECGMILDataset, lead_mode_num_channels
from MIL_classification_inference import _extract_state_dict, compute_metrics
from net1d import build_net1d_backbone, load_net1d_checkpoint_flexible
from one_d_efficientnet import ECGModel_Attn
from tensorboard_utils import add_confusion_matrix_text, add_per_class_metrics, add_scalar_dict, create_summary_writer, flush_and_close


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _progress_enabled(args) -> bool:
    return (not args.no_tqdm) and tqdm is not None


def _make_progress(loader: DataLoader, args, desc: str):
    if not _progress_enabled(args):
        return loader, None
    total = len(loader) if hasattr(loader, "__len__") else None
    bar = tqdm(loader, total=total, desc=desc, dynamic_ncols=True, leave=False, mininterval=0.5)
    return bar, bar


def _update_progress(bar, metrics: Dict[str, float]) -> None:
    if bar is None:
        return
    bar.set_postfix({k: round(float(v), 4) for k, v in metrics.items()}, refresh=False)


def _check_finite_tensor(name: str, tensor: torch.Tensor, batch_ids=None) -> None:
    if torch.isfinite(tensor).all():
        return
    bad_count = int((~torch.isfinite(tensor)).sum().item())
    context = ""
    if batch_ids:
        preview = [str(x) for x in batch_ids[:3]]
        context = " batch_ids={}".format(preview)
    raise ValueError("Non-finite values found in '{}' (count={}){}.".format(name, bad_count, context))


def _load_sample_weights(
    csv_path: Optional[Path],
    key_name: str,
    value_name: str,
    min_value: float,
    max_value: float,
) -> Dict[str, float]:
    if csv_path is None:
        return {}

    weights: Dict[str, float] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if key_name not in reader.fieldnames or value_name not in reader.fieldnames:
            raise ValueError(
                "Weight CSV must contain columns '{}' and '{}'.".format(key_name, value_name)
            )
        for row in reader:
            key = str(row[key_name])
            try:
                value = float(row[value_name])
            except ValueError:
                continue
            value = max(min_value, min(max_value, value))
            weights[key] = value
    return weights


def _batch_sample_weights(
    batch: Dict,
    weight_map: Dict[str, float],
    key_name: str,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not weight_map:
        return None

    values = batch.get(key_name)
    if values is None:
        raise ValueError("Batch does not contain weight key '{}'.".format(key_name))

    weights = [float(weight_map.get(str(v), 1.0)) for v in values]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _build_model(args, device: torch.device) -> ECGModel_Attn:
    encoder = build_net1d_backbone(
        in_channels=args.in_channels,
        embedding_dim=args.embedding_dim,
        preset=args.encoder_preset,
        num_classes=None,
        use_bn=None,
        use_do=None,
    )
    model = ECGModel_Attn(
        encoder=encoder,
        d_embed=args.embedding_dim,
        out_dim=args.num_classes,
        pool_type=args.pool_type,
        pool_hidden=args.pool_hidden,
        dropout=args.dropout,
        quality_dim=args.quality_dim,
        quality_hidden=args.quality_hidden,
        quality_alpha=args.quality_alpha,
        topk=args.topk,
        mix_beta=args.mix_beta,
    ).to(device)

    if args.encoder_ckpt:
        missing, unexpected, adapted = load_net1d_checkpoint_flexible(
            model.encoder,
            str(args.encoder_ckpt),
            strict=False,
            adapt_input_channels=True,
        )
        print(
            json.dumps(
                {
                    "encoder_ckpt": str(args.encoder_ckpt),
                    "encoder_missing_keys": list(missing),
                    "encoder_unexpected_keys": list(unexpected),
                    "encoder_adapted_keys": list(adapted),
                },
                indent=2,
            )
        )

    if args.init_ckpt:
        checkpoint = torch.load(args.init_ckpt, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=args.strict_ckpt)

    return model


def _make_dataloader(csv_path: Path, args, shuffle: bool, is_train: bool, device: torch.device) -> DataLoader:
    dataset = ECGMILDataset(
        csv_path=csv_path,
        npy_root=args.root,
        base_seed=args.base_seed,
        K=args.K,
        seg_sec=args.seg_sec,
        deterministic=not is_train,
        sample_mode=args.train_sample_mode if is_train else "uniform",
        lead_mode=args.lead_mode,
        return_dict=True,
        return_seg_quality=args.pool_type != "attention",
        return_seg_starts=not is_train,
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else args.eval_batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _train_one_epoch(model, loader, optimizer, device, args, weight_map, epoch: int):
    model.train()
    running_loss = 0.0
    total_samples = 0

    class_weights = None
    if args.class_weights:
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)

    iterator, progress_bar = _make_progress(loader, args, desc="Epoch {}/{} train".format(epoch, args.epochs))
    total_steps = len(loader) if hasattr(loader, "__len__") else None

    for step, batch in enumerate(iterator, start=1):
        signal = batch["signal"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        seg_mask = batch["seg_mask"].to(device, non_blocking=True)
        batch_ids = batch.get("xml_file")
        _check_finite_tensor("signal", signal, batch_ids=batch_ids)
        seg_quality = batch.get("seg_quality")
        if seg_quality is not None:
            seg_quality = seg_quality.to(device, non_blocking=True)
            _check_finite_tensor("seg_quality", seg_quality, batch_ids=batch_ids)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(signal, mask=seg_mask, seg_quality=seg_quality)
        _check_finite_tensor("logits", logits, batch_ids=batch_ids)

        loss_vec = F.cross_entropy(logits, labels, reduction="none", weight=class_weights)
        _check_finite_tensor("loss_vec", loss_vec, batch_ids=batch_ids)
        sample_weights = _batch_sample_weights(
            batch=batch,
            weight_map=weight_map,
            key_name=args.train_weight_key,
            device=device,
        )
        if sample_weights is not None:
            loss = (loss_vec * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
        else:
            loss = loss_vec.mean()

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

        if progress_bar is not None:
            should_update = step == 1 or step % max(1, args.tqdm_update_interval) == 0 or (total_steps is not None and step == total_steps)
            if should_update:
                _update_progress(
                    progress_bar,
                    {
                        "loss": running_loss / max(1, total_samples),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                    },
                )

    if progress_bar is not None:
        progress_bar.close()

    return {"train_loss": running_loss / max(1, total_samples)}


def _evaluate(model, loader, device, args, epoch: int):
    model.eval()
    all_y: List[int] = []
    all_prob: List[np.ndarray] = []
    all_pred: List[int] = []
    running_loss = 0.0
    total_samples = 0

    class_weights = None
    if args.class_weights:
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)

    iterator, progress_bar = _make_progress(loader, args, desc="Epoch {}/{} val".format(epoch, args.epochs))
    total_steps = len(loader) if hasattr(loader, "__len__") else None

    with torch.no_grad():
        for step, batch in enumerate(iterator, start=1):
            signal = batch["signal"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            seg_mask = batch["seg_mask"].to(device, non_blocking=True)
            batch_ids = batch.get("xml_file")
            _check_finite_tensor("signal", signal, batch_ids=batch_ids)
            seg_quality = batch.get("seg_quality")
            if seg_quality is not None:
                seg_quality = seg_quality.to(device, non_blocking=True)
                _check_finite_tensor("seg_quality", seg_quality, batch_ids=batch_ids)

            logits, _ = model(signal, mask=seg_mask, seg_quality=seg_quality)
            _check_finite_tensor("logits", logits, batch_ids=batch_ids)
            loss = F.cross_entropy(logits, labels, reduction="mean", weight=class_weights)
            _check_finite_tensor("loss", loss, batch_ids=batch_ids)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            batch_size = labels.size(0)
            running_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            all_y.extend(labels.cpu().numpy().tolist())
            all_prob.extend(probs.cpu().numpy())
            all_pred.extend(preds.cpu().numpy().tolist())

            if progress_bar is not None:
                should_update = step == 1 or step % max(1, args.tqdm_update_interval) == 0 or (total_steps is not None and step == total_steps)
                if should_update:
                    _update_progress(progress_bar, {"loss": running_loss / max(1, total_samples)})

    if progress_bar is not None:
        progress_bar.close()

    metrics = compute_metrics(
        all_y=np.asarray(all_y, dtype=np.int64),
        all_prob=np.asarray(all_prob, dtype=np.float64),
        all_pred=np.asarray(all_pred, dtype=np.int64),
        num_classes=args.num_classes,
        out_dir=None,
        class_names=args.class_names or [f"class_{i}" for i in range(args.num_classes)],
    )
    metrics["val_loss"] = running_loss / max(1, total_samples)
    return metrics


def _monitor_value(metrics: Dict, metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None:
        raise ValueError("Metric '{}' is not available in validation metrics.".format(metric_name))
    return float(value)


def _save_checkpoint(model, optimizer, epoch: int, args, metrics: Dict, out_path: Path) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }
    torch.save(payload, out_path)


def train(args):
    device = _resolve_device(args.device)
    _set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer, writer_backend = create_summary_writer(
        log_dir=args.tensorboard_dir or (out_dir / "tensorboard"),
        enabled=not args.no_tensorboard,
        flush_secs=args.tb_flush_secs,
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "tqdm": not args.no_tqdm and tqdm is not None,
                "tensorboard": writer is not None,
                "tensorboard_backend": writer_backend,
                "tensorboard_dir": str(args.tensorboard_dir or (out_dir / "tensorboard")),
            },
            ensure_ascii=False,
        )
    )

    try:
        model = _build_model(args, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_loader = _make_dataloader(args.train_csv, args, shuffle=True, is_train=True, device=device)
        valid_loader = _make_dataloader(args.valid_csv, args, shuffle=False, is_train=False, device=device)

        weight_map = _load_sample_weights(
            csv_path=args.train_weight_csv,
            key_name=args.train_weight_key,
            value_name=args.train_weight_column,
            min_value=args.min_train_weight,
            max_value=args.max_train_weight,
        )

        best_value = None
        history = []

        for epoch in range(1, args.epochs + 1):
            train_stats = _train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                args=args,
                weight_map=weight_map,
                epoch=epoch,
            )
            val_metrics = _evaluate(model=model, loader=valid_loader, device=device, args=args, epoch=epoch)
            monitor_value = _monitor_value(val_metrics, args.monitor)

            epoch_record = {
                "epoch": epoch,
                "monitor": args.monitor,
                "monitor_value": monitor_value,
            }
            epoch_record.update(train_stats)
            epoch_record.update(val_metrics)
            history.append(epoch_record)

            add_scalar_dict(writer, train_stats, step=epoch)
            add_scalar_dict(
                writer,
                {
                    "val_loss": val_metrics.get("val_loss"),
                    "val_accuracy": val_metrics.get("accuracy"),
                    "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
                    "val_macro_f1": val_metrics.get("macro_f1"),
                    "val_macro_auroc": val_metrics.get("macro_auroc"),
                    "val_macro_auprc": val_metrics.get("macro_auprc"),
                    "val_macro_precision": val_metrics.get("macro_precision"),
                    "val_macro_recall": val_metrics.get("macro_recall"),
                    "val_macro_specificity": val_metrics.get("macro_specificity"),
                    "monitor_value": monitor_value,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                },
                step=epoch,
            )
            add_per_class_metrics(writer, val_metrics.get("per_class", []), step=epoch, prefix="val/per_class")
            add_confusion_matrix_text(writer, val_metrics.get("confusion_matrix"), step=epoch)

            (out_dir / "history.json").write_text(
                json.dumps(history, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (out_dir / "last_metrics.json").write_text(
                json.dumps(epoch_record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                metrics=epoch_record,
                out_path=out_dir / "last_model.pt",
            )

            is_best = best_value is None or monitor_value > best_value
            if is_best:
                best_value = monitor_value
                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    metrics=epoch_record,
                    out_path=out_dir / "best_model.pt",
                )
                (out_dir / "best_metrics.json").write_text(
                    json.dumps(epoch_record, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_stats["train_loss"],
                        "val_accuracy": val_metrics.get("accuracy"),
                        "val_balanced_accuracy": val_metrics.get("balanced_accuracy"),
                        "val_macro_f1": val_metrics.get("macro_f1"),
                        "monitor": args.monitor,
                        "monitor_value": monitor_value,
                        "best_monitor_value": best_value,
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        flush_and_close(writer)

    return history


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train MIL ECG heart-failure classification models with attention, quality-aware, or hybrid pooling."
    )
    parser.add_argument("--train-csv", type=Path, required=True, help="Training CSV manifest.")
    parser.add_argument("--valid-csv", type=Path, required=True, help="Validation CSV manifest.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory that contains ECG signal files referenced by the manifest.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for checkpoints and logs.")

    parser.add_argument("--num-classes", type=int, required=True, help="Number of target classes.")
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Optional class names in class-index order. Length must equal --num-classes.",
    )
    parser.add_argument(
        "--class-weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional per-class cross-entropy weights in class-index order.",
    )

    parser.add_argument("--in-channels", type=int, default=12, help="Encoder input channels.")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Encoder embedding dimension.")
    parser.add_argument("--encoder-preset", type=str, default="ecgfounder_large", help="Net1D preset name.")
    parser.add_argument("--encoder-ckpt", type=Path, default=None, help="Optional encoder checkpoint.")
    parser.add_argument("--init-ckpt", type=Path, default=None, help="Optional full model checkpoint.")
    parser.add_argument("--strict-ckpt", action="store_true", help="Require exact key match for --init-ckpt.")

    parser.add_argument(
        "--lead-mode",
        type=str,
        default="12",
        help="Lead selection mode. Examples: 12, I, II, I_1ch, II_1ch, V2_1ch.",
    )
    parser.add_argument("--seg-sec", type=float, default=4.0, help="Segment length in seconds.")
    parser.add_argument("--K", type=int, default=16, help="Number of MIL segments per record.")
    parser.add_argument(
        "--train-sample-mode",
        type=str,
        default="random",
        choices=["random", "uniform", "dense_nonoverlap"],
        help="Segment sampling mode for training.",
    )
    parser.add_argument("--base-seed", type=int, default=1234, help="Dataset sampling seed.")
    parser.add_argument("--seed", type=int, default=1234, help="Global random seed.")

    parser.add_argument(
        "--pool-type",
        type=str,
        default="attention",
        choices=["attention", "quality_attention", "hybrid"],
        help="Pooling head to train.",
    )
    parser.add_argument("--pool-hidden", type=int, default=128, help="Hidden size for pooling scorer MLP.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in the pooling scorer.")
    parser.add_argument("--quality-dim", type=int, default=4, help="Number of segment-quality features.")
    parser.add_argument("--quality-hidden", type=int, default=32, help="Hidden size for quality projection.")
    parser.add_argument("--quality-alpha", type=float, default=1.0, help="Scaling factor for quality logits.")
    parser.add_argument("--topk", type=int, default=4, help="Top-k size for hybrid pooling.")
    parser.add_argument("--mix-beta", type=float, default=0.5, help="Attention/top-k mixing ratio.")

    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value; <=0 disables it.")
    parser.add_argument("--device", type=str, default="auto", help="Device for training: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--tqdm-update-interval", type=int, default=10, help="Update tqdm postfix every N iterations.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard event logging.")
    parser.add_argument("--tensorboard-dir", type=Path, default=None, help="Optional TensorBoard log directory. Defaults to <out-dir>/tensorboard.")
    parser.add_argument("--tb-flush-secs", type=int, default=30, help="TensorBoard writer flush interval in seconds.")
    parser.add_argument(
        "--monitor",
        type=str,
        default="balanced_accuracy",
        help="Validation metric to monitor for best checkpoint selection.",
    )

    parser.add_argument("--train-weight-csv", type=Path, default=None, help="Optional CSV with training-only reliability weights.")
    parser.add_argument(
        "--train-weight-key",
        type=str,
        default="xml_file",
        choices=["xml_file", "patient_id"],
        help="Batch key used to join reliability weights.",
    )
    parser.add_argument(
        "--train-weight-column",
        type=str,
        default="weight",
        help="Column name in --train-weight-csv that stores the weight value.",
    )
    parser.add_argument("--min-train-weight", type=float, default=0.1, help="Minimum clipped training weight.")
    parser.add_argument("--max-train-weight", type=float, default=1.0, help="Maximum clipped training weight.")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.class_names is not None and len(args.class_names) != args.num_classes:
        raise ValueError("Length of --class-names must match --num-classes.")
    if args.class_weights is not None and len(args.class_weights) != args.num_classes:
        raise ValueError("Length of --class-weights must match --num-classes.")
    expected_in_channels = lead_mode_num_channels(args.lead_mode)
    if args.in_channels != expected_in_channels:
        raise ValueError(
            "--in-channels={} does not match lead mode '{}' which produces {} channel(s).".format(
                args.in_channels,
                args.lead_mode,
                expected_in_channels,
            )
        )

    train(args)


if __name__ == "__main__":
    main()
