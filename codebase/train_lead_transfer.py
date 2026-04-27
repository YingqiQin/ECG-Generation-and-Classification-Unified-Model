import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from ecg_dataset import ECGLeadTransferDataset, lead_mode_num_channels
from net1d import build_net1d_backbone, load_net1d_checkpoint, load_net1d_checkpoint_flexible
from tensorboard_utils import add_scalar_dict, create_summary_writer, flush_and_close


def _resolve_device(device_arg: str, local_rank: int = 0, use_ddp: bool = False) -> torch.device:
    if use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA is not available.")
        return torch.device("cuda", local_rank)
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", local_rank)
        return torch.device("cpu")
    return torch.device(device_arg)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _progress_enabled(args) -> bool:
    return (not args.no_tqdm) and tqdm is not None and _is_main_process()


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


class StudentAlignHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
        elif in_dim == out_dim:
            self.net = nn.Identity()
        else:
            self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def _get_rank() -> int:
    return dist.get_rank() if _is_dist_ready() else 0


def _is_main_process() -> bool:
    return _get_rank() == 0


def _should_use_ddp(args) -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return bool(args.ddp or world_size > 1)


def _init_distributed(args) -> Tuple[bool, int, int, int]:
    use_ddp = _should_use_ddp(args)
    if not use_ddp:
        return False, 0, 1, 0

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available, but DDP was requested.")

    if args.device == "cpu":
        raise ValueError("DDP lead-transfer is intended for CUDA single-node multi-GPU training, not CPU.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DDP lead-transfer training, but no CUDA device is available.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method="env://")
    return True, local_rank, rank, world_size


def _build_encoders(args, device: torch.device):
    teacher_encoder = build_net1d_backbone(
        in_channels=args.teacher_in_channels,
        embedding_dim=args.embedding_dim,
        preset=args.teacher_preset,
        num_classes=None,
    ).to(device)
    student_encoder = build_net1d_backbone(
        in_channels=args.student_in_channels,
        embedding_dim=args.embedding_dim,
        preset=args.student_preset,
        num_classes=None,
    ).to(device)
    align_head = StudentAlignHead(
        in_dim=args.embedding_dim,
        out_dim=args.embedding_dim,
        hidden_dim=args.align_hidden,
        dropout=args.align_dropout,
    ).to(device)

    if args.teacher_ckpt:
        missing, unexpected = load_net1d_checkpoint(
            teacher_encoder,
            str(args.teacher_ckpt),
            strict=args.strict_teacher_ckpt,
        )
        if _is_main_process():
            print(
                json.dumps(
                    {
                        "teacher_ckpt": str(args.teacher_ckpt),
                        "teacher_missing_keys": list(missing),
                        "teacher_unexpected_keys": list(unexpected),
                    },
                    indent=2,
                )
            )
    else:
        raise ValueError("--teacher-ckpt is required for lead-transfer training.")

    student_init_ckpt = args.student_init_ckpt
    if student_init_ckpt is None and args.init_student_from_teacher:
        student_init_ckpt = args.teacher_ckpt

    if student_init_ckpt:
        missing, unexpected, adapted = load_net1d_checkpoint_flexible(
            student_encoder,
            str(student_init_ckpt),
            strict=args.strict_student_init,
            adapt_input_channels=True,
        )
        if _is_main_process():
            print(
                json.dumps(
                    {
                        "student_init_ckpt": str(student_init_ckpt),
                        "student_missing_keys": list(missing),
                        "student_unexpected_keys": list(unexpected),
                        "student_adapted_keys": list(adapted),
                    },
                    indent=2,
                )
            )

    for param in teacher_encoder.parameters():
        param.requires_grad = False
    teacher_encoder.eval()
    return teacher_encoder, student_encoder, align_head


def _make_loader(
    csv_path: Path,
    args,
    shuffle: bool,
    is_train: bool,
    device: torch.device,
    use_ddp: bool,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    dataset = ECGLeadTransferDataset(
        csv_path=csv_path,
        npy_root=args.root,
        clip_sec=args.clip_sec,
        target_fs=args.target_fs,
        teacher_lead_mode=args.teacher_lead_mode,
        student_lead_mode=args.student_lead_mode,
        bandpass=not args.no_bandpass,
        norm=args.norm,
        deterministic=not is_train,
        base_seed=args.base_seed,
        random_crop=is_train,
        default_fs=args.default_fs,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=shuffle,
        drop_last=False,
    ) if use_ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size if is_train else args.eval_batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    return loader, sampler


def _apply_time_mask(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    if mask_ratio <= 0:
        return x
    batch_size, _, length = x.shape
    mask_len = max(1, int(round(length * mask_ratio)))
    if mask_len >= length:
        return torch.zeros_like(x)
    starts = torch.randint(0, length - mask_len + 1, (batch_size,), device=x.device)
    out = x.clone()
    for idx, start in enumerate(starts.tolist()):
        out[idx, :, start:start + mask_len] = 0.0
    return out


def _augment_student_view(x: torch.Tensor, noise_std: float, time_mask_ratio: float) -> torch.Tensor:
    out = x
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    out = _apply_time_mask(out, mask_ratio=time_mask_ratio)
    return out


def _cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 1.0 - F.cosine_similarity(x, y, dim=-1)


def _forward_transfer(
    teacher_encoder: nn.Module,
    student_encoder: nn.Module,
    align_head: nn.Module,
    teacher_signal: torch.Tensor,
    student_signal: torch.Tensor,
    args,
    training: bool,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        teacher_embedding = teacher_encoder(teacher_signal)

    student_view_a = _augment_student_view(
        student_signal,
        noise_std=args.student_noise_std if training else 0.0,
        time_mask_ratio=args.student_time_mask_ratio if training else 0.0,
    )
    student_view_b = _augment_student_view(
        student_signal,
        noise_std=args.student_noise_std if training else 0.0,
        time_mask_ratio=args.student_time_mask_ratio if training else 0.0,
    )

    student_embedding_a = align_head(student_encoder(student_view_a))
    student_embedding_b = align_head(student_encoder(student_view_b))

    align_loss_a = _cosine_distance(student_embedding_a, teacher_embedding)
    align_loss_b = _cosine_distance(student_embedding_b, teacher_embedding)
    align_loss = 0.5 * (align_loss_a.mean() + align_loss_b.mean())

    cons_loss = _cosine_distance(student_embedding_a, student_embedding_b).mean()
    total_loss = args.align_weight * align_loss + args.consistency_weight * cons_loss

    cosine = 1.0 - _cosine_distance(student_embedding_a, teacher_embedding).mean()
    return {
        "loss": total_loss,
        "align_loss": align_loss.detach(),
        "cons_loss": cons_loss.detach(),
        "cosine": cosine.detach(),
    }


def _run_epoch(
    teacher_encoder: nn.Module,
    student_encoder: nn.Module,
    align_head: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    args,
    training: bool,
    epoch: int,
) -> Dict[str, float]:
    if training:
        student_encoder.train()
        align_head.train()
    else:
        student_encoder.eval()
        align_head.eval()

    total_loss = 0.0
    total_align_loss = 0.0
    total_cons_loss = 0.0
    total_cosine = 0.0
    total_samples = 0

    split_name = "train" if training else "val"
    iterator, progress_bar = _make_progress(
        loader,
        args,
        desc="Epoch {}/{} {}".format(epoch, args.epochs, split_name),
    )
    total_steps = len(loader) if hasattr(loader, "__len__") else None

    for step, batch in enumerate(iterator, start=1):
        teacher_signal = batch["teacher_signal"].to(device, non_blocking=True)
        student_signal = batch["student_signal"].to(device, non_blocking=True)
        batch_ids = batch.get("record_id")
        _check_finite_tensor("teacher_signal", teacher_signal, batch_ids=batch_ids)
        _check_finite_tensor("student_signal", student_signal, batch_ids=batch_ids)
        batch_size = teacher_signal.size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)
            outputs = _forward_transfer(
                teacher_encoder=teacher_encoder,
                student_encoder=student_encoder,
                align_head=align_head,
                teacher_signal=teacher_signal,
                student_signal=student_signal,
                args=args,
                training=True,
            )
            _check_finite_tensor("transfer_loss", outputs["loss"], batch_ids=batch_ids)
            _check_finite_tensor("transfer_align_loss", outputs["align_loss"], batch_ids=batch_ids)
            _check_finite_tensor("transfer_cons_loss", outputs["cons_loss"], batch_ids=batch_ids)
            _check_finite_tensor("transfer_cosine", outputs["cosine"], batch_ids=batch_ids)
            outputs["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(student_encoder.parameters()) + list(align_head.parameters()),
                    args.grad_clip,
                )
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = _forward_transfer(
                    teacher_encoder=teacher_encoder,
                    student_encoder=student_encoder,
                    align_head=align_head,
                    teacher_signal=teacher_signal,
                    student_signal=student_signal,
                    args=args,
                    training=False,
                )
            _check_finite_tensor("transfer_loss", outputs["loss"], batch_ids=batch_ids)
            _check_finite_tensor("transfer_align_loss", outputs["align_loss"], batch_ids=batch_ids)
            _check_finite_tensor("transfer_cons_loss", outputs["cons_loss"], batch_ids=batch_ids)
            _check_finite_tensor("transfer_cosine", outputs["cosine"], batch_ids=batch_ids)

        total_loss += float(outputs["loss"].detach().cpu()) * batch_size
        total_align_loss += float(outputs["align_loss"].detach().cpu()) * batch_size
        total_cons_loss += float(outputs["cons_loss"].detach().cpu()) * batch_size
        total_cosine += float(outputs["cosine"].detach().cpu()) * batch_size
        total_samples += batch_size

        if progress_bar is not None:
            should_update = step == 1 or step % max(1, args.tqdm_update_interval) == 0 or (total_steps is not None and step == total_steps)
            if should_update:
                denom = max(1, total_samples)
                metrics = {
                    "loss": total_loss / denom,
                    "align": total_align_loss / denom,
                    "cons": total_cons_loss / denom,
                    "cos": total_cosine / denom,
                }
                if training and optimizer is not None:
                    metrics["lr"] = float(optimizer.param_groups[0]["lr"])
                _update_progress(progress_bar, metrics)

    if progress_bar is not None:
        progress_bar.close()

    if _is_dist_ready():
        reduced = torch.tensor(
            [total_loss, total_align_loss, total_cons_loss, total_cosine, float(total_samples)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        total_loss, total_align_loss, total_cons_loss, total_cosine, total_samples = reduced.tolist()

    prefix = "train" if training else "val"
    denom = max(1.0, total_samples)
    return {
        "{}_loss".format(prefix): total_loss / denom,
        "{}_align_loss".format(prefix): total_align_loss / denom,
        "{}_cons_loss".format(prefix): total_cons_loss / denom,
        "{}_cosine".format(prefix): total_cosine / denom,
    }


def _save_transfer_checkpoint(
    student_encoder: nn.Module,
    align_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args,
    metrics: Dict[str, float],
    out_path: Path,
) -> None:
    payload = {
        "epoch": epoch,
        "student_encoder": _unwrap_module(student_encoder).state_dict(),
        "align_head": _unwrap_module(align_head).state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
    }
    torch.save(payload, out_path)


def _save_student_encoder_checkpoint(
    student_encoder: nn.Module,
    epoch: int,
    args,
    metrics: Dict[str, float],
    out_path: Path,
) -> None:
    payload = {
        "model": _unwrap_module(student_encoder).state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "metrics": metrics,
    }
    torch.save(payload, out_path)


def train(args):
    use_ddp, local_rank, rank, world_size = _init_distributed(args)
    device = _resolve_device(args.device, local_rank=local_rank, use_ddp=use_ddp)
    _set_seed(args.seed + rank)

    out_dir = Path(args.out_dir)
    if _is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
    writer = None

    try:
        teacher_encoder, student_encoder, align_head = _build_encoders(args, device)
        if use_ddp and args.sync_bn:
            student_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(student_encoder)
            align_head = nn.SyncBatchNorm.convert_sync_batchnorm(align_head)

        if use_ddp:
            student_encoder = DDP(student_encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            align_head = DDP(align_head, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        optimizer = torch.optim.AdamW(
            list(student_encoder.parameters()) + list(align_head.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        train_loader, train_sampler = _make_loader(args.train_csv, args, shuffle=True, is_train=True, device=device, use_ddp=use_ddp)
        valid_loader, _ = _make_loader(args.valid_csv, args, shuffle=False, is_train=False, device=device, use_ddp=use_ddp)

        best_value = None
        history = []

        if _is_main_process():
            writer, writer_backend = create_summary_writer(
                log_dir=args.tensorboard_dir or (out_dir / "tensorboard"),
                enabled=not args.no_tensorboard,
                flush_secs=args.tb_flush_secs,
            )
            print(
                json.dumps(
                    {
                        "ddp": use_ddp,
                        "rank": rank,
                        "world_size": world_size,
                        "local_rank": local_rank,
                        "device": str(device),
                        "effective_train_batch_size": args.batch_size * world_size,
                        "effective_eval_batch_size": args.eval_batch_size * world_size,
                        "tqdm": not args.no_tqdm and tqdm is not None,
                        "tensorboard": writer is not None,
                        "tensorboard_backend": writer_backend,
                        "tensorboard_dir": str(args.tensorboard_dir or (out_dir / "tensorboard")),
                    },
                    ensure_ascii=False,
                )
            )

        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_stats = _run_epoch(
                teacher_encoder=teacher_encoder,
                student_encoder=student_encoder,
                align_head=align_head,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                args=args,
                training=True,
                epoch=epoch,
            )
            val_stats = _run_epoch(
                teacher_encoder=teacher_encoder,
                student_encoder=student_encoder,
                align_head=align_head,
                loader=valid_loader,
                optimizer=None,
                device=device,
                args=args,
                training=False,
                epoch=epoch,
            )

            epoch_record = {"epoch": epoch, "monitor": args.monitor}
            epoch_record.update(train_stats)
            epoch_record.update(val_stats)
            epoch_record["monitor_value"] = float(epoch_record[args.monitor])
            history.append(epoch_record)

            monitor_value = float(epoch_record[args.monitor])
            if args.monitor_mode == "max":
                is_best = best_value is None or monitor_value > best_value
            else:
                is_best = best_value is None or monitor_value < best_value

            if is_best:
                best_value = monitor_value

            if _is_main_process():
                add_scalar_dict(writer, train_stats, step=epoch)
                add_scalar_dict(writer, val_stats, step=epoch)
                if writer is not None:
                    writer.add_scalar("monitor/value", monitor_value, epoch)
                    writer.add_scalar("optim/lr", float(optimizer.param_groups[0]["lr"]), epoch)

                (out_dir / "history.json").write_text(
                    json.dumps(history, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                (out_dir / "last_metrics.json").write_text(
                    json.dumps(epoch_record, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                _save_transfer_checkpoint(
                    student_encoder=student_encoder,
                    align_head=align_head,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    metrics=epoch_record,
                    out_path=out_dir / "last_transfer.pt",
                )
                _save_student_encoder_checkpoint(
                    student_encoder=student_encoder,
                    epoch=epoch,
                    args=args,
                    metrics=epoch_record,
                    out_path=out_dir / "last_student_encoder.pt",
                )

                if is_best:
                    _save_transfer_checkpoint(
                        student_encoder=student_encoder,
                        align_head=align_head,
                        optimizer=optimizer,
                        epoch=epoch,
                        args=args,
                        metrics=epoch_record,
                        out_path=out_dir / "best_transfer.pt",
                    )
                    _save_student_encoder_checkpoint(
                        student_encoder=student_encoder,
                        epoch=epoch,
                        args=args,
                        metrics=epoch_record,
                        out_path=out_dir / "best_student_encoder.pt",
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
                            "val_loss": val_stats["val_loss"],
                            "val_cosine": val_stats["val_cosine"],
                            "monitor": args.monitor,
                            "monitor_value": monitor_value,
                            "best_monitor_value": best_value,
                        },
                        ensure_ascii=False,
                    )
                )
    finally:
        if _is_main_process():
            flush_and_close(writer)
        if _is_dist_ready():
            dist.barrier()
            dist.destroy_process_group()


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train a richer-lead teacher -> single-lead student transfer model for ECG lead adaptation."
    )
    parser.add_argument("--train-csv", type=Path, required=True, help="Transfer-training CSV manifest.")
    parser.add_argument("--valid-csv", type=Path, required=True, help="Transfer-validation CSV manifest.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory that contains ECG signal files, either .npy arrays or original WFDB records.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for checkpoints and logs.")

    parser.add_argument("--teacher-ckpt", type=Path, required=True, help="Pretrained richer-lead teacher checkpoint.")
    parser.add_argument("--student-init-ckpt", type=Path, default=None, help="Optional student initialization checkpoint.")
    parser.add_argument(
        "--init-student-from-teacher",
        action="store_true",
        help="If --student-init-ckpt is omitted, initialize the single-lead student from --teacher-ckpt with flexible channel adaptation.",
    )
    parser.add_argument("--strict-teacher-ckpt", action="store_true", help="Require exact key match for the teacher checkpoint.")
    parser.add_argument("--strict-student-init", action="store_true", help="Require exact key match for the student init checkpoint.")

    parser.add_argument("--teacher-preset", type=str, default="ecgfounder_large", help="Teacher Net1D preset.")
    parser.add_argument("--student-preset", type=str, default="ecgfounder_large", help="Student Net1D preset.")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Shared encoder embedding dimension.")
    parser.add_argument("--teacher-in-channels", type=int, default=12, help="Teacher encoder input channels.")
    parser.add_argument("--student-in-channels", type=int, default=1, help="Student encoder input channels.")
    parser.add_argument("--teacher-lead-mode", type=str, default="12", help="Teacher lead mode, usually 12.")
    parser.add_argument("--student-lead-mode", type=str, default="I_1ch", help="Student lead mode, for example I_1ch or II_1ch.")

    parser.add_argument("--clip-sec", type=float, default=10.0, help="Clip length in seconds for transfer training.")
    parser.add_argument("--target-fs", type=int, default=500, help="Target sampling rate.")
    parser.add_argument("--default-fs", type=int, default=None, help="Fallback fs if the manifest has no fs column.")
    parser.add_argument("--norm", type=str, default="zscore", choices=["zscore", "none"], help="Signal normalization mode.")
    parser.add_argument("--no-bandpass", action="store_true", help="Disable the lightweight bandpass filter.")
    parser.add_argument("--base-seed", type=int, default=1234, help="Dataset sampling seed.")
    parser.add_argument("--seed", type=int, default=1234, help="Global random seed.")

    parser.add_argument("--align-hidden", type=int, default=256, help="Hidden size of the student alignment head; 0 means linear/identity.")
    parser.add_argument("--align-dropout", type=float, default=0.1, help="Dropout used in the alignment head.")
    parser.add_argument("--align-weight", type=float, default=1.0, help="Weight for teacher-student alignment loss.")
    parser.add_argument("--consistency-weight", type=float, default=0.2, help="Weight for student-view consistency loss.")
    parser.add_argument("--student-noise-std", type=float, default=0.01, help="Gaussian noise std added to the student view during training.")
    parser.add_argument("--student-time-mask-ratio", type=float, default=0.05, help="Fraction of the student clip to zero-mask during training.")

    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Validation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value; <=0 disables it.")
    parser.add_argument("--device", type=str, default="auto", help="Device for training: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--ddp", action="store_true", help="Enable single-node multi-GPU DistributedDataParallel training. Launch with torchrun.")
    parser.add_argument("--sync-bn", action="store_true", help="Convert BatchNorm layers to SyncBatchNorm before DDP wrapping.")
    parser.add_argument("--dist-backend", type=str, default="nccl", choices=["nccl", "gloo"], help="torch.distributed backend.")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--tqdm-update-interval", type=int, default=10, help="Update tqdm postfix every N iterations.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard event logging.")
    parser.add_argument("--tensorboard-dir", type=Path, default=None, help="Optional TensorBoard log directory. Defaults to <out-dir>/tensorboard.")
    parser.add_argument("--tb-flush-secs", type=int, default=30, help="TensorBoard writer flush interval in seconds.")
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_cosine",
        choices=["val_cosine", "val_loss", "val_align_loss", "val_cons_loss"],
        help="Validation metric to monitor for best checkpoint selection.",
    )
    parser.add_argument(
        "--monitor-mode",
        type=str,
        default="max",
        choices=["max", "min"],
        help="Whether the monitor metric should be maximized or minimized.",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    expected_teacher_in = lead_mode_num_channels(args.teacher_lead_mode)
    expected_student_in = lead_mode_num_channels(args.student_lead_mode)
    if args.teacher_in_channels != expected_teacher_in:
        raise ValueError(
            "--teacher-in-channels={} does not match teacher lead mode '{}' which produces {} channel(s).".format(
                args.teacher_in_channels,
                args.teacher_lead_mode,
                expected_teacher_in,
            )
        )
    if args.student_in_channels != expected_student_in:
        raise ValueError(
            "--student-in-channels={} does not match student lead mode '{}' which produces {} channel(s).".format(
                args.student_in_channels,
                args.student_lead_mode,
                expected_student_in,
            )
        )

    train(args)


if __name__ == "__main__":
    main()
