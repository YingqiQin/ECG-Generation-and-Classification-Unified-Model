from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from mcma_torch.data.dataset import PTBXLSegmentsDataset
from mcma_torch.data.upperarm_csv import UpperArmCSVWindowsDataset, is_upperarm_dataset_type
from mcma_torch.models.mcma import MCMA
from mcma_torch.utils import dist as dist_utils
from mcma_torch.utils.config import load_config
from mcma_torch.utils.checkpoint import load_shape_matched_checkpoint
from mcma_torch.utils.io import append_csv, append_jsonl, ensure_dir
from mcma_torch.utils.meter import AverageMeter
from mcma_torch.utils.metrics import RunningMSE, RunningPCC
from mcma_torch.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MCMA model.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run a small number of batches to validate the pipeline.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides like trainer.epochs=1 optim.lr=1e-4",
    )
    return parser


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if "," in raw and not (raw.startswith("[") and raw.endswith("]")):
        return [_parse_value(chunk.strip()) for chunk in raw.split(",")]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _set_nested(config: dict, keys: list[str], value: Any) -> None:
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _apply_overrides(config: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item}")
        key, raw_value = item.split("=", 1)
        _set_nested(config, key.split("."), _parse_value(raw_value))
    return config


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _reduce_mean(value_sum: float, count: int, device: torch.device) -> float:
    total = torch.tensor([value_sum], device=device, dtype=torch.float64)
    total_count = torch.tensor([count], device=device, dtype=torch.float64)
    if dist_utils.is_dist_initialized():
        dist_utils.all_reduce_tensor(total)
        dist_utils.all_reduce_tensor(total_count)
    denom = max(total_count.item(), 1.0)
    return float(total.item() / denom)


def _build_dataloader(
    split: str,
    data_cfg: dict,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    sampler: torch.utils.data.Sampler | None,
) -> DataLoader:
    dataset = _build_dataset(split=split, data_cfg=data_cfg)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == "train"),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _build_dataset(split: str, data_cfg: dict) -> torch.utils.data.Dataset:
    dataset_type = data_cfg.get("dataset_type", "ptbxl")
    if is_upperarm_dataset_type(dataset_type):
        return UpperArmCSVWindowsDataset(
            csv_dir=data_cfg["csv_dir"],
            split=split,
            segment_length=int(data_cfg.get("segment_length", 1024)),
            segment_stride=int(data_cfg.get("segment_stride", data_cfg.get("segment_length", 1024))),
            input_channel=data_cfg.get("input_channel", "CH20"),
            target_channels=data_cfg.get("target_channels"),
            file_glob=data_cfg.get("file_glob", "emg_data_*.csv"),
            train_ratio=float(data_cfg.get("train_ratio", 0.8)),
            val_ratio=float(data_cfg.get("val_ratio", 0.2)),
            test_ratio=float(data_cfg.get("test_ratio", 0.0)),
            split_seed=int(data_cfg.get("split_seed", 42)),
            apply_filter=bool(data_cfg.get("apply_filter", True)),
            normalize_mode=data_cfg.get("normalize_mode", "zscore"),
            fallback_fs=float(data_cfg.get("fallback_fs", 250.0)),
            target_fs=data_cfg.get("target_fs"),
            segment_policy=data_cfg.get("segment_policy", "pad"),
            padding_mode=data_cfg.get("padding_mode", "zero"),
            max_files=data_cfg.get("max_files"),
            split_files=data_cfg.get("split_files"),
            dataset_type=dataset_type,
            segment_group_regex=data_cfg.get("segment_group_regex", r"^(?P<record>.+)_\d+s$"),
            segment_offset_regex=data_cfg.get("segment_offset_regex", r"_(?P<offset_seconds>\d+)s$"),
            npz_timestamp_key=data_cfg.get("npz_timestamp_key", "timestamp_ms"),
            npz_sampling_rate_key=data_cfg.get("npz_sampling_rate_key", "sampling_rate_hz"),
            npz_start_time_key=data_cfg.get("npz_start_time_key", "start_time_ms"),
            npz_signal_matrix_key=data_cfg.get("npz_signal_matrix_key"),
            npz_channel_names_key=data_cfg.get("npz_channel_names_key"),
            csv_dirs=data_cfg.get("csv_dirs"),
        )

    random_input = bool(data_cfg.get("random_input_lead", True))
    lead_mode = "random" if random_input else "fixed"
    return PTBXLSegmentsDataset(
        index_path=data_cfg["index_path"],
        ptbxl_root=data_cfg.get("ptbxl_root", "data/ptbxl"),
        segment_length=int(data_cfg.get("segment_length", 1024)),
        padding_mode=data_cfg.get("padding_mode", "zero"),
        input_pad_strategy=data_cfg.get("pad_strategy", "zero"),
        lead_mode=lead_mode,
        fixed_lead=data_cfg.get("fixed_lead", 0),
        split=split,
    )


def _infer_dataset_channels(dataset: torch.utils.data.Dataset) -> tuple[int, int]:
    in_channels = getattr(dataset, "input_channels", None)
    out_channels = getattr(dataset, "output_channels", None)
    if in_channels is not None and out_channels is not None:
        return int(in_channels), int(out_channels)

    sample_x, sample_y, _ = dataset[0]
    return int(sample_x.shape[0]), int(sample_y.shape[0])


def _run_validation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp: bool,
    num_leads: int,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, float, torch.Tensor, float]:
    model.eval()
    mse_meter = RunningMSE(num_leads=num_leads)
    pcc_meter = RunningPCC(num_leads=num_leads)
    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(dataloader):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp):
                y_hat = model(x)
            mse_meter.update(y_hat, y)
            pcc_meter.update(y_hat, y)
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break
    mse_meter.merge_ddp()
    pcc_meter.merge_ddp()
    mse_per_lead, mse_all = mse_meter.compute()
    pcc_per_lead, pcc_all = pcc_meter.compute()
    return mse_per_lead, mse_all, pcc_per_lead, pcc_all


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    try:
        import yaml  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required for nested config parsing. Install pyyaml."
        ) from exc

    config = load_config(config_path)
    if not isinstance(config, dict):
        raise ValueError("Config must parse into a dict.")
    config = _apply_overrides(config, args.overrides)

    seed = int(config.get("seed", 42))
    trainer_cfg = config.get("trainer", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    optim_cfg = config.get("optim", {})

    ddp_enabled = bool(trainer_cfg.get("ddp", False)) or dist_utils.distributed_enabled()
    if ddp_enabled:
        dist_utils.setup_distributed()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        local_rank = 0

    device_name = trainer_cfg.get("device", "cuda")
    if ddp_enabled and torch.cuda.is_available():
        device_name = f"cuda:{local_rank}"
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    set_seed(seed, deterministic=bool(trainer_cfg.get("deterministic", False)))

    out_dir = ensure_dir(config.get("out_dir", "artifacts/exp_mcma"))
    metrics_path = out_dir / "metrics.jsonl"
    per_lead_path = out_dir / "per_lead_metrics.csv"

    batch_size = int(data_cfg.get("batch_size", 256))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", device.type == "cuda"))

    train_sampler = None
    val_sampler = None
    if ddp_enabled:
        train_dataset = _build_dataset("train", data_cfg)
        val_dataset = _build_dataset("val", data_cfg)
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
    else:
        train_loader = _build_dataloader(
            "train", data_cfg, batch_size, num_workers, pin_memory, None
        )
        val_loader = _build_dataloader(
            "val", data_cfg, batch_size, num_workers, pin_memory, None
        )

    input_channels, output_channels = _infer_dataset_channels(train_loader.dataset)

    if dist_utils.is_main_process():
        print("Training settings:")
        print(f"  device: {device}")
        print(f"  batch_size: {batch_size}")
        print(f"  epochs: {trainer_cfg.get('epochs', 100)}")
        print(f"  lr: {optim_cfg.get('lr', 1e-3)}")
        print(f"  dataset_type: {data_cfg.get('dataset_type', 'ptbxl')}")
        print(f"  input_channels: {input_channels}")
        print(f"  output_channels: {output_channels}")
        print(f"  pad_strategy: {data_cfg.get('pad_strategy', 'zero')}")
        print(
            f"  random_input_lead: {data_cfg.get('random_input_lead', True)}"
        )
        print(f"  amp: {trainer_cfg.get('amp', True)}")
        print(f"  out_dir: {out_dir}")

    filters = model_cfg.get("filters", [16, 32, 64, 128, 256, 512])
    model = MCMA(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=int(model_cfg.get("kernel_size", 13)),
        window_size=int(model_cfg.get("window_size", 2)),
        filters=filters,
    ).to(device)
    init_ckpt = trainer_cfg.get("init_ckpt") or model_cfg.get("init_ckpt")
    if init_ckpt:
        init_report = load_shape_matched_checkpoint(model, ckpt_path=init_ckpt, device=device)
        if dist_utils.is_main_process():
            print(f"Initialized model from: {init_ckpt}")
            print(f"  matched keys: {init_report['matched_count']}")
            print(f"  skipped shape mismatch: {len(init_report['skipped_shape_keys'])}")
            print(f"  missing after load: {len(init_report['missing_keys'])}")
            print(f"  unexpected in ckpt: {len(init_report['unexpected_keys'])}")
    if ddp_enabled:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )
    mse_loss = nn.MSELoss(reduction="mean")
    l1_weight = float(optim_cfg.get("l1_weight", 0.0))
    l1_loss = nn.L1Loss(reduction="mean") if l1_weight > 0 else None

    amp_enabled = bool(trainer_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    grad_clip_norm = trainer_cfg.get("grad_clip_norm")
    log_every = int(trainer_cfg.get("log_every", 50))
    eval_every = int(trainer_cfg.get("eval_every", 1))
    epochs = int(trainer_cfg.get("epochs", 100))

    if args.dry_run:
        epochs = 1
        max_train_batches = 2
        max_val_batches = 1
    else:
        max_train_batches = None
        max_val_batches = None

    best_loss = float("inf")
    best_pcc = -float("inf")
    best_saved = False

    for epoch in range(1, epochs + 1):
        if ddp_enabled and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (x, y, _) in enumerate(train_loader):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                y_hat = model(x)
                loss = mse_loss(y_hat, y)
                if l1_loss is not None:
                    loss = loss + l1_weight * l1_loss(y_hat, y)
            if amp_enabled:
                scaler.scale(loss).backward()
                if grad_clip_norm:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            loss_meter.update(loss.item(), n=x.shape[0])
            if (
                log_every > 0
                and (batch_idx + 1) % log_every == 0
                and dist_utils.is_main_process()
            ):
                print(
                    f"Epoch {epoch} step {batch_idx + 1}: "
                    f"train_loss={loss_meter.avg:.6f}"
                )
            if max_train_batches is not None and (batch_idx + 1) >= max_train_batches:
                break

        train_loss = _reduce_mean(loss_meter.sum, loss_meter.count, device)

        if epoch % eval_every == 0:
            if ddp_enabled and val_sampler is not None:
                val_sampler.set_epoch(epoch)
            mse_per_lead, val_loss, pcc_per_lead, pcc_all = _run_validation(
                model,
                val_loader,
                device,
                amp_enabled,
                output_channels,
                max_batches=max_val_batches,
            )
            val_pcc_mean = float(pcc_per_lead.mean().item())
        else:
            mse_per_lead = torch.zeros(output_channels)
            pcc_per_lead = torch.zeros(output_channels)
            val_loss = float("nan")
            pcc_all = float("nan")
            val_pcc_mean = float("nan")

        if dist_utils.is_main_process():
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pcc_mean": val_pcc_mean,
                "val_pcc_all": pcc_all,
            }
            append_jsonl(metrics_path, record)

            lead_names = (
                getattr(train_loader.dataset, "lead_names", None) or list(range(output_channels))
            )
            rows = []
            for idx, name in enumerate(lead_names):
                rows.append(
                    {
                        "epoch": epoch,
                        "lead": name,
                        "pcc": float(pcc_per_lead[idx].item()),
                        "mse": float(mse_per_lead[idx].item()),
                    }
                )
            append_csv(per_lead_path, rows, fieldnames=["epoch", "lead", "pcc", "mse"])

            state = {
                "epoch": epoch,
                "model": _unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "val_loss": val_loss,
                "val_pcc": val_pcc_mean,
            }
            torch.save(state, out_dir / "last.pt")

            is_better = False
            if math.isfinite(val_loss):
                if val_loss < best_loss or (
                    val_loss == best_loss and val_pcc_mean > best_pcc
                ):
                    is_better = True
            elif not best_saved:
                is_better = True

            if is_better:
                if math.isfinite(val_loss):
                    best_loss = val_loss
                if math.isfinite(val_pcc_mean):
                    best_pcc = val_pcc_mean
                best_saved = True
                torch.save(state, out_dir / "best.pt")

        if args.dry_run:
            break

    if dist_utils.is_main_process():
        print("Training complete.")
        print(f"Artifacts saved to: {out_dir}")

    if ddp_enabled:
        dist_utils.cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
