import argparse
import datetime
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model

# Ensure repo root is importable when running this script from custom_vqvae_tokenizer/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codebook import codebook_model  # noqa: F401, needed for timm model registry
from codebook.codebook_engine import calculate_codebook_usage, evaluate, train_one_epoch
from codebook.optim_factory import create_optimizer
from codebook.utils import NativeScalerWithGradNormCount as NativeScaler
from codebook import utils


def get_args():
    parser = argparse.ArgumentParser("SIGMA-PPG tokenizer training from csv+npy", add_help=False)

    # Data input parameters
    parser.add_argument("--csv_path", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--data_root", type=str, default="", help="Base dir for relative ppg_path")
    parser.add_argument("--id_col", type=str, default="id_clean")
    parser.add_argument("--ppg_path_col", type=str, default="ppg_path")
    parser.add_argument("--i0_col", type=str, default="i0")
    parser.add_argument("--i1_col", type=str, default="i1")
    parser.add_argument("--n_ppg_col", type=str, default="n_ppg")
    parser.add_argument("--channel_index", type=int, default=0, help="Channel index from npy(T,C) or npy(C,T)")
    parser.add_argument("--channel_axis", type=str, default="auto", choices=["auto", "0", "1"])
    parser.add_argument("--split_mode", type=str, default="patient", choices=["patient", "random"])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--cache_size", type=int, default=16, help="Per-worker LRU cache of loaded npy files")
    parser.add_argument("--normalize", action="store_true", help="Per-segment min-max normalize to [-1, 1]")
    parser.add_argument("--max_rows", type=int, default=0, help="Debug: use first N rows")

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save_ckpt_freq", default=20, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default="vqnsp_encoder_base_decoder_3x250x12",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--codebook_n_emd", default=4096, type=int, metavar="MODEL", help="number of codebook")
    parser.add_argument("--codebook_emd_dim", default=32, type=int, metavar="MODEL", help="embedding dim")
    parser.add_argument("--ema_decay", default=0.99, type=float, metavar="MODEL", help="ema decay for quantizer")
    parser.add_argument("--quantize_kmeans_init", action="store_true", help="enable kmeans_init for quantizer")
    parser.add_argument("--use_consistency_loss", action="store_true")
    parser.add_argument("--consistency_weight", default=0.5, type=float)
    parser.add_argument("--input_size", default=12000, type=int, help="ppg input size for backbone")
    parser.add_argument("--reconstruct_phase", action="store_true")

    # Optimizer parameters
    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER")
    parser.add_argument("--opt_eps", default=1e-8, type=float, metavar="EPSILON")
    parser.add_argument("--opt_betas", default=None, type=float, nargs="+", metavar="BETA")
    parser.add_argument("--clip_grad", type=float, default=None, metavar="NORM")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weight_decay_end", type=float, default=None)
    parser.add_argument("--lr", type=float, default=5e-5, metavar="LR")
    parser.add_argument("--warmup_lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--min_lr", type=float, default=1e-5, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=5, metavar="N")
    parser.add_argument("--warmup_steps", type=int, default=-1, metavar="N")

    # Runtime parameters
    parser.add_argument("--output_dir", default="", help="path where to save")
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument("--dist_eval", action="store_true", default=True, help="Enable distributed evaluation")
    parser.add_argument("--disable_eval", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False, help="Evaluation only")
    parser.add_argument("--calculate_codebook_usage", action="store_true", default=False)
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distributed
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")

    return parser.parse_args()


def resolve_path(raw_path: str, base_dir: Path) -> str:
    p = Path(str(raw_path))
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def choose_channel_1d(arr: np.ndarray, channel_index: int, channel_axis: str) -> np.ndarray:
    if arr.ndim == 1:
        return np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported npy shape {arr.shape}. Expected 1D or 2D")

    if channel_axis == "0":
        return np.asarray(arr[channel_index, :], dtype=np.float32)
    if channel_axis == "1":
        return np.asarray(arr[:, channel_index], dtype=np.float32)

    # auto infer
    if arr.shape[1] <= 64 and arr.shape[0] > arr.shape[1]:
        return np.asarray(arr[:, channel_index], dtype=np.float32)  # (T, C)
    if arr.shape[0] <= 64 and arr.shape[1] > arr.shape[0]:
        return np.asarray(arr[channel_index, :], dtype=np.float32)  # (C, T)
    raise ValueError(f"Cannot infer channel axis from shape {arr.shape}, set --channel_axis explicitly")


def fill_nan_linear(sig: np.ndarray) -> np.ndarray:
    if not np.isnan(sig).any():
        return sig
    x = np.arange(sig.shape[0])
    mask = np.isnan(sig)
    if mask.all():
        return np.zeros_like(sig, dtype=np.float32)
    out = np.asarray(sig, dtype=np.float32).copy()
    out[mask] = np.interp(x[mask], x[~mask], out[~mask])
    return out


def normalize_minus_one_to_one(sig: np.ndarray) -> np.ndarray:
    mn = float(np.min(sig))
    mx = float(np.max(sig))
    if mx - mn < 1e-9:
        return np.zeros_like(sig, dtype=np.float32)
    return (2.0 * (sig - mn) / (mx - mn) - 1.0).astype(np.float32)


class CsvNPYSegmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        input_size: int,
        channel_index: int = 0,
        channel_axis: str = "auto",
        cache_size: int = 16,
        normalize: bool = False,
        n_ppg_col: str = "n_ppg",
    ):
        self.df = dataframe.reset_index(drop=True)
        self.input_size = input_size
        self.channel_index = channel_index
        self.channel_axis = channel_axis
        self.cache_size = max(1, cache_size)
        self.normalize = normalize
        self.n_ppg_col = n_ppg_col

        self.paths = self.df["_ppg_path_resolved"].astype(str).to_numpy()
        self.i0 = self.df["_i0"].to_numpy(dtype=np.int64)
        self.i1 = self.df["_i1"].to_numpy(dtype=np.int64)
        if self.n_ppg_col in self.df.columns:
            self.n_ppg = self.df[self.n_ppg_col].fillna(0).to_numpy(dtype=np.int64)
        else:
            self.n_ppg = None

        self._cache = OrderedDict()

    def __len__(self):
        return len(self.df)

    def get_ch_names(self):
        return ["PPG"]

    def _get_signal_1d(self, npy_path: str) -> np.ndarray:
        if npy_path in self._cache:
            self._cache.move_to_end(npy_path)
            return self._cache[npy_path]

        arr = np.load(npy_path, mmap_mode="r")
        sig = choose_channel_1d(arr, self.channel_index, self.channel_axis)
        sig = fill_nan_linear(sig)

        self._cache[npy_path] = sig
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return sig

    def __getitem__(self, idx: int):
        out = np.zeros(self.input_size, dtype=np.float32)
        npy_path = self.paths[idx]

        try:
            sig = self._get_signal_1d(npy_path)
            s = max(0, int(self.i0[idx]))
            e = min(int(self.i1[idx]), sig.shape[0])
            if e > s:
                seg = sig[s:e]
                if self.n_ppg is not None and idx < len(self.n_ppg):
                    n_ppg = int(self.n_ppg[idx])
                    if n_ppg > 0:
                        seg = seg[:n_ppg]
                valid = min(self.input_size, seg.shape[0])
                if valid > 0:
                    out[:valid] = seg[:valid].astype(np.float32)
        except Exception:
            # Keep zero segment if current row cannot be loaded.
            pass

        if self.normalize:
            out = normalize_minus_one_to_one(out)

        return torch.from_numpy(out).unsqueeze(0)


def get_model(args, **kwargs):
    print(f"Creating model: {args.model}")
    print(f"Consistency Loss Enabled: {args.use_consistency_loss}, Weight: {args.consistency_weight}")
    print(f"Phase Reconstruction Enabled: {args.reconstruct_phase}")
    model = create_model(
        args.model,
        pretrained=False,
        as_tokenzer=False,
        n_code=args.codebook_n_emd,
        code_dim=args.codebook_emd_dim,
        PPG_size=args.input_size,
        decay=args.ema_decay,
        quantize_kmeans_init=args.quantize_kmeans_init,
        use_consistency_loss=args.use_consistency_loss,
        consistency_weight=args.consistency_weight,
        reconstruct_phase=args.reconstruct_phase,
    )
    return model


def prepare_dataframe(args):
    csv_path = Path(args.csv_path).resolve()
    base_dir = Path(args.data_root).resolve() if args.data_root else csv_path.parent.resolve()

    df = pd.read_csv(csv_path)
    raw_rows = len(df)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    capped_rows = len(df)

    required = [args.ppg_path_col, args.i0_col, args.i1_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # parse and clean indices
    df["_i0"] = pd.to_numeric(df[args.i0_col], errors="coerce")
    df["_i1"] = pd.to_numeric(df[args.i1_col], errors="coerce")
    df = df.dropna(subset=["_i0", "_i1"]).copy()
    df["_i0"] = df["_i0"].astype(np.int64)
    df["_i1"] = df["_i1"].astype(np.int64)
    df = df[df["_i1"] > df["_i0"]].copy()
    rows_after_index_filter = len(df)

    # resolve paths with unique mapping
    raw_paths = df[args.ppg_path_col].astype(str)
    unique_raw = raw_paths.unique().tolist()
    path_map = {}
    exists_map = {}
    for rp in unique_raw:
        resolved = resolve_path(rp, base_dir)
        path_map[rp] = resolved
        exists_map[rp] = Path(resolved).exists()

    df["_ppg_path_resolved"] = raw_paths.map(path_map)
    df["_ppg_exists"] = raw_paths.map(exists_map)
    missing_paths = int((~df["_ppg_exists"]).sum())
    df = df[df["_ppg_exists"]].copy()
    rows_after_path_filter = len(df)

    if args.n_ppg_col in df.columns:
        df[args.n_ppg_col] = pd.to_numeric(df[args.n_ppg_col], errors="coerce").fillna(0).astype(np.int64)

    print(
        f"CSV rows raw={raw_rows}, after_max_rows={capped_rows}, "
        f"after_index_filter={rows_after_index_filter}, after_path_filter={rows_after_path_filter}, "
        f"missing_path_rows={missing_paths}"
    )
    print(f"Remaining rows for training pipeline: {rows_after_path_filter}")
    print(f"Unique npy files: {df['_ppg_path_resolved'].nunique()}")

    if len(df) == 0:
        raise ValueError("No valid rows after filtering.")

    return df


def split_dataframe(df, args):
    rng = np.random.RandomState(args.seed)
    train_ratio = float(args.train_ratio)
    train_ratio = min(max(train_ratio, 0.05), 0.95)

    if args.split_mode == "patient" and args.id_col in df.columns:
        ids = df[args.id_col].astype(str).fillna("nan").unique()
        rng.shuffle(ids)
        n_train = int(len(ids) * train_ratio)
        if len(ids) >= 2:
            n_train = min(max(1, n_train), len(ids) - 1)
        train_ids = set(ids[:n_train])
        is_train = df[args.id_col].astype(str).isin(train_ids)
        train_df = df[is_train].copy()
        val_df = df[~is_train].copy()
        print(f"Split by patient id ({args.id_col}): train_ids={len(train_ids)}, val_ids={len(ids) - len(train_ids)}")
    else:
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        if len(idx) >= 2:
            n_train = min(max(1, n_train), len(idx) - 1)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:]
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        print(f"Split by random rows: train={len(train_df)}, val={len(val_df)}")

    if len(val_df) == 0:
        print("Validation split is empty, evaluation will be disabled.")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    df = prepare_dataframe(args)
    train_df, val_df = split_dataframe(df, args)
    if len(val_df) == 0:
        args.disable_eval = True

    model = get_model(args)

    dataset_train = CsvNPYSegmentDataset(
        train_df,
        input_size=args.input_size,
        channel_index=args.channel_index,
        channel_axis=args.channel_axis,
        cache_size=args.cache_size,
        normalize=args.normalize,
        n_ppg_col=args.n_ppg_col,
    )
    print(f"Training segments: {len(dataset_train)}")

    dataset_val = None
    if not args.disable_eval:
        dataset_val = CsvNPYSegmentDataset(
            val_df,
            input_size=args.input_size,
            channel_index=args.channel_index,
            channel_axis=args.channel_axis,
            cache_size=args.cache_size,
            normalize=args.normalize,
            n_ppg_col=args.n_ppg_col,
        )
        print(f"Validation segments: {len(dataset_val)}")
    else:
        print("Evaluation disabled.")

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    num_training_steps_per_epoch = max(1, len(dataset_train) // args.batch_size // max(1, num_tasks))

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if dataset_val is not None:
            if args.dist_eval:
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
                )
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_val = None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = None
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    model.to(device)
    model_without_ddp = model
    if not args.eval:
        print("Model = %s" % str(model_without_ddp))

    for part in ["encoder", "decoder"]:
        model_part = eval(f"model.{part}")
        n_learnable_parameters = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
        n_fix_parameters = sum(p.numel() for p in model_part.parameters() if not p.requires_grad)
        print(f"number of learnable params in model.{part}: {n_learnable_parameters / 1e6} M")
        print(f"number of fixed params in model.{part}: {n_fix_parameters / 1e6} M")

    n_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fix_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"total number of learnable params: {n_learnable_parameters / 1e6} M")
    print(f"total number of fixed params: {n_fix_parameters / 1e6} M")

    total_batch_size = args.batch_size * max(1, utils.get_world_size())
    args.lr = args.lr * math.sqrt(total_batch_size / 128)
    print("LR = %.8f" % args.lr)
    print("Min LR = %.8f" % args.min_lr)
    print("Weight Decay = %.8f" % args.weight_decay)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    print("Use step level LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )

    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    train_loader_list = [data_loader_train]
    train_ch_names_list = [dataset_train.get_ch_names()]

    val_loader_list = [data_loader_val] if data_loader_val is not None else None
    val_ch_names_list = [dataset_val.get_ch_names()] if dataset_val is not None else None

    if args.eval:
        if val_loader_list is None:
            raise ValueError("No validation dataset available for --eval")
        test_stats = evaluate(val_loader_list, model, device, log_writer, 0, ch_names_list=val_ch_names_list, args=args)
        print(test_stats)
        return

    if args.calculate_codebook_usage:
        if val_loader_list is None:
            raise ValueError("No validation dataset available for --calculate_codebook_usage")
        usage_stats = calculate_codebook_usage(val_loader_list[0], model, device, log_writer, 0, args=args)
        print(usage_stats)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model,
            train_loader_list,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            ch_names_list=train_ch_names_list,
            args=args,
        )

        if args.output_dir:
            utils.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                save_ckpt_freq=args.save_ckpt_freq,
            )

        if val_loader_list is not None:
            test_stats = evaluate(
                val_loader_list,
                model,
                device,
                log_writer,
                epoch,
                ch_names_list=val_ch_names_list,
                args=args,
            )
            print(f"Validation loss on {len(dataset_val)} segments: {test_stats['loss']:.4f}")

            if log_writer is not None:
                log_writer.update(**test_stats, head="val/loss")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_learnable_parameters,
            }
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_learnable_parameters,
            }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
