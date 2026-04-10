from __future__ import annotations

import argparse
import csv
import gc
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mcma_torch.data.upperarm_csv import prepare_upperarm_record
from mcma_torch.eval.reconstruct_upperarm import (
    build_reconstruction_metrics_row,
    build_upperarm_model,
    reconstruct_record,
    save_focus_lead_plot,
    save_latent_space_plot,
    save_reconstruction_comparison_plot,
)
from mcma_torch.train.fit import main as fit_main
from mcma_torch.utils.checkpoint import load_shape_matched_checkpoint
from mcma_torch.utils.config import load_config
from mcma_torch.utils.io import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run file-level cross-validation for zero-shot versus fine-tuned upper-arm lead reconstruction."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--pretrained-ckpt",
        default=None,
        help="Optional PTB-XL checkpoint used for shape-matched initialization and zero-shot evaluation.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides like data.csv_dir=/path crossval.num_folds=5 trainer.epochs=10",
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


def _discover_files(csv_dir: str | Path, file_glob: str) -> list[Path]:
    files = sorted(Path(csv_dir).glob(file_glob))
    if not files:
        raise RuntimeError(f"No files matching {file_glob} found in {csv_dir}")
    return files


def _build_folds(files: list[Path], num_folds: int, seed: int) -> list[list[Path]]:
    if num_folds <= 1:
        raise ValueError("num_folds must be greater than 1")
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    folds = [[] for _ in range(num_folds)]
    for idx, path in enumerate(shuffled):
        folds[idx % num_folds].append(path)
    return [sorted(fold) for fold in folds if fold]


def _split_train_and_val(files: list[Path], val_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not files:
        raise RuntimeError("No files available for train/val split")
    if len(files) == 1:
        return list(files), list(files)

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    val_files = sorted(shuffled[:val_count])
    train_files = sorted(shuffled[val_count:])
    return train_files, val_files


def _append_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_fieldnames(target_channels: list[str], corr_method: str) -> list[str]:
    fieldnames = [
        "fold",
        "stage",
        "file_name",
        "original_sampling_rate_hz",
        "effective_sampling_rate_hz",
        "corr_method",
        "plot_path",
        "focus_lead",
        "focus_plot_path",
        "latent_plot_path",
        "matched_pretrained_keys",
        "skipped_shape_keys",
    ]
    for lead_name in target_channels:
        fieldnames.extend(
            [
                f"{lead_name}_mse",
                f"{lead_name}_mae",
                f"{lead_name}_rmse",
                f"{lead_name}_{corr_method}",
            ]
        )
    fieldnames.extend(["mean_mse", "mean_mae", "mean_rmse", f"mean_{corr_method}"])
    return fieldnames


def _evaluate_checkpoint_on_files(
    config: dict,
    ckpt_path: Path,
    files: list[Path],
    corr_method: str,
    plot_dir: Path | None = None,
    max_plot_samples: int = 4000,
    plot_dpi: int = 150,
    focus_plot_dir: Path | None = None,
    focus_lead: str | None = None,
    focus_num_beats: int = 4,
    focus_window_ms: float = 900.0,
    focus_plot_dpi: int = 180,
    latent_plot_dir: Path | None = None,
    latent_max_windows_per_signal: int = 24,
    latent_plot_dpi: int = 180,
) -> list[dict[str, object]]:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    trainer_cfg = config.get("trainer", {})
    reconstruct_cfg = config.get("reconstruct", {})

    device_name = reconstruct_cfg.get("device", trainer_cfg.get("device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    target_channels = list(data_cfg.get("target_channels") or [f"CH{i}" for i in range(1, 9)])

    model = build_upperarm_model(model_cfg=model_cfg, target_channels=target_channels, device=device)
    load_report = load_shape_matched_checkpoint(model, ckpt_path=ckpt_path, device=device)
    model.eval()

    segment_length = int(data_cfg.get("segment_length", 1024))
    segment_stride = int(data_cfg.get("segment_stride", segment_length))
    segment_policy = data_cfg.get("segment_policy", "pad")
    padding_mode = data_cfg.get("padding_mode", "zero")
    batch_size = int(reconstruct_cfg.get("batch_size", data_cfg.get("batch_size", 64)))

    rows: list[dict[str, object]] = []
    for path in files:
        record = prepare_upperarm_record(
            path=path,
            input_channel=data_cfg.get("input_channel", "CH20"),
            target_channels=target_channels,
            apply_filter=bool(data_cfg.get("apply_filter", True)),
            normalize_mode=data_cfg.get("normalize_mode", "zscore"),
            fallback_fs=float(data_cfg.get("fallback_fs", 250.0)),
            target_fs=data_cfg.get("target_fs"),
        )
        reconstructed = reconstruct_record(
            model=model,
            record=record,
            segment_length=segment_length,
            segment_stride=segment_stride,
            segment_policy=segment_policy,
            padding_mode=padding_mode,
            batch_size=batch_size,
            device=device,
        )
        row = build_reconstruction_metrics_row(
            record=record,
            target_channels=target_channels,
            reconstructed=reconstructed,
            corr_method=corr_method,
        )
        if plot_dir is not None:
            plot_path = plot_dir / f"{path.stem}_comparison.png"
            save_reconstruction_comparison_plot(
                output_path=plot_path,
                record=record,
                target_channels=target_channels,
                reconstructed=reconstructed,
                metrics_row=row,
                max_plot_samples=max_plot_samples,
                dpi=plot_dpi,
            )
            row["plot_path"] = str(plot_path)
        if focus_plot_dir is not None:
            focus_plot_path = focus_plot_dir / f"{path.stem}_focus.png"
            selected_focus_lead, _ = save_focus_lead_plot(
                output_path=focus_plot_path,
                record=record,
                target_channels=target_channels,
                reconstructed=reconstructed,
                metrics_row=row,
                focus_lead=focus_lead,
                num_beats=focus_num_beats,
                window_ms=focus_window_ms,
                max_plot_samples=max_plot_samples,
                dpi=focus_plot_dpi,
            )
            row["focus_lead"] = selected_focus_lead
            row["focus_plot_path"] = str(focus_plot_path)
        if latent_plot_dir is not None:
            latent_plot_path = latent_plot_dir / f"{path.stem}_latent.png"
            save_latent_space_plot(
                output_path=latent_plot_path,
                model=model,
                record=record,
                target_channels=target_channels,
                reconstructed=reconstructed,
                segment_length=segment_length,
                segment_stride=segment_stride,
                segment_policy=segment_policy,
                padding_mode=padding_mode,
                batch_size=batch_size,
                device=device,
                max_windows_per_signal=latent_max_windows_per_signal,
                dpi=latent_plot_dpi,
            )
            row["latent_plot_path"] = str(latent_plot_path)
        row["matched_pretrained_keys"] = load_report["matched_count"]
        row["skipped_shape_keys"] = len(load_report["skipped_shape_keys"])
        rows.append(row)
    return rows


def _summarize_rows(rows: list[dict[str, object]], corr_method: str) -> list[dict[str, object]]:
    if not rows:
        return []
    metrics = ["mean_mse", "mean_mae", "mean_rmse", f"mean_{corr_method}"]
    summaries: list[dict[str, object]] = []
    for stage in sorted({str(row["stage"]) for row in rows}):
        stage_rows = [row for row in rows if row["stage"] == stage]
        summary: dict[str, object] = {
            "stage": stage,
            "num_files": len(stage_rows),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in stage_rows], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = float(np.nanstd(values))
        summaries.append(summary)
    return summaries


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    if not isinstance(config, dict):
        raise ValueError("Config must parse into a dict")
    config = _apply_overrides(config, args.overrides)

    data_cfg = config.get("data", {})
    crossval_cfg = config.get("crossval", {})
    target_channels = list(data_cfg.get("target_channels") or [f"CH{i}" for i in range(1, 9)])
    corr_method = crossval_cfg.get("corr_method", config.get("reconstruct", {}).get("corr_method", "pearson"))
    save_plots = bool(crossval_cfg.get("save_plots", True))
    max_plot_samples = int(crossval_cfg.get("max_plot_samples", config.get("reconstruct", {}).get("max_plot_samples", 4000)))
    plot_dpi = int(crossval_cfg.get("plot_dpi", config.get("reconstruct", {}).get("plot_dpi", 150)))
    save_focus_plots = bool(crossval_cfg.get("save_focus_plots", config.get("reconstruct", {}).get("save_focus_plots", True)))
    focus_lead = crossval_cfg.get("focus_lead", config.get("reconstruct", {}).get("focus_lead"))
    focus_num_beats = int(crossval_cfg.get("focus_num_beats", config.get("reconstruct", {}).get("focus_num_beats", 4)))
    focus_window_ms = float(crossval_cfg.get("focus_window_ms", config.get("reconstruct", {}).get("focus_window_ms", 900.0)))
    focus_plot_dpi = int(crossval_cfg.get("focus_plot_dpi", config.get("reconstruct", {}).get("focus_plot_dpi", max(plot_dpi, 180))))
    save_latent_plots = bool(crossval_cfg.get("save_latent_plots", config.get("reconstruct", {}).get("save_latent_plots", True)))
    latent_max_windows_per_signal = int(crossval_cfg.get("latent_max_windows_per_signal", config.get("reconstruct", {}).get("latent_max_windows_per_signal", 24)))
    latent_plot_dpi = int(crossval_cfg.get("latent_plot_dpi", config.get("reconstruct", {}).get("latent_plot_dpi", max(plot_dpi, 180))))

    files = _discover_files(
        csv_dir=data_cfg["csv_dir"],
        file_glob=data_cfg.get("file_glob", "emg_data_*.csv"),
    )
    num_folds = min(int(crossval_cfg.get("num_folds", 5)), len(files))
    folds = _build_folds(files, num_folds=num_folds, seed=int(crossval_cfg.get("seed", config.get("seed", 42))))

    output_dir = ensure_dir(crossval_cfg.get("output_dir", "artifacts/upperarm_crossval"))
    fold_metrics_path = output_dir / "fold_metrics.csv"
    summary_path = output_dir / "summary.csv"
    fieldnames = _build_fieldnames(target_channels=target_channels, corr_method=corr_method)

    all_rows: list[dict[str, object]] = []
    for fold_idx, test_files in enumerate(folds):
        remaining = [path for path in files if path not in test_files]
        train_files, val_files = _split_train_and_val(
            remaining,
            val_ratio=float(crossval_cfg.get("inner_val_ratio", 0.2)),
            seed=int(crossval_cfg.get("seed", config.get("seed", 42))) + fold_idx,
        )

        if args.pretrained_ckpt:
            zero_shot_rows = _evaluate_checkpoint_on_files(
                config=config,
                ckpt_path=Path(args.pretrained_ckpt),
                files=test_files,
                corr_method=corr_method,
                plot_dir=(output_dir / f"fold_{fold_idx:02d}" / "zero_shot") if save_plots else None,
                max_plot_samples=max_plot_samples,
                plot_dpi=plot_dpi,
                focus_plot_dir=(output_dir / f"fold_{fold_idx:02d}" / "zero_shot_focus") if save_focus_plots else None,
                focus_lead=focus_lead,
                focus_num_beats=focus_num_beats,
                focus_window_ms=focus_window_ms,
                focus_plot_dpi=focus_plot_dpi,
                latent_plot_dir=(output_dir / f"fold_{fold_idx:02d}" / "zero_shot_latent") if save_latent_plots else None,
                latent_max_windows_per_signal=latent_max_windows_per_signal,
                latent_plot_dpi=latent_plot_dpi,
            )
            for row in zero_shot_rows:
                row["fold"] = fold_idx
                row["stage"] = "zero_shot"
            _append_csv(fold_metrics_path, zero_shot_rows, fieldnames=fieldnames)
            all_rows.extend(zero_shot_rows)

        fold_train_dir = output_dir / f"fold_{fold_idx:02d}" / "train"
        fit_overrides = list(args.overrides)
        fit_overrides.extend(
            [
                f"out_dir={fold_train_dir}",
                f"data.split_files.train={','.join(path.name for path in train_files)}",
                f"data.split_files.val={','.join(path.name for path in val_files)}",
                "data.train_ratio=1.0",
                "data.val_ratio=0.0",
                "data.test_ratio=0.0",
            ]
        )
        fine_tune_epochs = crossval_cfg.get("fine_tune_epochs")
        if fine_tune_epochs is not None:
            fit_overrides.append(f"trainer.epochs={fine_tune_epochs}")
        if args.pretrained_ckpt:
            fit_overrides.append(f"trainer.init_ckpt={args.pretrained_ckpt}")

        fit_status = fit_main(["--config", str(config_path), *fit_overrides])
        if fit_status != 0:
            raise RuntimeError(f"Fine-tuning failed on fold {fold_idx} with status {fit_status}")

        fine_tuned_ckpt = fold_train_dir / "best.pt"
        if not fine_tuned_ckpt.exists():
            fine_tuned_ckpt = fold_train_dir / "last.pt"
        if not fine_tuned_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint produced for fold {fold_idx}: {fold_train_dir}")

        fine_tuned_rows = _evaluate_checkpoint_on_files(
            config=config,
            ckpt_path=fine_tuned_ckpt,
            files=test_files,
            corr_method=corr_method,
            plot_dir=(output_dir / f"fold_{fold_idx:02d}" / "fine_tuned") if save_plots else None,
            max_plot_samples=max_plot_samples,
            plot_dpi=plot_dpi,
            focus_plot_dir=(output_dir / f"fold_{fold_idx:02d}" / "fine_tuned_focus") if save_focus_plots else None,
            focus_lead=focus_lead,
            focus_num_beats=focus_num_beats,
            focus_window_ms=focus_window_ms,
            focus_plot_dpi=focus_plot_dpi,
            latent_plot_dir=(output_dir / f"fold_{fold_idx:02d}" / "fine_tuned_latent") if save_latent_plots else None,
            latent_max_windows_per_signal=latent_max_windows_per_signal,
            latent_plot_dpi=latent_plot_dpi,
        )
        for row in fine_tuned_rows:
            row["fold"] = fold_idx
            row["stage"] = "fine_tuned"
        _append_csv(fold_metrics_path, fine_tuned_rows, fieldnames=fieldnames)
        all_rows.extend(fine_tuned_rows)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summaries = _summarize_rows(all_rows, corr_method=corr_method)
    summary_fields = ["stage", "num_files"]
    for metric in ["mean_mse", "mean_mae", "mean_rmse", f"mean_{corr_method}"]:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    _append_csv(summary_path, summaries, fieldnames=summary_fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
