from __future__ import annotations

import argparse
import csv
import gc
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mcma_torch.data.upperarm_csv import (
    UpperArmSourceUnit,
    discover_upperarm_source_units_from_dirs,
    is_upperarm_dataset_type,
    prepare_upperarm_record_from_unit,
)
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


def _discover_units(data_cfg: dict) -> list[UpperArmSourceUnit]:
    dataset_type = str(data_cfg.get("dataset_type", "upperarm_csv"))
    if not is_upperarm_dataset_type(dataset_type):
        raise ValueError(f"cv_upperarm requires an upper-arm dataset type, got {dataset_type}")
    roots = data_cfg.get("csv_dirs") or data_cfg["csv_dir"]
    units = discover_upperarm_source_units_from_dirs(
        csv_dirs=roots,
        file_glob=data_cfg.get("file_glob", "emg_data_*.csv"),
        dataset_type=dataset_type,
        max_files=data_cfg.get("max_files"),
        segment_group_regex=data_cfg.get("segment_group_regex", r"^(?P<record>.+)_\d+s$"),
        segment_offset_regex=data_cfg.get("segment_offset_regex", r"_(?P<offset_seconds>\d+)s$"),
    )
    if not units:
        raise RuntimeError(f"No files matching {data_cfg.get('file_glob', 'emg_data_*.csv')} found in {data_cfg['csv_dir']}")
    return units


def _build_folds(files: list[UpperArmSourceUnit], num_folds: int, seed: int) -> list[list[UpperArmSourceUnit]]:
    if num_folds <= 1:
        raise ValueError("num_folds must be greater than 1")
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    folds = [[] for _ in range(num_folds)]
    for idx, path in enumerate(shuffled):
        folds[idx % num_folds].append(path)
    return [sorted(fold, key=lambda unit: unit.path.name) for fold in folds if fold]


def _split_train_and_val(
    files: list[UpperArmSourceUnit],
    val_ratio: float,
    seed: int,
) -> tuple[list[UpperArmSourceUnit], list[UpperArmSourceUnit]]:
    if not files:
        raise RuntimeError("No files available for train/val split")
    if len(files) == 1:
        return list(files), list(files)

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_count = min(val_count, len(shuffled) - 1)
    val_files = sorted(shuffled[:val_count], key=lambda unit: unit.path.name)
    train_files = sorted(shuffled[val_count:], key=lambda unit: unit.path.name)
    return train_files, val_files


def _split_holdout_train_val_test(
    files: list[UpperArmSourceUnit],
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[UpperArmSourceUnit], list[UpperArmSourceUnit], list[UpperArmSourceUnit]]:
    if len(files) < 2:
        raise RuntimeError("Holdout protocol requires at least 2 logical records")
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * test_ratio)))
    test_count = min(test_count, len(shuffled) - 1)
    test_files = sorted(shuffled[:test_count], key=lambda unit: unit.path.name)
    remaining = shuffled[test_count:]
    train_files, val_files = _split_train_and_val(remaining, val_ratio=val_ratio, seed=seed + 1)
    return train_files, val_files, test_files


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
        "lag_metrics_enabled",
        "lag_search_window_ms",
        "rpeak_tolerance_ms",
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
                f"{lead_name}_max_xcorr",
                f"{lead_name}_best_lag_samples",
                f"{lead_name}_best_lag_ms",
                f"{lead_name}_lag_corrected_{corr_method}",
                f"{lead_name}_lag_corrected_rmse",
                f"{lead_name}_lag_corrected_mae",
                f"{lead_name}_dtw_distance",
                f"{lead_name}_lag_corrected_rpeak_timing_mae_ms",
                f"{lead_name}_lag_corrected_rpeak_match_fraction",
            ]
        )
    fieldnames.extend(
        [
            "mean_mse",
            "mean_mae",
            "mean_rmse",
            f"mean_{corr_method}",
            "mean_max_xcorr",
            "mean_abs_best_lag_ms",
            f"mean_lag_corrected_{corr_method}",
            "mean_lag_corrected_rmse",
            "mean_lag_corrected_mae",
            "mean_dtw_distance",
            "mean_lag_corrected_rpeak_timing_mae_ms",
            "mean_lag_corrected_rpeak_match_fraction",
        ]
    )
    return fieldnames


def _evaluate_checkpoint_on_files(
    config: dict,
    ckpt_path: Path,
    files: list[UpperArmSourceUnit],
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
    latent_projection_method: str = "umap_like",
    latent_projection_neighbors: int = 12,
    latent_projection_seed: int = 42,
) -> list[dict[str, object]]:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    trainer_cfg = config.get("trainer", {})
    reconstruct_cfg = config.get("reconstruct", {})

    dataset_type = str(data_cfg.get("dataset_type", "upperarm_csv"))
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
    lag_search_window_ms = float(reconstruct_cfg.get("lag_search_window_ms", 150.0))
    dtw_max_points = int(reconstruct_cfg.get("dtw_max_points", 2000))
    rpeak_tolerance_ms = float(reconstruct_cfg.get("rpeak_tolerance_ms", 120.0))
    visual_filter_mode = str(reconstruct_cfg.get("visual_filter_mode", "recon_only"))
    enable_lag_metrics = bool(reconstruct_cfg.get("enable_lag_metrics", True))

    rows: list[dict[str, object]] = []
    for unit in files:
        record = prepare_upperarm_record_from_unit(
            unit=unit,
            dataset_type=dataset_type,
            input_channel=data_cfg.get("input_channel", "CH20"),
            target_channels=target_channels,
            apply_filter=bool(data_cfg.get("apply_filter", True)),
            normalize_mode=data_cfg.get("normalize_mode", "zscore"),
            fallback_fs=float(data_cfg.get("fallback_fs", 250.0)),
            target_fs=data_cfg.get("target_fs"),
            npz_timestamp_key=data_cfg.get("npz_timestamp_key", "timestamp_ms"),
            npz_sampling_rate_key=data_cfg.get("npz_sampling_rate_key", "sampling_rate_hz"),
            npz_start_time_key=data_cfg.get("npz_start_time_key", "start_time_ms"),
            npz_signal_matrix_key=data_cfg.get("npz_signal_matrix_key"),
            npz_channel_names_key=data_cfg.get("npz_channel_names_key"),
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
            enable_lag_metrics=enable_lag_metrics,
            lag_search_window_ms=lag_search_window_ms,
            dtw_max_points=dtw_max_points,
            rpeak_tolerance_ms=rpeak_tolerance_ms,
        )
        if plot_dir is not None:
            plot_path = plot_dir / f"{unit.path.stem}_comparison.png"
            save_reconstruction_comparison_plot(
                output_path=plot_path,
                record=record,
                target_channels=target_channels,
                reconstructed=reconstructed,
                metrics_row=row,
                visual_filter_mode=visual_filter_mode,
                max_plot_samples=max_plot_samples,
                dpi=plot_dpi,
            )
            row["plot_path"] = str(plot_path)
        if focus_plot_dir is not None:
            focus_plot_path = focus_plot_dir / f"{unit.path.stem}_focus.png"
            selected_focus_lead, _ = save_focus_lead_plot(
                output_path=focus_plot_path,
                record=record,
                target_channels=target_channels,
                reconstructed=reconstructed,
                metrics_row=row,
                focus_lead=focus_lead,
                visual_filter_mode=visual_filter_mode,
                num_beats=focus_num_beats,
                window_ms=focus_window_ms,
                max_plot_samples=max_plot_samples,
                dpi=focus_plot_dpi,
            )
            row["focus_lead"] = selected_focus_lead
            row["focus_plot_path"] = str(focus_plot_path)
        if latent_plot_dir is not None:
            latent_plot_path = latent_plot_dir / f"{unit.path.stem}_latent.png"
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
                projection_method=latent_projection_method,
                projection_neighbors=latent_projection_neighbors,
                projection_seed=latent_projection_seed,
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
    metrics = [
        "mean_mse",
        "mean_mae",
        "mean_rmse",
        f"mean_{corr_method}",
        "mean_max_xcorr",
        "mean_abs_best_lag_ms",
        f"mean_lag_corrected_{corr_method}",
        "mean_lag_corrected_rmse",
        "mean_lag_corrected_mae",
        "mean_dtw_distance",
        "mean_lag_corrected_rpeak_timing_mae_ms",
        "mean_lag_corrected_rpeak_match_fraction",
    ]
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


def _run_one_split(
    config: dict,
    config_path: Path,
    pretrained_ckpt: str | None,
    overrides: list[str],
    split_label: str,
    split_dir_name: str,
    train_files: list[UpperArmSourceUnit],
    val_files: list[UpperArmSourceUnit],
    test_files: list[UpperArmSourceUnit],
    output_dir: Path,
    fieldnames: list[str],
    corr_method: str,
    save_plots: bool,
    max_plot_samples: int,
    plot_dpi: int,
    save_focus_plots: bool,
    focus_lead: str | None,
    focus_num_beats: int,
    focus_window_ms: float,
    focus_plot_dpi: int,
    save_latent_plots: bool,
    latent_max_windows_per_signal: int,
    latent_plot_dpi: int,
    latent_projection_method: str,
    latent_projection_neighbors: int,
    latent_projection_seed: int,
    fine_tune_epochs: object,
    metrics_path: Path,
) -> list[dict[str, object]]:
    split_rows: list[dict[str, object]] = []
    split_root = output_dir / split_dir_name

    if pretrained_ckpt:
        zero_shot_rows = _evaluate_checkpoint_on_files(
            config=config,
            ckpt_path=Path(pretrained_ckpt),
            files=test_files,
            corr_method=corr_method,
            plot_dir=(split_root / "zero_shot") if save_plots else None,
            max_plot_samples=max_plot_samples,
            plot_dpi=plot_dpi,
            focus_plot_dir=(split_root / "zero_shot_focus") if save_focus_plots else None,
            focus_lead=focus_lead,
            focus_num_beats=focus_num_beats,
            focus_window_ms=focus_window_ms,
            focus_plot_dpi=focus_plot_dpi,
            latent_plot_dir=(split_root / "zero_shot_latent") if save_latent_plots else None,
            latent_max_windows_per_signal=latent_max_windows_per_signal,
            latent_plot_dpi=latent_plot_dpi,
            latent_projection_method=latent_projection_method,
            latent_projection_neighbors=latent_projection_neighbors,
            latent_projection_seed=latent_projection_seed,
        )
        for row in zero_shot_rows:
            row["fold"] = split_label
            row["stage"] = "zero_shot"
        _append_csv(metrics_path, zero_shot_rows, fieldnames=fieldnames)
        split_rows.extend(zero_shot_rows)

    fold_train_dir = split_root / "train"
    fit_overrides = list(overrides)
    fit_overrides.extend(
        [
            f"out_dir={fold_train_dir}",
            f"data.split_files.train={','.join(path.path.name for path in train_files)}",
            f"data.split_files.val={','.join(path.path.name for path in val_files)}",
            "data.train_ratio=1.0",
            "data.val_ratio=0.0",
            "data.test_ratio=0.0",
        ]
    )
    if fine_tune_epochs is not None:
        fit_overrides.append(f"trainer.epochs={fine_tune_epochs}")
    if pretrained_ckpt:
        fit_overrides.append(f"trainer.init_ckpt={pretrained_ckpt}")

    fit_status = fit_main(["--config", str(config_path), *fit_overrides])
    if fit_status != 0:
        raise RuntimeError(f"Fine-tuning failed on split {split_label} with status {fit_status}")

    fine_tuned_ckpt = fold_train_dir / "best.pt"
    if not fine_tuned_ckpt.exists():
        fine_tuned_ckpt = fold_train_dir / "last.pt"
    if not fine_tuned_ckpt.exists():
        raise FileNotFoundError(f"No checkpoint produced for split {split_label}: {fold_train_dir}")

    fine_tuned_rows = _evaluate_checkpoint_on_files(
        config=config,
        ckpt_path=fine_tuned_ckpt,
        files=test_files,
        corr_method=corr_method,
        plot_dir=(split_root / "fine_tuned") if save_plots else None,
        max_plot_samples=max_plot_samples,
        plot_dpi=plot_dpi,
        focus_plot_dir=(split_root / "fine_tuned_focus") if save_focus_plots else None,
        focus_lead=focus_lead,
        focus_num_beats=focus_num_beats,
        focus_window_ms=focus_window_ms,
        focus_plot_dpi=focus_plot_dpi,
        latent_plot_dir=(split_root / "fine_tuned_latent") if save_latent_plots else None,
        latent_max_windows_per_signal=latent_max_windows_per_signal,
        latent_plot_dpi=latent_plot_dpi,
        latent_projection_method=latent_projection_method,
        latent_projection_neighbors=latent_projection_neighbors,
        latent_projection_seed=latent_projection_seed,
    )
    for row in fine_tuned_rows:
        row["fold"] = split_label
        row["stage"] = "fine_tuned"
    _append_csv(metrics_path, fine_tuned_rows, fieldnames=fieldnames)
    split_rows.extend(fine_tuned_rows)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return split_rows


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
    latent_projection_method = str(crossval_cfg.get("latent_projection_method", config.get("reconstruct", {}).get("latent_projection_method", "umap_like")))
    latent_projection_neighbors = int(crossval_cfg.get("latent_projection_neighbors", config.get("reconstruct", {}).get("latent_projection_neighbors", 12)))
    latent_projection_seed = int(crossval_cfg.get("latent_projection_seed", config.get("reconstruct", {}).get("latent_projection_seed", config.get("seed", 42))))
    protocol = str(crossval_cfg.get("protocol", "kfold")).lower()
    if protocol not in {"kfold", "holdout"}:
        raise ValueError(f"Unsupported crossval.protocol={protocol}; expected kfold or holdout")

    files = _discover_units(data_cfg=data_cfg)

    output_dir = ensure_dir(crossval_cfg.get("output_dir", "artifacts/upperarm_crossval"))
    fold_metrics_path = output_dir / "fold_metrics.csv"
    summary_path = output_dir / "summary.csv"
    fieldnames = _build_fieldnames(target_channels=target_channels, corr_method=corr_method)

    all_rows: list[dict[str, object]] = []
    seed = int(crossval_cfg.get("seed", config.get("seed", 42)))
    fine_tune_epochs = crossval_cfg.get("fine_tune_epochs")
    if protocol == "holdout":
        train_files, val_files, test_files = _split_holdout_train_val_test(
            files=files,
            test_ratio=float(crossval_cfg.get("holdout_test_ratio", data_cfg.get("test_ratio", 0.2) or 0.2)),
            val_ratio=float(crossval_cfg.get("inner_val_ratio", 0.2)),
            seed=seed,
        )
        all_rows.extend(
            _run_one_split(
                config=config,
                config_path=config_path,
                pretrained_ckpt=args.pretrained_ckpt,
                overrides=args.overrides,
                split_label="holdout",
                split_dir_name="holdout",
                train_files=train_files,
                val_files=val_files,
                test_files=test_files,
                output_dir=output_dir,
                fieldnames=fieldnames,
                corr_method=corr_method,
                save_plots=save_plots,
                max_plot_samples=max_plot_samples,
                plot_dpi=plot_dpi,
                save_focus_plots=save_focus_plots,
                focus_lead=focus_lead,
                focus_num_beats=focus_num_beats,
                focus_window_ms=focus_window_ms,
                focus_plot_dpi=focus_plot_dpi,
                save_latent_plots=save_latent_plots,
                latent_max_windows_per_signal=latent_max_windows_per_signal,
                latent_plot_dpi=latent_plot_dpi,
                latent_projection_method=latent_projection_method,
                latent_projection_neighbors=latent_projection_neighbors,
                latent_projection_seed=latent_projection_seed,
                fine_tune_epochs=fine_tune_epochs,
                metrics_path=fold_metrics_path,
            )
        )
    else:
        num_folds = min(int(crossval_cfg.get("num_folds", 5)), len(files))
        folds = _build_folds(files, num_folds=num_folds, seed=seed)
        for fold_idx, test_files in enumerate(folds):
            remaining = [path for path in files if path not in test_files]
            train_files, val_files = _split_train_and_val(
                remaining,
                val_ratio=float(crossval_cfg.get("inner_val_ratio", 0.2)),
                seed=seed + fold_idx,
            )
            all_rows.extend(
                _run_one_split(
                    config=config,
                    config_path=config_path,
                    pretrained_ckpt=args.pretrained_ckpt,
                    overrides=args.overrides,
                    split_label=str(fold_idx),
                    split_dir_name=f"fold_{fold_idx:02d}",
                    train_files=train_files,
                    val_files=val_files,
                    test_files=test_files,
                    output_dir=output_dir,
                    fieldnames=fieldnames,
                    corr_method=corr_method,
                    save_plots=save_plots,
                    max_plot_samples=max_plot_samples,
                    plot_dpi=plot_dpi,
                    save_focus_plots=save_focus_plots,
                    focus_lead=focus_lead,
                    focus_num_beats=focus_num_beats,
                    focus_window_ms=focus_window_ms,
                    focus_plot_dpi=focus_plot_dpi,
                    save_latent_plots=save_latent_plots,
                    latent_max_windows_per_signal=latent_max_windows_per_signal,
                    latent_plot_dpi=latent_plot_dpi,
                    latent_projection_method=latent_projection_method,
                    latent_projection_neighbors=latent_projection_neighbors,
                    latent_projection_seed=latent_projection_seed,
                    fine_tune_epochs=fine_tune_epochs,
                    metrics_path=fold_metrics_path,
                )
            )

    summaries = _summarize_rows(all_rows, corr_method=corr_method)
    summary_fields = ["stage", "num_files"]
    summary_metrics = [
        "mean_mse",
        "mean_mae",
        "mean_rmse",
        f"mean_{corr_method}",
        "mean_max_xcorr",
        "mean_abs_best_lag_ms",
        f"mean_lag_corrected_{corr_method}",
        "mean_lag_corrected_rmse",
        "mean_lag_corrected_mae",
        "mean_dtw_distance",
        "mean_lag_corrected_rpeak_timing_mae_ms",
        "mean_lag_corrected_rpeak_match_fraction",
    ]
    for metric in summary_metrics:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    _append_csv(summary_path, summaries, fieldnames=summary_fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
