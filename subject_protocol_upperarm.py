from __future__ import annotations

import argparse
import csv
import fnmatch
import itertools
from pathlib import Path
from typing import Any

from mcma_torch.data.upperarm_csv import discover_upperarm_source_units, is_upperarm_dataset_type
from mcma_torch.eval.cv_upperarm import (
    _apply_overrides,
    _build_fieldnames,
    _evaluate_checkpoint_on_files,
    _split_train_and_val,
    _summarize_rows,
    main as cv_main,
)
from mcma_torch.train.fit import main as fit_main
from mcma_torch.utils.config import load_config
from mcma_torch.utils.io import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run per-subject and cross-subject upper-arm reconstruction benchmarks."
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
        help="Config overrides like data.csv_dir=/path subject_protocol.output_dir=artifacts/run_01",
    )
    return parser


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _discover_subject_dirs(root_dir: Path, subject_dir_glob: str, file_glob: str) -> list[Path]:
    candidates = sorted(path for path in root_dir.glob(subject_dir_glob) if path.is_dir())
    subjects = [path for path in candidates if any(path.glob(file_glob))]
    if len(subjects) < 2:
        raise RuntimeError(
            f"Expected at least 2 subject folders in {root_dir} matching {subject_dir_glob} with files {file_glob}; "
            f"found {len(subjects)}"
        )
    return subjects


def _tag_rows(rows: list[dict[str, object]], **extra: object) -> list[dict[str, object]]:
    tagged: list[dict[str, object]] = []
    for row in rows:
        enriched = dict(row)
        enriched.update(extra)
        tagged.append(enriched)
    return tagged


def _resolve_checkpoint(train_dir: Path) -> Path:
    best_ckpt = train_dir / "best.pt"
    if best_ckpt.exists():
        return best_ckpt
    last_ckpt = train_dir / "last.pt"
    if last_ckpt.exists():
        return last_ckpt
    raise FileNotFoundError(f"No checkpoint found in {train_dir}")


def _summary_metric_names(corr_method: str) -> list[str]:
    return [
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


def _matches_any_glob(name: str, patterns: str) -> bool:
    candidates = [item.strip() for item in str(patterns).split(",") if item.strip()]
    if not candidates:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in candidates)


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
    dataset_type = str(data_cfg.get("dataset_type", "upperarm_csv"))
    if not is_upperarm_dataset_type(dataset_type):
        raise ValueError(f"subject_protocol_upperarm requires an upper-arm dataset type, got {dataset_type}")
    subject_cfg = config.get("subject_protocol", {})

    subject_root_dir = Path(subject_cfg.get("subject_root_dir", data_cfg.get("csv_dir", ".")))
    file_glob = str(data_cfg.get("file_glob", "emg_data_*.csv"))
    subject_dir_glob = str(subject_cfg.get("subject_dir_glob", "*"))
    output_dir = ensure_dir(subject_cfg.get("output_dir", "artifacts/upperarm_subject_protocol"))
    run_per_subject_cv = bool(subject_cfg.get("run_per_subject_cv", True))
    run_cross_subject = bool(subject_cfg.get("run_cross_subject", True))
    run_leave_one_subject_out = bool(subject_cfg.get("run_leave_one_subject_out", False))
    min_files_per_subject = int(subject_cfg.get("min_files_per_subject", 2))
    corr_method = str(crossval_cfg.get("corr_method", config.get("reconstruct", {}).get("corr_method", "pearson")))
    target_channels = list(data_cfg.get("target_channels") or [f"CH{i}" for i in range(1, 9)])

    latent_projection_method = str(
        crossval_cfg.get(
            "latent_projection_method",
            config.get("reconstruct", {}).get("latent_projection_method", "umap"),
        )
    )
    latent_projection_neighbors = int(
        crossval_cfg.get(
            "latent_projection_neighbors",
            config.get("reconstruct", {}).get("latent_projection_neighbors", 12),
        )
    )
    latent_projection_seed = int(
        crossval_cfg.get(
            "latent_projection_seed",
            config.get("reconstruct", {}).get("latent_projection_seed", config.get("seed", 42)),
        )
    )
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

    subject_dirs = _discover_subject_dirs(
        root_dir=subject_root_dir,
        subject_dir_glob=subject_dir_glob,
        file_glob=file_glob,
    )
    subject_files = {
        subject_dir.name: discover_upperarm_source_units(
            csv_dir=subject_dir,
            file_glob=file_glob,
            dataset_type=dataset_type,
            max_files=data_cfg.get("max_files"),
            segment_group_regex=data_cfg.get("segment_group_regex", r"^(?P<record>.+)_\d+s$"),
            segment_offset_regex=data_cfg.get("segment_offset_regex", r"_(?P<offset_seconds>\d+)s$"),
        )
        for subject_dir in subject_dirs
    }
    for subject_name, files in subject_files.items():
        if len(files) < min_files_per_subject:
            raise RuntimeError(
                f"Subject {subject_name} has only {len(files)} files, fewer than min_files_per_subject={min_files_per_subject}"
            )

    per_subject_summary_rows: list[dict[str, object]] = []
    if run_per_subject_cv:
        per_subject_root = ensure_dir(output_dir / "per_subject")
        for subject_dir in subject_dirs:
            subject_output_dir = per_subject_root / subject_dir.name
            cv_args: list[str] = ["--config", str(config_path)]
            if args.pretrained_ckpt:
                cv_args.extend(["--pretrained-ckpt", args.pretrained_ckpt])
            cv_args.extend(args.overrides)
            cv_args.extend(
                [
                    f"data.csv_dir={subject_dir}",
                    f"crossval.output_dir={subject_output_dir}",
                ]
            )
            status = cv_main(cv_args)
            if status != 0:
                raise RuntimeError(f"Per-subject CV failed for {subject_dir.name} with status {status}")
            summary_rows = _read_csv_rows(subject_output_dir / "summary.csv")
            per_subject_summary_rows.extend(
                _tag_rows(
                    [dict(row) for row in summary_rows],
                    subject=subject_dir.name,
                    subject_dir=str(subject_dir),
                    num_subject_files=len(subject_files[subject_dir.name]),
                )
            )
            print(f"Completed per-subject CV for {subject_dir.name} -> {subject_output_dir}")

        if per_subject_summary_rows:
            summary_fields = ["subject", "subject_dir", "num_subject_files", "stage", "num_files"]
            for metric in _summary_metric_names(corr_method):
                summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
            _write_csv(output_dir / "per_subject_summary.csv", per_subject_summary_rows, fieldnames=summary_fields)

    cross_subject_rows: list[dict[str, object]] = []
    cross_subject_summary_rows: list[dict[str, object]] = []
    if run_cross_subject:
        latent_projection_method = str(
            crossval_cfg.get(
                "latent_projection_method",
                config.get("reconstruct", {}).get("latent_projection_method", "umap"),
            )
        )
        latent_projection_neighbors = int(
            crossval_cfg.get(
                "latent_projection_neighbors",
                config.get("reconstruct", {}).get("latent_projection_neighbors", 12),
            )
        )
        latent_projection_seed = int(
            crossval_cfg.get(
                "latent_projection_seed",
                config.get("reconstruct", {}).get("latent_projection_seed", config.get("seed", 42)),
            )
        )
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
        cross_subject_val_ratio = float(subject_cfg.get("cross_subject_val_ratio", crossval_cfg.get("inner_val_ratio", 0.2)))
        fine_tune_epochs = subject_cfg.get("cross_subject_fine_tune_epochs", crossval_cfg.get("fine_tune_epochs"))
        fieldnames = [
            "source_subject",
            "target_subject",
            *[field for field in _build_fieldnames(target_channels=target_channels, corr_method=corr_method) if field != "fold"],
        ]

        cross_subject_root = ensure_dir(output_dir / "cross_subject")
        for source_dir, target_dir in itertools.permutations(subject_dirs, 2):
            pair_name = f"{source_dir.name}_to_{target_dir.name}"
            pair_output_dir = ensure_dir(cross_subject_root / pair_name)
            pair_rows: list[dict[str, object]] = []

            if args.pretrained_ckpt:
                zero_shot_rows = _evaluate_checkpoint_on_files(
                    config=config,
                    ckpt_path=Path(args.pretrained_ckpt),
                    files=subject_files[target_dir.name],
                    corr_method=corr_method,
                    plot_dir=(pair_output_dir / "zero_shot") if save_plots else None,
                    max_plot_samples=max_plot_samples,
                    plot_dpi=plot_dpi,
                    focus_plot_dir=(pair_output_dir / "zero_shot_focus") if save_focus_plots else None,
                    focus_lead=focus_lead,
                    focus_num_beats=focus_num_beats,
                    focus_window_ms=focus_window_ms,
                    focus_plot_dpi=focus_plot_dpi,
                    latent_plot_dir=(pair_output_dir / "zero_shot_latent") if save_latent_plots else None,
                    latent_max_windows_per_signal=latent_max_windows_per_signal,
                    latent_plot_dpi=latent_plot_dpi,
                    latent_projection_method=latent_projection_method,
                    latent_projection_neighbors=latent_projection_neighbors,
                    latent_projection_seed=latent_projection_seed,
                )
                pair_rows.extend(
                    _tag_rows(
                        zero_shot_rows,
                        source_subject=source_dir.name,
                        target_subject=target_dir.name,
                        stage="zero_shot",
                    )
                )

            train_files, val_files = _split_train_and_val(
                subject_files[source_dir.name],
                val_ratio=cross_subject_val_ratio,
                seed=int(config.get("seed", 42)),
            )
            train_output_dir = pair_output_dir / "train"
            fit_overrides = list(args.overrides)
            fit_overrides.extend(
                [
                    f"data.csv_dir={source_dir}",
                    f"out_dir={train_output_dir}",
                    f"data.split_files.train={','.join(path.path.name for path in train_files)}",
                    f"data.split_files.val={','.join(path.path.name for path in val_files)}",
                    "data.train_ratio=1.0",
                    "data.val_ratio=0.0",
                    "data.test_ratio=0.0",
                ]
            )
            if fine_tune_epochs is not None:
                fit_overrides.append(f"trainer.epochs={fine_tune_epochs}")
            if args.pretrained_ckpt:
                fit_overrides.append(f"trainer.init_ckpt={args.pretrained_ckpt}")

            fit_status = fit_main(["--config", str(config_path), *fit_overrides])
            if fit_status != 0:
                raise RuntimeError(f"Cross-subject fine-tuning failed for {pair_name} with status {fit_status}")

            fine_tuned_ckpt = _resolve_checkpoint(train_output_dir)
            fine_tuned_rows = _evaluate_checkpoint_on_files(
                config=config,
                ckpt_path=fine_tuned_ckpt,
                files=subject_files[target_dir.name],
                corr_method=corr_method,
                plot_dir=(pair_output_dir / "fine_tuned") if save_plots else None,
                max_plot_samples=max_plot_samples,
                plot_dpi=plot_dpi,
                focus_plot_dir=(pair_output_dir / "fine_tuned_focus") if save_focus_plots else None,
                focus_lead=focus_lead,
                focus_num_beats=focus_num_beats,
                focus_window_ms=focus_window_ms,
                focus_plot_dpi=focus_plot_dpi,
                latent_plot_dir=(pair_output_dir / "fine_tuned_latent") if save_latent_plots else None,
                latent_max_windows_per_signal=latent_max_windows_per_signal,
                latent_plot_dpi=latent_plot_dpi,
                latent_projection_method=latent_projection_method,
                latent_projection_neighbors=latent_projection_neighbors,
                latent_projection_seed=latent_projection_seed,
            )
            pair_rows.extend(
                _tag_rows(
                    fine_tuned_rows,
                    source_subject=source_dir.name,
                    target_subject=target_dir.name,
                    stage="fine_tuned",
                )
            )

            pair_summaries = _tag_rows(
                _summarize_rows(pair_rows, corr_method=corr_method),
                source_subject=source_dir.name,
                target_subject=target_dir.name,
                train_num_files=len(subject_files[source_dir.name]),
                test_num_files=len(subject_files[target_dir.name]),
            )
            pair_summary_fields = ["source_subject", "target_subject", "train_num_files", "test_num_files", "stage", "num_files"]
            for metric in _summary_metric_names(corr_method):
                pair_summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
            _write_csv(pair_output_dir / "summary.csv", pair_summaries, fieldnames=pair_summary_fields)
            _write_csv(pair_output_dir / "file_metrics.csv", pair_rows, fieldnames=fieldnames)
            cross_subject_rows.extend(pair_rows)
            cross_subject_summary_rows.extend(pair_summaries)
            print(f"Completed cross-subject transfer {pair_name} -> {pair_output_dir}")

        if cross_subject_rows:
            _write_csv(output_dir / "cross_subject_file_metrics.csv", cross_subject_rows, fieldnames=fieldnames)
        if cross_subject_summary_rows:
            pair_summary_fields = ["source_subject", "target_subject", "train_num_files", "test_num_files", "stage", "num_files"]
            for metric in _summary_metric_names(corr_method):
                pair_summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
            _write_csv(output_dir / "cross_subject_summary.csv", cross_subject_summary_rows, fieldnames=pair_summary_fields)

    loso_rows: list[dict[str, object]] = []
    loso_summary_rows: list[dict[str, object]] = []
    if run_leave_one_subject_out:
        train_subject_glob = str(subject_cfg.get("leave_one_subject_train_subject_glob", "*"))
        test_subject_glob = str(subject_cfg.get("leave_one_subject_test_subject_glob", "*"))
        candidate_train_dirs = [
            subject_dir for subject_dir in subject_dirs if _matches_any_glob(subject_dir.name, train_subject_glob)
        ]
        candidate_test_dirs = [
            subject_dir for subject_dir in subject_dirs if _matches_any_glob(subject_dir.name, test_subject_glob)
        ]
        if not candidate_test_dirs:
            raise RuntimeError(f"No held-out subjects matched leave_one_subject_test_subject_glob={test_subject_glob}")

        loso_val_ratio = float(subject_cfg.get("leave_one_subject_val_ratio", crossval_cfg.get("inner_val_ratio", 0.2)))
        loso_epochs = subject_cfg.get("leave_one_subject_fine_tune_epochs", crossval_cfg.get("fine_tune_epochs"))
        loso_root = ensure_dir(output_dir / "leave_one_subject_out")
        loso_fieldnames = [
            "heldout_subject",
            "train_subjects",
            "train_num_subjects",
            "train_num_files",
            "test_num_files",
            *[field for field in _build_fieldnames(target_channels=target_channels, corr_method=corr_method) if field != "fold"],
        ]

        for target_dir in candidate_test_dirs:
            source_dirs = [subject_dir for subject_dir in candidate_train_dirs if subject_dir != target_dir]
            if not source_dirs:
                raise RuntimeError(f"No source subjects available after leaving out {target_dir.name}")

            pooled_source_files = [
                unit
                for source_dir in source_dirs
                for unit in subject_files[source_dir.name]
            ]
            train_files, val_files = _split_train_and_val(
                pooled_source_files,
                val_ratio=loso_val_ratio,
                seed=int(config.get("seed", 42)),
            )
            train_subjects_label = ",".join(subject_dir.name for subject_dir in source_dirs)
            loso_output_dir = ensure_dir(loso_root / f"leave_out_{target_dir.name}")
            target_rows: list[dict[str, object]] = []

            if args.pretrained_ckpt:
                zero_shot_rows = _evaluate_checkpoint_on_files(
                    config=config,
                    ckpt_path=Path(args.pretrained_ckpt),
                    files=subject_files[target_dir.name],
                    corr_method=corr_method,
                    plot_dir=(loso_output_dir / "zero_shot") if save_plots else None,
                    max_plot_samples=max_plot_samples,
                    plot_dpi=plot_dpi,
                    focus_plot_dir=(loso_output_dir / "zero_shot_focus") if save_focus_plots else None,
                    focus_lead=focus_lead,
                    focus_num_beats=focus_num_beats,
                    focus_window_ms=focus_window_ms,
                    focus_plot_dpi=focus_plot_dpi,
                    latent_plot_dir=(loso_output_dir / "zero_shot_latent") if save_latent_plots else None,
                    latent_max_windows_per_signal=latent_max_windows_per_signal,
                    latent_plot_dpi=latent_plot_dpi,
                    latent_projection_method=latent_projection_method,
                    latent_projection_neighbors=latent_projection_neighbors,
                    latent_projection_seed=latent_projection_seed,
                )
                target_rows.extend(
                    _tag_rows(
                        zero_shot_rows,
                        heldout_subject=target_dir.name,
                        train_subjects=train_subjects_label,
                        train_num_subjects=len(source_dirs),
                        train_num_files=len(pooled_source_files),
                        test_num_files=len(subject_files[target_dir.name]),
                        stage="zero_shot",
                    )
                )

            train_output_dir = loso_output_dir / "train"
            fit_overrides = list(args.overrides)
            fit_overrides.extend(
                [
                    f"data.csv_dirs={','.join(str(subject_dir) for subject_dir in source_dirs)}",
                    f"out_dir={train_output_dir}",
                    f"data.split_files.train={','.join(str(unit.path) for unit in train_files)}",
                    f"data.split_files.val={','.join(str(unit.path) for unit in val_files)}",
                    "data.train_ratio=1.0",
                    "data.val_ratio=0.0",
                    "data.test_ratio=0.0",
                ]
            )
            if loso_epochs is not None:
                fit_overrides.append(f"trainer.epochs={loso_epochs}")
            if args.pretrained_ckpt:
                fit_overrides.append(f"trainer.init_ckpt={args.pretrained_ckpt}")

            fit_status = fit_main(["--config", str(config_path), *fit_overrides])
            if fit_status != 0:
                raise RuntimeError(f"LOSO fine-tuning failed for held-out subject {target_dir.name} with status {fit_status}")

            fine_tuned_ckpt = _resolve_checkpoint(train_output_dir)
            fine_tuned_rows = _evaluate_checkpoint_on_files(
                config=config,
                ckpt_path=fine_tuned_ckpt,
                files=subject_files[target_dir.name],
                corr_method=corr_method,
                plot_dir=(loso_output_dir / "fine_tuned") if save_plots else None,
                max_plot_samples=max_plot_samples,
                plot_dpi=plot_dpi,
                focus_plot_dir=(loso_output_dir / "fine_tuned_focus") if save_focus_plots else None,
                focus_lead=focus_lead,
                focus_num_beats=focus_num_beats,
                focus_window_ms=focus_window_ms,
                focus_plot_dpi=focus_plot_dpi,
                latent_plot_dir=(loso_output_dir / "fine_tuned_latent") if save_latent_plots else None,
                latent_max_windows_per_signal=latent_max_windows_per_signal,
                latent_plot_dpi=latent_plot_dpi,
                latent_projection_method=latent_projection_method,
                latent_projection_neighbors=latent_projection_neighbors,
                latent_projection_seed=latent_projection_seed,
            )
            target_rows.extend(
                _tag_rows(
                    fine_tuned_rows,
                    heldout_subject=target_dir.name,
                    train_subjects=train_subjects_label,
                    train_num_subjects=len(source_dirs),
                    train_num_files=len(pooled_source_files),
                    test_num_files=len(subject_files[target_dir.name]),
                    stage="fine_tuned",
                )
            )

            target_summaries = _tag_rows(
                _summarize_rows(target_rows, corr_method=corr_method),
                heldout_subject=target_dir.name,
                train_subjects=train_subjects_label,
                train_num_subjects=len(source_dirs),
                train_num_files=len(pooled_source_files),
                test_num_files=len(subject_files[target_dir.name]),
            )
            loso_summary_fields = [
                "heldout_subject",
                "train_subjects",
                "train_num_subjects",
                "train_num_files",
                "test_num_files",
                "stage",
                "num_files",
            ]
            for metric in _summary_metric_names(corr_method):
                loso_summary_fields.extend([f"{metric}_mean", f"{metric}_std"])

            _write_csv(loso_output_dir / "file_metrics.csv", target_rows, fieldnames=loso_fieldnames)
            _write_csv(loso_output_dir / "summary.csv", target_summaries, fieldnames=loso_summary_fields)
            loso_rows.extend(target_rows)
            loso_summary_rows.extend(target_summaries)
            print(f"Completed leave-one-subject-out for held-out subject {target_dir.name} -> {loso_output_dir}")

        if loso_rows:
            _write_csv(output_dir / "leave_one_subject_out_file_metrics.csv", loso_rows, fieldnames=loso_fieldnames)
        if loso_summary_rows:
            loso_summary_fields = [
                "heldout_subject",
                "train_subjects",
                "train_num_subjects",
                "train_num_files",
                "test_num_files",
                "stage",
                "num_files",
            ]
            for metric in _summary_metric_names(corr_method):
                loso_summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
            _write_csv(output_dir / "leave_one_subject_out_summary.csv", loso_summary_rows, fieldnames=loso_summary_fields)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
