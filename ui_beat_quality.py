from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any


def _as_string(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_script_path(project_root: Path) -> Path:
    return project_root / "pipeline" / "extract_quality_segments.py"


def _bundled_project_root() -> Path:
    return Path(__file__).resolve().parents[1] / "vendor" / "ui_beat_share"


def _quality_output_dir(csv_path: Path, cfg: dict[str, Any]) -> Path:
    output_root = _as_string(cfg.get("quality_preprocess_output_root"))
    output_subdir = _as_string(cfg.get("quality_preprocess_output_subdir"), "quality_segments")
    if output_root is None:
        return csv_path.parent / output_subdir
    return Path(output_root) / csv_path.parent.name / output_subdir


def _quality_report_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_quality_report.csv")


def _find_segment_files(output_dir: Path, base_name: str) -> list[Path]:
    return sorted(output_dir.glob(f"{base_name}_seg*.npz"))


def _build_command(csv_path: Path, output_dir: Path, cfg: dict[str, Any]) -> tuple[list[str], Path]:
    project_root_raw = _as_string(cfg.get("quality_preprocess_project_root"))
    project_root = Path(project_root_raw) if project_root_raw is not None else _bundled_project_root()

    script_path_raw = _as_string(cfg.get("quality_preprocess_script_path"))
    script_path = Path(script_path_raw) if script_path_raw else _default_script_path(project_root)
    if not script_path.exists():
        raise FileNotFoundError(f"UI_Beat script not found: {script_path}")

    python_executable = _as_string(cfg.get("quality_preprocess_python"), sys.executable)
    fs = int(cfg.get("quality_preprocess_fs", 1000))
    uc_thr = _as_string(cfg.get("quality_preprocess_uc_thr"), "auto")
    gpu = _as_string(cfg.get("quality_preprocess_gpu"), "0")
    infer_batch = int(cfg.get("quality_preprocess_infer_batch", 16))
    step_sec = float(cfg.get("quality_preprocess_step_sec", 8.0))

    command = [
        python_executable,
        str(script_path),
        "--csv",
        str(csv_path),
        "--fs",
        str(fs),
        "--out_dir",
        str(output_dir),
        "--gpu",
        gpu,
        "--infer_batch",
        str(infer_batch),
        "--step",
        str(step_sec),
    ]
    if uc_thr is not None:
        command.extend(["--uc_thr", uc_thr])
    return command, project_root


def ensure_ui_beat_quality_segments(
    csv_path: str | Path,
    cfg: dict[str, Any],
) -> tuple[Path, tuple[Path, ...]]:
    csv_path = Path(csv_path)
    output_dir = _ensure_dir(_quality_output_dir(csv_path=csv_path, cfg=cfg))
    report_path = _quality_report_path(csv_path)
    use_cache = bool(cfg.get("quality_preprocess_use_cache", True))
    force = bool(cfg.get("quality_preprocess_force", False))

    if use_cache and not force:
        cached_segments = _find_segment_files(output_dir=output_dir, base_name=csv_path.stem)
        if cached_segments or report_path.exists():
            return csv_path, tuple(cached_segments)

    command, project_root = _build_command(csv_path=csv_path, output_dir=output_dir, cfg=cfg)
    result = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"return code {result.returncode}"
        raise RuntimeError(f"UI_Beat preprocessing failed for {csv_path.name}: {detail}")

    generated_segments = _find_segment_files(output_dir=output_dir, base_name=csv_path.stem)
    return csv_path, tuple(generated_segments)
