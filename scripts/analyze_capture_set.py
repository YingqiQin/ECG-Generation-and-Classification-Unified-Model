#!/usr/bin/env python3
"""Batch-run capture analysis utilities on a set of WAV files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a set of capture WAV files.")
    parser.add_argument(
        "--interim-dir",
        default="data/interim",
        help="Directory where per-file analysis artifacts will be written",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="WAV files or directories containing WAV files",
    )
    return parser.parse_args()


def collect_wavs(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix.lower() in {".wav", ".wave"}:
            resolved = path.resolve()
            if resolved not in seen:
                found.append(path)
                seen.add(resolved)
        elif path.is_dir():
            for wav_path in sorted(path.rglob("*.wav")):
                resolved = wav_path.resolve()
                if resolved not in seen:
                    found.append(wav_path)
                    seen.add(resolved)
    return found


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    inspect_script = repo_root / "scripts" / "inspect_wav.py"
    stationary_script = repo_root / "scripts" / "extract_stationary_hr.py"

    wavs = collect_wavs(args.paths)
    if not wavs:
        raise SystemExit("No WAV files found.")

    for wav_path in wavs:
        print(f"Analyzing {wav_path}")
        subprocess.run(
            [sys.executable, str(inspect_script), str(wav_path), "--output-dir", args.interim_dir],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(stationary_script), str(wav_path), "--output-dir", args.interim_dir],
            check=True,
        )


if __name__ == "__main__":
    main()
