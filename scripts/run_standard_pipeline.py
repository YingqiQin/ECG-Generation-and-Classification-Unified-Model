#!/usr/bin/env python3
"""Run the standard proof-of-concept pipeline on a capture set.

This orchestration script:
1. runs the existing waveform inspector and stationary HR extractor
2. aggregates the resulting summaries
3. ranks channels against an optional reference BPM
4. writes a compact SVG dashboard and JSON summary
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standard earable HR proof-of-concept pipeline.")
    parser.add_argument("wavs", nargs="+", help="Input WAV files")
    parser.add_argument(
        "--reference-bpm",
        type=float,
        default=None,
        help="Optional reference BPM from a watch or other device",
    )
    parser.add_argument(
        "--session-name",
        default="latest_session",
        help="Name of the output session directory",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("data/interim"),
        help="Directory where per-file analysis artifacts are stored",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/poc_pipeline"),
        help="Base output directory",
    )
    return parser.parse_args()


def run_analysis(repo_root: Path, wavs: list[str], interim_dir: Path) -> None:
    analyze_script = repo_root / "scripts" / "analyze_capture_set.py"
    subprocess.run(
        [sys.executable, str(analyze_script), "--interim-dir", str(interim_dir), *wavs],
        check=True,
        cwd=repo_root,
    )


def choose_branch(branches: dict, reference_bpm: float | None) -> tuple[str, dict]:
    best_name = None
    best_branch = None
    best_score = None
    for branch_name, branch in branches.items():
        dft = branch.get("dft_estimate", {})
        autocorr = branch.get("autocorr_estimate", {})
        score = 0.0
        if reference_bpm is not None and dft.get("candidate_bpm") is not None:
            score += max(0.0, 20.0 - abs(dft["candidate_bpm"] - reference_bpm))
        if dft.get("spectral_power") is not None:
            score += math.log10(max(dft["spectral_power"], 1.0))
        score += max((autocorr.get("autocorrelation") or 0.0) * 10.0, 0.0)
        if best_score is None or score > best_score:
            best_score = score
            best_name = branch_name
            best_branch = branch
    return best_name or "unknown", best_branch or {}


def load_channel_result(interim_dir: Path, wav_path: Path, reference_bpm: float | None) -> dict:
    stem = wav_path.stem
    inspect_summary = json.loads((interim_dir / stem / "summary.json").read_text(encoding="utf-8"))
    stationary_summary = json.loads(
        (interim_dir / stem / "stationary_hr" / "summary.json").read_text(encoding="utf-8")
    )
    sample_summary = inspect_summary["channel_summaries"][0]["sample_summary"]
    coarse = inspect_summary["channel_summaries"][0]["coarse_spectral_summary"]
    stationary = stationary_summary["stationary_preprocessing"]
    if "branches" in stationary:
        selected_branch_name, selected_branch = choose_branch(stationary["branches"], reference_bpm)
        autocorr = selected_branch.get("autocorr_estimate", {})
        dft = selected_branch.get("dft_estimate", {})
        stationary_artifacts = selected_branch.get("artifacts", {})
    else:
        selected_branch_name = "legacy_envelope"
        selected_branch = stationary
        autocorr = stationary.get("autocorr_estimate", {})
        dft = stationary.get("dft_estimate", {})
        stationary_artifacts = stationary_summary.get("artifacts", {})
    return {
        "file_name": wav_path.name,
        "stem": stem,
        "duration_seconds": inspect_summary["metadata"]["duration_seconds"],
        "rms": sample_summary["rms"],
        "peak_abs": sample_summary["peak_abs"],
        "near_zero_ratio": sample_summary["near_zero_ratio_abs_le_4"],
        "zero_crossing_rate_hz": sample_summary["zero_crossing_rate_hz_estimate"],
        "dominant_excerpt_frequency_hz": coarse["dominant_frequency_hz_on_first_excerpt"],
        "band_0_20_fraction": coarse["band_0_20_hz_fraction"],
        "band_20_80_fraction": coarse["band_20_80_hz_fraction"],
        "selected_branch": selected_branch_name,
        "branch_method": selected_branch.get("method"),
        "autocorr_bpm": autocorr.get("candidate_bpm"),
        "autocorr_strength": autocorr.get("autocorrelation"),
        "dft_bpm": dft.get("candidate_bpm"),
        "dft_power": dft.get("spectral_power"),
        "stationary_artifacts": stationary_artifacts,
    }


def score_channel(result: dict, reference_bpm: float | None) -> float:
    score = 0.0
    score += min(result["rms"] / 10.0, 5.0)
    score += min((result["autocorr_strength"] or 0.0) * 10.0, 5.0)
    if reference_bpm is not None and result["dft_bpm"] is not None:
        score += max(0.0, 10.0 - abs(result["dft_bpm"] - reference_bpm))
    return score


def make_dashboard_svg(results: list[dict], reference_bpm: float | None, out_path: Path) -> None:
    width = 1200
    row_height = 140
    height = 140 + row_height * len(results)
    rows = []
    for idx, result in enumerate(results):
        y = 120 + idx * row_height
        dft_bpm = result["dft_bpm"]
        autocorr_bpm = result["autocorr_bpm"]
        ref_marker = ""
        if reference_bpm is not None:
            ref_x = 720 + (reference_bpm / 200.0) * 360.0
            ref_marker = f'<line x1="{ref_x:.1f}" y1="{y - 42}" x2="{ref_x:.1f}" y2="{y + 26}" stroke="#2b8a3e" stroke-width="2" stroke-dasharray="6 4" />'
        dft_x = 720 + ((dft_bpm or 0.0) / 200.0) * 360.0
        auto_x = 720 + ((autocorr_bpm or 0.0) / 200.0) * 360.0
        rows.append(
            f"""
  <text x="40" y="{y - 8}" font-size="24" fill="#212529">{result['file_name']}</text>
  <text x="40" y="{y + 22}" font-size="18" fill="#495057">RMS {result['rms']:.2f} | DFT {dft_bpm:.1f} bpm | autocorr {autocorr_bpm:.1f} bpm</text>
  <rect x="720" y="{y - 26}" width="360" height="16" fill="#e9ecef" rx="8" />
  {ref_marker}
  <circle cx="{dft_x:.1f}" cy="{y - 18}" r="7" fill="#1864ab" />
  <circle cx="{auto_x:.1f}" cy="{y + 10}" r="7" fill="#c92a2a" />
  <text x="1095" y="{y - 12}" font-size="16" text-anchor="end" fill="#1864ab">DFT</text>
  <text x="1095" y="{y + 16}" font-size="16" text-anchor="end" fill="#c92a2a">Autocorr</text>
"""
        )
    reference_text = f"Reference HR: about {reference_bpm:.1f} bpm" if reference_bpm is not None else "Reference HR: not provided"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8f9fa" />
  <text x="40" y="50" font-size="36" fill="#212529">Earable HR Proof-of-Concept Dashboard</text>
  <text x="40" y="86" font-size="20" fill="#495057">{reference_text}</text>
  <text x="720" y="86" font-size="18" fill="#495057">BPM scale 0-200</text>
  {''.join(rows)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    wavs = [Path(wav) for wav in args.wavs]
    run_analysis(repo_root, [str(wav) for wav in wavs], args.interim_dir)

    results = [load_channel_result(args.interim_dir, wav, args.reference_bpm) for wav in wavs]
    for result in results:
        result["pipeline_score"] = score_channel(result, args.reference_bpm)
    results.sort(key=lambda item: item["pipeline_score"], reverse=True)

    best = results[0]
    output_session_dir = args.output_dir / args.session_name
    output_session_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "session_name": args.session_name,
        "reference_bpm": args.reference_bpm,
        "ranked_results": results,
        "best_channel": best,
        "selection_reason": (
            "highest combined score from signal strength, autocorrelation stability, and DFT closeness to reference"
        ),
    }
    (output_session_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    make_dashboard_svg(results, args.reference_bpm, output_session_dir / "dashboard.svg")
    print(f"Saved pipeline summary to {output_session_dir / 'summary.json'}")
    print(f"Saved dashboard to {output_session_dir / 'dashboard.svg'}")


if __name__ == "__main__":
    main()
