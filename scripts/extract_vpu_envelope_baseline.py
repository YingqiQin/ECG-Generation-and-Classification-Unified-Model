#!/usr/bin/env python3
"""VPU breath-rate baseline using carrier Hilbert-envelope + DFT.

This is the clean VPU-facing wrapper around the envelope logic that worked
empirically in `extract_stationary_hr.py`. It intentionally does not implement
the experimental RespEar-style SSA selector. Use this script as the current
VPU baseline for stationary and condition-prior motion captures.

Expected input shape: mono or multi-channel PCM16 WAV. If a file is
multi-channel, channel 0 is used. The processing is non-causal because it
analyzes the whole recording.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_stationary_hr import (
    analytic_signal,
    centered,
    decimate_mean,
    estimate_autocorr_bpm,
    estimate_dft_bpm,
    fir_filter,
    load_mono_pcm16,
    make_bandpass_fir,
    make_series_svg,
    moving_average,
    save_branch_fragments,
)


CONDITION_PRESETS: dict[str, dict[str, float]] = {
    "stationary": {"cpm_min": 6.0, "cpm_max": 40.0},
    "motion": {"cpm_min": 14.0, "cpm_max": 50.0},
    "running": {"cpm_min": 14.0, "cpm_max": 55.0},
    "cycling": {"cpm_min": 14.0, "cpm_max": 55.0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract VPU breath-rate baseline candidates from a WAV file.")
    parser.add_argument("wav_path", type=Path, help="Path to VPU WAV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim"),
        help="Base output directory",
    )
    parser.add_argument(
        "--condition",
        choices=["stationary", "motion", "running", "cycling", "custom"],
        default="stationary",
        help="Capture condition used only to choose the default CPM search range",
    )
    parser.add_argument(
        "--cpm-min",
        type=float,
        default=None,
        help="Lower CPM bound. Overrides the condition preset.",
    )
    parser.add_argument(
        "--cpm-max",
        type=float,
        default=None,
        help="Upper CPM bound. Overrides the condition preset.",
    )
    parser.add_argument(
        "--analysis-rate",
        type=float,
        default=400.0,
        help="Target sample rate before envelope processing",
    )
    parser.add_argument(
        "--envelope-rate",
        type=float,
        default=50.0,
        help="Target sample rate for the breath-rate envelope curve",
    )
    parser.add_argument(
        "--carrier-low-hz",
        type=float,
        default=8.0,
        help="Lower cutoff for the carrier bandpass before Hilbert envelope extraction",
    )
    parser.add_argument(
        "--carrier-high-hz",
        type=float,
        default=40.0,
        help="Upper cutoff for the carrier bandpass before Hilbert envelope extraction",
    )
    parser.add_argument(
        "--fragment-seconds",
        type=float,
        default=20.0,
        help="Length of saved zoom fragments for the envelope breath curve",
    )
    parser.add_argument(
        "--expected-cpm",
        type=float,
        default=None,
        help="Optional reference CPM for evaluation only; not used for selecting the prediction",
    )
    return parser.parse_args()


def resolve_cpm_bounds(args: argparse.Namespace) -> tuple[float, float, str]:
    if args.condition == "custom":
        if args.cpm_min is None or args.cpm_max is None:
            raise SystemExit("--condition custom requires both --cpm-min and --cpm-max.")
        source = "custom"
        cpm_min = args.cpm_min
        cpm_max = args.cpm_max
    else:
        preset = CONDITION_PRESETS[args.condition]
        cpm_min = preset["cpm_min"] if args.cpm_min is None else args.cpm_min
        cpm_max = preset["cpm_max"] if args.cpm_max is None else args.cpm_max
        source = "condition_preset" if args.cpm_min is None and args.cpm_max is None else "condition_preset_with_override"

    if cpm_min <= 0.0:
        raise SystemExit("--cpm-min must be positive.")
    if cpm_max <= cpm_min:
        raise SystemExit("--cpm-max must be greater than --cpm-min.")
    return cpm_min, cpm_max, source


def peak_dominance(dft_estimate: dict) -> float:
    peaks = dft_estimate.get("top_peaks") or []
    if not peaks:
        return 0.0
    best = peaks[0].get("power", 0.0)
    second = max((peak.get("power", 0.0) for peak in peaks[1:]), default=0.0)
    return best / max(second, 1e-12)


def make_cpm_spectrum_svg(
    spectrum: list[dict],
    out_path: Path,
    width: int,
    height: int,
    title: str,
    peak_cpm: float | None,
) -> None:
    if not spectrum:
        out_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='600' height='120'></svg>", encoding="utf-8")
        return

    max_power = max(item["power"] for item in spectrum) or 1.0
    min_cpm = min(item["bpm"] for item in spectrum)
    max_cpm = max(item["bpm"] for item in spectrum)
    usable_width = max(width - 80, 1)
    usable_height = max(height - 70, 1)

    points = []
    for item in spectrum:
        x = 50 + ((item["bpm"] - min_cpm) / max(max_cpm - min_cpm, 1e-9)) * usable_width
        y = height - 30 - ((item["power"] / max_power) * usable_height)
        points.append(f"{x:.2f},{y:.2f}")

    peak_marker = ""
    peak_label = ""
    if peak_cpm is not None:
        peak_item = min(spectrum, key=lambda item: abs(item["bpm"] - peak_cpm))
        peak_x = 50 + ((peak_item["bpm"] - min_cpm) / max(max_cpm - min_cpm, 1e-9)) * usable_width
        peak_y = height - 30 - ((peak_item["power"] / max_power) * usable_height)
        peak_marker = f'<circle cx="{peak_x:.2f}" cy="{peak_y:.2f}" r="6" fill="#c92a2a" />'
        peak_label = (
            f'<text x="{peak_x + 10:.2f}" y="{max(28.0, peak_y - 10):.2f}" '
            f'font-size="14" fill="#c92a2a">peak {peak_item["bpm"]:.1f} CPM</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8f9fa" />
  <text x="12" y="20" font-size="14" fill="#343a40">{title}</text>
  <line x1="50" y1="{height - 30}" x2="{width - 20}" y2="{height - 30}" stroke="#adb5bd" stroke-width="1" />
  <line x1="50" y1="30" x2="50" y2="{height - 30}" stroke="#adb5bd" stroke-width="1" />
  <text x="50" y="{height - 8}" font-size="12" fill="#495057">{min_cpm:.0f}</text>
  <text x="{width - 30}" y="{height - 8}" font-size="12" text-anchor="end" fill="#495057">{max_cpm:.0f} CPM</text>
  <polyline fill="none" stroke="#1864ab" stroke-width="2" points="{' '.join(points)}" />
  {peak_marker}
  {peak_label}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def build_vpu_envelope_baseline(
    analysis_signal: list[float],
    analysis_rate: float,
    envelope_rate_target: float,
    carrier_low_hz: float,
    carrier_high_hz: float,
    cpm_min: float,
    cpm_max: float,
    out_dir: Path,
    fragment_seconds: float,
) -> dict:
    """Extract VPU breath CPM from a carrier-band Hilbert envelope.

    Input shape: mono analysis signal `[time]`, already centered and decimated.
    The operation is non-causal and uses a DFT search over the full envelope
    curve. Output rates are cycles per minute.
    """
    taps = make_bandpass_fir(analysis_rate, carrier_low_hz, carrier_high_hz, num_taps=129)
    bandpassed = fir_filter(analysis_signal, taps)
    analytic = analytic_signal(bandpassed)
    hilbert_envelope = [abs(value) for value in analytic]

    envelope_stride = max(int(analysis_rate / envelope_rate_target), 1)
    envelope_ds = decimate_mean(hilbert_envelope, envelope_stride)
    envelope_rate = analysis_rate / envelope_stride
    envelope_smooth = moving_average(envelope_ds, max(int(envelope_rate * 0.20), 1))
    slow_trend = moving_average(envelope_smooth, max(int(envelope_rate * 2.0), 1))
    breath_curve = [value - trend for value, trend in zip(envelope_smooth, slow_trend)]
    breath_curve = moving_average(breath_curve, max(int(envelope_rate * 0.30), 1))

    autocorr_est = estimate_autocorr_bpm(breath_curve, envelope_rate, bpm_min=cpm_min, bpm_max=cpm_max)
    dft_est = estimate_dft_bpm(breath_curve, envelope_rate, bpm_min=cpm_min, bpm_max=cpm_max)
    dominance = peak_dominance(dft_est)
    dft_cpm = dft_est.get("candidate_bpm")
    autocorr_cpm = autocorr_est.get("candidate_bpm")
    autocorr_delta = abs(dft_cpm - autocorr_cpm) if dft_cpm is not None and autocorr_cpm is not None else None

    confidence = min(1.0, 0.35 + 0.18 * min(dominance, 3.0))
    status = "valid"
    warnings: list[str] = []
    if dft_cpm is None:
        status = "rejected"
        confidence = 0.0
        warnings.append("no_dft_candidate")
    if dft_cpm is not None and dft_cpm - cpm_min < 2.0:
        warnings.append("candidate_near_cpm_min_check_condition_prior")
        confidence = min(confidence, 0.55)
    if autocorr_delta is not None and autocorr_delta <= 4.0:
        confidence = min(1.0, confidence + 0.12)
    elif autocorr_delta is not None:
        warnings.append("autocorr_disagrees_with_dft")
        confidence = min(confidence, 0.72)

    make_series_svg(bandpassed, out_dir / "vpu_carrier_bandpassed_waveform.svg", 1200, 280, "VPU carrier-band waveform")
    make_series_svg(envelope_ds, out_dir / "vpu_hilbert_envelope.svg", 1200, 280, "VPU Hilbert envelope")
    make_series_svg(breath_curve, out_dir / "vpu_breath_curve.svg", 1200, 280, "VPU breath-focused envelope curve")
    make_cpm_spectrum_svg(
        dft_est.get("spectrum", []),
        out_dir / "vpu_envelope_dft_spectrum.svg",
        1200,
        320,
        "VPU envelope DFT spectrum",
        dft_cpm,
    )
    fragment_meta = save_branch_fragments(
        breath_curve,
        envelope_rate,
        out_dir,
        "vpu_breath_curve",
        "VPU breath curve",
        fragment_seconds,
    )

    return {
        "method": "vpu_carrier_hilbert_envelope_dft_baseline",
        "carrier_bandpass_low_hz": carrier_low_hz,
        "carrier_bandpass_high_hz": carrier_high_hz,
        "carrier_bandpass_num_taps": 129,
        "envelope_sample_rate_hz": envelope_rate,
        "envelope_smoothing_window_seconds": max(int(envelope_rate * 0.20), 1) / envelope_rate,
        "slow_trend_window_seconds": max(int(envelope_rate * 2.0), 1) / envelope_rate,
        "breath_curve_smoothing_window_seconds": max(int(envelope_rate * 0.30), 1) / envelope_rate,
        "cpm_search_min": cpm_min,
        "cpm_search_max": cpm_max,
        "dft_estimate": dft_est,
        "autocorr_estimate": autocorr_est,
        "dft_peak_dominance_ratio": dominance,
        "autocorr_delta_cpm": autocorr_delta,
        "status": status,
        "confidence": confidence,
        "warnings": warnings,
        "artifacts": {
            "carrier_bandpassed_waveform_svg": "vpu_carrier_bandpassed_waveform.svg",
            "hilbert_envelope_svg": "vpu_hilbert_envelope.svg",
            "breath_curve_svg": "vpu_breath_curve.svg",
            "dft_spectrum_svg": "vpu_envelope_dft_spectrum.svg",
            "fragment_svgs": fragment_meta,
        },
    }


def main() -> None:
    args = parse_args()
    if args.analysis_rate <= 0.0:
        raise SystemExit("--analysis-rate must be positive.")
    if args.envelope_rate <= 0.0:
        raise SystemExit("--envelope-rate must be positive.")
    if args.fragment_seconds <= 0.0:
        raise SystemExit("--fragment-seconds must be positive.")

    cpm_min, cpm_max, cpm_bound_source = resolve_cpm_bounds(args)
    if not (0.0 < args.carrier_low_hz < args.carrier_high_hz):
        raise SystemExit("--carrier-low-hz must be positive and lower than --carrier-high-hz.")

    out_dir = args.output_dir / args.wav_path.stem / "vpu_envelope_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, metadata = load_mono_pcm16(args.wav_path)
    raw = [float(sample) for sample in samples]

    analysis_stride = max(int(metadata["sample_rate_hz"] / args.analysis_rate), 1)
    analysis_signal = centered(decimate_mean(raw, analysis_stride))
    analysis_rate = metadata["sample_rate_hz"] / analysis_stride
    if args.carrier_high_hz >= analysis_rate / 2.0:
        raise SystemExit("--carrier-high-hz must be below the analysis Nyquist frequency.")

    make_series_svg(analysis_signal, out_dir / "analysis_signal.svg", 1200, 280, "VPU decimated analysis waveform")
    baseline = build_vpu_envelope_baseline(
        analysis_signal,
        analysis_rate,
        args.envelope_rate,
        args.carrier_low_hz,
        args.carrier_high_hz,
        cpm_min,
        cpm_max,
        out_dir,
        args.fragment_seconds,
    )

    predicted_cpm = baseline["dft_estimate"].get("candidate_bpm")
    error_vs_expected = predicted_cpm - args.expected_cpm if predicted_cpm is not None and args.expected_cpm is not None else None
    final_prediction = {
        "status": baseline["status"],
        "breath_cpm": predicted_cpm,
        "confidence": baseline["confidence"],
        "condition": args.condition,
        "cpm_bound_source": cpm_bound_source,
        "cpm_search_min": cpm_min,
        "cpm_search_max": cpm_max,
        "dft_peak_dominance_ratio": baseline["dft_peak_dominance_ratio"],
        "autocorr_cpm": baseline["autocorr_estimate"].get("candidate_bpm"),
        "autocorr_delta_cpm": baseline["autocorr_delta_cpm"],
        "expected_cpm": args.expected_cpm,
        "error_vs_expected_cpm": error_vs_expected,
        "warnings": baseline["warnings"],
    }

    summary = {
        "file_name": args.wav_path.name,
        "metadata": metadata,
        "vpu_envelope_baseline": {
            "method": "vpu_carrier_hilbert_envelope_dft_baseline",
            "analysis_sample_rate_hz": analysis_rate,
            "condition": args.condition,
            "cpm_bound_source": cpm_bound_source,
            "condition_presets": CONDITION_PRESETS,
            "expected_cpm": args.expected_cpm,
            "final_prediction": final_prediction,
            "baseline": baseline,
            "artifacts": {
                "analysis_signal_svg": "analysis_signal.svg",
            },
        },
    }
    prediction = {
        "file_name": args.wav_path.name,
        "method": "vpu_carrier_hilbert_envelope_dft_baseline",
        "final_prediction": final_prediction,
        "artifacts": {
            "summary_json": "summary.json",
            "analysis_signal_svg": "analysis_signal.svg",
            "breath_curve_svg": "vpu_breath_curve.svg",
            "dft_spectrum_svg": "vpu_envelope_dft_spectrum.svg",
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "prediction.json").write_text(json.dumps(prediction, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved VPU envelope baseline summary to {out_dir / 'summary.json'}")
    print(f"Saved compact prediction to {out_dir / 'prediction.json'}")
    if predicted_cpm is None:
        print("FINAL_VPU_CPM rejected confidence=0.000")
    else:
        print(
            "FINAL_VPU_CPM "
            f"{predicted_cpm:.2f} "
            f"status={baseline['status']} "
            f"confidence={baseline['confidence']:.3f} "
            f"condition={args.condition} "
            f"range={cpm_min:.1f}-{cpm_max:.1f}"
        )


if __name__ == "__main__":
    main()
