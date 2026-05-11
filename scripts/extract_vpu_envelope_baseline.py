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
MOTION_CONDITIONS = {"motion", "running", "cycling"}


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
    parser.add_argument(
        "--selection-mode",
        choices=["max_peak", "candidate_ranker"],
        default="candidate_ranker",
        help="Use the maximum DFT peak directly, or rank competing peaks with stability evidence",
    )
    parser.add_argument(
        "--competitive-peak-ratio",
        type=float,
        default=0.75,
        help="A DFT local peak is considered competitive when its power is at least this fraction of the top peak",
    )
    parser.add_argument(
        "--peak-merge-cpm",
        type=float,
        default=2.0,
        help="Merge DFT local peaks within this CPM distance before ambiguity detection",
    )
    parser.add_argument(
        "--stationary-min-peak-dominance",
        type=float,
        default=1.05,
        help="Minimum top/runner-up DFT peak power ratio required for stationary captures",
    )
    parser.add_argument(
        "--motion-min-peak-dominance",
        type=float,
        default=1.25,
        help="Minimum top/runner-up DFT peak power ratio required for motion captures",
    )
    parser.add_argument(
        "--ranker-max-candidates",
        type=int,
        default=5,
        help="Maximum separated DFT peak candidates to score in candidate_ranker mode",
    )
    parser.add_argument(
        "--ranker-min-score",
        type=float,
        default=0.50,
        help="Minimum evidence score required for candidate_ranker to output a valid CPM",
    )
    parser.add_argument(
        "--ranker-score-margin",
        type=float,
        default=0.08,
        help="Minimum score gap between the best and runner-up ranked candidate",
    )
    parser.add_argument(
        "--candidate-tolerance-cpm",
        type=float,
        default=2.5,
        help="Tolerance for treating supporting peaks as matching a candidate CPM",
    )
    parser.add_argument(
        "--subwindow-seconds",
        type=float,
        default=20.0,
        help="Internal subwindow length for candidate stability checks",
    )
    parser.add_argument(
        "--subwindow-hop-seconds",
        type=float,
        default=10.0,
        help="Internal subwindow hop for candidate stability checks",
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


def rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return (sum(value * value for value in values) / len(values)) ** 0.5


def summarize_dft_peaks(
    dft_estimate: dict,
    competitive_peak_ratio: float,
    peak_merge_cpm: float,
    max_peaks: int = 8,
) -> dict:
    """Summarize separated local DFT peaks for ambiguity detection."""
    spectrum = dft_estimate.get("spectrum") or []
    if not spectrum:
        return {
            "peaks": [],
            "competitive_peaks": [],
            "competitive_peak_count": 0,
            "dominance_ratio": 0.0,
        }

    local_peaks = []
    for idx, item in enumerate(spectrum):
        power = item["power"]
        left_power = spectrum[idx - 1]["power"] if idx > 0 else None
        right_power = spectrum[idx + 1]["power"] if idx + 1 < len(spectrum) else None
        if left_power is None:
            is_peak = right_power is None or power >= right_power
        elif right_power is None:
            is_peak = power >= left_power
        else:
            is_peak = power >= left_power and power > right_power
        if is_peak:
            local_peaks.append(item)

    if not local_peaks:
        local_peaks = dft_estimate.get("top_peaks") or []

    merged_peaks = []
    for peak in sorted(local_peaks, key=lambda item: item["power"], reverse=True):
        if all(abs(peak["bpm"] - existing["bpm"]) > peak_merge_cpm for existing in merged_peaks):
            merged_peaks.append(dict(peak))
        if len(merged_peaks) >= max_peaks:
            break

    if not merged_peaks:
        return {
            "peaks": [],
            "competitive_peaks": [],
            "competitive_peak_count": 0,
            "dominance_ratio": 0.0,
        }

    top_power = max(merged_peaks[0]["power"], 1e-12)
    for rank, peak in enumerate(merged_peaks, start=1):
        peak["rank"] = rank
        peak["relative_power"] = peak["power"] / top_power

    competitive = [peak for peak in merged_peaks if peak["relative_power"] >= competitive_peak_ratio]
    second_power = merged_peaks[1]["power"] if len(merged_peaks) > 1 else 0.0
    return {
        "peaks": merged_peaks,
        "competitive_peaks": competitive,
        "competitive_peak_count": len(competitive),
        "dominance_ratio": top_power / max(second_power, 1e-12),
    }


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


def extract_envelope_curve(
    analysis_signal: list[float],
    analysis_rate: float,
    envelope_rate_target: float,
    carrier_low_hz: float,
    carrier_high_hz: float,
) -> dict:
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

    return {
        "bandpassed": bandpassed,
        "hilbert_envelope": hilbert_envelope,
        "envelope_ds": envelope_ds,
        "breath_curve": breath_curve,
        "envelope_rate_hz": envelope_rate,
        "band_rms": rms(bandpassed),
    }


def relative_power_at_cpm(dft_estimate: dict, candidate_cpm: float) -> float:
    spectrum = dft_estimate.get("spectrum") or []
    if not spectrum:
        return 0.0
    max_power = max(item["power"] for item in spectrum) or 1e-12
    closest = min(spectrum, key=lambda item: abs(item["bpm"] - candidate_cpm))
    return closest["power"] / max_power


def build_carrier_subbands(carrier_low_hz: float, carrier_high_hz: float) -> list[tuple[float, float]]:
    span = carrier_high_hz - carrier_low_hz
    if span < 9.0:
        return [(carrier_low_hz, carrier_high_hz)]
    step = span / 3.0
    return [
        (carrier_low_hz, carrier_low_hz + step),
        (carrier_low_hz + step, carrier_low_hz + 2.0 * step),
        (carrier_low_hz + 2.0 * step, carrier_high_hz),
    ]


def build_band_profiles(
    analysis_signal: list[float],
    analysis_rate: float,
    envelope_rate_target: float,
    carrier_low_hz: float,
    carrier_high_hz: float,
    cpm_min: float,
    cpm_max: float,
    energy_ratio_min: float = 0.03,
) -> list[dict]:
    full_rms = max(rms(analysis_signal), 1e-12)
    profiles = []
    for low_hz, high_hz in build_carrier_subbands(carrier_low_hz, carrier_high_hz):
        if high_hz <= low_hz:
            continue
        curve = extract_envelope_curve(analysis_signal, analysis_rate, envelope_rate_target, low_hz, high_hz)
        energy_ratio = curve["band_rms"] / full_rms
        if energy_ratio < energy_ratio_min:
            continue
        dft = estimate_dft_bpm(curve["breath_curve"], curve["envelope_rate_hz"], bpm_min=cpm_min, bpm_max=cpm_max)
        profiles.append(
            {
                "band_low_hz": low_hz,
                "band_high_hz": high_hz,
                "energy_ratio": energy_ratio,
                "dft_estimate": dft,
            }
        )
    return profiles


def build_subwindow_profiles(
    breath_curve: list[float],
    envelope_rate_hz: float,
    cpm_min: float,
    cpm_max: float,
    subwindow_seconds: float,
    subwindow_hop_seconds: float,
) -> list[dict]:
    window_len = max(int(subwindow_seconds * envelope_rate_hz), 1)
    hop_len = max(int(subwindow_hop_seconds * envelope_rate_hz), 1)
    if len(breath_curve) < window_len or window_len < max(int(envelope_rate_hz * 8.0), 1):
        return []

    profiles = []
    start = 0
    while start + window_len <= len(breath_curve):
        chunk = breath_curve[start : start + window_len]
        dft = estimate_dft_bpm(chunk, envelope_rate_hz, bpm_min=cpm_min, bpm_max=cpm_max)
        profiles.append(
            {
                "start_seconds": start / envelope_rate_hz,
                "end_seconds": (start + window_len) / envelope_rate_hz,
                "dft_estimate": dft,
            }
        )
        start += hop_len
    return profiles


def summarize_candidate_support(
    profiles: list[dict],
    candidate_cpm: float,
    support_threshold: float,
) -> dict:
    if not profiles:
        return {
            "support_count": 0,
            "profile_count": 0,
            "support_fraction": 0.5,
            "mean_relative_power": 0.5,
            "relative_powers": [],
        }
    relative_powers = [relative_power_at_cpm(profile["dft_estimate"], candidate_cpm) for profile in profiles]
    support_count = sum(1 for value in relative_powers if value >= support_threshold)
    return {
        "support_count": support_count,
        "profile_count": len(profiles),
        "support_fraction": support_count / len(profiles),
        "mean_relative_power": sum(relative_powers) / len(relative_powers),
        "relative_powers": relative_powers,
    }


def rank_dft_candidates(
    peak_summary: dict,
    autocorr_estimate: dict,
    band_profiles: list[dict],
    subwindow_profiles: list[dict],
    cpm_min: float,
    condition: str,
    candidate_tolerance_cpm: float,
    max_candidates: int,
) -> list[dict]:
    peaks = peak_summary.get("peaks") or []
    candidates = peaks[:max_candidates]
    ranked = []
    autocorr_cpm = autocorr_estimate.get("candidate_bpm")
    for candidate in candidates:
        cpm = candidate["bpm"]
        relative_power = candidate.get("relative_power", 0.0)
        band_support = summarize_candidate_support(band_profiles, cpm, support_threshold=0.50)
        subwindow_support = summarize_candidate_support(subwindow_profiles, cpm, support_threshold=0.45)
        autocorr_score = 0.0
        if autocorr_cpm is not None:
            autocorr_delta = abs(cpm - autocorr_cpm)
            autocorr_score = max(0.0, 1.0 - autocorr_delta / max(candidate_tolerance_cpm * 2.0, 1e-9))
        else:
            autocorr_delta = None

        boundary_distance = cpm - cpm_min
        boundary_penalty = 0.0
        evidence: list[str] = []
        if relative_power >= 0.90:
            evidence.append("high_fullband_power")
        if band_support["support_fraction"] >= 0.67:
            evidence.append("carrier_band_stable")
        if subwindow_support["support_fraction"] >= 0.60:
            evidence.append("subwindow_stable")
        if autocorr_score >= 0.60:
            evidence.append("autocorr_agrees")
        if boundary_distance < 2.0:
            boundary_penalty = 0.12 if condition in MOTION_CONDITIONS else 0.08
            evidence.append("near_cpm_min_penalty")

        score = (
            0.15 * relative_power
            + 0.45 * band_support["support_fraction"]
            + 0.25 * band_support["mean_relative_power"]
            + 0.05 * subwindow_support["support_fraction"]
            + 0.10 * autocorr_score
            - boundary_penalty
        )
        ranked.append(
            {
                "cpm": cpm,
                "score": max(0.0, min(1.0, score)),
                "source_rank": candidate.get("rank"),
                "relative_power": relative_power,
                "band_support": band_support,
                "subwindow_support": subwindow_support,
                "autocorr_delta_cpm": autocorr_delta,
                "boundary_distance_cpm": boundary_distance,
                "evidence": evidence,
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def build_vpu_envelope_baseline(
    analysis_signal: list[float],
    analysis_rate: float,
    envelope_rate_target: float,
    carrier_low_hz: float,
    carrier_high_hz: float,
    cpm_min: float,
    cpm_max: float,
    condition: str,
    selection_mode: str,
    competitive_peak_ratio: float,
    peak_merge_cpm: float,
    stationary_min_peak_dominance: float,
    motion_min_peak_dominance: float,
    ranker_max_candidates: int,
    ranker_min_score: float,
    ranker_score_margin: float,
    candidate_tolerance_cpm: float,
    subwindow_seconds: float,
    subwindow_hop_seconds: float,
    out_dir: Path,
    fragment_seconds: float,
) -> dict:
    """Extract VPU breath CPM from a carrier-band Hilbert envelope.

    Input shape: mono analysis signal `[time]`, already centered and decimated.
    The operation is non-causal and uses a DFT search over the full envelope
    curve. Output rates are cycles per minute.
    """
    curve_bundle = extract_envelope_curve(
        analysis_signal,
        analysis_rate,
        envelope_rate_target,
        carrier_low_hz,
        carrier_high_hz,
    )
    bandpassed = curve_bundle["bandpassed"]
    hilbert_envelope = curve_bundle["hilbert_envelope"]
    envelope_ds = curve_bundle["envelope_ds"]
    breath_curve = curve_bundle["breath_curve"]
    envelope_rate = curve_bundle["envelope_rate_hz"]

    autocorr_est = estimate_autocorr_bpm(breath_curve, envelope_rate, bpm_min=cpm_min, bpm_max=cpm_max)
    dft_est = estimate_dft_bpm(breath_curve, envelope_rate, bpm_min=cpm_min, bpm_max=cpm_max)
    peak_summary = summarize_dft_peaks(dft_est, competitive_peak_ratio, peak_merge_cpm)
    dominance = peak_summary["dominance_ratio"]
    dft_cpm = dft_est.get("candidate_bpm")
    autocorr_cpm = autocorr_est.get("candidate_bpm")
    autocorr_delta = abs(dft_cpm - autocorr_cpm) if dft_cpm is not None and autocorr_cpm is not None else None

    candidate_ranking: list[dict] = []
    band_profiles: list[dict] = []
    subwindow_profiles: list[dict] = []
    selected_cpm = dft_cpm
    selected_by = "max_peak"
    confidence = min(1.0, 0.35 + 0.18 * min(dominance, 3.0))
    status = "valid"
    warnings: list[str] = []
    min_required_dominance = motion_min_peak_dominance if condition in MOTION_CONDITIONS else stationary_min_peak_dominance
    if dft_cpm is None:
        status = "rejected"
        selected_cpm = None
        confidence = 0.0
        warnings.append("no_dft_candidate")
    elif selection_mode == "candidate_ranker":
        band_profiles = build_band_profiles(
            analysis_signal,
            analysis_rate,
            envelope_rate_target,
            carrier_low_hz,
            carrier_high_hz,
            cpm_min,
            cpm_max,
        )
        subwindow_profiles = build_subwindow_profiles(
            breath_curve,
            envelope_rate,
            cpm_min,
            cpm_max,
            subwindow_seconds,
            subwindow_hop_seconds,
        )
        candidate_ranking = rank_dft_candidates(
            peak_summary,
            autocorr_est,
            band_profiles,
            subwindow_profiles,
            cpm_min,
            condition,
            candidate_tolerance_cpm,
            ranker_max_candidates,
        )
        if not candidate_ranking:
            status = "rejected"
            selected_cpm = None
            confidence = 0.0
            warnings.append("no_rankable_candidate")
        else:
            best = candidate_ranking[0]
            runner_up = candidate_ranking[1] if len(candidate_ranking) > 1 else None
            margin = best["score"] - (runner_up["score"] if runner_up else 0.0)
            selected_cpm = best["cpm"]
            selected_by = "candidate_ranker"
            confidence = min(1.0, 0.30 + 0.55 * best["score"] + 0.20 * max(margin, 0.0))
            if best["score"] < ranker_min_score:
                status = "ambiguous"
                selected_cpm = None
                confidence = min(confidence, 0.42)
                warnings.append("candidate_ranker_low_score")
            elif runner_up is not None and margin < ranker_score_margin:
                status = "ambiguous"
                selected_cpm = None
                confidence = min(confidence, 0.45)
                warnings.append("candidate_ranker_small_margin")
            elif peak_summary["competitive_peak_count"] > 1:
                warnings.append("resolved_competing_dft_peaks_with_candidate_ranker")
            if status == "valid" and dft_cpm is not None and abs(best["cpm"] - dft_cpm) > candidate_tolerance_cpm:
                warnings.append("selected_non_top_candidate_with_candidate_ranker")
    elif peak_summary["competitive_peak_count"] > 1 and dominance < min_required_dominance:
        status = "ambiguous"
        selected_cpm = None
        confidence = min(confidence, 0.40)
        warnings.append("multiple_competing_dft_peaks")
        if condition in MOTION_CONDITIONS:
            warnings.append("motion_artifact_may_dominate_envelope")
    if dft_cpm is not None and dft_cpm - cpm_min < 2.0:
        warnings.append("candidate_near_cpm_min_check_condition_prior")
        confidence = min(confidence, 0.55)
    if autocorr_delta is not None and autocorr_delta <= 4.0:
        confidence = min(1.0, confidence + 0.12)
    elif autocorr_delta is not None:
        warnings.append("autocorr_disagrees_with_dft")
        confidence = min(confidence, 0.72)
    if status == "ambiguous":
        selected_cpm = None
        confidence = min(confidence, 0.45)

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
        "dft_peak_summary": peak_summary,
        "selection_mode": selection_mode,
        "selected_by": selected_by,
        "selected_cpm": selected_cpm,
        "candidate_ranking": candidate_ranking,
        "band_profiles": [
            {
                "band_low_hz": profile["band_low_hz"],
                "band_high_hz": profile["band_high_hz"],
                "energy_ratio": profile["energy_ratio"],
                "candidate_cpm": profile["dft_estimate"].get("candidate_bpm"),
            }
            for profile in band_profiles
        ],
        "subwindow_profiles": [
            {
                "start_seconds": profile["start_seconds"],
                "end_seconds": profile["end_seconds"],
                "candidate_cpm": profile["dft_estimate"].get("candidate_bpm"),
            }
            for profile in subwindow_profiles
        ],
        "min_required_peak_dominance_ratio": min_required_dominance,
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
    if not (0.0 < args.competitive_peak_ratio <= 1.0):
        raise SystemExit("--competitive-peak-ratio must be in (0, 1].")
    if args.peak_merge_cpm <= 0.0:
        raise SystemExit("--peak-merge-cpm must be positive.")
    if args.stationary_min_peak_dominance < 1.0 or args.motion_min_peak_dominance < 1.0:
        raise SystemExit("--stationary-min-peak-dominance and --motion-min-peak-dominance must be at least 1.")
    if args.ranker_max_candidates <= 0:
        raise SystemExit("--ranker-max-candidates must be positive.")
    if not (0.0 <= args.ranker_min_score <= 1.0):
        raise SystemExit("--ranker-min-score must be in [0, 1].")
    if args.ranker_score_margin < 0.0:
        raise SystemExit("--ranker-score-margin must be non-negative.")
    if args.candidate_tolerance_cpm <= 0.0:
        raise SystemExit("--candidate-tolerance-cpm must be positive.")
    if args.subwindow_seconds <= 0.0 or args.subwindow_hop_seconds <= 0.0:
        raise SystemExit("--subwindow-seconds and --subwindow-hop-seconds must be positive.")

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
        args.condition,
        args.selection_mode,
        args.competitive_peak_ratio,
        args.peak_merge_cpm,
        args.stationary_min_peak_dominance,
        args.motion_min_peak_dominance,
        args.ranker_max_candidates,
        args.ranker_min_score,
        args.ranker_score_margin,
        args.candidate_tolerance_cpm,
        args.subwindow_seconds,
        args.subwindow_hop_seconds,
        out_dir,
        args.fragment_seconds,
    )

    top_candidate_cpm = baseline["dft_estimate"].get("candidate_bpm")
    selected_candidate_cpm = baseline["selected_cpm"]
    final_breath_cpm = selected_candidate_cpm if baseline["status"] == "valid" else None
    error_vs_expected = (
        final_breath_cpm - args.expected_cpm
        if final_breath_cpm is not None and args.expected_cpm is not None
        else None
    )
    top_candidate_error_vs_expected = (
        top_candidate_cpm - args.expected_cpm
        if top_candidate_cpm is not None and args.expected_cpm is not None
        else None
    )
    final_prediction = {
        "status": baseline["status"],
        "breath_cpm": final_breath_cpm,
        "selected_candidate_cpm": selected_candidate_cpm,
        "top_candidate_cpm": top_candidate_cpm,
        "selected_by": baseline["selected_by"],
        "selection_mode": args.selection_mode,
        "confidence": baseline["confidence"],
        "condition": args.condition,
        "cpm_bound_source": cpm_bound_source,
        "cpm_search_min": cpm_min,
        "cpm_search_max": cpm_max,
        "dft_peak_dominance_ratio": baseline["dft_peak_dominance_ratio"],
        "min_required_peak_dominance_ratio": baseline["min_required_peak_dominance_ratio"],
        "competitive_peak_count": baseline["dft_peak_summary"]["competitive_peak_count"],
        "competitive_peaks": baseline["dft_peak_summary"]["competitive_peaks"],
        "top_peaks": baseline["dft_peak_summary"]["peaks"],
        "candidate_ranking": baseline["candidate_ranking"],
        "autocorr_cpm": baseline["autocorr_estimate"].get("candidate_bpm"),
        "autocorr_delta_cpm": baseline["autocorr_delta_cpm"],
        "expected_cpm": args.expected_cpm,
        "error_vs_expected_cpm": error_vs_expected,
        "top_candidate_error_vs_expected_cpm": top_candidate_error_vs_expected,
        "warnings": baseline["warnings"],
    }

    summary = {
        "file_name": args.wav_path.name,
        "metadata": metadata,
        "vpu_envelope_baseline": {
            "method": "vpu_carrier_hilbert_envelope_dft_baseline",
            "analysis_sample_rate_hz": analysis_rate,
            "condition": args.condition,
            "selection_mode": args.selection_mode,
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
    if top_candidate_cpm is None:
        print("FINAL_VPU_CPM rejected confidence=0.000")
    elif baseline["status"] == "ambiguous":
        candidates = ",".join(f"{peak['bpm']:.1f}" for peak in baseline["dft_peak_summary"]["competitive_peaks"])
        print(
            "FINAL_VPU_CPM ambiguous "
            f"top_candidate={top_candidate_cpm:.2f} "
            f"confidence={baseline['confidence']:.3f} "
            f"condition={args.condition} "
            f"range={cpm_min:.1f}-{cpm_max:.1f} "
            f"competitive_candidates={candidates}"
        )
    else:
        selected_note = ""
        if selected_candidate_cpm is not None and abs(selected_candidate_cpm - top_candidate_cpm) > 1e-9:
            selected_note = f" top_candidate={top_candidate_cpm:.2f}"
        print(
            "FINAL_VPU_CPM "
            f"{selected_candidate_cpm:.2f} "
            f"status={baseline['status']} "
            f"confidence={baseline['confidence']:.3f} "
            f"condition={args.condition} "
            f"range={cpm_min:.1f}-{cpm_max:.1f} "
            f"selected_by={baseline['selected_by']}"
            f"{selected_note}"
        )


if __name__ == "__main__":
    main()
