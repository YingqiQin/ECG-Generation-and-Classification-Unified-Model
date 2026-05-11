#!/usr/bin/env python3
"""RespEar-inspired respiration extraction for VPU WAV files.

This script is separate from the in-ear HR pipeline. It treats VPU respiration
as a weak periodic rhythm problem and adds motion-aware candidate scoring plus
SSA decomposition so motion-dominant low-frequency peaks do not automatically
win over plausible breathing components.

Expected input shape: mono or multi-channel PCM16 WAV. If a file is
multi-channel, channel 0 is used. The processing is non-causal because it
analyzes the whole recording or full analysis windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract VPU respiration-rate candidates from a WAV file.")
    parser.add_argument("wav_path", type=Path, help="Path to VPU WAV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim"),
        help="Base output directory",
    )
    parser.add_argument(
        "--analysis-rate",
        type=float,
        default=100.0,
        help="Target sample rate for low-frequency VPU analysis",
    )
    parser.add_argument(
        "--respiration-rate",
        type=float,
        default=20.0,
        help="Target sample rate for respiration curves before CPM estimation",
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["whole", "windows"],
        default="whole",
        help="Analyze the whole file once by default, or split long recordings into windows",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=60.0,
        help="Analysis window length in seconds when --analysis-mode windows is used",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=30.0,
        help="Hop length between analysis windows when --analysis-mode windows is used",
    )
    parser.add_argument(
        "--cpm-min",
        type=float,
        default=8.0,
        help="Lower breaths-per-minute bound",
    )
    parser.add_argument(
        "--cpm-max",
        type=float,
        default=40.0,
        help="Upper breaths-per-minute bound",
    )
    parser.add_argument(
        "--direct-low-hz",
        type=float,
        default=0.10,
        help="Lower cutoff for direct respiration branch",
    )
    parser.add_argument(
        "--direct-high-hz",
        type=float,
        default=0.70,
        help="Upper cutoff for direct respiration branch",
    )
    parser.add_argument(
        "--carrier-envelope-low-hz",
        type=float,
        default=8.0,
        help="Lower carrier bandpass cutoff for the stationary Hilbert envelope branch",
    )
    parser.add_argument(
        "--carrier-envelope-high-hz",
        type=float,
        default=40.0,
        help="Upper carrier bandpass cutoff for the stationary Hilbert envelope branch",
    )
    parser.add_argument(
        "--carrier-min-energy-ratio",
        type=float,
        default=0.03,
        help="Minimum carrier-band RMS/raw RMS ratio required before using carrier envelope candidates",
    )
    parser.add_argument(
        "--motion-low-hz",
        type=float,
        default=0.05,
        help="Lower bound for motion-spectrum inspection",
    )
    parser.add_argument(
        "--motion-high-hz",
        type=float,
        default=3.0,
        help="Upper bound for motion-spectrum inspection",
    )
    parser.add_argument(
        "--preset",
        choices=["auto", "stationary", "motion"],
        default="auto",
        help="Condition prior used only for scoring/rejection",
    )
    parser.add_argument(
        "--selection-policy",
        choices=["auto", "stationary_legacy_envelope", "motion_robust"],
        default="auto",
        help=(
            "Final candidate policy. auto uses the legacy Hilbert-envelope branch first for stationary VPU, "
            "while motion_robust keeps the multi-branch RespEar-style selector."
        ),
    )
    parser.add_argument(
        "--legacy-envelope-rate",
        type=float,
        default=50.0,
        help="Target curve rate for the stationary legacy Hilbert-envelope branch",
    )
    parser.add_argument(
        "--boundary-guard-cpm",
        type=float,
        default=2.0,
        help="Reject final candidates closer than this to --cpm-min as lower-bound locks",
    )
    parser.add_argument(
        "--legacy-min-dominance",
        type=float,
        default=1.05,
        help="Minimum DFT peak dominance for accepting the stationary legacy envelope branch",
    )
    parser.add_argument(
        "--motion-min-cpm",
        type=float,
        default=14.0,
        help="Motion windows below this CPM are treated as suspicious unless strongly supported",
    )
    parser.add_argument(
        "--expected-cpm",
        type=float,
        default=None,
        help="Optional reference CPM for evaluation only; not used for selecting candidates",
    )
    parser.add_argument(
        "--ssa-window-seconds",
        type=float,
        default=6.0,
        help="SSA embedding window in seconds on the respiration-rate curve",
    )
    parser.add_argument(
        "--ssa-components",
        type=int,
        default=8,
        help="Maximum SSA components to inspect",
    )
    return parser.parse_args()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) * (value - mean) for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def peak_dominance_from_dft(dft_estimate: dict, candidate_cpm: float | None = None) -> float:
    peaks = dft_estimate.get("top_peaks") or []
    if not peaks:
        return 0.0
    if candidate_cpm is None:
        first_power = peaks[0].get("power", 0.0)
    else:
        first_power = min(peaks, key=lambda item: abs(item["bpm"] - candidate_cpm)).get("power", 0.0)
    competitors = [peak.get("power", 0.0) for peak in peaks if abs(peak["bpm"] - (candidate_cpm or peaks[0]["bpm"])) > 1e-9]
    second_power = max(competitors, default=0.0)
    return first_power / max(second_power, 1e-12)


def estimate_peak_count_cpm(values: list[float], sample_rate_hz: float, cpm_min: float, cpm_max: float) -> dict:
    """Estimate CPM by counting separated local maxima.

    This is deliberately conservative and used as a supporting check, not as a
    standalone estimator.
    """
    if len(values) < 3:
        return {"status": "insufficient_data"}
    signal = centered(values)
    sigma = stddev(signal)
    if sigma <= 0.0:
        return {"status": "flat_signal"}
    threshold = 0.25 * sigma
    min_gap = max(int(sample_rate_hz * 60.0 / cpm_max * 0.65), 1)
    peaks: list[int] = []
    last_peak = -min_gap
    for idx in range(1, len(signal) - 1):
        if idx - last_peak < min_gap:
            continue
        if signal[idx] > threshold and signal[idx] >= signal[idx - 1] and signal[idx] > signal[idx + 1]:
            peaks.append(idx)
            last_peak = idx
    duration_seconds = len(values) / sample_rate_hz
    if duration_seconds <= 0.0 or not peaks:
        return {"status": "no_peak"}
    cpm = len(peaks) * 60.0 / duration_seconds
    return {
        "status": "candidate_only",
        "peak_count": len(peaks),
        "candidate_cpm": cpm,
        "duration_seconds": duration_seconds,
    }


def make_rate_spectrum_svg(
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


def jacobi_eigen_symmetric(matrix: list[list[float]], max_sweeps: int = 80, tolerance: float = 1e-10) -> list[tuple[float, list[float]]]:
    """Return eigenpairs for a small symmetric matrix using Jacobi rotations."""
    n = len(matrix)
    if n == 0:
        return []
    a = [row[:] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_sweeps):
        p = 0
        q = 1 if n > 1 else 0
        max_offdiag = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                value = abs(a[i][j])
                if value > max_offdiag:
                    max_offdiag = value
                    p = i
                    q = j
        if max_offdiag < tolerance or p == q:
            break

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]
        tau = (aqq - app) / (2.0 * apq) if abs(apq) > 1e-20 else 0.0
        t = math.copysign(1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)), tau) if tau != 0.0 else 1.0
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        for k in range(n):
            if k != p and k != q:
                akp = a[k][p]
                akq = a[k][q]
                a[k][p] = c * akp - s * akq
                a[p][k] = a[k][p]
                a[k][q] = s * akp + c * akq
                a[q][k] = a[k][q]

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = 0.0
        a[q][p] = 0.0

        for k in range(n):
            vkp = vectors[k][p]
            vkq = vectors[k][q]
            vectors[k][p] = c * vkp - s * vkq
            vectors[k][q] = s * vkp + c * vkq

    eigenpairs = []
    for idx in range(n):
        eigenvector = [vectors[row][idx] for row in range(n)]
        eigenpairs.append((max(a[idx][idx], 0.0), eigenvector))
    eigenpairs.sort(key=lambda item: item[0], reverse=True)
    return eigenpairs


def ssa_decompose(values: list[float], window_length: int, max_components: int) -> list[dict]:
    """Decompose a 1D signal into SSA reconstructed components.

    Input shape: one respiration-rate curve, `[time]`.
    Output shape: list of components, each with a reconstructed `[time]` series.
    """
    n = len(values)
    window_length = max(2, min(window_length, n // 2 if n >= 4 else 2))
    if n < 8 or window_length >= n:
        return []

    signal = centered(values)
    k_count = n - window_length + 1
    covariance = [[0.0 for _ in range(window_length)] for _ in range(window_length)]
    for i in range(window_length):
        for j in range(i, window_length):
            total = 0.0
            for k in range(k_count):
                total += signal[i + k] * signal[j + k]
            value = total / k_count
            covariance[i][j] = value
            covariance[j][i] = value

    eigenpairs = jacobi_eigen_symmetric(covariance)
    total_eigenvalue = sum(value for value, _ in eigenpairs) or 1.0
    components = []
    for component_index, (eigenvalue, eigenvector) in enumerate(eigenpairs[:max_components]):
        pcs = []
        for k in range(k_count):
            pcs.append(sum(eigenvector[i] * signal[i + k] for i in range(window_length)))

        reconstructed = [0.0] * n
        counts = [0] * n
        for i in range(window_length):
            for k, pc in enumerate(pcs):
                reconstructed[i + k] += eigenvector[i] * pc
                counts[i + k] += 1
        for idx, count in enumerate(counts):
            if count:
                reconstructed[idx] /= count

        components.append(
            {
                "component_index": component_index,
                "eigenvalue": eigenvalue,
                "energy_fraction": eigenvalue / total_eigenvalue,
                "values": reconstructed,
            }
        )
    return components


def build_respiration_curves(
    window_signal: list[float],
    analysis_rate: float,
    respiration_rate: float,
    low_hz: float,
    high_hz: float,
    carrier_low_hz: float,
    carrier_high_hz: float,
) -> dict:
    taps = make_bandpass_fir(analysis_rate, low_hz, high_hz, num_taps=257)
    direct_bandpassed = fir_filter(window_signal, taps)
    stride = max(int(analysis_rate / respiration_rate), 1)
    direct_ds = decimate_mean(direct_bandpassed, stride)
    curve_rate = analysis_rate / stride
    direct_curve = moving_average(direct_ds, max(int(curve_rate * 0.25), 1))

    abs_envelope = moving_average([abs(value) for value in window_signal], max(int(analysis_rate * 0.20), 1))
    envelope_ds = decimate_mean(abs_envelope, stride)
    envelope_smooth = moving_average(envelope_ds, max(int(curve_rate * 0.50), 1))
    slow_trend = moving_average(envelope_smooth, max(int(curve_rate * 5.0), 1))
    envelope_curve = [value - trend for value, trend in zip(envelope_smooth, slow_trend)]
    envelope_curve = moving_average(envelope_curve, max(int(curve_rate * 0.50), 1))

    carrier_bandpassed: list[float] = []
    carrier_envelope_curve: list[float] = []
    carrier_energy_ratio = 0.0
    nyquist = analysis_rate / 2.0
    safe_carrier_high = min(carrier_high_hz, nyquist * 0.90)
    if 0.0 < carrier_low_hz < safe_carrier_high:
        carrier_taps = make_bandpass_fir(analysis_rate, carrier_low_hz, safe_carrier_high, num_taps=129)
        carrier_bandpassed = fir_filter(window_signal, carrier_taps)
        carrier_energy_ratio = rms(carrier_bandpassed) / max(rms(window_signal), 1e-12)
        carrier_analytic = analytic_signal(carrier_bandpassed)
        carrier_envelope = [abs(value) for value in carrier_analytic]
        carrier_envelope_ds = decimate_mean(carrier_envelope, stride)
        carrier_envelope_smooth = moving_average(carrier_envelope_ds, max(int(curve_rate * 0.20), 1))
        carrier_slow_trend = moving_average(carrier_envelope_smooth, max(int(curve_rate * 2.0), 1))
        carrier_envelope_curve = [
            value - trend for value, trend in zip(carrier_envelope_smooth, carrier_slow_trend)
        ]
        carrier_envelope_curve = moving_average(carrier_envelope_curve, max(int(curve_rate * 0.30), 1))

    return {
        "direct_bandpassed": direct_bandpassed,
        "direct_curve": direct_curve,
        "envelope_curve": envelope_curve,
        "carrier_bandpassed": carrier_bandpassed,
        "carrier_envelope_curve": carrier_envelope_curve,
        "carrier_envelope_low_hz": carrier_low_hz,
        "carrier_envelope_high_hz": safe_carrier_high,
        "carrier_energy_ratio": carrier_energy_ratio,
        "curve_rate_hz": curve_rate,
    }


def build_stationary_legacy_envelope_curve(
    window_signal: list[float],
    analysis_rate: float,
    target_curve_rate: float,
    carrier_low_hz: float,
    carrier_high_hz: float,
) -> dict:
    """Mirror extract_stationary_hr.build_envelope_branch for stationary VPU files.

    Input shape: one mono analysis window, `[time]`, already centered and
    decimated. The branch is non-causal and returns a one-dimensional envelope
    curve for CPM estimation.
    """
    nyquist = analysis_rate / 2.0
    safe_carrier_high = min(carrier_high_hz, nyquist * 0.90)
    if target_curve_rate <= 0.0 or not (0.0 < carrier_low_hz < safe_carrier_high):
        return {
            "status": "invalid_band_or_rate",
            "bandpassed": [],
            "hilbert_envelope": [],
            "curve": [],
            "curve_rate_hz": 0.0,
            "carrier_energy_ratio": 0.0,
            "carrier_envelope_low_hz": carrier_low_hz,
            "carrier_envelope_high_hz": safe_carrier_high,
        }

    taps = make_bandpass_fir(analysis_rate, carrier_low_hz, safe_carrier_high, num_taps=129)
    bandpassed = fir_filter(window_signal, taps)
    analytic = analytic_signal(bandpassed)
    hilbert_envelope = [abs(value) for value in analytic]

    stride = max(int(analysis_rate / target_curve_rate), 1)
    envelope_ds = decimate_mean(hilbert_envelope, stride)
    curve_rate = analysis_rate / stride
    envelope_smooth = moving_average(envelope_ds, max(int(curve_rate * 0.20), 1))
    slow_trend = moving_average(envelope_smooth, max(int(curve_rate * 2.0), 1))
    legacy_curve = [value - trend for value, trend in zip(envelope_smooth, slow_trend)]
    legacy_curve = moving_average(legacy_curve, max(int(curve_rate * 0.30), 1))

    return {
        "status": "ok",
        "bandpassed": bandpassed,
        "hilbert_envelope": hilbert_envelope,
        "curve": legacy_curve,
        "curve_rate_hz": curve_rate,
        "carrier_energy_ratio": rms(bandpassed) / max(rms(window_signal), 1e-12),
        "carrier_envelope_low_hz": carrier_low_hz,
        "carrier_envelope_high_hz": safe_carrier_high,
    }


def inspect_motion(window_signal: list[float], analysis_rate: float, motion_low_hz: float, motion_high_hz: float) -> dict:
    motion_min_cpm = motion_low_hz * 60.0
    motion_max_cpm = motion_high_hz * 60.0
    dft = estimate_dft_bpm(window_signal, analysis_rate, bpm_min=motion_min_cpm, bpm_max=motion_max_cpm)
    dominance = peak_dominance_from_dft(dft)
    return {
        "dft_estimate": dft,
        "dominant_motion_cpm": dft.get("candidate_bpm"),
        "dominance_ratio": dominance,
    }


def make_candidate(
    source: str,
    cpm: float,
    power: float,
    dominance: float,
    score_seed: float,
    extra: dict | None = None,
) -> dict:
    return {
        "source": source,
        "candidate_cpm": cpm,
        "power": power,
        "dominance_ratio": dominance,
        "score_seed": score_seed,
        **(extra or {}),
    }


def dft_candidates_from_signal(
    source: str,
    signal: list[float],
    sample_rate_hz: float,
    cpm_min: float,
    cpm_max: float,
    max_candidates: int = 5,
) -> tuple[list[dict], dict, dict, dict]:
    dft = estimate_dft_bpm(signal, sample_rate_hz, bpm_min=cpm_min, bpm_max=cpm_max)
    autocorr = estimate_autocorr_bpm(signal, sample_rate_hz, bpm_min=cpm_min, bpm_max=cpm_max)
    peak_count = estimate_peak_count_cpm(signal, sample_rate_hz, cpm_min, cpm_max)
    candidates: list[dict] = []
    peaks = dft.get("top_peaks") or []
    for rank, peak in enumerate(peaks[:max_candidates]):
        dominance = peak_dominance_from_dft(dft, candidate_cpm=peak["bpm"])
        seed = math.log10(max(peak["power"], 1.0)) + min(dominance, 6.0)
        lower_edge_distance_cpm = peak["bpm"] - cpm_min
        upper_edge_distance_cpm = cpm_max - peak["bpm"]
        candidates.append(
            make_candidate(
                source,
                peak["bpm"],
                peak["power"],
                dominance,
                seed,
                {
                    "rank": rank + 1,
                    "autocorr_cpm": autocorr.get("candidate_bpm"),
                    "autocorr_strength": autocorr.get("autocorrelation"),
                    "peak_count_cpm": peak_count.get("candidate_cpm"),
                    "lower_edge_distance_cpm": lower_edge_distance_cpm,
                    "upper_edge_distance_cpm": upper_edge_distance_cpm,
                },
            )
        )
    return candidates, dft, autocorr, peak_count


def close_to_any(value: float, targets: list[float], tolerance: float) -> bool:
    return any(abs(value - target) <= tolerance for target in targets if target > 0.0)


def score_candidate(
    candidate: dict,
    motion_profile: dict,
    preset: str,
    motion_min_cpm: float,
    boundary_guard_cpm: float,
) -> tuple[float, list[str]]:
    score = candidate["score_seed"]
    reasons: list[str] = []
    cpm = candidate["candidate_cpm"]
    lower_edge_distance = candidate.get("lower_edge_distance_cpm")

    if lower_edge_distance is not None and lower_edge_distance < boundary_guard_cpm:
        if candidate["source"] in {"direct", "envelope"}:
            score -= 4.0
            reasons.append("near_lower_bound_penalty")
        elif candidate["source"].startswith("ssa"):
            score -= 2.0
            reasons.append("near_lower_bound_penalty")
        elif candidate["source"] == "stationary_legacy_envelope":
            score -= 8.0
            reasons.append("near_lower_bound_penalty")

    if candidate["source"] == "direct":
        if preset == "motion":
            score += 5.0
            reasons.append("direct_vpu_motion_support")
        elif preset == "stationary":
            score += 5.0
            reasons.append("direct_vpu_stationary_support")
        elif preset == "auto":
            score += 2.0
            reasons.append("direct_vpu_auto_support")

    if candidate["source"] == "carrier_envelope":
        if preset == "stationary":
            score += 10.0
            reasons.append("carrier_envelope_stationary_support")
        elif preset == "auto":
            score += 2.0
            reasons.append("carrier_envelope_auto_support")
        else:
            score += 1.0
            reasons.append("carrier_envelope_support")

    if candidate["source"] == "stationary_legacy_envelope":
        if preset == "stationary":
            score += 14.0
            reasons.append("legacy_envelope_stationary_support")
        elif preset == "auto":
            score += 3.0
            reasons.append("legacy_envelope_auto_support")
        else:
            score += 1.0
            reasons.append("legacy_envelope_support")

    autocorr_cpm = candidate.get("autocorr_cpm")
    autocorr_strength = candidate.get("autocorr_strength") or 0.0
    if autocorr_cpm is not None:
        delta = abs(cpm - autocorr_cpm)
        if delta <= 3.0:
            score += 3.0 + max(autocorr_strength, 0.0)
            reasons.append("autocorr_agrees")
        elif delta <= 7.0:
            score += 1.0
            reasons.append("autocorr_nearby")

    peak_count_cpm = candidate.get("peak_count_cpm")
    if peak_count_cpm is not None:
        delta = abs(cpm - peak_count_cpm)
        if delta <= 4.0:
            score += 1.5
            reasons.append("peak_count_agrees")

    if candidate["source"].startswith("ssa"):
        score += 1.0 + 3.0 * min(candidate.get("energy_fraction", 0.0), 0.5)
        reasons.append("ssa_support")

    motion_like = preset == "motion"
    if preset == "auto":
        motion_cpm = motion_profile.get("dominant_motion_cpm")
        dominance = motion_profile.get("dominance_ratio") or 0.0
        motion_like = bool(motion_cpm and dominance >= 8.0 and (motion_cpm < 8.0 or motion_cpm > 45.0))

    motion_cpm = motion_profile.get("dominant_motion_cpm") or 0.0
    motion_targets = [motion_cpm, motion_cpm / 2.0, motion_cpm / 3.0, motion_cpm * 2.0]
    if motion_like and cpm < motion_min_cpm:
        score -= 8.0
        reasons.append("motion_low_cpm_penalty")
    if motion_like and candidate["source"] != "direct" and cpm <= motion_min_cpm + 1.0:
        score -= 8.0
        reasons.append("motion_low_nondirect_penalty")
    if motion_like and close_to_any(cpm, motion_targets, tolerance=1.5):
        score -= 3.0
        reasons.append("near_motion_or_subharmonic")

    return score, reasons


def source_family(source: str) -> str:
    if source == "direct":
        return "direct"
    if source == "envelope":
        return "envelope"
    if source == "carrier_envelope":
        return "carrier_envelope"
    if source == "stationary_legacy_envelope":
        return "stationary_legacy_envelope"
    if source == "ssa_aggregate":
        return "ssa_aggregate"
    if source.startswith("ssa_component"):
        return "ssa_component"
    return source


def combine_scored_candidates(
    scored: list[dict],
    preset: str,
    boundary_guard_cpm: float,
    tolerance_cpm: float = 2.5,
) -> list[dict]:
    groups: list[dict] = []
    for candidate in sorted(scored, key=lambda item: item["score"], reverse=True):
        target_group = None
        for group in groups:
            if abs(candidate["candidate_cpm"] - group["candidate_cpm"]) <= tolerance_cpm:
                target_group = group
                break
        if target_group is None:
            target_group = {
                "candidate_cpm": candidate["candidate_cpm"],
                "support_sources": [],
                "members": [],
                "score_weighted_total": 0.0,
                "weight_total": 0.0,
            }
            groups.append(target_group)
        weight = max(candidate["score"], 0.1)
        target_group["score_weighted_total"] += candidate["candidate_cpm"] * weight
        target_group["weight_total"] += weight
        target_group["support_sources"].append(candidate["source"])
        target_group["members"].append(candidate)

    for group in groups:
        if group["weight_total"] > 0.0:
            group["candidate_cpm"] = group["score_weighted_total"] / group["weight_total"]
        family_scores: dict[str, float] = {}
        for member in group["members"]:
            family = source_family(member["source"])
            family_scores[family] = max(family_scores.get(family, -1e12), member["score"])
        group["support_families"] = sorted(family_scores)
        group["support_count"] = len(group["support_families"])
        group["combined_score"] = sum(family_scores.values()) + 1.5 * max(group["support_count"] - 1, 0)
        lower_edge_distances = [
            member.get("lower_edge_distance_cpm")
            for member in group["members"]
            if member.get("lower_edge_distance_cpm") is not None
        ]
        group["min_lower_edge_distance_cpm"] = min(lower_edge_distances) if lower_edge_distances else None
        if (
            group["min_lower_edge_distance_cpm"] is not None
            and group["min_lower_edge_distance_cpm"] < boundary_guard_cpm
        ):
            group["boundary_warning"] = "near_cpm_min"
        group["support_sources"] = sorted(set(group["support_sources"]))
        del group["score_weighted_total"]
        del group["weight_total"]

    if preset in {"stationary", "motion"}:
        direct_groups = [group for group in groups if "direct" in group.get("support_families", [])]
        for group in groups:
            if "direct" in group.get("support_families", []):
                continue
            if preset == "stationary" and "carrier_envelope" in group.get("support_families", []):
                continue
            for direct_group in direct_groups:
                direct_cpm = direct_group["candidate_cpm"]
                if abs(group["candidate_cpm"] - 2.0 * direct_cpm) <= tolerance_cpm:
                    group["combined_score"] -= 55.0
                    group["harmonic_penalty"] = "near_double_direct_candidate"
                    break
                if preset == "motion" and 0.0 < group["candidate_cpm"] - direct_cpm <= tolerance_cpm * 2.0:
                    group["combined_score"] -= 8.0
                    group["harmonic_penalty"] = "near_upper_neighbor_of_direct_candidate"
                    break

    groups.sort(key=lambda item: item["combined_score"], reverse=True)
    return groups


def select_final_candidate(groups: list[dict]) -> dict:
    if not groups:
        return {
            "status": "rejected",
            "rr_cpm": None,
            "confidence": 0.0,
            "reject_reason": "no_candidate",
        }
    best = groups[0]
    second_score = groups[1]["combined_score"] if len(groups) > 1 else 0.0
    margin = best["combined_score"] - second_score
    confidence = clamp(0.18 * margin + 0.12 * best["support_count"] + 0.02 * best["combined_score"], 0.0, 1.0)
    status = "valid" if confidence >= 0.35 else "low_confidence"
    reject_reason = None if status == "valid" else "weak_candidate_margin_or_support"
    if best.get("boundary_warning") == "near_cpm_min":
        return {
            "status": "rejected",
            "rr_cpm": None,
            "confidence": 0.0,
            "reject_reason": "candidate_near_cpm_min_boundary",
            "selected_group": best,
            "runner_up": groups[1] if len(groups) > 1 else None,
        }
    return {
        "status": status,
        "rr_cpm": best["candidate_cpm"],
        "confidence": confidence,
        "reject_reason": reject_reason,
        "selected_group": best,
        "runner_up": groups[1] if len(groups) > 1 else None,
    }


def make_policy_final(candidate: dict, source: str, selection_policy: str, confidence: float) -> dict:
    group = {
        "candidate_cpm": candidate["candidate_cpm"],
        "support_sources": [source],
        "support_families": [source_family(source)],
        "support_count": 1,
        "combined_score": candidate.get("score", candidate.get("score_seed", 0.0)),
        "members": [candidate],
    }
    return {
        "status": "valid",
        "rr_cpm": candidate["candidate_cpm"],
        "confidence": confidence,
        "reject_reason": None,
        "selection_policy": selection_policy,
        "selected_group": group,
        "runner_up": None,
    }


def select_stationary_legacy_envelope(
    legacy_candidates: list[dict],
    carrier_energy_ratio: float,
    min_energy_ratio: float,
    boundary_guard_cpm: float,
    min_dominance: float,
    selection_policy: str,
) -> dict:
    if carrier_energy_ratio < min_energy_ratio:
        return {
            "status": "rejected",
            "rr_cpm": None,
            "confidence": 0.0,
            "reject_reason": "stationary_legacy_envelope_low_carrier_energy",
            "selection_policy": selection_policy,
        }
    if not legacy_candidates:
        return {
            "status": "rejected",
            "rr_cpm": None,
            "confidence": 0.0,
            "reject_reason": "stationary_legacy_envelope_no_candidate",
            "selection_policy": selection_policy,
        }

    best = sorted(legacy_candidates, key=lambda item: item.get("rank", 999))[0]
    lower_edge_distance = best.get("lower_edge_distance_cpm")
    if lower_edge_distance is not None and lower_edge_distance < boundary_guard_cpm:
        return {
            "status": "rejected",
            "rr_cpm": None,
            "confidence": 0.0,
            "reject_reason": "stationary_legacy_envelope_near_cpm_min_boundary",
            "selection_policy": selection_policy,
            "selected_group": {
                "candidate_cpm": best["candidate_cpm"],
                "support_sources": ["stationary_legacy_envelope"],
                "support_families": ["stationary_legacy_envelope"],
                "support_count": 1,
                "boundary_warning": "near_cpm_min",
                "members": [best],
            },
        }

    dominance = best.get("dominance_ratio", 0.0)
    if dominance < min_dominance:
        return {
            "status": "low_confidence",
            "rr_cpm": best["candidate_cpm"],
            "confidence": 0.25,
            "reject_reason": "stationary_legacy_envelope_weak_dft_dominance",
            "selection_policy": selection_policy,
            "selected_group": {
                "candidate_cpm": best["candidate_cpm"],
                "support_sources": ["stationary_legacy_envelope"],
                "support_families": ["stationary_legacy_envelope"],
                "support_count": 1,
                "members": [best],
            },
        }

    confidence = clamp(0.48 + 0.09 * min(dominance, 4.0) + 0.08 * min(carrier_energy_ratio, 1.0), 0.0, 1.0)
    return make_policy_final(best, "stationary_legacy_envelope", selection_policy, confidence)


def choose_window_final(
    groups: list[dict],
    legacy_candidates: list[dict],
    legacy_energy_ratio: float,
    args: argparse.Namespace,
) -> dict:
    default_final = select_final_candidate(groups)
    default_final.setdefault("selection_policy", "motion_robust" if args.preset == "motion" else "multi_branch")

    use_legacy_first = args.selection_policy == "stationary_legacy_envelope" or (
        args.selection_policy == "auto" and args.preset == "stationary"
    )
    if not use_legacy_first:
        default_final["selection_policy"] = args.selection_policy
        return default_final

    legacy_final = select_stationary_legacy_envelope(
        legacy_candidates,
        legacy_energy_ratio,
        args.carrier_min_energy_ratio,
        args.boundary_guard_cpm,
        args.legacy_min_dominance,
        "stationary_legacy_envelope",
    )
    if legacy_final["status"] == "valid":
        return legacy_final
    if args.selection_policy == "stationary_legacy_envelope":
        return legacy_final

    default_final["legacy_fallback_reason"] = legacy_final.get("reject_reason")
    return default_final


def analyze_window(
    window_signal: list[float],
    analysis_rate: float,
    args: argparse.Namespace,
    out_dir: Path,
    window_index: int,
    start_seconds: float,
) -> dict:
    prefix = f"window_{window_index:02d}"
    curves = build_respiration_curves(
        window_signal,
        analysis_rate,
        args.respiration_rate,
        args.direct_low_hz,
        args.direct_high_hz,
        args.carrier_envelope_low_hz,
        args.carrier_envelope_high_hz,
    )
    direct_curve = curves["direct_curve"]
    envelope_curve = curves["envelope_curve"]
    carrier_envelope_curve = curves["carrier_envelope_curve"]
    curve_rate = curves["curve_rate_hz"]
    legacy_curves = build_stationary_legacy_envelope_curve(
        window_signal,
        analysis_rate,
        args.legacy_envelope_rate,
        args.carrier_envelope_low_hz,
        args.carrier_envelope_high_hz,
    )
    legacy_envelope_curve = legacy_curves["curve"]
    legacy_curve_rate = legacy_curves["curve_rate_hz"]

    motion_profile = inspect_motion(window_signal, analysis_rate, args.motion_low_hz, args.motion_high_hz)
    direct_candidates, direct_dft, direct_autocorr, direct_peak_count = dft_candidates_from_signal(
        "direct",
        direct_curve,
        curve_rate,
        args.cpm_min,
        args.cpm_max,
    )
    envelope_candidates, envelope_dft, envelope_autocorr, envelope_peak_count = dft_candidates_from_signal(
        "envelope",
        envelope_curve,
        curve_rate,
        args.cpm_min,
        args.cpm_max,
    )
    if curves["carrier_energy_ratio"] >= args.carrier_min_energy_ratio:
        carrier_candidates, carrier_dft, carrier_autocorr, carrier_peak_count = dft_candidates_from_signal(
            "carrier_envelope",
            carrier_envelope_curve,
            curve_rate,
            args.cpm_min,
            args.cpm_max,
        )
    else:
        carrier_candidates = []
        carrier_dft = {
            "status": "skipped_low_carrier_energy",
            "carrier_energy_ratio": curves["carrier_energy_ratio"],
            "required_min_energy_ratio": args.carrier_min_energy_ratio,
        }
        carrier_autocorr = {"status": "skipped_low_carrier_energy"}
        carrier_peak_count = {"status": "skipped_low_carrier_energy"}
    if legacy_envelope_curve:
        legacy_candidates, legacy_dft, legacy_autocorr, legacy_peak_count = dft_candidates_from_signal(
            "stationary_legacy_envelope",
            legacy_envelope_curve,
            legacy_curve_rate,
            args.cpm_min,
            args.cpm_max,
        )
    else:
        legacy_candidates = []
        legacy_dft = {"status": legacy_curves["status"]}
        legacy_autocorr = {"status": legacy_curves["status"]}
        legacy_peak_count = {"status": legacy_curves["status"]}

    ssa_window = max(int(curve_rate * args.ssa_window_seconds), 8)
    ssa_window = min(ssa_window, 80)
    components = ssa_decompose(envelope_curve, ssa_window, args.ssa_components)
    ssa_candidates: list[dict] = []
    retained_components: list[dict] = []
    aggregate = [0.0 for _ in envelope_curve]
    for component in components:
        component_candidates, component_dft, component_autocorr, component_peak_count = dft_candidates_from_signal(
            f"ssa_component_{component['component_index']:02d}",
            component["values"],
            curve_rate,
            args.cpm_min,
            args.cpm_max,
            max_candidates=2,
        )
        best_cpm = component_dft.get("candidate_bpm")
        dominance = peak_dominance_from_dft(component_dft)
        peak_count_cpm = component_peak_count.get("candidate_cpm")
        is_plausible = best_cpm is not None and args.cpm_min <= best_cpm <= args.cpm_max
        if peak_count_cpm is not None and abs(best_cpm - peak_count_cpm) > 10.0:
            is_plausible = False
        if dominance < 0.35:
            is_plausible = False
        component_meta = {
            "component_index": component["component_index"],
            "energy_fraction": component["energy_fraction"],
            "dft_estimate": component_dft,
            "autocorr_estimate": component_autocorr,
            "peak_count_estimate": component_peak_count,
            "retained": is_plausible,
        }
        if component["component_index"] < 4:
            component_svg = f"{prefix}_ssa_component_{component['component_index']:02d}.svg"
            make_series_svg(
                component["values"],
                out_dir / component_svg,
                1200,
                240,
                f"{prefix} SSA component {component['component_index']}",
            )
            component_meta["svg"] = component_svg
        if is_plausible:
            retained_components.append(component_meta)
            for idx, value in enumerate(component["values"]):
                aggregate[idx] += value
            for candidate in component_candidates:
                candidate["energy_fraction"] = component["energy_fraction"]
                ssa_candidates.append(candidate)
        else:
            retained_components.append(component_meta)

    aggregate_candidates: list[dict] = []
    aggregate_dft: dict = {"status": "no_candidate"}
    if any(value != 0.0 for value in aggregate):
        aggregate_candidates, aggregate_dft, _, _ = dft_candidates_from_signal(
            "ssa_aggregate",
            aggregate,
            curve_rate,
            args.cpm_min,
            args.cpm_max,
            max_candidates=3,
        )

    raw_candidates = direct_candidates + envelope_candidates + carrier_candidates + ssa_candidates + aggregate_candidates
    scored: list[dict] = []
    for candidate in raw_candidates:
        score, reasons = score_candidate(
            candidate,
            motion_profile,
            args.preset,
            args.motion_min_cpm,
            args.boundary_guard_cpm,
        )
        scored.append({**candidate, "score": score, "score_reasons": reasons})
    groups = combine_scored_candidates(scored, args.preset, args.boundary_guard_cpm)
    final = choose_window_final(groups, legacy_candidates, legacy_curves["carrier_energy_ratio"], args)

    make_series_svg(window_signal, out_dir / f"{prefix}_analysis_signal.svg", 1200, 260, f"{prefix} VPU analysis signal")
    make_series_svg(direct_curve, out_dir / f"{prefix}_direct_curve.svg", 1200, 260, f"{prefix} direct respiration curve")
    make_series_svg(envelope_curve, out_dir / f"{prefix}_envelope_curve.svg", 1200, 260, f"{prefix} envelope respiration curve")
    make_series_svg(
        carrier_envelope_curve,
        out_dir / f"{prefix}_carrier_envelope_curve.svg",
        1200,
        260,
        f"{prefix} carrier Hilbert-envelope respiration curve",
    )
    make_series_svg(
        legacy_envelope_curve,
        out_dir / f"{prefix}_stationary_legacy_envelope_curve.svg",
        1200,
        260,
        f"{prefix} stationary legacy Hilbert-envelope curve",
    )
    make_series_svg(aggregate, out_dir / f"{prefix}_ssa_aggregate.svg", 1200, 260, f"{prefix} retained SSA aggregate")
    make_rate_spectrum_svg(
        direct_dft.get("spectrum", []),
        out_dir / f"{prefix}_direct_dft_spectrum.svg",
        1200,
        320,
        f"{prefix} direct branch CPM spectrum",
        direct_dft.get("candidate_bpm"),
    )
    make_rate_spectrum_svg(
        envelope_dft.get("spectrum", []),
        out_dir / f"{prefix}_envelope_dft_spectrum.svg",
        1200,
        320,
        f"{prefix} envelope branch CPM spectrum",
        envelope_dft.get("candidate_bpm"),
    )
    make_rate_spectrum_svg(
        carrier_dft.get("spectrum", []),
        out_dir / f"{prefix}_carrier_envelope_dft_spectrum.svg",
        1200,
        320,
        f"{prefix} carrier envelope CPM spectrum",
        carrier_dft.get("candidate_bpm"),
    )
    make_rate_spectrum_svg(
        legacy_dft.get("spectrum", []),
        out_dir / f"{prefix}_stationary_legacy_envelope_dft_spectrum.svg",
        1200,
        320,
        f"{prefix} stationary legacy envelope CPM spectrum",
        legacy_dft.get("candidate_bpm"),
    )
    make_rate_spectrum_svg(
        motion_profile["dft_estimate"].get("spectrum", []),
        out_dir / f"{prefix}_motion_spectrum.svg",
        1200,
        320,
        f"{prefix} motion inspection spectrum",
        motion_profile.get("dominant_motion_cpm"),
    )
    make_rate_spectrum_svg(
        aggregate_dft.get("spectrum", []),
        out_dir / f"{prefix}_ssa_aggregate_spectrum.svg",
        1200,
        320,
        f"{prefix} retained SSA aggregate CPM spectrum",
        aggregate_dft.get("candidate_bpm"),
    )

    expected_error = None
    if args.expected_cpm is not None and final.get("rr_cpm") is not None:
        expected_error = final["rr_cpm"] - args.expected_cpm

    return {
        "window_index": window_index,
        "start_seconds": start_seconds,
        "end_seconds": start_seconds + len(window_signal) / analysis_rate,
        "duration_seconds": len(window_signal) / analysis_rate,
        "preset": args.preset,
        "rms": rms(window_signal),
        "motion_profile": motion_profile,
        "branches": {
            "direct": {
                "dft_estimate": direct_dft,
                "autocorr_estimate": direct_autocorr,
                "peak_count_estimate": direct_peak_count,
            },
            "envelope": {
                "dft_estimate": envelope_dft,
                "autocorr_estimate": envelope_autocorr,
                "peak_count_estimate": envelope_peak_count,
            },
            "carrier_envelope": {
                "bandpass_low_hz": curves["carrier_envelope_low_hz"],
                "bandpass_high_hz": curves["carrier_envelope_high_hz"],
                "carrier_energy_ratio": curves["carrier_energy_ratio"],
                "carrier_min_energy_ratio": args.carrier_min_energy_ratio,
                "dft_estimate": carrier_dft,
                "autocorr_estimate": carrier_autocorr,
                "peak_count_estimate": carrier_peak_count,
            },
            "stationary_legacy_envelope": {
                "method": "legacy_hilbert_envelope_matching_extract_stationary_hr",
                "bandpass_low_hz": legacy_curves["carrier_envelope_low_hz"],
                "bandpass_high_hz": legacy_curves["carrier_envelope_high_hz"],
                "carrier_energy_ratio": legacy_curves["carrier_energy_ratio"],
                "carrier_min_energy_ratio": args.carrier_min_energy_ratio,
                "curve_rate_hz": legacy_curve_rate,
                "dft_estimate": legacy_dft,
                "autocorr_estimate": legacy_autocorr,
                "peak_count_estimate": legacy_peak_count,
            },
            "ssa": {
                "embedding_window_samples": ssa_window,
                "embedding_window_seconds": ssa_window / curve_rate,
                "retained_components": retained_components,
                "aggregate_dft_estimate": aggregate_dft,
            },
        },
        "candidate_groups": groups,
        "final_estimate": {
            **final,
            "expected_cpm": args.expected_cpm,
            "error_vs_expected_cpm": expected_error,
        },
        "artifacts": {
            "analysis_signal_svg": f"{prefix}_analysis_signal.svg",
            "direct_curve_svg": f"{prefix}_direct_curve.svg",
            "envelope_curve_svg": f"{prefix}_envelope_curve.svg",
            "carrier_envelope_curve_svg": f"{prefix}_carrier_envelope_curve.svg",
            "stationary_legacy_envelope_curve_svg": f"{prefix}_stationary_legacy_envelope_curve.svg",
            "direct_dft_spectrum_svg": f"{prefix}_direct_dft_spectrum.svg",
            "envelope_dft_spectrum_svg": f"{prefix}_envelope_dft_spectrum.svg",
            "carrier_envelope_dft_spectrum_svg": f"{prefix}_carrier_envelope_dft_spectrum.svg",
            "stationary_legacy_envelope_dft_spectrum_svg": f"{prefix}_stationary_legacy_envelope_dft_spectrum.svg",
            "motion_spectrum_svg": f"{prefix}_motion_spectrum.svg",
            "ssa_aggregate_svg": f"{prefix}_ssa_aggregate.svg",
            "ssa_aggregate_spectrum_svg": f"{prefix}_ssa_aggregate_spectrum.svg",
        },
    }


def build_windows(signal: list[float], sample_rate_hz: float, window_seconds: float, hop_seconds: float) -> list[tuple[int, list[float]]]:
    window_len = max(int(window_seconds * sample_rate_hz), 1)
    hop_len = max(int(hop_seconds * sample_rate_hz), 1)
    if len(signal) <= window_len:
        return [(0, signal)]
    windows = []
    start = 0
    while start + window_len <= len(signal):
        windows.append((start, signal[start : start + window_len]))
        start += hop_len
    if windows and windows[-1][0] + window_len < len(signal):
        final_start = max(len(signal) - window_len, 0)
        if final_start != windows[-1][0]:
            windows.append((final_start, signal[final_start:]))
    return windows


def build_analysis_segments(
    signal: list[float],
    sample_rate_hz: float,
    analysis_mode: str,
    window_seconds: float,
    hop_seconds: float,
) -> list[tuple[int, list[float]]]:
    if analysis_mode == "whole":
        return [(0, signal)]
    return build_windows(signal, sample_rate_hz, window_seconds, hop_seconds)


def make_trend_svg(windows: list[dict], out_path: Path, expected_cpm: float | None) -> None:
    width = 1200
    height = 320
    valid_points = [
        (
            (window["start_seconds"] + window["end_seconds"]) / 2.0,
            window["final_estimate"].get("rr_cpm"),
            window["final_estimate"].get("confidence", 0.0),
            window["final_estimate"].get("status"),
        )
        for window in windows
        if window["final_estimate"].get("rr_cpm") is not None
    ]
    if not valid_points:
        out_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='600' height='120'></svg>", encoding="utf-8")
        return
    min_time = min(point[0] for point in valid_points)
    max_time = max(point[0] for point in valid_points)
    min_cpm = min(point[1] for point in valid_points if point[1] is not None)
    max_cpm = max(point[1] for point in valid_points if point[1] is not None)
    if expected_cpm is not None:
        min_cpm = min(min_cpm, expected_cpm)
        max_cpm = max(max_cpm, expected_cpm)
    min_cpm -= 3.0
    max_cpm += 3.0
    usable_width = width - 100
    usable_height = height - 80

    circles = []
    line_points = []
    for time_s, cpm, confidence, status in valid_points:
        x = 60 + ((time_s - min_time) / max(max_time - min_time, 1e-9)) * usable_width
        y = height - 40 - ((cpm - min_cpm) / max(max_cpm - min_cpm, 1e-9)) * usable_height
        color = "#2b8a3e" if status == "valid" else "#f08c00"
        radius = 4.0 + 5.0 * confidence
        circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" opacity="0.85" />')
        line_points.append(f"{x:.2f},{y:.2f}")

    expected_line = ""
    if expected_cpm is not None:
        y = height - 40 - ((expected_cpm - min_cpm) / max(max_cpm - min_cpm, 1e-9)) * usable_height
        expected_line = (
            f'<line x1="60" y1="{y:.2f}" x2="{width - 40}" y2="{y:.2f}" '
            f'stroke="#c92a2a" stroke-width="2" stroke-dasharray="6 4" />'
            f'<text x="{width - 44}" y="{y - 8:.2f}" font-size="13" text-anchor="end" fill="#c92a2a">expected {expected_cpm:.1f} CPM</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8f9fa" />
  <text x="12" y="24" font-size="16" fill="#343a40">VPU respiration trend</text>
  <line x1="60" y1="{height - 40}" x2="{width - 40}" y2="{height - 40}" stroke="#adb5bd" stroke-width="1" />
  <line x1="60" y1="40" x2="60" y2="{height - 40}" stroke="#adb5bd" stroke-width="1" />
  <polyline fill="none" stroke="#1864ab" stroke-width="2" points="{' '.join(line_points)}" />
  {''.join(circles)}
  {expected_line}
  <text x="60" y="{height - 12}" font-size="12" fill="#495057">{min_time:.0f}s</text>
  <text x="{width - 40}" y="{height - 12}" font-size="12" text-anchor="end" fill="#495057">{max_time:.0f}s</text>
  <text x="8" y="55" font-size="12" fill="#495057">{max_cpm:.1f} CPM</text>
  <text x="8" y="{height - 42}" font-size="12" fill="#495057">{min_cpm:.1f} CPM</text>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def summarize_overall(windows: list[dict]) -> dict:
    usable = [
        window["final_estimate"]
        for window in windows
        if window["final_estimate"].get("rr_cpm") is not None and window["final_estimate"].get("status") == "valid"
    ]
    if not usable:
        return {"status": "no_valid_windows"}
    rates = sorted(item["rr_cpm"] for item in usable)
    median = rates[len(rates) // 2]
    mean = sum(rates) / len(rates)
    return {
        "status": "valid_windows_present",
        "valid_window_count": len(usable),
        "total_window_count": len(windows),
        "mean_rr_cpm": mean,
        "median_rr_cpm": median,
        "min_rr_cpm": min(rates),
        "max_rr_cpm": max(rates),
    }


def choose_file_prediction(windows: list[dict]) -> dict:
    valid = [
        window
        for window in windows
        if window["final_estimate"].get("rr_cpm") is not None and window["final_estimate"].get("status") == "valid"
    ]
    usable = valid or [
        window for window in windows if window["final_estimate"].get("rr_cpm") is not None
    ]
    if not usable:
        return {
            "status": "rejected",
            "rr_cpm": None,
            "confidence": 0.0,
            "basis": "no_window_candidate",
            "valid_window_count": 0,
            "total_window_count": len(windows),
        }

    groups: list[dict] = []
    for window in usable:
        estimate = window["final_estimate"]
        selected_group = estimate.get("selected_group") or {}
        selected_sources = selected_group.get("support_sources") or []
        rr_cpm = estimate["rr_cpm"]
        confidence = estimate.get("confidence", 0.0)
        target_group = None
        for group in groups:
            if abs(rr_cpm - group["rr_cpm"]) <= 3.0:
                target_group = group
                break
        if target_group is None:
            target_group = {
                "rr_cpm": rr_cpm,
                "score": 0.0,
                "weighted_sum": 0.0,
                "weight_total": 0.0,
                "windows": [],
            }
            groups.append(target_group)
        source_bonus = 0.0
        if "direct" in selected_sources:
            source_bonus += 0.6
        if "stationary_legacy_envelope" in selected_sources:
            source_bonus += 0.8
        if "carrier_envelope" in selected_sources:
            source_bonus += 0.4
        support_bonus = 0.2 * max(len(selected_sources) - 1, 0)
        weight = max(confidence, 0.05)
        target_group["score"] += weight + source_bonus + support_bonus
        target_group["weighted_sum"] += rr_cpm * weight
        target_group["weight_total"] += weight
        target_group["windows"].append(window["window_index"])

    for group in groups:
        if group["weight_total"] > 0.0:
            group["rr_cpm"] = group["weighted_sum"] / group["weight_total"]
        del group["weighted_sum"]
        del group["weight_total"]
    groups.sort(key=lambda item: item["score"], reverse=True)
    best_group = groups[0]
    second_score = groups[1]["score"] if len(groups) > 1 else 0.0
    margin = best_group["score"] - second_score
    selected_windows = [window for window in usable if window["window_index"] in best_group["windows"]]
    mean_confidence = sum(window["final_estimate"].get("confidence", 0.0) for window in selected_windows) / len(selected_windows)
    if len(groups) > 1 and margin < 0.5:
        mean_confidence = min(mean_confidence, 0.45)
    status = "valid" if valid else "low_confidence"
    if len(groups) > 1 and margin < 0.5:
        status = "low_confidence"
    return {
        "status": status,
        "rr_cpm": best_group["rr_cpm"],
        "confidence": mean_confidence,
        "basis": "best_consensus_window_group" if valid else "best_consensus_candidate_window_group",
        "valid_window_count": len(valid),
        "candidate_window_count": len(usable),
        "total_window_count": len(windows),
        "selected_window_indexes": best_group["windows"],
        "window_group_count": len(groups),
        "window_group_margin": margin,
    }


def compact_window_prediction(window: dict) -> dict:
    estimate = window["final_estimate"]
    selected_group = estimate.get("selected_group") or {}
    branches = window.get("branches", {})
    return {
        "window_index": window["window_index"],
        "start_seconds": window["start_seconds"],
        "end_seconds": window["end_seconds"],
        "status": estimate.get("status"),
        "rr_cpm": estimate.get("rr_cpm"),
        "confidence": estimate.get("confidence"),
        "reject_reason": estimate.get("reject_reason"),
        "selection_policy": estimate.get("selection_policy"),
        "selected_sources": selected_group.get("support_sources", []),
        "direct_dft_cpm": branches.get("direct", {}).get("dft_estimate", {}).get("candidate_bpm"),
        "envelope_dft_cpm": branches.get("envelope", {}).get("dft_estimate", {}).get("candidate_bpm"),
        "carrier_envelope_dft_cpm": branches.get("carrier_envelope", {}).get("dft_estimate", {}).get("candidate_bpm"),
        "stationary_legacy_envelope_dft_cpm": branches.get("stationary_legacy_envelope", {})
        .get("dft_estimate", {})
        .get("candidate_bpm"),
        "expected_cpm": estimate.get("expected_cpm"),
        "error_vs_expected_cpm": estimate.get("error_vs_expected_cpm"),
    }


def write_window_predictions_csv(windows: list[dict], out_path: Path) -> None:
    fieldnames = [
        "window_index",
        "start_seconds",
        "end_seconds",
        "status",
        "rr_cpm",
        "confidence",
        "reject_reason",
        "selection_policy",
        "selected_sources",
        "direct_dft_cpm",
        "envelope_dft_cpm",
        "carrier_envelope_dft_cpm",
        "stationary_legacy_envelope_dft_cpm",
        "expected_cpm",
        "error_vs_expected_cpm",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for window in windows:
            row = compact_window_prediction(window)
            row["selected_sources"] = "+".join(row["selected_sources"])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.cpm_min <= 0.0:
        raise SystemExit("--cpm-min must be positive.")
    if args.cpm_max <= args.cpm_min:
        raise SystemExit("--cpm-max must be greater than --cpm-min.")
    if args.analysis_mode == "windows" and (args.window_seconds <= 0.0 or args.hop_seconds <= 0.0):
        raise SystemExit("--window-seconds and --hop-seconds must be positive.")
    if args.legacy_envelope_rate <= 0.0:
        raise SystemExit("--legacy-envelope-rate must be positive.")
    if args.boundary_guard_cpm < 0.0:
        raise SystemExit("--boundary-guard-cpm must be non-negative.")

    out_dir = args.output_dir / args.wav_path.stem / "vpu_respiration"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, metadata = load_mono_pcm16(args.wav_path)
    raw = [float(sample) for sample in samples]
    analysis_stride = max(int(metadata["sample_rate_hz"] / args.analysis_rate), 1)
    analysis_signal = centered(decimate_mean(raw, analysis_stride))
    analysis_rate = metadata["sample_rate_hz"] / analysis_stride
    make_series_svg(analysis_signal, out_dir / "analysis_signal.svg", 1200, 280, "full VPU decimated analysis signal")

    window_results = []
    for window_index, (start_sample, window_signal) in enumerate(
        build_analysis_segments(
            analysis_signal,
            analysis_rate,
            args.analysis_mode,
            args.window_seconds,
            args.hop_seconds,
        )
    ):
        window_results.append(
            analyze_window(
                window_signal,
                analysis_rate,
                args,
                out_dir,
                window_index,
                start_sample / analysis_rate,
            )
        )

    make_trend_svg(window_results, out_dir / "rr_trend.svg", args.expected_cpm)
    file_prediction = choose_file_prediction(window_results)
    compact_windows = [compact_window_prediction(window) for window in window_results]

    summary = {
        "file_name": args.wav_path.name,
        "metadata": metadata,
        "vpu_respiration": {
            "method": "vpu_stationary_legacy_envelope_or_motion_aware_ssa",
            "analysis_rate_hz": analysis_rate,
            "respiration_curve_rate_hz": args.respiration_rate,
            "cpm_search_min": args.cpm_min,
            "cpm_search_max": args.cpm_max,
            "analysis_mode": args.analysis_mode,
            "window_seconds": args.window_seconds,
            "hop_seconds": args.hop_seconds,
            "preset": args.preset,
            "selection_policy": args.selection_policy,
            "boundary_guard_cpm": args.boundary_guard_cpm,
            "legacy_envelope_rate_hz": args.legacy_envelope_rate,
            "expected_cpm": args.expected_cpm,
            "file_prediction": file_prediction,
            "overall": summarize_overall(window_results),
            "windows": window_results,
            "artifacts": {
                "analysis_signal_svg": "analysis_signal.svg",
                "rr_trend_svg": "rr_trend.svg",
            },
        },
    }
    prediction = {
        "file_name": args.wav_path.name,
        "method": "vpu_stationary_legacy_envelope_or_motion_aware_ssa",
        "preset": args.preset,
        "selection_policy": args.selection_policy,
        "analysis_mode": args.analysis_mode,
        "expected_cpm": args.expected_cpm,
        "file_prediction": file_prediction,
        "windows": compact_windows,
        "artifacts": {
            "full_summary_json": "summary.json",
            "window_predictions_csv": "window_predictions.csv",
            "rr_trend_svg": "rr_trend.svg",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "prediction.json").write_text(json.dumps(prediction, indent=2, sort_keys=True), encoding="utf-8")
    write_window_predictions_csv(window_results, out_dir / "window_predictions.csv")
    print(f"Saved VPU respiration summary to {out_dir / 'summary.json'}")
    print(f"Saved compact prediction to {out_dir / 'prediction.json'}")
    print(f"Saved window prediction table to {out_dir / 'window_predictions.csv'}")
    print(f"Saved VPU respiration trend to {out_dir / 'rr_trend.svg'}")
    if file_prediction["rr_cpm"] is None:
        print("FINAL_RR_CPM rejected confidence=0.000")
    else:
        print(
            "FINAL_RR_CPM "
            f"{file_prediction['rr_cpm']:.2f} "
            f"status={file_prediction['status']} "
            f"confidence={file_prediction['confidence']:.3f} "
            f"valid_windows={file_prediction['valid_window_count']}/{file_prediction['total_window_count']}"
        )


if __name__ == "__main__":
    main()
