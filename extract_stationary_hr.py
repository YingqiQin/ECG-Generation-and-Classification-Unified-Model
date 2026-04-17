#!/usr/bin/env python3
"""Stationary HR-oriented preprocessing for mono WAV files.

This script evaluates two candidate branches in parallel:

1. direct_lowfreq:
   Treat the waveform as a pulse-like low-frequency signal and estimate HR
   directly from a low-frequency bandpassed waveform.

2. envelope_higher_band:
   Treat the waveform as a higher-frequency carrier whose amplitude envelope
   contains the HR rhythm, using FIR bandpass + FFT-based Hilbert transform.

The script is standard-library-only so it can run on sparse remote servers.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract stationary HR-oriented features from a WAV file.")
    parser.add_argument("wav_path", type=Path, help="Path to WAV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim"),
        help="Base output directory",
    )
    parser.add_argument(
        "--analysis-rate",
        type=float,
        default=400.0,
        help="Target sample rate before branch-specific processing",
    )
    parser.add_argument(
        "--target-hr-rate",
        type=float,
        default=50.0,
        help="Target sample rate for HR-band estimation",
    )
    parser.add_argument(
        "--direct-low-hz",
        type=float,
        default=0.7,
        help="Lower cutoff for the direct low-frequency branch",
    )
    parser.add_argument(
        "--direct-high-hz",
        type=float,
        default=4.0,
        help="Upper cutoff for the direct low-frequency branch",
    )
    parser.add_argument(
        "--envelope-bandpass-low-hz",
        type=float,
        default=8.0,
        help="Lower cutoff for the higher-band Hilbert branch",
    )
    parser.add_argument(
        "--envelope-bandpass-high-hz",
        type=float,
        default=40.0,
        help="Upper cutoff for the higher-band Hilbert branch",
    )
    parser.add_argument(
        "--fragment-seconds",
        type=float,
        default=20.0,
        help="Length of saved zoom fragments for HR-focused plots",
    )
    return parser.parse_args()


def load_mono_pcm16(wav_path: Path) -> tuple[list[int], dict]:
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got {sample_width}-byte samples")

    values = struct.unpack("<" + ("h" * (n_frames * channels)), raw)
    mono = list(values if channels == 1 else values[0::channels])
    metadata = {
        "channels_in_file": channels,
        "sample_rate_hz": sample_rate,
        "sample_width_bytes": sample_width,
        "n_frames": n_frames,
        "duration_seconds": n_frames / sample_rate,
    }
    return mono, metadata


def moving_average(values: list[float], window_size: int) -> list[float]:
    if not values:
        return []
    window_size = max(window_size, 1)
    out: list[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window_size:
            running -= values[idx - window_size]
        denom = window_size if idx >= window_size - 1 else (idx + 1)
        out.append(running / denom)
    return out


def decimate_mean(values: list[float], stride: int) -> list[float]:
    stride = max(stride, 1)
    out: list[float] = []
    for start in range(0, len(values), stride):
        chunk = values[start : start + stride]
        if chunk:
            out.append(sum(chunk) / len(chunk))
    return out


def centered(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    return [value - mean for value in values]


def sinc(x: float) -> float:
    if abs(x) < 1e-12:
        return 1.0
    return math.sin(math.pi * x) / (math.pi * x)


def make_lowpass_fir(sample_rate_hz: float, cutoff_hz: float, num_taps: int) -> list[float]:
    mid = (num_taps - 1) / 2.0
    taps: list[float] = []
    normalized_cutoff = cutoff_hz / sample_rate_hz
    for idx in range(num_taps):
        n = idx - mid
        ideal = 2.0 * normalized_cutoff * sinc(2.0 * normalized_cutoff * n)
        window = 0.54 - 0.46 * math.cos(2.0 * math.pi * idx / (num_taps - 1))
        taps.append(ideal * window)
    scale = sum(taps) or 1.0
    return [tap / scale for tap in taps]


def make_bandpass_fir(sample_rate_hz: float, low_hz: float, high_hz: float, num_taps: int) -> list[float]:
    lowpass_high = make_lowpass_fir(sample_rate_hz, high_hz, num_taps)
    lowpass_low = make_lowpass_fir(sample_rate_hz, low_hz, num_taps)
    return [hi - lo for hi, lo in zip(lowpass_high, lowpass_low)]


def fir_filter(values: list[float], taps: list[float]) -> list[float]:
    out = [0.0] * len(values)
    for idx in range(len(values)):
        total = 0.0
        for tap_idx, tap in enumerate(taps):
            source_idx = idx - tap_idx
            if source_idx < 0:
                break
            total += tap * values[source_idx]
        out[idx] = total
    return out


def next_power_of_two(n: int) -> int:
    size = 1
    while size < n:
        size <<= 1
    return size


def fft_inplace(values: list[complex], inverse: bool = False) -> list[complex]:
    n = len(values)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            values[i], values[j] = values[j], values[i]

    length = 2
    sign = 1.0 if inverse else -1.0
    while length <= n:
        angle = sign * 2.0 * math.pi / length
        w_len = complex(math.cos(angle), math.sin(angle))
        half = length // 2
        for start in range(0, n, length):
            w = 1.0 + 0.0j
            for offset in range(half):
                u = values[start + offset]
                v = values[start + offset + half] * w
                values[start + offset] = u + v
                values[start + offset + half] = u - v
                w *= w_len
        length <<= 1

    if inverse:
        for idx in range(n):
            values[idx] /= n
    return values


def analytic_signal(values: list[float]) -> list[complex]:
    n = len(values)
    fft_len = next_power_of_two(n)
    padded = [complex(value, 0.0) for value in values] + [0.0j] * (fft_len - n)
    spectrum = fft_inplace(padded, inverse=False)

    h = [0.0] * fft_len
    h[0] = 1.0
    if fft_len % 2 == 0:
        h[fft_len // 2] = 1.0
        for idx in range(1, fft_len // 2):
            h[idx] = 2.0
    else:
        for idx in range(1, (fft_len + 1) // 2):
            h[idx] = 2.0

    analytic_spec = [value * h[idx] for idx, value in enumerate(spectrum)]
    analytic = fft_inplace(analytic_spec, inverse=True)
    return analytic[:n]


def pearson_autocorr(values: list[float], lag: int) -> float | None:
    if lag <= 0 or lag >= len(values):
        return None
    a = values[:-lag]
    b = values[lag:]
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    sum_ab = 0.0
    sum_aa = 0.0
    sum_bb = 0.0
    for idx in range(len(a)):
        da = a[idx] - mean_a
        db = b[idx] - mean_b
        sum_ab += da * db
        sum_aa += da * da
        sum_bb += db * db
    if sum_aa <= 0.0 or sum_bb <= 0.0:
        return None
    return sum_ab / math.sqrt(sum_aa * sum_bb)


def estimate_autocorr_bpm(values: list[float], sample_rate_hz: float, bpm_min: float, bpm_max: float) -> dict:
    lag_min = max(int(sample_rate_hz * 60.0 / bpm_max), 1)
    lag_max = min(int(sample_rate_hz * 60.0 / bpm_min), len(values) // 2)
    best_lag = None
    best_corr = None
    for lag in range(lag_min, lag_max + 1):
        corr = pearson_autocorr(values, lag)
        if corr is None:
            continue
        if best_corr is None or corr > best_corr:
            best_corr = corr
            best_lag = lag
    if best_lag is None or best_corr is None:
        return {"status": "no_candidate"}
    return {
        "status": "candidate_only",
        "best_lag_samples": best_lag,
        "autocorrelation": best_corr,
        "candidate_bpm": sample_rate_hz * 60.0 / best_lag,
    }


def estimate_dft_bpm(values: list[float], sample_rate_hz: float, bpm_min: float, bpm_max: float) -> dict:
    if not values:
        return {"status": "no_candidate"}
    centered_values = centered(values)
    best_freq = None
    best_power = None
    freq_step_hz = 0.01
    start_hz = bpm_min / 60.0
    end_hz = bpm_max / 60.0
    steps = int((end_hz - start_hz) / freq_step_hz) + 1
    spectrum: list[dict] = []
    for step_idx in range(steps):
        freq = start_hz + step_idx * freq_step_hz
        real = 0.0
        imag = 0.0
        for idx, value in enumerate(centered_values):
            angle = -2.0 * math.pi * freq * idx / sample_rate_hz
            real += value * math.cos(angle)
            imag += value * math.sin(angle)
        power = real * real + imag * imag
        spectrum.append({"freq_hz": freq, "bpm": freq * 60.0, "power": power})
        if best_power is None or power > best_power:
            best_power = power
            best_freq = freq
    if best_freq is None or best_power is None:
        return {"status": "no_candidate"}
    top_peaks = sorted(spectrum, key=lambda item: item["power"], reverse=True)[:5]
    return {
        "status": "candidate_only",
        "candidate_bpm": best_freq * 60.0,
        "spectral_power": best_power,
        "peak_frequency_hz": best_freq,
        "top_peaks": top_peaks,
        "spectrum": spectrum,
    }


def strongest_fragment_starts(values: list[float], sample_rate_hz: float, fragment_seconds: float, max_fragments: int) -> list[int]:
    fragment_len = max(int(sample_rate_hz * fragment_seconds), 1)
    if len(values) <= fragment_len:
        return [0]
    stride = max(fragment_len // 2, 1)
    candidates: list[tuple[float, int]] = []
    for start in range(0, len(values) - fragment_len + 1, stride):
        chunk = values[start : start + fragment_len]
        energy = sum(value * value for value in chunk) / len(chunk)
        candidates.append((energy, start))
    candidates.sort(reverse=True)
    selected: list[int] = []
    for _, start in candidates:
        if all(abs(start - existing) >= fragment_len for existing in selected):
            selected.append(start)
        if len(selected) >= max_fragments:
            break
    if 0 not in selected:
        selected.append(0)
    selected.sort()
    return selected[:max_fragments]


def make_series_svg(values: list[float], out_path: Path, width: int, height: int, title: str) -> None:
    if not values:
        out_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='600' height='120'></svg>", encoding="utf-8")
        return
    bound = max(abs(value) for value in values) or 1.0
    x_span = max(width - 20, 1)
    points = []
    for pixel_x in range(x_span):
        start = int(pixel_x * len(values) / x_span)
        end = int((pixel_x + 1) * len(values) / x_span)
        if start >= len(values):
            break
        chunk = values[start:end] or [values[start]]
        value = sum(chunk) / len(chunk)
        x = 10 + pixel_x
        y = (height / 2.0) - ((value / bound) * (height * 0.38))
        points.append(f"{x:.2f},{y:.2f}")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8f9fa" />
  <text x="12" y="20" font-size="14" fill="#343a40">{title}</text>
  <line x1="0" y1="{height / 2.0:.2f}" x2="{width}" y2="{height / 2.0:.2f}" stroke="#adb5bd" stroke-width="1" />
  <polyline fill="none" stroke="#1864ab" stroke-width="2" points="{' '.join(points)}" />
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def make_spectrum_svg(spectrum: list[dict], out_path: Path, width: int, height: int, title: str, peak_bpm: float | None) -> None:
    if not spectrum:
        out_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='600' height='120'></svg>", encoding="utf-8")
        return

    max_power = max(item["power"] for item in spectrum) or 1.0
    min_bpm = min(item["bpm"] for item in spectrum)
    max_bpm = max(item["bpm"] for item in spectrum)
    usable_width = max(width - 80, 1)
    usable_height = max(height - 70, 1)

    points = []
    for item in spectrum:
        x = 50 + ((item["bpm"] - min_bpm) / max(max_bpm - min_bpm, 1e-9)) * usable_width
        y = height - 30 - ((item["power"] / max_power) * usable_height)
        points.append(f"{x:.2f},{y:.2f}")

    peak_marker = ""
    peak_label = ""
    if peak_bpm is not None:
        peak_item = min(spectrum, key=lambda item: abs(item["bpm"] - peak_bpm))
        peak_x = 50 + ((peak_item["bpm"] - min_bpm) / max(max_bpm - min_bpm, 1e-9)) * usable_width
        peak_y = height - 30 - ((peak_item["power"] / max_power) * usable_height)
        peak_marker = f'<circle cx="{peak_x:.2f}" cy="{peak_y:.2f}" r="6" fill="#c92a2a" />'
        peak_label = (
            f'<text x="{peak_x + 10:.2f}" y="{max(28.0, peak_y - 10):.2f}" '
            f'font-size="14" fill="#c92a2a">peak {peak_item["bpm"]:.1f} bpm</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8f9fa" />
  <text x="12" y="20" font-size="14" fill="#343a40">{title}</text>
  <line x1="50" y1="{height - 30}" x2="{width - 20}" y2="{height - 30}" stroke="#adb5bd" stroke-width="1" />
  <line x1="50" y1="30" x2="50" y2="{height - 30}" stroke="#adb5bd" stroke-width="1" />
  <text x="50" y="{height - 8}" font-size="12" fill="#495057">{min_bpm:.0f}</text>
  <text x="{width - 30}" y="{height - 8}" font-size="12" text-anchor="end" fill="#495057">{max_bpm:.0f} bpm</text>
  <polyline fill="none" stroke="#1864ab" stroke-width="2" points="{' '.join(points)}" />
  {peak_marker}
  {peak_label}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def save_branch_fragments(
    signal: list[float],
    sample_rate_hz: float,
    out_dir: Path,
    prefix: str,
    title_prefix: str,
    fragment_seconds: float,
) -> list[dict]:
    starts = strongest_fragment_starts(signal, sample_rate_hz, fragment_seconds, 3)
    fragment_len = max(int(sample_rate_hz * fragment_seconds), 1)
    fragment_meta = []
    for fragment_index, start in enumerate(starts):
        end = min(start + fragment_len, len(signal))
        fragment = signal[start:end]
        name = f"{prefix}_fragment_{fragment_index:02d}.svg"
        make_series_svg(
            fragment,
            out_dir / name,
            1200,
            280,
            f"{title_prefix} fragment {fragment_index} {start / sample_rate_hz:.2f}-{end / sample_rate_hz:.2f}s",
        )
        fragment_meta.append(
            {
                "svg": name,
                "start_seconds": start / sample_rate_hz,
                "end_seconds": end / sample_rate_hz,
            }
        )
    return fragment_meta


def build_direct_branch(
    analysis_signal: list[float],
    analysis_rate: float,
    target_hr_rate: float,
    low_hz: float,
    high_hz: float,
    out_dir: Path,
    fragment_seconds: float,
) -> dict:
    taps = make_bandpass_fir(analysis_rate, low_hz, high_hz, num_taps=257)
    direct_bandpassed = fir_filter(analysis_signal, taps)
    hr_stride = max(int(analysis_rate / target_hr_rate), 1)
    direct_ds = decimate_mean(direct_bandpassed, hr_stride)
    direct_rate = analysis_rate / hr_stride
    direct_smooth = moving_average(direct_ds, max(int(direct_rate * 0.15), 1))
    autocorr_est = estimate_autocorr_bpm(direct_smooth, direct_rate, bpm_min=40.0, bpm_max=180.0)
    dft_est = estimate_dft_bpm(direct_smooth, direct_rate, bpm_min=40.0, bpm_max=180.0)

    make_series_svg(direct_bandpassed, out_dir / "direct_bandpassed_waveform.svg", 1200, 280, "direct low-frequency bandpassed waveform")
    make_series_svg(direct_smooth, out_dir / "direct_hr_band.svg", 1200, 280, "direct HR-focused waveform")
    make_spectrum_svg(
        dft_est.get("spectrum", []),
        out_dir / "direct_dft_spectrum.svg",
        1200,
        320,
        "direct branch DFT spectrum",
        dft_est.get("candidate_bpm"),
    )
    fragment_meta = save_branch_fragments(
        direct_smooth,
        direct_rate,
        out_dir,
        "direct_hr",
        "direct HR waveform",
        fragment_seconds,
    )

    return {
        "method": "direct_low_frequency_pulse_branch",
        "bandpass_low_hz": low_hz,
        "bandpass_high_hz": high_hz,
        "bandpass_num_taps": 257,
        "signal_sample_rate_hz": direct_rate,
        "smoothing_window_seconds": max(int(direct_rate * 0.15), 1) / direct_rate,
        "autocorr_estimate": autocorr_est,
        "dft_estimate": dft_est,
        "artifacts": {
            "bandpassed_waveform_svg": "direct_bandpassed_waveform.svg",
            "hr_waveform_svg": "direct_hr_band.svg",
            "dft_spectrum_svg": "direct_dft_spectrum.svg",
            "fragment_svgs": fragment_meta,
        },
    }


def build_envelope_branch(
    analysis_signal: list[float],
    analysis_rate: float,
    target_hr_rate: float,
    low_hz: float,
    high_hz: float,
    out_dir: Path,
    fragment_seconds: float,
) -> dict:
    taps = make_bandpass_fir(analysis_rate, low_hz, high_hz, num_taps=129)
    bandpassed = fir_filter(analysis_signal, taps)
    analytic = analytic_signal(bandpassed)
    hilbert_envelope = [abs(value) for value in analytic]

    envelope_stride = max(int(analysis_rate / target_hr_rate), 1)
    envelope_ds = decimate_mean(hilbert_envelope, envelope_stride)
    envelope_rate = analysis_rate / envelope_stride
    envelope_smooth = moving_average(envelope_ds, max(int(envelope_rate * 0.20), 1))
    slow_trend = moving_average(envelope_smooth, max(int(envelope_rate * 2.0), 1))
    envelope_hr = [value - trend for value, trend in zip(envelope_smooth, slow_trend)]
    envelope_hr = moving_average(envelope_hr, max(int(envelope_rate * 0.30), 1))

    autocorr_est = estimate_autocorr_bpm(envelope_hr, envelope_rate, bpm_min=40.0, bpm_max=180.0)
    dft_est = estimate_dft_bpm(envelope_hr, envelope_rate, bpm_min=40.0, bpm_max=180.0)

    make_series_svg(bandpassed, out_dir / "bandpassed_waveform.svg", 1200, 280, "higher-band bandpassed waveform")
    make_series_svg(envelope_ds, out_dir / "hilbert_envelope.svg", 1200, 280, "Hilbert envelope")
    make_series_svg(envelope_hr, out_dir / "envelope_hr_band.svg", 1200, 280, "envelope HR-focused waveform")
    make_spectrum_svg(
        dft_est.get("spectrum", []),
        out_dir / "envelope_dft_spectrum.svg",
        1200,
        320,
        "envelope branch DFT spectrum",
        dft_est.get("candidate_bpm"),
    )
    fragment_meta = save_branch_fragments(
        envelope_hr,
        envelope_rate,
        out_dir,
        "envelope_hr",
        "envelope HR waveform",
        fragment_seconds,
    )

    return {
        "method": "higher_band_hilbert_envelope_branch",
        "bandpass_low_hz": low_hz,
        "bandpass_high_hz": high_hz,
        "bandpass_num_taps": 129,
        "envelope_sample_rate_hz": envelope_rate,
        "envelope_smoothing_window_seconds": max(int(envelope_rate * 0.20), 1) / envelope_rate,
        "slow_trend_window_seconds": max(int(envelope_rate * 2.0), 1) / envelope_rate,
        "hr_smoothing_window_seconds": max(int(envelope_rate * 0.30), 1) / envelope_rate,
        "autocorr_estimate": autocorr_est,
        "dft_estimate": dft_est,
        "artifacts": {
            "bandpassed_waveform_svg": "bandpassed_waveform.svg",
            "hilbert_envelope_svg": "hilbert_envelope.svg",
            "hr_waveform_svg": "envelope_hr_band.svg",
            "dft_spectrum_svg": "envelope_dft_spectrum.svg",
            "fragment_svgs": fragment_meta,
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir / args.wav_path.stem / "stationary_hr"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, metadata = load_mono_pcm16(args.wav_path)
    raw = [float(sample) for sample in samples]

    analysis_stride = max(int(metadata["sample_rate_hz"] / args.analysis_rate), 1)
    analysis_signal = decimate_mean(raw, analysis_stride)
    analysis_rate = metadata["sample_rate_hz"] / analysis_stride
    analysis_signal = centered(analysis_signal)
    make_series_svg(analysis_signal, out_dir / "analysis_signal.svg", 1200, 280, "decimated analysis waveform")

    direct_branch = build_direct_branch(
        analysis_signal,
        analysis_rate,
        args.target_hr_rate,
        args.direct_low_hz,
        args.direct_high_hz,
        out_dir,
        args.fragment_seconds,
    )
    envelope_branch = build_envelope_branch(
        analysis_signal,
        analysis_rate,
        args.target_hr_rate,
        args.envelope_bandpass_low_hz,
        args.envelope_bandpass_high_hz,
        out_dir,
        args.fragment_seconds,
    )

    summary = {
        "file_name": args.wav_path.name,
        "metadata": metadata,
        "stationary_preprocessing": {
            "method": "dual_path_direct_and_hilbert",
            "analysis_sample_rate_hz": analysis_rate,
            "target_hr_rate_hz": args.target_hr_rate,
            "branches": {
                "direct_lowfreq": direct_branch,
                "envelope_higher_band": envelope_branch,
            },
            "artifacts": {
                "analysis_signal_svg": "analysis_signal.svg",
            },
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved stationary HR summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
