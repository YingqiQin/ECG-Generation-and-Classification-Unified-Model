#!/usr/bin/env python3
"""Summarize WAV metadata and save quick-look plots for each channel.

The implementation intentionally relies on the Python standard library so it
remains usable in a sparse early-stage research environment.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import wave
from pathlib import Path


INT16_MAX = 32767
INT16_MIN = -32768
FULL_SCALE_BY_WIDTH = {
    1: 128.0,
    2: 32768.0,
    4: 2147483648.0,
}
STRUCT_FORMAT_BY_WIDTH = {
    1: "B",
    2: "h",
    4: "i",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a WAV file.")
    parser.add_argument("wav_path", type=Path, help="Path to a WAV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim"),
        help="Directory where summary artifacts are written",
    )
    parser.add_argument(
        "--svg-width",
        type=int,
        default=1200,
        help="Width of generated waveform SVGs",
    )
    parser.add_argument(
        "--svg-height",
        type=int,
        default=260,
        help="Height of each generated waveform SVG",
    )
    parser.add_argument(
        "--fragment-seconds",
        type=float,
        default=8.0,
        help="Duration of each zoomed fragment SVG",
    )
    parser.add_argument(
        "--max-fragments",
        type=int,
        default=4,
        help="Maximum number of zoomed fragments per channel",
    )
    return parser.parse_args()


def load_wav_channels(wav_path: Path) -> tuple[list[list[int]], dict]:
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        comptype = wf.getcomptype()
        compname = wf.getcompname()
        raw = wf.readframes(n_frames)

    if sample_width not in STRUCT_FORMAT_BY_WIDTH:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")

    total_samples = n_frames * channels
    fmt = "<" + STRUCT_FORMAT_BY_WIDTH[sample_width] * total_samples
    unpacked = struct.unpack(fmt, raw)

    channel_samples = [[] for _ in range(channels)]
    if sample_width == 1:
        for idx, value in enumerate(unpacked):
            channel_samples[idx % channels].append(int(value) - 128)
    else:
        for idx, value in enumerate(unpacked):
            channel_samples[idx % channels].append(int(value))

    metadata = {
        "channels_in_file": channels,
        "sample_width_bytes": sample_width,
        "sample_rate_hz": sample_rate,
        "n_frames": n_frames,
        "duration_seconds": n_frames / sample_rate,
        "compression_type": comptype,
        "compression_name": compname,
        "payload_bitrate_bps": sample_rate * sample_width * 8 * channels,
    }
    return channel_samples, metadata


def summarize_channel(samples: list[int], sample_rate: int, full_scale: float) -> dict:
    count = len(samples)
    if count == 0:
        raise ValueError("Channel is empty")

    sample_min = samples[0]
    sample_max = samples[0]
    abs_peak = abs(samples[0])
    clip_pos = 0
    clip_neg = 0
    near_zero = 0
    zero_crossings = 0
    total = 0.0
    total_sq = 0.0
    prev = samples[0]
    pos_clip_value = int(full_scale - 1)
    neg_clip_value = int(-full_scale)

    for s in samples:
        if s < sample_min:
            sample_min = s
        if s > sample_max:
            sample_max = s
        if abs(s) > abs_peak:
            abs_peak = abs(s)
        clip_pos += int(s >= pos_clip_value)
        clip_neg += int(s <= neg_clip_value)
        near_zero += int(abs(s) <= 4)
        total += s
        total_sq += s * s
        if (prev < 0 <= s) or (prev > 0 >= s):
            zero_crossings += 1
        prev = s

    mean = total / count
    rms = math.sqrt(total_sq / count)
    variance = max((total_sq / count) - (mean * mean), 0.0)

    return {
        "num_samples": count,
        "min": sample_min,
        "max": sample_max,
        "peak_abs": abs_peak,
        "mean": mean,
        "std": math.sqrt(variance),
        "rms": rms,
        "dc_offset_ratio_to_full_scale": mean / full_scale,
        "rms_ratio_to_full_scale": rms / full_scale,
        "clipping_ratio_pos": clip_pos / count,
        "clipping_ratio_neg": clip_neg / count,
        "near_zero_ratio_abs_le_4": near_zero / count,
        "zero_crossing_rate_hz_estimate": zero_crossings / (2.0 * (count / sample_rate)),
    }


def summarize_window_rms(samples: list[int], sample_rate: int, window_seconds: float = 1.0) -> dict:
    window_size = max(int(sample_rate * window_seconds), 1)
    rms_values: list[float] = []
    for start in range(0, len(samples) - window_size + 1, window_size):
        total_sq = 0.0
        for sample in samples[start : start + window_size]:
            total_sq += sample * sample
        rms_values.append(math.sqrt(total_sq / window_size))

    if not rms_values:
        return {
            "window_seconds": window_seconds,
            "num_windows": 0,
            "values_preview": [],
        }

    sorted_values = sorted(rms_values)
    median = sorted_values[len(sorted_values) // 2]
    return {
        "window_seconds": window_seconds,
        "num_windows": len(rms_values),
        "rms_min": min(rms_values),
        "rms_median": median,
        "rms_max": max(rms_values),
        "values_preview": [round(v, 3) for v in rms_values[:10]],
    }


def summarize_coarse_spectrum(samples: list[int], sample_rate: int) -> dict:
    """Estimate coarse spectral fractions from a short excerpt."""
    target_len = min(len(samples), 2048)
    if target_len < 16:
        return {}

    excerpt = samples[:target_len]
    excerpt_mean = sum(excerpt) / target_len
    centered = [value - excerpt_mean for value in excerpt]

    freqs: list[float] = []
    powers: list[float] = []
    for k in range(target_len // 2 + 1):
        real = 0.0
        imag = 0.0
        for n, value in enumerate(centered):
            angle = -2.0 * math.pi * k * n / target_len
            real += value * math.cos(angle)
            imag += value * math.sin(angle)
        power = real * real + imag * imag
        freqs.append(k * sample_rate / target_len)
        powers.append(power)

    total_power = sum(powers[1:]) or 1.0
    peak_freq = 0.0
    peak_power = -1.0
    for freq, power in zip(freqs[1:], powers[1:]):
        if power > peak_power:
            peak_freq = freq
            peak_power = power

    bands = [
        (0.0, 20.0),
        (20.0, 80.0),
        (80.0, 300.0),
        (300.0, 3000.0),
        (3000.0, sample_rate / 2.0),
    ]
    summary = {}
    for lo, hi in bands:
        band_power = 0.0
        for freq, power in zip(freqs, powers):
            if lo <= freq < hi:
                band_power += power
        summary[f"band_{int(lo)}_{int(hi)}_hz_fraction"] = band_power / total_power
    summary["dominant_frequency_hz_on_first_excerpt"] = peak_freq
    return summary


def moving_average_abs(samples: list[int], window_size: int, stride: int) -> list[float]:
    """Compute a smoothed absolute-amplitude envelope."""
    if not samples:
        return []

    abs_samples = [abs(sample) for sample in samples]
    window_size = max(window_size, 1)
    stride = max(stride, 1)
    values: list[float] = []
    running = 0.0
    for idx, value in enumerate(abs_samples):
        running += value
        if idx >= window_size:
            running -= abs_samples[idx - window_size]
        if idx >= window_size - 1 and ((idx - (window_size - 1)) % stride == 0):
            values.append(running / window_size)
    return values


def estimate_hr_candidate_from_envelope(samples: list[int], sample_rate: int) -> dict:
    """Estimate a weak rhythm candidate from the smoothed envelope.

    This is exploratory only. A candidate BPM here is not evidence of heart rate.
    """
    env_stride = max(sample_rate // 50, 1)
    envelope = moving_average_abs(
        samples,
        window_size=max(sample_rate // 20, 1),
        stride=env_stride,
    )
    if len(envelope) < 32:
        return {"status": "insufficient_data"}

    mean_env = sum(envelope) / len(envelope)
    centered = [value - mean_env for value in envelope]
    env_rate_hz = sample_rate / env_stride
    lag_min = max(int(env_rate_hz * 60.0 / 180.0), 1)
    lag_max = min(int(env_rate_hz * 60.0 / 40.0), len(centered) // 2)
    if lag_max <= lag_min:
        return {"status": "insufficient_lag_range"}

    best_lag = None
    best_score = None
    for lag in range(lag_min, lag_max + 1):
        total = 0.0
        for idx in range(len(centered) - lag):
            total += centered[idx] * centered[idx + lag]
        if best_score is None or total > best_score:
            best_score = total
            best_lag = lag

    if best_lag is None or best_lag <= 0:
        return {"status": "no_candidate"}

    candidate_hz = env_rate_hz / best_lag
    return {
        "status": "candidate_only",
        "envelope_sample_rate_hz": env_rate_hz,
        "best_lag_samples": best_lag,
        "candidate_frequency_hz": candidate_hz,
        "candidate_bpm": candidate_hz * 60.0,
        "autocorrelation_score": best_score,
    }


def pearson_corr(a: list[int], b: list[int]) -> float | None:
    count = min(len(a), len(b))
    if count == 0:
        return None
    mean_a = sum(a[:count]) / count
    mean_b = sum(b[:count]) / count
    sum_ab = 0.0
    sum_aa = 0.0
    sum_bb = 0.0
    for idx in range(count):
        da = a[idx] - mean_a
        db = b[idx] - mean_b
        sum_ab += da * db
        sum_aa += da * da
        sum_bb += db * db
    if sum_aa <= 0.0 or sum_bb <= 0.0:
        return None
    return sum_ab / math.sqrt(sum_aa * sum_bb)


def channel_pairwise_correlations(channels: list[list[int]]) -> list[dict]:
    pairs = []
    for left_idx in range(len(channels)):
        for right_idx in range(left_idx + 1, len(channels)):
            pairs.append(
                {
                    "channel_a": left_idx,
                    "channel_b": right_idx,
                    "pearson_corr_zero_lag": pearson_corr(channels[left_idx], channels[right_idx]),
                }
            )
    return pairs


def strongest_fragment_starts(
    samples: list[int],
    sample_rate: int,
    fragment_seconds: float,
    max_fragments: int,
) -> list[int]:
    fragment_len = max(int(sample_rate * fragment_seconds), 1)
    if len(samples) <= fragment_len:
        return [0]

    stride = max(fragment_len // 2, 1)
    candidates: list[tuple[float, int]] = []
    for start in range(0, len(samples) - fragment_len + 1, stride):
        total_sq = 0.0
        for sample in samples[start : start + fragment_len]:
            total_sq += sample * sample
        rms = math.sqrt(total_sq / fragment_len)
        candidates.append((rms, start))

    candidates.sort(reverse=True)
    selected: list[int] = []
    min_gap = fragment_len
    for _, start in candidates:
        if all(abs(start - existing) >= min_gap for existing in selected):
            selected.append(start)
        if len(selected) >= max_fragments:
            break

    if 0 not in selected:
        selected.append(0)
    selected.sort()
    return selected[:max_fragments]


def make_waveform_svg(
    samples: list[int],
    out_path: Path,
    width: int,
    height: int,
    full_scale: float,
    amplitude_scale: float | None = None,
    title: str | None = None,
) -> None:
    step = max(len(samples) // width, 1)
    mid_y = height / 2.0
    scale_limit = amplitude_scale if amplitude_scale is not None else full_scale
    scale_limit = max(scale_limit, 1.0)
    scale_y = (height * 0.42) / scale_limit

    vertical_lines = []
    x = 0
    for start in range(0, len(samples), step):
        chunk = samples[start : start + step]
        if not chunk:
            continue
        chunk_min = min(chunk)
        chunk_max = max(chunk)
        y_top = mid_y - (chunk_max * scale_y)
        y_bottom = mid_y - (chunk_min * scale_y)
        vertical_lines.append(
            f'<line x1="{x}" y1="{y_top:.2f}" x2="{x}" y2="{y_bottom:.2f}" stroke="#0b7285" stroke-width="1" />'
        )
        x += 1
        if x >= width:
            break

    title_text = ""
    if title:
        title_text = f'<text x="12" y="20" font-size="14" fill="#343a40">{title}</text>'

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8f9fa" />
  <line x1="0" y1="{mid_y:.2f}" x2="{width}" y2="{mid_y:.2f}" stroke="#adb5bd" stroke-width="1" />
  {title_text}
  {"".join(vertical_lines)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def make_envelope_svg(
    envelope: list[float],
    out_path: Path,
    width: int,
    height: int,
    title: str,
) -> None:
    if not envelope:
        out_path.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='600' height='120'></svg>", encoding="utf-8")
        return

    max_value = max(envelope) or 1.0
    x_span = max(width - 20, 1)
    points = []
    for pixel_x in range(x_span):
        start = int(pixel_x * len(envelope) / x_span)
        end = int((pixel_x + 1) * len(envelope) / x_span)
        if start >= len(envelope):
            break
        chunk = envelope[start:end] or [envelope[start]]
        value = sum(chunk) / len(chunk)
        x = 10 + pixel_x
        y = height - 20 - ((value / max_value) * (height - 40))
        points.append(f"{x:.2f},{y:.2f}")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8f9fa" />
  <text x="12" y="20" font-size="14" fill="#343a40">{title}</text>
  <polyline fill="none" stroke="#c92a2a" stroke-width="2" points="{' '.join(points)}" />
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def write_json(data: dict, out_path: Path) -> None:
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / args.wav_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    channels, metadata = load_wav_channels(args.wav_path)
    full_scale = FULL_SCALE_BY_WIDTH[metadata["sample_width_bytes"]]

    channel_summaries = []
    for channel_index, channel_samples in enumerate(channels):
        fragment_len = max(int(metadata["sample_rate_hz"] * args.fragment_seconds), 1)
        fragment_starts = strongest_fragment_starts(
            channel_samples,
            metadata["sample_rate_hz"],
            args.fragment_seconds,
            args.max_fragments,
        )
        fragment_artifacts = []
        for fragment_index, start in enumerate(fragment_starts):
            end = min(start + fragment_len, len(channel_samples))
            fragment = channel_samples[start:end]
            local_scale = max((abs(sample) for sample in fragment), default=1)
            fragment_name = f"channel_{channel_index:02d}_fragment_{fragment_index:02d}.svg"
            make_waveform_svg(
                fragment,
                output_dir / fragment_name,
                args.svg_width,
                args.svg_height,
                full_scale,
                amplitude_scale=local_scale,
                title=(
                    f"channel {channel_index} fragment {fragment_index} "
                    f"{start / metadata['sample_rate_hz']:.2f}-{end / metadata['sample_rate_hz']:.2f}s autoscaled"
                ),
            )
            fragment_artifacts.append(
                {
                    "start_seconds": start / metadata["sample_rate_hz"],
                    "end_seconds": end / metadata["sample_rate_hz"],
                    "svg": fragment_name,
                    "autoscale_peak_abs": local_scale,
                }
            )

        envelope = moving_average_abs(
            channel_samples,
            window_size=max(metadata["sample_rate_hz"] // 20, 1),
            stride=max(metadata["sample_rate_hz"] // 50, 1),
        )
        envelope_name = f"channel_{channel_index:02d}_envelope.svg"
        make_envelope_svg(
            envelope,
            output_dir / envelope_name,
            args.svg_width,
            args.svg_height,
            title=f"channel {channel_index} smoothed abs-amplitude envelope",
        )
        channel_summary = {
            "channel_index": channel_index,
            "sample_summary": summarize_channel(channel_samples, metadata["sample_rate_hz"], full_scale),
            "window_rms_1s": summarize_window_rms(channel_samples, metadata["sample_rate_hz"]),
            "coarse_spectral_summary": summarize_coarse_spectrum(channel_samples, metadata["sample_rate_hz"]),
            "rhythm_candidate_from_envelope": estimate_hr_candidate_from_envelope(
                channel_samples,
                metadata["sample_rate_hz"],
            ),
            "artifact_paths": {
                "waveform_svg": f"channel_{channel_index:02d}_waveform.svg",
                "envelope_svg": envelope_name,
                "fragment_svgs": fragment_artifacts,
            },
        }
        channel_summaries.append(channel_summary)
        make_waveform_svg(
            channel_samples,
            output_dir / f"channel_{channel_index:02d}_waveform.svg",
            args.svg_width,
            args.svg_height,
            full_scale,
            title=f"channel {channel_index} full recording fixed scale",
        )

    summary = {
        "file_name": args.wav_path.name,
        "metadata": metadata,
        "channel_summaries": channel_summaries,
        "pairwise_correlations": channel_pairwise_correlations(channels),
    }
    write_json(summary, output_dir / "summary.json")

    print(f"Saved summary to {output_dir / 'summary.json'}")
    for channel_index in range(len(channels)):
        print(f"Saved waveform SVG to {output_dir / f'channel_{channel_index:02d}_waveform.svg'}")
        print(f"Saved envelope SVG to {output_dir / f'channel_{channel_index:02d}_envelope.svg'}")


if __name__ == "__main__":
    main()
