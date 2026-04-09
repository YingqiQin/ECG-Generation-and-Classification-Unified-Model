from __future__ import annotations

import argparse
import html
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


STANDARD_CHANNELS = [f"CH{i}" for i in range(1, 9)]
UPPER_ARM_CHANNEL = "CH20"
ALL_SIGNAL_CHANNELS = [*STANDARD_CHANNELS, UPPER_ARM_CHANNEL]
REQUIRED_COLUMNS = ["timestamp_ms", *ALL_SIGNAL_CHANNELS]
FILENAME_PATTERN = re.compile(r"emg_data_(\d{8})_(\d{6})\.csv$")
DEFAULT_SEGMENT_SECONDS = 8.0


@dataclass
class RecordingSummary:
    path: Path
    capture_label: str
    duration_seconds: float
    sample_count: int
    removed_rows: int
    sampling_rate_hz: float
    sampling_interval_ms: float
    sampling_jitter_ms: float
    reference_channel: str
    reference_quality_score: float
    upperarm_quality_score: float
    upperarm_vs_reference_corr: float
    upperarm_best_match_channel: str
    upperarm_best_match_corr: float
    upperarm_bpm: float
    reconstruction_available: bool
    reconstruction_corr: float
    reconstruction_mae: float
    reconstruction_rmse: float
    report_row: dict[str, object]
    raw_plot_path: Path
    filtered_plot_path: Path
    beat_plot_path: Path
    quality_plot_path: Path
    similarity_plot_path: Path
    reconstruction_plot_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize and analyze ECG CSV recordings with standard ECG channels "
            "CH1-CH8 and upper-arm ECG channel CH20."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing files like emg_data_YYYYMMDD_HHMMSS.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where generated figures and reports will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of CSV files to process.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=12000,
        help="Downsample each plot to at most this many points for speed.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=DEFAULT_SEGMENT_SECONDS,
        help="Length of the beat inspection window in seconds.",
    )
    parser.add_argument(
        "--recon-dir",
        type=Path,
        default=None,
        help=(
            "Optional folder containing reconstructed ECG CSV files with the same "
            "filenames as the input recordings."
        ),
    )
    parser.add_argument(
        "--recon-column",
        type=str,
        default="reconstructed_signal",
        help=(
            "Column name to read from reconstruction CSV files. If absent and the file "
            "has exactly one non-timestamp column, that column is used automatically."
        ),
    )
    return parser.parse_args()


def find_csv_files(input_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(input_dir.glob("emg_data_*.csv"))
    if limit is not None:
        files = files[:limit]
    return files


def extract_capture_label(path: Path) -> str:
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return path.stem
    date_part, time_part = match.groups()
    return (
        f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} "
        f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
    )


def format_float(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.{digits}f}"


def read_csv(path: Path) -> tuple[pd.DataFrame, int]:
    raw_df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in raw_df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{path.name} is missing required columns: {missing_text}")

    df = raw_df[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    removed_rows = before - len(df)
    if df.empty:
        raise ValueError(f"{path.name} has no valid numeric rows after cleaning")
    return df.reset_index(drop=True), removed_rows


def read_reconstruction_csv(path: Path, recon_column: str) -> pd.DataFrame:
    raw_df = pd.read_csv(path)
    if "timestamp_ms" not in raw_df.columns:
        raise ValueError(f"{path.name} is missing required column timestamp_ms")

    if recon_column in raw_df.columns:
        signal_column = recon_column
    else:
        candidate_columns = [column for column in raw_df.columns if column != "timestamp_ms"]
        if len(candidate_columns) != 1:
            raise ValueError(
                f"{path.name} must contain column {recon_column} or exactly one "
                "non-timestamp signal column"
            )
        signal_column = candidate_columns[0]

    df = raw_df[["timestamp_ms", signal_column]].copy()
    df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
    df[signal_column] = pd.to_numeric(df[signal_column], errors="coerce")
    df = df.dropna(subset=["timestamp_ms", signal_column]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path.name} has no valid rows after cleaning")

    return df.rename(columns={signal_column: "reconstructed_signal"})


def maybe_downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = math.ceil(len(df) / max_points)
    return df.iloc[::step].reset_index(drop=True)


def estimate_sampling_rate(df: pd.DataFrame) -> tuple[float, float, float]:
    diffs_ms = df["timestamp_ms"].diff().dropna()
    if diffs_ms.empty:
        return 0.0, 0.0, 0.0
    median_diff_ms = float(diffs_ms.median())
    sampling_rate_hz = 1000.0 / median_diff_ms if median_diff_ms > 0 else 0.0
    jitter_ms = float((diffs_ms - median_diff_ms).abs().median())
    return sampling_rate_hz, median_diff_ms, jitter_ms


def rolling_mean(series: pd.Series, window_samples: int) -> pd.Series:
    window_samples = max(1, int(window_samples))
    return series.rolling(window=window_samples, center=True, min_periods=1).mean()


def filter_signal(series: pd.Series, sampling_rate_hz: float) -> pd.Series:
    if sampling_rate_hz <= 0:
        return series - series.mean()

    highpass_window = max(3, int(round(sampling_rate_hz * 0.6)))
    lowpass_window = max(1, int(round(sampling_rate_hz * 0.03)))

    baseline = rolling_mean(series, highpass_window)
    highpassed = series - baseline
    filtered = rolling_mean(highpassed, lowpass_window)
    return filtered


def normalize_signal(series: pd.Series) -> pd.Series:
    centered = series - float(series.median())
    scale = float(centered.std())
    if scale <= 1e-8:
        scale = float(centered.abs().quantile(0.9))
    if scale <= 1e-8:
        scale = 1.0
    return centered / scale


def detect_peaks(signal: pd.Series, sampling_rate_hz: float) -> list[int]:
    if len(signal) < 5 or sampling_rate_hz <= 0:
        return []

    normalized = normalize_signal(signal).abs().reset_index(drop=True)
    threshold = max(0.8, float(normalized.quantile(0.93)))
    refractory = max(1, int(round(0.30 * sampling_rate_hz)))

    peaks: list[int] = []
    last_peak = -refractory
    for idx in range(1, len(normalized) - 1):
        value = float(normalized.iloc[idx])
        if value < threshold:
            continue
        if value < float(normalized.iloc[idx - 1]) or value < float(normalized.iloc[idx + 1]):
            continue
        if idx - last_peak < refractory:
            if peaks and value > float(normalized.iloc[peaks[-1]]):
                peaks[-1] = idx
                last_peak = idx
            continue
        peaks.append(idx)
        last_peak = idx
    return peaks


def compute_channel_metrics(
    df: pd.DataFrame,
    raw_series: pd.Series,
    filtered_series: pd.Series,
    channel: str,
    sampling_rate_hz: float,
) -> dict[str, float]:
    values = raw_series.reset_index(drop=True)
    filtered = filtered_series.reset_index(drop=True)
    diffs = values.diff().dropna()
    amplitude_range = float(values.max() - values.min())
    std_raw = float(values.std())
    std_filtered = float(filtered.std())
    baseline_component = values - filtered
    baseline_std = float(baseline_component.std())

    flat_threshold = max(amplitude_range * 1e-5, 1e-9)
    flatline_ratio = float((diffs.abs() <= flat_threshold).mean()) if not diffs.empty else 1.0

    q01 = float(values.quantile(0.01))
    q99 = float(values.quantile(0.99))
    clip_margin = max((q99 - q01) * 0.002, 1e-9)
    clipping_ratio = float(((values <= q01 + clip_margin) | (values >= q99 - clip_margin)).mean())

    peaks = detect_peaks(filtered, sampling_rate_hz=sampling_rate_hz)
    duration_seconds = max(
        (float(df["timestamp_ms"].iloc[-1]) - float(df["timestamp_ms"].iloc[0])) / 1000.0,
        1e-9,
    )
    bpm = float("nan")
    rr_cv = float("nan")
    if len(peaks) >= 2:
        peak_times_s = df["timestamp_ms"].iloc[peaks].reset_index(drop=True) / 1000.0
        rr_intervals = peak_times_s.diff().dropna()
        if not rr_intervals.empty:
            bpm = float(60.0 / rr_intervals.median())
            rr_cv = float(rr_intervals.std() / rr_intervals.mean()) if rr_intervals.mean() > 0 else float("nan")
    elif len(peaks) == 1:
        bpm = float(60.0 / duration_seconds)

    baseline_ratio = baseline_std / (std_filtered + 1e-9)
    dynamic_ratio = std_filtered / (std_raw + 1e-9)

    quality_score = 100.0
    quality_score -= min(35.0, flatline_ratio * 160.0)
    quality_score -= min(20.0, clipping_ratio * 80.0)
    quality_score -= min(18.0, baseline_ratio * 12.0)
    quality_score -= min(12.0, abs(dynamic_ratio - 1.0) * 20.0)
    if pd.isna(bpm) or bpm < 35 or bpm > 220:
        quality_score -= 25.0
    if not pd.isna(rr_cv):
        quality_score -= min(10.0, rr_cv * 20.0)
    quality_score = max(0.0, min(100.0, quality_score))

    return {
        "channel": channel,
        "amplitude_range": amplitude_range,
        "std_raw": std_raw,
        "std_filtered": std_filtered,
        "baseline_ratio": baseline_ratio,
        "dynamic_ratio": dynamic_ratio,
        "flatline_ratio": flatline_ratio,
        "clipping_ratio": clipping_ratio,
        "peak_count": float(len(peaks)),
        "estimated_bpm": bpm,
        "rr_cv": rr_cv,
        "quality_score": quality_score,
    }


def compute_pairwise_correlations(filtered_df: pd.DataFrame, source_channel: str) -> dict[str, float]:
    source = normalize_signal(filtered_df[source_channel]).reset_index(drop=True)
    correlations: dict[str, float] = {}
    for channel in STANDARD_CHANNELS:
        target = normalize_signal(filtered_df[channel]).reset_index(drop=True)
        correlations[channel] = float(source.corr(target))
    return correlations


def compute_cross_channel_metrics(
    filtered_df: pd.DataFrame,
    reference_channel: str,
    upperarm_channel: str,
) -> tuple[float, float]:
    ref = normalize_signal(filtered_df[reference_channel]).reset_index(drop=True)
    upper = normalize_signal(filtered_df[upperarm_channel]).reset_index(drop=True)
    corr = float(ref.corr(upper))

    ref_abs = ref.abs()
    upper_abs = upper.abs()
    energy_ratio = float(upper_abs.mean() / (ref_abs.mean() + 1e-9))
    return corr, energy_ratio


def evaluate_reconstruction(
    filtered_df: pd.DataFrame,
    recon_df: pd.DataFrame,
    target_channel: str,
    sampling_rate_hz: float,
) -> dict[str, float]:
    merged = filtered_df[["timestamp_ms", target_channel]].merge(recon_df, on="timestamp_ms", how="inner")
    if merged.empty:
        raise ValueError("Reconstruction file does not align with input timestamps")

    target_filtered = filter_signal(merged[target_channel], sampling_rate_hz=sampling_rate_hz)
    recon_filtered = filter_signal(merged["reconstructed_signal"], sampling_rate_hz=sampling_rate_hz)
    target_norm = normalize_signal(target_filtered).reset_index(drop=True)
    recon_norm = normalize_signal(recon_filtered).reset_index(drop=True)

    corr = float(target_norm.corr(recon_norm))
    diff = target_norm - recon_norm
    mae = float(diff.abs().mean())
    rmse = float((diff.pow(2).mean()) ** 0.5)
    cosine = float((target_norm * recon_norm).mean() / ((target_norm.pow(2).mean() ** 0.5) * (recon_norm.pow(2).mean() ** 0.5) + 1e-9))

    return {
        "aligned_sample_count": float(len(merged)),
        "corr": corr,
        "mae": mae,
        "rmse": rmse,
        "cosine_similarity": cosine,
    }


def build_filtered_df(df: pd.DataFrame, sampling_rate_hz: float) -> pd.DataFrame:
    filtered_df = pd.DataFrame({"timestamp_ms": df["timestamp_ms"]})
    for channel in ALL_SIGNAL_CHANNELS:
        filtered_df[channel] = filter_signal(df[channel], sampling_rate_hz=sampling_rate_hz)
    return filtered_df


def select_reference_channel(channel_metrics: dict[str, dict[str, float]]) -> str:
    ranked = sorted(
        STANDARD_CHANNELS,
        key=lambda channel: (
            channel_metrics[channel]["quality_score"],
            channel_metrics[channel]["std_filtered"],
        ),
        reverse=True,
    )
    return ranked[0]


def select_best_match_channel(correlations: dict[str, float]) -> tuple[str, float]:
    ranked = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)
    return ranked[0]


def plot_raw_recording(
    df: pd.DataFrame,
    summary_title: str,
    output_path: Path,
    dpi: int,
    max_points: int,
) -> None:
    plot_df = maybe_downsample(df, max_points=max_points)
    time_s = (plot_df["timestamp_ms"] - plot_df["timestamp_ms"].iloc[0]) / 1000.0

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 10), sharex=True, constrained_layout=True)

    for channel in STANDARD_CHANNELS[:4]:
        axes[0].plot(time_s, plot_df[channel], linewidth=0.8, label=channel)
    axes[0].set_title("Raw Standard ECG Channels CH1-CH4")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", ncol=4)

    for channel in STANDARD_CHANNELS[4:]:
        axes[1].plot(time_s, plot_df[channel], linewidth=0.8, label=channel)
    axes[1].set_title("Raw Standard ECG Channels CH5-CH8")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right", ncol=4)

    axes[2].plot(time_s, plot_df[UPPER_ARM_CHANNEL], linewidth=0.9, color="black", label=UPPER_ARM_CHANNEL)
    axes[2].set_title("Raw Upper-Arm ECG Channel CH20")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right")

    fig.suptitle(summary_title)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_filtered_stacked(
    filtered_df: pd.DataFrame,
    summary_title: str,
    output_path: Path,
    dpi: int,
    max_points: int,
) -> None:
    plot_df = maybe_downsample(filtered_df, max_points=max_points)
    time_s = (plot_df["timestamp_ms"] - plot_df["timestamp_ms"].iloc[0]) / 1000.0

    fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)
    colors = plt.cm.tab10.colors
    offset_step = 4.0

    for idx, channel in enumerate(ALL_SIGNAL_CHANNELS):
        normalized = normalize_signal(plot_df[channel]) + idx * offset_step
        color = "black" if channel == UPPER_ARM_CHANNEL else colors[idx % len(colors)]
        linewidth = 1.0 if channel == UPPER_ARM_CHANNEL else 0.9
        ax.plot(time_s, normalized, label=channel, linewidth=linewidth, color=color)

    ax.set_title("Filtered and Normalized Stacked Comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Amplitude + Offset")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=3)
    fig.suptitle(summary_title)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def choose_segment_start_seconds(
    df: pd.DataFrame,
    reference_channel: str,
    filtered_df: pd.DataFrame,
    segment_seconds: float,
) -> float:
    total_duration_s = max(
        (float(df["timestamp_ms"].iloc[-1]) - float(df["timestamp_ms"].iloc[0])) / 1000.0,
        0.0,
    )
    if total_duration_s <= segment_seconds:
        return 0.0

    time_s = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0
    abs_signal = normalize_signal(filtered_df[reference_channel]).abs()
    window = max(3, int(len(df) * segment_seconds / max(total_duration_s, 1e-9) / 3))
    rolling = abs_signal.rolling(window=window, min_periods=1).mean()
    best_idx = int(rolling.idxmax())
    center_s = float(time_s.iloc[best_idx])
    start_s = max(0.0, min(total_duration_s - segment_seconds, center_s - segment_seconds / 2.0))
    return start_s


def plot_beat_inspection(
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    summary_title: str,
    reference_channel: str,
    output_path: Path,
    dpi: int,
    segment_seconds: float,
    sampling_rate_hz: float,
) -> None:
    start_s = choose_segment_start_seconds(df, reference_channel, filtered_df, segment_seconds)
    time_s = (df["timestamp_ms"] - df["timestamp_ms"].iloc[0]) / 1000.0
    mask = (time_s >= start_s) & (time_s <= start_s + segment_seconds)

    seg_time = time_s.loc[mask].reset_index(drop=True)
    ref_signal = filtered_df.loc[mask, reference_channel].reset_index(drop=True)
    upper_signal = filtered_df.loc[mask, UPPER_ARM_CHANNEL].reset_index(drop=True)

    ref_norm = normalize_signal(ref_signal)
    upper_norm = normalize_signal(upper_signal)
    ref_peaks = detect_peaks(ref_signal, sampling_rate_hz=sampling_rate_hz)
    upper_peaks = detect_peaks(upper_signal, sampling_rate_hz=sampling_rate_hz)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8), sharex=True, constrained_layout=True)

    axes[0].plot(seg_time, ref_norm, color="#22577A", linewidth=1.0, label=reference_channel)
    if ref_peaks:
        axes[0].scatter(seg_time.iloc[ref_peaks], ref_norm.iloc[ref_peaks], color="#D1495B", s=18, label="Detected peaks")
    axes[0].set_title(f"Beat Inspection: Reference Channel {reference_channel}")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(seg_time, upper_norm, color="black", linewidth=1.0, label=UPPER_ARM_CHANNEL)
    if upper_peaks:
        axes[1].scatter(seg_time.iloc[upper_peaks], upper_norm.iloc[upper_peaks], color="#D1495B", s=18, label="Detected peaks")
    axes[1].set_title("Beat Inspection: Upper-Arm Channel CH20")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Normalized amplitude")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.suptitle(f"{summary_title} | Inspection window starts at {start_s:.2f}s")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_quality_bars(
    channel_metrics: dict[str, dict[str, float]],
    summary_title: str,
    output_path: Path,
    dpi: int,
) -> None:
    channels = ALL_SIGNAL_CHANNELS
    quality_scores = [channel_metrics[channel]["quality_score"] for channel in channels]
    estimated_bpms = [channel_metrics[channel]["estimated_bpm"] for channel in channels]
    baseline_ratios = [channel_metrics[channel]["baseline_ratio"] for channel in channels]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 11), constrained_layout=True)

    axes[0].bar(channels, quality_scores, color=["black" if channel == UPPER_ARM_CHANNEL else "#5DA271" for channel in channels])
    axes[0].set_title("Quality Score by Channel")
    axes[0].set_ylabel("Score (0-100)")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(channels, [0.0 if pd.isna(item) else item for item in estimated_bpms], color="#3A7CA5")
    axes[1].set_title("Estimated BPM by Channel")
    axes[1].set_ylabel("BPM")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(channels, baseline_ratios, color="#B56576")
    axes[2].set_title("Baseline Wander Ratio by Channel")
    axes[2].set_ylabel("Baseline / Filtered Std")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle(summary_title)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_similarity_heatmap(
    correlations: dict[str, float],
    summary_title: str,
    best_match_channel: str,
    output_path: Path,
    dpi: int,
) -> None:
    ordered_values = [[correlations[channel] for channel in STANDARD_CHANNELS]]
    fig, ax = plt.subplots(figsize=(12, 2.8), constrained_layout=True)
    image = ax.imshow(ordered_values, cmap="coolwarm", aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_title(f"CH20 Similarity to Standard Channels | Best match {best_match_channel}")
    ax.set_xticks(range(len(STANDARD_CHANNELS)))
    ax.set_xticklabels(STANDARD_CHANNELS)
    ax.set_yticks([0])
    ax.set_yticklabels([UPPER_ARM_CHANNEL])
    for idx, channel in enumerate(STANDARD_CHANNELS):
        ax.text(idx, 0, f"{correlations[channel]:.2f}", ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03, label="Correlation")
    fig.suptitle(summary_title)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_reconstruction_comparison(
    filtered_df: pd.DataFrame,
    recon_df: pd.DataFrame,
    target_channel: str,
    summary_title: str,
    output_path: Path,
    dpi: int,
    max_points: int,
    sampling_rate_hz: float,
) -> None:
    merged = filtered_df[["timestamp_ms", target_channel]].merge(recon_df, on="timestamp_ms", how="inner")
    if merged.empty:
        return

    merged["target_filtered"] = filter_signal(merged[target_channel], sampling_rate_hz=sampling_rate_hz)
    merged["recon_filtered"] = filter_signal(merged["reconstructed_signal"], sampling_rate_hz=sampling_rate_hz)
    plot_df = maybe_downsample(merged[["timestamp_ms", "target_filtered", "recon_filtered"]], max_points=max_points)
    time_s = (plot_df["timestamp_ms"] - plot_df["timestamp_ms"].iloc[0]) / 1000.0

    fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
    ax.plot(time_s, normalize_signal(plot_df["target_filtered"]), label=f"Target {target_channel}", color="#22577A", linewidth=1.0)
    ax.plot(time_s, normalize_signal(plot_df["recon_filtered"]), label="AutoEncoder reconstruction", color="#D1495B", linewidth=1.0, alpha=0.85)
    ax.set_title(f"Reconstruction vs Most Similar Channel ({target_channel})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.suptitle(summary_title)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_overview(summaries: list[RecordingSummary], output_dir: Path, dpi: int) -> Path:
    labels = [item.capture_label for item in summaries]
    durations = [item.duration_seconds for item in summaries]
    sample_counts = [item.sample_count for item in summaries]
    upperarm_scores = [item.upperarm_quality_score for item in summaries]
    corrs = [item.upperarm_best_match_corr for item in summaries]

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(18, 14), constrained_layout=True)

    axes[0].bar(range(len(summaries)), durations, color="#3A7CA5")
    axes[0].set_title("Recording Duration by File")
    axes[0].set_ylabel("Seconds")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(range(len(summaries)), sample_counts, color="#7FB069")
    axes[1].set_title("Sample Count by File")
    axes[1].set_ylabel("Rows")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(range(len(summaries)), upperarm_scores, color="#1F1F1F")
    axes[2].set_title("Upper-Arm CH20 Quality Score by File")
    axes[2].set_ylabel("Score (0-100)")
    axes[2].grid(axis="y", alpha=0.25)

    axes[3].bar(range(len(summaries)), corrs, color="#B56576")
    axes[3].set_title("Best CH20 Correlation to Any Standard Channel")
    axes[3].set_ylabel("Correlation")
    axes[3].grid(axis="y", alpha=0.25)

    for axis in axes:
        axis.set_xticks(range(len(summaries)))
        axis.set_xticklabels(labels, rotation=45, ha="right")

    output_path = output_dir / "recording_overview.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def build_summary_table(summaries: list[RecordingSummary]) -> pd.DataFrame:
    return pd.DataFrame([summary.report_row for summary in summaries])


def write_html_report(summaries: list[RecordingSummary], overview_path: Path, output_dir: Path) -> Path:
    def rel(path: Path) -> str:
        return path.relative_to(output_dir).as_posix()

    rows = []
    for summary in summaries:
        reconstruction_text = (
            f"{summary.reconstruction_corr:.3f}" if summary.reconstruction_available else "n/a"
        )
        reconstruction_link = (
            f"<a href=\"{html.escape(rel(summary.reconstruction_plot_path))}\">recon</a>"
            if summary.reconstruction_plot_path
            else "n/a"
        )
        row = (
            "<tr>"
            f"<td>{html.escape(summary.path.name)}</td>"
            f"<td>{html.escape(summary.capture_label)}</td>"
            f"<td>{summary.duration_seconds:.2f}</td>"
            f"<td>{summary.sample_count}</td>"
            f"<td>{summary.sampling_rate_hz:.2f}</td>"
            f"<td>{html.escape(summary.upperarm_best_match_channel)}</td>"
            f"<td>{summary.upperarm_best_match_corr:.3f}</td>"
            f"<td>{summary.upperarm_quality_score:.1f}</td>"
            f"<td>{summary.upperarm_bpm:.1f}</td>"
            f"<td>{reconstruction_text}</td>"
            f"<td><a href=\"{html.escape(rel(summary.raw_plot_path))}\">raw</a></td>"
            f"<td><a href=\"{html.escape(rel(summary.filtered_plot_path))}\">filtered</a></td>"
            f"<td><a href=\"{html.escape(rel(summary.beat_plot_path))}\">beats</a></td>"
            f"<td><a href=\"{html.escape(rel(summary.quality_plot_path))}\">quality</a></td>"
            f"<td><a href=\"{html.escape(rel(summary.similarity_plot_path))}\">similarity</a></td>"
            f"<td>{reconstruction_link}</td>"
            "</tr>"
        )
        rows.append(row)

    detail_sections = []
    for summary in summaries:
        reconstruction_paragraph = (
            f"<p>Reconstruction corr {summary.reconstruction_corr:.3f} | "
            f"MAE {summary.reconstruction_mae:.3f} | RMSE {summary.reconstruction_rmse:.3f}</p>"
            if summary.reconstruction_available
            else "<p>No reconstruction file matched this recording.</p>"
        )
        reconstruction_image = (
            f"<img src=\"{html.escape(rel(summary.reconstruction_plot_path))}\" alt=\"Reconstruction plot for {html.escape(summary.path.name)}\">"
            if summary.reconstruction_plot_path
            else ""
        )
        detail_sections.append(
            "\n".join(
                [
                    "<section class=\"card\">",
                    f"<h2>{html.escape(summary.path.name)}</h2>",
                    f"<p>Capture {html.escape(summary.capture_label)} | Best standard quality channel {html.escape(summary.reference_channel)} | "
                    f"Most similar to CH20: {html.escape(summary.upperarm_best_match_channel)} ({summary.upperarm_best_match_corr:.3f})</p>",
                    reconstruction_paragraph,
                    f"<img src=\"{html.escape(rel(summary.raw_plot_path))}\" alt=\"Raw plot for {html.escape(summary.path.name)}\">",
                    f"<img src=\"{html.escape(rel(summary.filtered_plot_path))}\" alt=\"Filtered plot for {html.escape(summary.path.name)}\">",
                    f"<img src=\"{html.escape(rel(summary.beat_plot_path))}\" alt=\"Beat plot for {html.escape(summary.path.name)}\">",
                    f"<img src=\"{html.escape(rel(summary.quality_plot_path))}\" alt=\"Quality plot for {html.escape(summary.path.name)}\">",
                    f"<img src=\"{html.escape(rel(summary.similarity_plot_path))}\" alt=\"Similarity plot for {html.escape(summary.path.name)}\">",
                    reconstruction_image,
                    "</section>",
                ]
            )
        )

    html_text = "\n".join(
        [
            "<!DOCTYPE html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            "<title>ECG Visualization Report</title>",
            "<style>",
            "body { font-family: Helvetica, Arial, sans-serif; margin: 24px; color: #1c1c1c; background: #fafafa; }",
            "h1, h2 { margin-bottom: 0.4rem; }",
            ".card { background: #ffffff; border: 1px solid #dddddd; border-radius: 12px; padding: 18px; margin: 24px 0; }",
            "table { border-collapse: collapse; width: 100%; background: #ffffff; }",
            "th, td { border: 1px solid #dddddd; padding: 8px 10px; text-align: left; font-size: 14px; }",
            "th { background: #f1f5f9; }",
            "img { width: 100%; margin: 12px 0; border: 1px solid #dddddd; border-radius: 8px; background: #ffffff; }",
            "a { color: #22577a; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>ECG Visualization and Quality Report</h1>",
            "<p>Evaluation focuses on two questions: which standard lead looks most similar to self-captured CH20, and how well an AutoEncoder reconstruction matches that most similar lead.</p>",
            "<section class=\"card\">",
            "<h2>Overview</h2>",
            f"<img src=\"{html.escape(rel(overview_path))}\" alt=\"Overview plot\">",
            "<table>",
            "<thead><tr><th>File</th><th>Capture</th><th>Duration s</th><th>Samples</th><th>Hz</th>"
            "<th>Best CH20 match</th><th>Best corr</th><th>CH20 score</th><th>CH20 BPM</th><th>Recon corr</th>"
            "<th>Raw</th><th>Filtered</th><th>Beats</th><th>Quality</th><th>Similarity</th><th>Recon</th></tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            "</section>",
            *detail_sections,
            "</body>",
            "</html>",
        ]
    )

    report_path = output_dir / "report.html"
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def analyze_recording(
    path: Path,
    output_dir: Path,
    dpi: int,
    max_points: int,
    segment_seconds: float,
    recon_dir: Path | None,
    recon_column: str,
) -> RecordingSummary:
    df, removed_rows = read_csv(path)
    capture_label = extract_capture_label(path)
    duration_seconds = max((df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]) / 1000.0, 0.0)
    sampling_rate_hz, sampling_interval_ms, sampling_jitter_ms = estimate_sampling_rate(df)
    filtered_df = build_filtered_df(df, sampling_rate_hz=sampling_rate_hz)

    channel_metrics: dict[str, dict[str, float]] = {}
    for channel in ALL_SIGNAL_CHANNELS:
        channel_metrics[channel] = compute_channel_metrics(
            df=df,
            raw_series=df[channel],
            filtered_series=filtered_df[channel],
            channel=channel,
            sampling_rate_hz=sampling_rate_hz,
        )

    reference_channel = select_reference_channel(channel_metrics)
    upperarm_correlations = compute_pairwise_correlations(filtered_df, source_channel=UPPER_ARM_CHANNEL)
    upperarm_best_match_channel, upperarm_best_match_corr = select_best_match_channel(upperarm_correlations)
    upper_corr, upper_energy_ratio = compute_cross_channel_metrics(
        filtered_df=filtered_df,
        reference_channel=reference_channel,
        upperarm_channel=UPPER_ARM_CHANNEL,
    )

    summary_title = (
        f"{path.name} | Capture {capture_label} | Duration {duration_seconds:.2f}s | "
        f"Samples {len(df)} | Fs {sampling_rate_hz:.2f} Hz"
    )

    raw_plot_path = output_dir / f"{path.stem}_raw.png"
    filtered_plot_path = output_dir / f"{path.stem}_filtered.png"
    beat_plot_path = output_dir / f"{path.stem}_beats.png"
    quality_plot_path = output_dir / f"{path.stem}_quality.png"
    similarity_plot_path = output_dir / f"{path.stem}_similarity.png"
    reconstruction_plot_path = None

    plot_raw_recording(df=df, summary_title=summary_title, output_path=raw_plot_path, dpi=dpi, max_points=max_points)
    plot_filtered_stacked(
        filtered_df=filtered_df,
        summary_title=summary_title,
        output_path=filtered_plot_path,
        dpi=dpi,
        max_points=max_points,
    )
    plot_beat_inspection(
        df=df,
        filtered_df=filtered_df,
        summary_title=summary_title,
        reference_channel=reference_channel,
        output_path=beat_plot_path,
        dpi=dpi,
        segment_seconds=segment_seconds,
        sampling_rate_hz=sampling_rate_hz,
    )
    plot_quality_bars(channel_metrics=channel_metrics, summary_title=summary_title, output_path=quality_plot_path, dpi=dpi)
    plot_similarity_heatmap(
        correlations=upperarm_correlations,
        summary_title=summary_title,
        best_match_channel=upperarm_best_match_channel,
        output_path=similarity_plot_path,
        dpi=dpi,
    )

    reconstruction_available = False
    reconstruction_corr = float("nan")
    reconstruction_mae = float("nan")
    reconstruction_rmse = float("nan")
    reconstruction_cosine = float("nan")
    reconstruction_aligned_count = float("nan")

    if recon_dir is not None:
        recon_path = recon_dir / path.name
        if recon_path.exists():
            recon_df = read_reconstruction_csv(recon_path, recon_column=recon_column)
            recon_metrics = evaluate_reconstruction(
                filtered_df=filtered_df,
                recon_df=recon_df,
                target_channel=upperarm_best_match_channel,
                sampling_rate_hz=sampling_rate_hz,
            )
            reconstruction_available = True
            reconstruction_corr = recon_metrics["corr"]
            reconstruction_mae = recon_metrics["mae"]
            reconstruction_rmse = recon_metrics["rmse"]
            reconstruction_cosine = recon_metrics["cosine_similarity"]
            reconstruction_aligned_count = recon_metrics["aligned_sample_count"]
            reconstruction_plot_path = output_dir / f"{path.stem}_reconstruction.png"
            plot_reconstruction_comparison(
                filtered_df=filtered_df,
                recon_df=recon_df,
                target_channel=upperarm_best_match_channel,
                summary_title=summary_title,
                output_path=reconstruction_plot_path,
                dpi=dpi,
                max_points=max_points,
                sampling_rate_hz=sampling_rate_hz,
            )

    reference_metrics = channel_metrics[reference_channel]
    upper_metrics = channel_metrics[UPPER_ARM_CHANNEL]

    report_row: dict[str, object] = {
        "file_name": path.name,
        "capture_label": capture_label,
        "duration_seconds": duration_seconds,
        "sample_count": len(df),
        "removed_rows": removed_rows,
        "sampling_rate_hz": sampling_rate_hz,
        "sampling_interval_ms": sampling_interval_ms,
        "sampling_jitter_ms": sampling_jitter_ms,
        "reference_channel": reference_channel,
        "reference_quality_score": reference_metrics["quality_score"],
        "reference_estimated_bpm": reference_metrics["estimated_bpm"],
        "upperarm_quality_score": upper_metrics["quality_score"],
        "upperarm_estimated_bpm": upper_metrics["estimated_bpm"],
        "upperarm_baseline_ratio": upper_metrics["baseline_ratio"],
        "upperarm_flatline_ratio": upper_metrics["flatline_ratio"],
        "upperarm_clipping_ratio": upper_metrics["clipping_ratio"],
        "upperarm_vs_reference_corr": upper_corr,
        "upperarm_vs_reference_energy_ratio": upper_energy_ratio,
        "upperarm_best_match_channel": upperarm_best_match_channel,
        "upperarm_best_match_corr": upperarm_best_match_corr,
        "reconstruction_available": reconstruction_available,
        "reconstruction_target_channel": upperarm_best_match_channel if reconstruction_available else "",
        "reconstruction_corr": reconstruction_corr,
        "reconstruction_mae": reconstruction_mae,
        "reconstruction_rmse": reconstruction_rmse,
        "reconstruction_cosine_similarity": reconstruction_cosine,
        "reconstruction_aligned_sample_count": reconstruction_aligned_count,
    }

    for channel in STANDARD_CHANNELS:
        report_row[f"ch20_vs_{channel.lower()}_corr"] = upperarm_correlations[channel]

    for channel in ALL_SIGNAL_CHANNELS:
        metrics = channel_metrics[channel]
        prefix = channel.lower()
        report_row[f"{prefix}_quality_score"] = metrics["quality_score"]
        report_row[f"{prefix}_estimated_bpm"] = metrics["estimated_bpm"]
        report_row[f"{prefix}_baseline_ratio"] = metrics["baseline_ratio"]
        report_row[f"{prefix}_flatline_ratio"] = metrics["flatline_ratio"]
        report_row[f"{prefix}_clipping_ratio"] = metrics["clipping_ratio"]
        report_row[f"{prefix}_peak_count"] = metrics["peak_count"]

    return RecordingSummary(
        path=path,
        capture_label=capture_label,
        duration_seconds=duration_seconds,
        sample_count=len(df),
        removed_rows=removed_rows,
        sampling_rate_hz=sampling_rate_hz,
        sampling_interval_ms=sampling_interval_ms,
        sampling_jitter_ms=sampling_jitter_ms,
        reference_channel=reference_channel,
        reference_quality_score=reference_metrics["quality_score"],
        upperarm_quality_score=upper_metrics["quality_score"],
        upperarm_vs_reference_corr=upper_corr,
        upperarm_best_match_channel=upperarm_best_match_channel,
        upperarm_best_match_corr=upperarm_best_match_corr,
        upperarm_bpm=0.0 if pd.isna(upper_metrics["estimated_bpm"]) else upper_metrics["estimated_bpm"],
        reconstruction_available=reconstruction_available,
        reconstruction_corr=reconstruction_corr,
        reconstruction_mae=reconstruction_mae,
        reconstruction_rmse=reconstruction_rmse,
        report_row=report_row,
        raw_plot_path=raw_plot_path,
        filtered_plot_path=filtered_plot_path,
        beat_plot_path=beat_plot_path,
        quality_plot_path=quality_plot_path,
        similarity_plot_path=similarity_plot_path,
        reconstruction_plot_path=reconstruction_plot_path,
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    recon_dir = args.recon_dir.expanduser().resolve() if args.recon_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = find_csv_files(input_dir, args.limit)
    if not csv_files:
        raise SystemExit(f"No files matching emg_data_*.csv found in {input_dir}")

    summaries: list[RecordingSummary] = []
    for path in csv_files:
        summary = analyze_recording(
            path=path,
            output_dir=output_dir,
            dpi=args.dpi,
            max_points=args.max_points,
            segment_seconds=args.segment_seconds,
            recon_dir=recon_dir,
            recon_column=args.recon_column,
        )
        summaries.append(summary)
        recon_text = (
            f" | recon_corr={format_float(summary.reconstruction_corr, 3)}"
            if summary.reconstruction_available
            else ""
        )
        print(
            f"Analyzed {path.name} | fs={format_float(summary.sampling_rate_hz, 2)} Hz | "
            f"best_match={summary.upperarm_best_match_channel} ({format_float(summary.upperarm_best_match_corr, 3)}) | "
            f"CH20_score={format_float(summary.upperarm_quality_score, 1)}{recon_text}"
        )

    summary_df = build_summary_table(summaries)
    summary_csv_path = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    overview_path = plot_overview(summaries, output_dir=output_dir, dpi=args.dpi)
    html_report_path = write_html_report(summaries, overview_path=overview_path, output_dir=output_dir)

    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {overview_path}")
    print(f"Wrote {html_report_path}")


if __name__ == "__main__":
    main()
