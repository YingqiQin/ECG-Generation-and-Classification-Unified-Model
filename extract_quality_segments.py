#!/usr/bin/env python3
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""
extract_quality_segments.py -- 基于 PN-QRS 不确定性评估 CH20 信号质量，提取高质量 10 秒片段

原理：
  PN-QRS 在对每个 10 秒窗口推理时，模型内部同时计算两种不确定性：
    U_E (认知不确定性 Epistemic) = mi_est(logits)  模型对预测的自信程度
    U_A (偶然不确定性 Aleatoric) = en_est(logits)  信号本身的噪声程度
  mean(U_E + U_A) 低 → 信号干净 → 高质量窗口
  mean(U_E + U_A) 高 → 信号嘈杂 → 低质量窗口

  此方法只需要单路 CH20 信号，无需 12 导联参考。

用法：
  python extract_quality_segments.py --csv /path/to/data.csv --fs 1000
  python extract_quality_segments.py --csv /path/to/data.csv --fs 1000 --uc_thr 0.5
  python extract_quality_segments.py --csv /path/to/data.csv --fs 1000 --out_dir /path/to/output
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

PNQRS_ROOT = Path(__file__).parent.parent   # pipeline/ -> PN-QRS root
CKPT_PATH  = PNQRS_ROOT / "experiments/logs_real/zy2lki18/models/best_model.pt"
sys.path.insert(0, str(PNQRS_ROOT))

try:
    from dataset.dataset import preprocess_ecg
    from models.multi_head import decoder4qrs, encoder4qrs, phi_qrs
    from models.qrs_model import QRSModel
    from utils.qrs_post_process import correct, uncertain_est, mi_est, en_est
except ModuleNotFoundError:
    from ..dataset.dataset import preprocess_ecg
    from ..models.multi_head import decoder4qrs, encoder4qrs, phi_qrs
    from ..models.qrs_model import QRSModel
    from ..utils.qrs_post_process import correct, uncertain_est, mi_est, en_est

FS_MODEL      = 50
WIN_SEC       = 10
STEP_SEC_DEF  = 8          # 默认滑动步长（2 秒 overlap）；嘈杂数据可用 --step 1~2
UC_THR_DEF    = 1.0        # mean(U_E+U_A) 阈值，高于此值视为低质量
BEAT_MIN      = 5          # 10s 内最少心拍 (~30 bpm)
BEAT_MAX      = 25         # 10s 内最多心拍 (~150 bpm)
COLOR_GOOD    = "#2ecc71"
COLOR_BAD     = "#e74c3c"


# ──────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────

def load_model(device):
    ckpt  = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model = QRSModel(encoder4qrs(), decoder4qrs(), phi_qrs()).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model.eval()


# ──────────────────────────────────────────────
# 窗口预处理（CPU，返回 numpy (1, T_50hz)）
# ──────────────────────────────────────────────

def _preprocess_window(window_1d: np.ndarray, fs: int) -> np.ndarray:
    """Robustly preprocess one window for PN-QRS inference."""
    proc = preprocess_ecg(window_1d, fs=fs)
    if proc.ndim == 1:
        proc = proc[np.newaxis, :]
    elif proc.shape[0] > proc.shape[1]:
        proc = proc.T
    return proc.astype(np.float32)   # (1, T_50hz)


def _robust_display_limits(signal: np.ndarray) -> tuple[float, float] | None:
    values = np.asarray(signal, dtype=np.float32)
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    low = float(np.quantile(finite, 0.005))
    high = float(np.quantile(finite, 0.995))
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if high <= low:
        center = float(np.median(finite))
        span = max(float(np.std(finite)), 1.0)
        return center - span, center + span
    pad = max((high - low) * 0.08, 1e-3)
    return low - pad, high + pad


# ──────────────────────────────────────────────
# 自动阈值：Otsu's method（最大化组间方差）
# ──────────────────────────────────────────────

def otsu_threshold(values: list) -> float:
    """
    对 mean_uc 值列表做 1D Otsu 二值化，返回最优分割阈值。
    把"好窗口"（低 uc）和"坏窗口"（高 uc）尽可能分开。
    """
    arr = np.sort(np.array(values, dtype=float))
    n   = len(arr)
    if n < 2:
        return arr[0]

    # 累积和用于 O(n) 计算
    cumsum  = np.cumsum(arr)
    cumsum2 = np.cumsum(arr ** 2)

    best_thresh    = arr[0]
    best_var_inter = -1.0

    for i in range(1, n):
        w1 = i / n
        w2 = 1.0 - w1
        if w1 == 0 or w2 == 0:
            continue
        mu1 = cumsum[i - 1] / i
        mu2 = (cumsum[-1] - cumsum[i - 1]) / (n - i)
        var_inter = w1 * w2 * (mu1 - mu2) ** 2
        if var_inter > best_var_inter:
            best_var_inter = var_inter
            best_thresh    = float((arr[i - 1] + arr[i]) / 2)

    return best_thresh


# ──────────────────────────────────────────────
# 推理（不含阈值判断）+ 阈值应用（分离）
# ──────────────────────────────────────────────

def run_inference(signal: np.ndarray, fs: int, model, device,
                  infer_batch: int = 16, step_sec: float = STEP_SEC_DEF) -> list:
    """
    滑动窗口批量推理，返回每个窗口的 mean_uc 和 n_beats。
    infer_batch 个窗口一起做 GPU forward，比逐个推快 5-10x。
    step_sec 越小，overlap 越大，发现有效窗口的机会越多（嘈杂数据推荐 1~2s）。
    is_good 暂不填（由 apply_threshold 填入）。
    """
    ws      = int(WIN_SEC   * fs)
    ss      = int(step_sec  * fs)
    sig_len = len(signal)

    # ── 阶段 1：切片 + CPU 预处理 ─────────────────
    meta_list   = []   # (start_samp, end_samp)
    tensor_list = []   # list of (1, T_50hz) float32

    pos = 0
    while pos + ws <= sig_len:   # 不足一整窗的尾巴直接跳过，不推理不画 bar
        w = signal[pos: pos + ws]
        tensor_list.append(_preprocess_window(w, fs))
        meta_list.append((pos, pos + ws))
        pos += ss

    if not tensor_list:
        return []

    # ── 阶段 2：分批 GPU forward ──────────────────
    all_logits = []   # list of (n_classes, T_50hz) numpy
    for i in range(0, len(tensor_list), infer_batch):
        batch_np = np.stack(tensor_list[i: i + infer_batch], axis=0)  # (B, 1, T_50hz)
        batch_t  = torch.from_numpy(batch_np).to(device)
        with torch.no_grad():
            out = model(batch_t, return_projection=True)   # (B, n_classes, T_50hz[, 1])
            out = out.squeeze(-1).cpu().numpy()            # (B, n_classes, T_50hz)
        for j in range(out.shape[0]):
            all_logits.append(out[j])                      # (n_classes, T_50hz)

    # ── 阶段 3：CPU 后处理（uncertain_est + correct）─
    windows = []
    for (pos, actual_end), logits_i in zip(meta_list, all_logits):
        uc     = uncertain_est(logits_i)   # clamped U_E + U_A
        au     = en_est(logits_i)          # raw U_A (aleatoric)
        eu     = mi_est(logits_i)          # raw U_E (epistemic, before clamping)
        r_50hz = correct(logits_i[0], uc)
        r_orig = np.round(np.array(r_50hz) * (fs / FS_MODEL)).astype(int)
        r_rel  = r_orig[r_orig < (actual_end - pos)]

        windows.append(dict(
            start_samp  = pos,
            end_samp    = actual_end,
            start_s     = pos / fs,
            end_s       = actual_end / fs,
            n_beats     = len(r_rel),
            mean_uc     = round(float(np.mean(uc)), 4),
            mean_ue     = round(float(np.mean(eu)), 4),   # 论文 Stage-1 指标
            mean_ua     = round(float(np.mean(au)), 4),   # 论文 Stage-2 指标
            is_good     = False,
            r_peaks_abs = r_rel + pos,
        ))

    return windows


def apply_threshold(windows: list, uc_thr: float) -> list:
    """根据阈值设置每个窗口的 is_good。"""
    for w in windows:
        w["is_good"] = (w["mean_uc"] <= uc_thr) and (BEAT_MIN <= w["n_beats"] <= BEAT_MAX)
    return windows


def assess_quality(signal: np.ndarray, fs: int, model, device,
                   uc_thr: float, infer_batch: int = 16,
                   step_sec: float = STEP_SEC_DEF):
    """推理 + 立即应用阈值（单文件固定阈值模式用）。"""
    return apply_threshold(
        run_inference(signal, fs, model, device, infer_batch, step_sec), uc_thr
    )


# ──────────────────────────────────────────────
# 保存高质量片段 (NPZ)
# ──────────────────────────────────────────────

def save_segments(signal, windows, fs, out_dir, base_name, df=None):
    """高质量片段各存一个 NPZ 文件，包含 CH20 及可用的 CH1-CH8。"""
    os.makedirs(out_dir, exist_ok=True)

    # 找出 df 中存在的标准导联列（CH1~CH8），按编号排序
    std_cols = []
    if df is not None:
        std_cols = sorted(
            [c for c in df.columns if str(c).upper() in {f"CH{i}" for i in range(1, 9)}],
            key=lambda c: int(str(c)[2:])
        )

    good = [w for w in windows if w["is_good"]]
    for i, w in enumerate(good):
        s, e  = w["start_samp"], w["end_samp"]
        fname = f"{base_name}_seg{i:03d}_{int(w['start_s'])}s.npz"

        arrays = {
            "CH20":    signal[s:e],                          # 上臂导联
            "fs":      np.array(fs),
            "start_s": np.array(w["start_s"]),
            "mean_uc": np.array(w["mean_uc"]),
            "n_beats": np.array(w["n_beats"]),
            "r_peaks": w["r_peaks_abs"] - s,                 # 相对于片段起点
        }
        # 加入标准导联（若有）
        for col in std_cols:
            arrays[str(col).upper()] = df[col].values[s:e].astype(np.float32)

        np.savez(os.path.join(out_dir, fname), **arrays)
    return good


# ──────────────────────────────────────────────
# 可视化 1：全局概览（完整信号 + 窗口颜色 + 不确定性柱状图）
# ──────────────────────────────────────────────

def _coverage_spans(signal_len: int, windows: list, good: bool):
    """
    把 windows 里所有 is_good==good 的样本范围合并成连续区段列表，
    返回 [(start_s, end_s), ...] 供 axvspan 绘制。
    适用于大量重叠窗口，避免重复绘制。
    """
    mask = np.zeros(signal_len, dtype=bool)
    for w in windows:
        if w["is_good"] == good:
            mask[w["start_samp"]: w["end_samp"]] = True

    spans = []
    in_span = False
    for i, v in enumerate(mask):
        if v and not in_span:
            start = i; in_span = True
        elif not v and in_span:
            spans.append((start, i)); in_span = False
    if in_span:
        spans.append((start, signal_len))
    return spans


def plot_overview(signal, windows, fs, uc_thr, out_path):
    t        = np.arange(len(signal)) / fs
    n_good   = sum(w["is_good"] for w in windows)
    n_total  = len(windows)
    step_sec = windows[1]["start_s"] - windows[0]["start_s"] if len(windows) > 1 else 8.0
    dense    = step_sec < 4.0          # 密集滑动窗口模式

    fig, (ax_ecg, ax_uc) = plt.subplots(
        2, 1, figsize=(20, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]}
    )

    # ── ECG 信号 ─────────────────────────────────
    ax_ecg.plot(t, signal, lw=0.35, color="steelblue", alpha=0.85, rasterized=True)
    display_limits = _robust_display_limits(signal)
    if display_limits is not None:
        ax_ecg.set_ylim(*display_limits)

    # R-peak 红点（汇集所有窗口的 r_peaks_abs，去重）
    all_rp = np.unique(np.concatenate([
        w["r_peaks_abs"] for w in windows if len(w.get("r_peaks_abs", [])) > 0
    ])) if windows else np.array([], dtype=int)
    valid_rp = all_rp[(all_rp >= 0) & (all_rp < len(signal))]
    if len(valid_rp):
        ax_ecg.scatter(valid_rp / fs, signal[valid_rp],
                       color="red", s=8, zorder=5, linewidths=0, alpha=0.7)

    # 连续区段着色（无论窗口多密集都只画有限个矩形）
    for s, e in _coverage_spans(len(signal), windows, good=True):
        ax_ecg.axvspan(s / fs, e / fs, alpha=0.18, color=COLOR_GOOD, lw=0)
    for s, e in _coverage_spans(len(signal), windows, good=False):
        ax_ecg.axvspan(s / fs, e / fs, alpha=0.10, color=COLOR_BAD,  lw=0)

    ax_ecg.set_ylabel("CH20 (ADC counts, robust display)", fontsize=9)
    ax_ecg.set_title(
        f"{os.path.basename(out_path)}  |  "
        f"Green=good ({n_good}/{n_total})  Red=low-quality  "
        f"threshold mean(U_E+U_A) <= {uc_thr:.3f}  step={step_sec:.0f}s",
        fontsize=8.5, pad=5
    )
    ax_ecg.tick_params(labelsize=8)
    ax_ecg.spines["top"].set_visible(False)
    ax_ecg.spines["right"].set_visible(False)

    # ── 不确定性图 ────────────────────────────────
    ucs = np.array([min(w["mean_uc"], uc_thr * 3) for w in windows])

    if dense:
        # 把每个窗口的 mean_uc 铺回到它覆盖的每个样本（多窗口重叠时取均值），
        # 与上方 ECG panel 的时间轴完全对齐。
        sig_len     = len(signal)
        uc_map      = np.zeros(sig_len, dtype=np.float32)
        cnt_map     = np.zeros(sig_len, dtype=np.float32)
        good_map    = np.zeros(sig_len, dtype=bool)
        for w in windows:
            s, e = w["start_samp"], w["end_samp"]
            uc_map[s:e]  += w["mean_uc"]
            cnt_map[s:e] += 1
            if w["is_good"]:
                good_map[s:e] = True
        valid       = cnt_map > 0
        uc_map[valid] /= cnt_map[valid]
        uc_map      = np.clip(uc_map, 0, uc_thr * 3)

        ax_uc.fill_between(t, uc_map, 0, where=valid & good_map,
                           color=COLOR_GOOD, alpha=0.55, lw=0, label="good")
        ax_uc.fill_between(t, uc_map, 0, where=valid & ~good_map,
                           color=COLOR_BAD,  alpha=0.45, lw=0, label="low-quality")
        ax_uc.plot(t[valid], uc_map[valid], lw=0.6, color="steelblue", alpha=0.7)
    else:
        # 柱状图：左对齐到 start_s，宽度 = 实际窗口宽度（end_s - start_s）
        # 相邻窗口物理上有 step_sec 步长带来的 (WIN_SEC - step_sec)=2s 重叠，
        # bar 宽度取实际窗口宽度后，所有 bar（包括尾窗）都恰好重叠 2s，视觉一致。
        xs     = np.array([w["start_s"] for w in windows])
        bar_ws = np.array([w["end_s"] - w["start_s"] for w in windows])
        colors = [COLOR_GOOD if w["is_good"] else COLOR_BAD for w in windows]
        ax_uc.bar(xs, ucs, width=bar_ws, align="edge",
                  color=colors, alpha=0.75, edgecolor="none")

    ax_uc.axhline(uc_thr, color="gray", lw=1.0, linestyle="--",
                  label=f"threshold {uc_thr:.3f}")
    ax_uc.set_ylabel("mean(U_E+U_A)", fontsize=8)
    ax_uc.set_xlabel("Time (s)", fontsize=9)
    ax_uc.tick_params(labelsize=8)
    ax_uc.legend(fontsize=7, loc="upper right")
    ax_uc.spines["top"].set_visible(False)
    ax_uc.spines["right"].set_visible(False)

    plt.tight_layout(h_pad=0.5)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [概览图] {out_path}")


# ──────────────────────────────────────────────
# 可视化 2：高质量片段网格（每片段一个子图，带 R-peak 红点）
# ──────────────────────────────────────────────

def plot_good_segments(signal, good_windows, fs, out_path, max_cols=3):
    if not good_windows:
        print("  [片段图] 无高质量片段，跳过绘图")
        return

    n     = len(good_windows)
    ncols = min(n, max_cols)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(6 * ncols, 3 * nrows))
    gs  = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.3)

    for idx, w in enumerate(good_windows):
        r = idx // ncols
        c = idx  % ncols
        ax = fig.add_subplot(gs[r, c])

        seg = signal[w["start_samp"]: w["end_samp"]]
        t_ax = np.arange(len(seg)) / fs

        ax.plot(t_ax, seg, lw=0.6, color="steelblue", alpha=0.9)
        display_limits = _robust_display_limits(seg)
        if display_limits is not None:
            ax.set_ylim(*display_limits)

        # R-peak 红点（相对于片段起点）
        rp_rel = w["r_peaks_abs"] - w["start_samp"]
        rp_rel = rp_rel[(rp_rel >= 0) & (rp_rel < len(seg))]
        if len(rp_rel):
            ax.scatter(rp_rel / fs, seg[rp_rel],
                       color="red", s=20, zorder=5, linewidths=0)

        # 心率标注
        hr_str = ""
        if len(rp_rel) > 1:
            mean_hr = 60 / np.mean(np.diff(rp_rel) / fs)
            hr_str  = f"  {mean_hr:.0f} bpm"

        ax.set_title(
            f"seg{idx:03d}  {w['start_s']:.0f}-{w['end_s']:.0f}s\n"
            f"beats={w['n_beats']}{hr_str}  uc={w['mean_uc']:.3f}",
            fontsize=7.5
        )
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 隐藏多余子图格子
    total_cells = nrows * ncols
    for idx in range(n, total_cells):
        r = idx // ncols
        c = idx  % ncols
        fig.add_subplot(gs[r, c]).set_visible(False)

    fig.suptitle(
        f"Good segments (n={n})  |  low mean(U_E+U_A) + normal beat count  |  red dot = R-peak",
        fontsize=9, y=1.01
    )
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [片段图] {out_path}")


# ──────────────────────────────────────────────
# 可视化 3：mean_uc 分布图（判断 Otsu 阈值是否可信）
# ──────────────────────────────────────────────

def plot_uc_distribution(uc_values: list, uc_thr: float, out_path: str,
                         title: str = ""):
    """
    绘制 mean_uc 直方图 + Otsu 阈值线。
    uc range 跨度大 → 双峰明显 → 阈值可信；跨度小 → 单峰 → 提示手动设置。
    """
    ucs = np.array(uc_values, dtype=float)
    ucs = ucs[np.isfinite(ucs)]   # 过滤 NaN / inf
    n   = len(ucs)
    if n == 0:
        return

    uc_min, uc_max = ucs.min(), ucs.max()
    uc_range = uc_max - uc_min

    # 判断可信度
    bimodal    = uc_range > 1.0      # 跨度 > 1 认为双峰明显
    trust_note = ("Bimodal distribution — threshold reliable"
                  if bimodal else
                  "Unimodal distribution — threshold may be unreliable, consider setting --uc_thr manually")

    # 直方图 bin 数量
    n_bins = min(max(int(n / 2), 20), 80)

    fig, ax = plt.subplots(figsize=(8, 4))

    # 画两段直方图：阈值左边（绿）和右边（红）
    bins = np.linspace(min(uc_min, 0), min(uc_max, uc_thr * 4), n_bins + 1)
    good_mask = ucs <= uc_thr
    ax.hist(ucs[good_mask],  bins=bins, color=COLOR_GOOD, alpha=0.7,
            label=f"good  (n={good_mask.sum()})", edgecolor="none")
    ax.hist(ucs[~good_mask], bins=bins, color=COLOR_BAD,  alpha=0.7,
            label=f"low-quality  (n={(~good_mask).sum()})", edgecolor="none")

    # Otsu 阈值线
    ax.axvline(uc_thr, color="black", lw=1.5, linestyle="--",
               label=f"threshold = {uc_thr:.3f}")

    # 可信度注释
    color_note = COLOR_GOOD if bimodal else COLOR_BAD
    ax.text(0.98, 0.95, trust_note,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=color_note,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_note, alpha=0.8))

    ax.set_xlabel("mean(U_E + U_A) per window", fontsize=9)
    ax.set_ylabel("Window count", fontsize=9)
    ax.set_title(
        f"{title}  |  n={n} windows  |  uc range: {uc_min:.3f} – {uc_max:.3f}",
        fontsize=9
    )
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [分布图] {out_path}  {'✓ bimodal' if bimodal else '⚠ unimodal'}")


# ──────────────────────────────────────────────
# 单文件处理（单模式和批量模式共用）
# ──────────────────────────────────────────────

def _read_csv_robust(path: str) -> pd.DataFrame:
    """pd.read_csv，自动截断末尾多余空列，不丢任何数据行。"""
    import io
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.readlines()
    if not raw:
        return pd.DataFrame()
    ncols = len(raw[0].split(","))
    fixed = []
    for line in raw:
        fields = line.rstrip("\n").split(",")
        if len(fields) != ncols:
            fields = fields[:ncols]
        fixed.append(",".join(fields))
    return pd.read_csv(io.StringIO("\n".join(fixed)))


def load_signal(csv_path: str):
    """读取文件，返回 (signal, df) 或 (None, None) 若无 CH20 列。"""
    if csv_path.endswith(".csv"):
        df = _read_csv_robust(csv_path)
    else:
        df = pd.read_excel(csv_path)
    upper_col = next((c for c in df.columns if str(c).upper() == "CH20"), None)
    if upper_col is None:
        return None, None
    return df[upper_col].values.astype(np.float32), df


def process_one_file(csv_path: str, fs: int, model, device,
                     uc_thr: float, out_dir: str = None,
                     precomputed_windows: list = None,
                     infer_batch: int = 16,
                     step_sec: float = STEP_SEC_DEF):
    """
    处理单个文件：推理（或复用预计算窗口） → 阈值 → 保存 → 可视化。
    precomputed_windows: 批量 auto 模式下已推理好的 windows，直接复用。
    返回统计 dict 或 None（跳过）。
    """
    signal, df = load_signal(csv_path)
    if signal is None:
        print(f"  [跳过] 未找到 CH20 列：{csv_path}")
        return None

    base_name  = os.path.basename(csv_path).replace(".csv", "").replace(".xlsx", "")
    file_dir   = os.path.dirname(os.path.abspath(csv_path))
    seg_dir    = out_dir or os.path.join(file_dir, "quality_segments")
    duration_s = len(signal) / fs

    if duration_s < WIN_SEC:
        print(f"  [跳过] 时长 {duration_s:.1f}s < {WIN_SEC}s（最小窗口长度）：{os.path.basename(csv_path)}")
        qr_path = os.path.join(file_dir, base_name + "_quality_report.csv")
        _REPORT_COLS = ["start_samp", "end_samp", "start_s", "end_s",
                        "n_beats", "mean_uc", "mean_ue", "mean_ua", "is_good"]
        pd.DataFrame(columns=_REPORT_COLS).to_csv(qr_path, index=False)
        return None

    print(f"\n>> {csv_path}")
    print(f"   fs={fs}Hz  duration={duration_s:.1f}s  uc_thr={uc_thr:.3f}")

    if precomputed_windows is not None:
        windows = apply_threshold(precomputed_windows, uc_thr)
    else:
        windows = assess_quality(signal, fs, model, device, uc_thr, infer_batch, step_sec)
    n_good       = sum(w["is_good"] for w in windows)
    n_total      = len(windows)
    good_windows = save_segments(signal, windows, fs, seg_dir, base_name, df=df)

    # 统计指标
    good_ucs   = [w["mean_uc"] for w in windows if w["is_good"]]
    all_ucs    = [w["mean_uc"] for w in windows]
    good_beats = [w["n_beats"] for w in windows if w["is_good"]]
    mean_uc_good = float(np.mean(good_ucs))  if good_ucs  else float("nan")
    mean_uc_all  = float(np.mean(all_ucs))   if all_ucs   else float("nan")
    mean_beats   = float(np.mean(good_beats)) if good_beats else float("nan")
    good_ratio   = n_good / max(n_total, 1) * 100

    pd.DataFrame([
        {k: v for k, v in w.items() if k != "r_peaks_abs"}
        for w in windows
    ]).to_csv(os.path.join(file_dir, base_name + "_quality_report.csv"), index=False)

    plot_overview(
        signal, windows, fs, uc_thr,
        os.path.join(file_dir, base_name + "_quality_overview.png")
    )
    plot_good_segments(
        signal, good_windows, fs,
        os.path.join(file_dir, base_name + "_quality_segments.png")
    )
    plot_uc_distribution(
        all_ucs, uc_thr,
        os.path.join(file_dir, base_name + "_uc_distribution.png"),
        title=base_name
    )

    print(f"   good={n_good}/{n_total} ({good_ratio:.1f}%)  "
          f"mean_uc(good)={mean_uc_good:.3f}  NPZ → {seg_dir}")

    return dict(
        file          = csv_path,
        duration_s    = round(duration_s, 1),
        n_windows     = n_total,
        n_good        = n_good,
        n_bad         = n_total - n_good,
        good_ratio_pct= round(good_ratio, 1),
        mean_uc_good  = round(mean_uc_good, 4) if not np.isnan(mean_uc_good) else None,
        mean_uc_all   = round(mean_uc_all,  4) if not np.isnan(mean_uc_all)  else None,
        mean_beats_good = round(mean_beats, 1) if not np.isnan(mean_beats)   else None,
    )


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────
# 多卡 worker（顶层函数，spawn 进程可序列化）
# ──────────────────────────────────────────────

def _worker_infer(gpu_id, files, fs, infer_batch, step_sec):
    """Pass-1 worker：只做推理，不应用阈值，返回 {csv_path: windows}。"""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[GPU{gpu_id}] 加载模型…", flush=True)
    model = load_model(device)
    print(f"[GPU{gpu_id}] 模型就绪，处理 {len(files)} 个文件", flush=True)
    result = {}
    n = len(files)
    for i, csv_path in enumerate(files, 1):
        signal, _ = load_signal(csv_path)
        if signal is None:
            print(f"[GPU{gpu_id}] [{i}/{n}] [skip] 无 CH20: {os.path.basename(csv_path)}", flush=True)
            continue
        if len(signal) / fs < WIN_SEC:
            print(f"[GPU{gpu_id}] [{i}/{n}] [skip] 时长 {len(signal)/fs:.1f}s < {WIN_SEC}s: {os.path.basename(csv_path)}", flush=True)
            continue
        print(f"[GPU{gpu_id}] [{i}/{n}] 推理: {os.path.basename(csv_path)}", flush=True)
        result[csv_path] = run_inference(signal, fs, model, device, infer_batch, step_sec)
    return result


def _worker_process(gpu_id, file_windows, fs, uc_thr, out_dir, infer_batch, step_sec):
    """Pass-2 worker：应用阈值 + 保存，返回 [(csv_path, stat), ...]。"""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)
    stats  = []
    items  = list(file_windows.items())
    n = len(items)
    for i, (csv_path, windows) in enumerate(items, 1):
        print(f"[GPU{gpu_id}] [{i}/{n}] {os.path.basename(csv_path)}", flush=True)
        stat = process_one_file(
            csv_path, fs, model, device, uc_thr, out_dir,
            precomputed_windows=windows, infer_batch=infer_batch, step_sec=step_sec
        )
        if stat:
            stats.append((csv_path, stat))
    return stats


def _worker_fixed(gpu_id, files, fs, uc_thr, out_dir, infer_batch, step_sec):
    """固定阈值 worker：单次推理 + 保存，返回 [(csv_path, stat), ...]。"""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[GPU{gpu_id}] 加载模型…", flush=True)
    model  = load_model(device)
    print(f"[GPU{gpu_id}] 模型就绪，处理 {len(files)} 个文件", flush=True)
    stats  = []
    n = len(files)
    for i, csv_path in enumerate(files, 1):
        print(f"[GPU{gpu_id}] [{i}/{n}] {os.path.basename(csv_path)}", flush=True)
        stat = process_one_file(
            csv_path, fs, model, device, uc_thr, out_dir,
            infer_batch=infer_batch, step_sec=step_sec
        )
        if stat:
            stats.append((csv_path, stat))
    return stats


def main():
    ap = argparse.ArgumentParser(
        description="基于 PN-QRS 不确定性提取 CH20 高质量 10 秒片段"
    )

    # 模式互斥：单文件 vs 批量
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--csv",      help="单文件模式：原始 ECG CSV/Excel 文件路径")
    mode.add_argument("--data_dir", help="批量模式（需同时加 --batch）：递归扫描该目录下所有 CSV/Excel")

    ap.add_argument("--batch",   action="store_true",
                    help="开启批量模式，递归处理 --data_dir 下所有 CSV/Excel 文件")
    ap.add_argument("--fs",      required=True, type=int,    help="采样率 Hz")
    ap.add_argument("--uc_thr",  default=str(UC_THR_DEF),
                    help=f"不确定性阈值，默认 {UC_THR_DEF}；填 'auto' 自动用 Otsu 方法决定")
    ap.add_argument("--out_dir", default=None,
                    help="NPZ 保存目录（单文件模式默认：CSV 同目录/quality_segments/；"
                         "批量模式默认：各 CSV 同目录下各自建 quality_segments/）")
    ap.add_argument("--gpu",         default="0")
    ap.add_argument("--infer_batch", default=16, type=int,
                    help="每次 GPU forward 的窗口数，越大越快（默认 16）")
    ap.add_argument("--step",        default=STEP_SEC_DEF, type=float,
                    help=f"滑动窗口步长（秒），默认 {STEP_SEC_DEF}s；"
                         f"嘈杂数据可设 1~2s 以发现更多有效片段")
    args = ap.parse_args()

    # 参数校验
    if args.batch and args.data_dir is None:
        ap.error("--batch 需要同时指定 --data_dir")
    if not args.batch and args.csv is None:
        ap.error("单文件模式需要指定 --csv")

    gpu_ids = [int(g.strip()) for g in str(args.gpu).split(",")]
    n_gpus  = len(gpu_ids)
    gpu_id  = gpu_ids[0]
    device  = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"加载模型... 设备: {device} (共 {n_gpus} 张 GPU: {gpu_ids})", flush=True)
    model = load_model(device)

    # 解析 uc_thr（float 或 'auto'）
    auto_thr = (args.uc_thr.strip().lower() == "auto")
    uc_thr   = None if auto_thr else float(args.uc_thr)

    # ── 单文件模式 ──────────────────────────────
    if not args.batch:
        if auto_thr:
            signal, _ = load_signal(args.csv)
            if signal is None:
                print("未找到 CH20 列")
                return
            print("Auto threshold: running inference...")
            windows  = run_inference(signal, args.fs, model, device,
                                     args.infer_batch, args.step)
            uc_vals  = [w["mean_uc"] for w in windows]
            uc_thr   = otsu_threshold(uc_vals)
            print(f"Otsu threshold = {uc_thr:.4f}  "
                  f"(uc range: {min(uc_vals):.3f} – {max(uc_vals):.3f})")
            stat = process_one_file(
                args.csv, args.fs, model, device, uc_thr, args.out_dir,
                precomputed_windows=windows, infer_batch=args.infer_batch,
                step_sec=args.step
            )
        else:
            stat = process_one_file(
                args.csv, args.fs, model, device, uc_thr, args.out_dir,
                infer_batch=args.infer_batch, step_sec=args.step
            )
        if stat:
            print(f"\nDone. good={stat['n_good']}/{stat['n_windows']} "
                  f"({stat['good_ratio_pct']}%)  uc_thr={uc_thr:.4f}")
        return

    # ── 批量模式 ────────────────────────────────
    import glob as _glob
    patterns = ["**/*.csv", "**/*.xlsx"]
    all_files = []
    for pat in patterns:
        all_files += _glob.glob(os.path.join(args.data_dir, pat), recursive=True)

    # 过滤掉脚本自身产生的输出文件
    skip_suffixes = (
        "_quality_report.csv", "_CH1-8_rpeaks.csv", "_CH20_rpeaks.csv",
        "_quality_overview.png", "_quality_segments.png",
    )
    all_files = sorted(
        f for f in all_files
        if not any(f.endswith(s) for s in skip_suffixes)
    )

    if not all_files:
        print(f"No CSV/Excel files found under {args.data_dir}")
        return

    print(f"\nFound {len(all_files)} files (recursive scan: {args.data_dir})")
    print(f"uc_thr={args.uc_thr}  fs={args.fs}Hz\n{'─'*60}")

    data_dir_abs = os.path.abspath(args.data_dir)
    stats = []

    def _attach_meta(csv_path, stat):
        rel = os.path.relpath(csv_path, data_dir_abs)
        stat["activity"] = Path(rel).parts[0] if len(Path(rel).parts) > 1 else "(root)"
        stat["rel_path"] = rel
        return stat

    use_multi = n_gpus > 1 and torch.cuda.is_available()

    if auto_thr:
        # ── 批量 auto 模式：Pass-1 推理 → Otsu → Pass-2 保存 ─────────────────
        n_files = len(all_files)
        print(f"Auto threshold (batch): pass 1 — inferring {n_files} files on {n_gpus} GPU(s)...")

        if use_multi:
            chunks = [all_files[i::n_gpus] for i in range(n_gpus)]
            ctx    = mp.get_context("spawn")
            p1_args = [(gid, ch, args.fs, args.infer_batch, args.step)
                       for gid, ch in zip(gpu_ids, chunks) if ch]
            with ctx.Pool(len(p1_args)) as pool:
                p1_results = pool.starmap(_worker_infer, p1_args)
            all_file_data = {}
            for r in p1_results:
                all_file_data.update(r)
        else:
            all_file_data = {}
            for idx, csv_path in enumerate(all_files, 1):
                signal, _ = load_signal(csv_path)
                if signal is None:
                    print(f"  [{idx}/{n_files}] [skip] 无 CH20: {csv_path}", flush=True)
                    continue
                print(f"  [{idx}/{n_files}] 推理: {os.path.relpath(csv_path, data_dir_abs)}", flush=True)
                all_file_data[csv_path] = run_inference(
                    signal, args.fs, model, device, args.infer_batch, args.step)

        all_ucs = [w["mean_uc"] for wins in all_file_data.values() for w in wins]
        if not all_ucs:
            print("No valid windows found.")
            return

        uc_thr = otsu_threshold(all_ucs)
        print(f"\nOtsu threshold (pooled, {len(all_ucs)} windows) = {uc_thr:.4f}  "
              f"(range: {min(all_ucs):.3f} – {max(all_ucs):.3f})")
        plot_uc_distribution(
            all_ucs, uc_thr,
            os.path.join(args.data_dir, "batch_uc_distribution.png"),
            title=f"Pooled distribution — {len(all_file_data)} files"
        )
        print(f"pass 2 — applying threshold and generating outputs ({n_gpus} GPU(s))…\n{'─'*60}")

        if use_multi:
            file_list = list(all_file_data.items())
            chunks_p2 = [{} for _ in range(n_gpus)]
            for i, (path, wins) in enumerate(file_list):
                chunks_p2[i % n_gpus][path] = wins
            ctx = mp.get_context("spawn")
            p2_args = [(gid, ch, args.fs, uc_thr, args.out_dir, args.infer_batch, args.step)
                       for gid, ch in zip(gpu_ids, chunks_p2) if ch]
            with ctx.Pool(len(p2_args)) as pool:
                p2_results = pool.starmap(_worker_process, p2_args)
            for gpu_stats in p2_results:
                for csv_path, stat in gpu_stats:
                    stats.append(_attach_meta(csv_path, stat))
        else:
            items = list(all_file_data.items())
            for idx2, (csv_path, windows) in enumerate(items, 1):
                print(f"  [{idx2}/{len(items)}] {os.path.relpath(csv_path, data_dir_abs)}", flush=True)
                stat = process_one_file(
                    csv_path, args.fs, model, device, uc_thr, args.out_dir,
                    precomputed_windows=windows, infer_batch=args.infer_batch,
                    step_sec=args.step
                )
                if stat:
                    stats.append(_attach_meta(csv_path, stat))
    else:
        # ── 批量固定阈值模式 ───────────────────────────────────────────────────
        n_files = len(all_files)
        if use_multi:
            chunks = [all_files[i::n_gpus] for i in range(n_gpus)]
            ctx    = mp.get_context("spawn")
            fx_args = [(gid, ch, args.fs, uc_thr, args.out_dir, args.infer_batch, args.step)
                       for gid, ch in zip(gpu_ids, chunks) if ch]
            with ctx.Pool(len(fx_args)) as pool:
                fx_results = pool.starmap(_worker_fixed, fx_args)
            for gpu_stats in fx_results:
                for csv_path, stat in gpu_stats:
                    stats.append(_attach_meta(csv_path, stat))
        else:
            for idx, csv_path in enumerate(all_files, 1):
                print(f"[{idx}/{n_files}] {os.path.relpath(csv_path, data_dir_abs)}", flush=True)
                stat = process_one_file(
                    csv_path, args.fs, model, device, uc_thr, args.out_dir,
                    infer_batch=args.infer_batch, step_sec=args.step
                )
                if stat:
                    stats.append(_attach_meta(csv_path, stat))

    if not stats:
        print("No files processed.")
        return

    # ── 批量汇总报告 CSV ────────────────────────
    summary_path = os.path.join(args.data_dir, "batch_quality_summary.csv")
    col_order = ["activity", "rel_path", "duration_s", "n_windows",
                 "n_good", "n_bad", "good_ratio_pct",
                 "mean_uc_good", "mean_uc_all", "mean_beats_good"]
    pd.DataFrame(stats)[col_order].to_csv(summary_path, index=False)

    # ── 按行为分组汇总打印 ──────────────────────
    from collections import defaultdict
    groups = defaultdict(list)
    for s in stats:
        groups[s["activity"]].append(s)

    def _group_row(label, rows):
        gw = sum(r["n_windows"] for r in rows)
        gg = sum(r["n_good"]    for r in rows)
        gd = sum(r["duration_s"] for r in rows)
        gr = gg / max(gw, 1) * 100
        uc_vals = [r["mean_uc_good"] for r in rows if r["mean_uc_good"] is not None]
        uc_str  = f"{np.mean(uc_vals):.3f}" if uc_vals else "  n/a"
        return label, gd, gw, gg, gr, uc_str

    file_w = max(len(s["rel_path"]) for s in stats)
    ACT_W  = max(max(len(s["activity"]) for s in stats), 8)
    HDR = (f"  {'file':<{file_w}}  {'dur(s)':>7}  {'windows':>7}  "
           f"{'good':>5}  {'ratio%':>7}  {'uc_good':>8}  {'beats':>6}")
    SEP = "─" * (len(HDR) + ACT_W + 2)

    print(f"\n{SEP}")
    print(f"{'activity':<{ACT_W}}{HDR}")
    print(SEP)

    act_totals = []
    for activity in sorted(groups):
        rows = groups[activity]
        # 行为小计行
        lbl, gd, gw, gg, gr, uc_str = _group_row(activity, rows)
        print(f"\033[1m{lbl:<{ACT_W}}"
              f"  {'':>{file_w}}  {gd:>7.1f}  {gw:>7}  "
              f"{gg:>5}  {gr:>6.1f}%  {uc_str:>8}\033[0m")
        # 每文件明细行（缩进）
        for s in rows:
            uc_g  = f"{s['mean_uc_good']:.3f}"   if s["mean_uc_good"]    is not None else "   n/a"
            beats = f"{s['mean_beats_good']:.1f}" if s["mean_beats_good"] is not None else "  n/a"
            fname = os.path.basename(s["file"])
            indent_file = f"  └ {fname}"
            print(f"{'':>{ACT_W}}"
                  f"  {indent_file:<{file_w}}  {s['duration_s']:>7.1f}  {s['n_windows']:>7}  "
                  f"{s['n_good']:>5}  {s['good_ratio_pct']:>6.1f}%  {uc_g:>8}  {beats:>6}")
        act_totals.append((activity, gd, gw, gg, gr))

    # 总计行
    total_windows = sum(s["n_windows"]  for s in stats)
    total_good    = sum(s["n_good"]     for s in stats)
    total_dur_s   = sum(s["duration_s"] for s in stats)
    overall_ratio = total_good / max(total_windows, 1) * 100
    print(SEP)
    print(f"\033[1m{'TOTAL':<{ACT_W}}"
          f"  {'':>{file_w}}  {total_dur_s:>7.1f}  {total_windows:>7}  "
          f"{total_good:>5}  {overall_ratio:>6.1f}%\033[0m")
    print(f"\nBatch summary saved → {summary_path}")


if __name__ == "__main__":
    main()
