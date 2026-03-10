import math
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

try:
    from scipy.signal import butter, sosfiltfilt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# =========================
# 基础工具
# =========================

def me_std_mae(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = pred - true
    me = float(err.mean())
    std = float(err.std(ddof=1)) if len(err) > 1 else np.nan
    mae = float(np.abs(err).mean())
    return me, std, mae


def fmt(name, triple):
    me, std, mae = triple
    return f"{name}: ME={me:+.3f}, STD={std:.3f}, MAE={mae:.3f}"


@lru_cache(maxsize=256)
def load_ppg_memmap(ppg_path: str):
    return np.load(ppg_path, mmap_mode="r")


@lru_cache(maxsize=256)
def load_time_memmap(time_path: str):
    return np.load(time_path, mmap_mode="r")


def get_ppg1d(arr: np.ndarray, channel_idx: int = 2):
    """
    支持:
      [N]
      [N, C]
      [C, N]
    """
    if arr.ndim == 1:
        return arr

    if arr.ndim != 2:
        raise ValueError(f"ppg array must be 1D or 2D, got shape={arr.shape}")

    # [N, C]
    if arr.shape[0] >= 1000 and arr.shape[1] > channel_idx:
        return arr[:, channel_idx]

    # [C, N]
    if arr.shape[1] >= 1000 and arr.shape[0] > channel_idx:
        return arr[channel_idx, :]

    # fallback
    if arr.shape[1] > channel_idx:
        return arr[:, channel_idx]
    return arr[channel_idx, :]


def build_bandpass(fs=100, lo=0.5, hi=8.0, order=3):
    if not _HAS_SCIPY:
        return None
    nyq = 0.5 * fs
    lo_n = max(1e-6, lo / nyq)
    hi_n = min(0.999999, hi / nyq)
    if hi_n <= lo_n:
        return None
    return butter(order, [lo_n, hi_n], btype="band", output="sos")


def apply_bandpass_1d(x: np.ndarray, sos):
    if sos is None:
        return x.astype(np.float32, copy=False)
    x = np.asarray(x, dtype=np.float64)
    if x.size < 16:
        return x.astype(np.float32, copy=False)
    padlen = min(21, x.size - 1)
    y = sosfiltfilt(sos, x, padlen=padlen)
    return y.astype(np.float32, copy=False)


def event_minmax_norm(x: np.ndarray, p_lo=1.0, p_hi=99.0, eps=1e-6):
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    scale = max(hi - lo, eps)
    y = (x - lo) / scale
    y = np.clip(y, 0.0, 1.0)
    return y.astype(np.float32, copy=False)


def robust_hour_agg(arr: np.ndarray, mode="median", trim_ratio=0.1):
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return np.nan
    if mode == "mean":
        return float(arr.mean())
    if mode == "median":
        return float(np.median(arr))
    if mode == "trimmed_mean":
        n = len(arr)
        k = int(math.floor(n * trim_ratio))
        if 2 * k >= n:
            return float(arr.mean())
        arr = np.sort(arr)
        return float(arr[k:n-k].mean())
    raise ValueError(f"unknown agg mode={mode}")


# =========================
# 自然小时内 120s 窗口定义
# =========================

def make_hour_window_starts(
    hour_len_sec=3600,
    win_sec=120,
    mode="half_hour_sampled",
):
    """
    返回相对 hour_start 的窗口起点（单位：秒）

    mode:
      - half_hour_sampled: 从 30 个 non-overlap 120s 窗口中均匀取 15 个（总计约半小时数据）
      - all_nonoverlap:    取全部 30 个 non-overlap 120s 窗口
    """
    starts = list(range(0, hour_len_sec - win_sec + 1, win_sec))  # 0,120,...,3480 共30个

    if mode == "all_nonoverlap":
        return starts

    if mode == "half_hour_sampled":
        idx = np.linspace(0, len(starts) - 1, num=15)
        idx = np.round(idx).astype(int)
        idx = np.clip(idx, 0, len(starts) - 1)

        chosen = []
        for i in idx.tolist():
            s = starts[i]
            if s not in chosen:
                chosen.append(s)

        for s in starts:
            if len(chosen) >= 15:
                break
            if s not in chosen:
                chosen.append(s)

        return chosen

    raise ValueError(f"unknown mode={mode}")


# =========================
# 一个 120s 窗口 -> [K,1,L]
# =========================

def build_model_input_from_120s_window(
    sig_120s: np.ndarray,
    fs=100,
    seg_sec=8,
    K=8,
    sample_mode="uniform",   # uniform / random / dense_nonoverlap
    base_seed=1234,
    bandpass_sos=None,
    normalize="event_minmax",
):
    """
    输入:
      sig_120s: [T]
    输出:
      x: [K,1,L]
    """
    x = sig_120s.astype(np.float32, copy=False)

    # bandpass on whole 120s window
    x = apply_bandpass_1d(x, bandpass_sos)

    # normalize on whole 120s window
    if normalize == "event_minmax":
        x = event_minmax_norm(x)
    elif normalize == "none":
        x = x.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unknown normalize={normalize}")

    L = fs * seg_sec
    T = len(x)

    if T < L:
        x = np.pad(x, (0, L - T), mode="edge")
        T = len(x)

    max_start = max(0, T - L)

    if sample_mode == "uniform":
        if K == 1:
            starts = [max_start // 2]
        else:
            starts = np.linspace(0, max_start, num=K).astype(int).tolist()

    elif sample_mode == "random":
        rng = np.random.RandomState(base_seed)
        starts = rng.randint(0, max_start + 1, size=K).tolist()

    elif sample_mode == "dense_nonoverlap":
        starts = list(range(0, max_start + 1, L))
        if len(starts) >= K:
            starts = starts[:K]
        else:
            starts = starts + [starts[-1]] * (K - len(starts))

    else:
        raise ValueError(f"unknown sample_mode={sample_mode}")

    segs = []
    for s in starts:
        seg = x[s:s+L]  # [L]
        segs.append(seg[None, :])  # [1,L]

    arr = np.stack(segs, axis=0).astype(np.float32, copy=False)  # [K,1,L]
    return arr


@torch.no_grad()
def predict_bp_for_120s_windows(
    model,
    window_list,
    device="cuda",
    fs=100,
    seg_sec=8,
    K=8,
    sample_mode="uniform",
    base_seed=1234,
    bandpass_sos=None,
    normalize="event_minmax",
    batch_size=16,
):
    xs = []
    for i, w in enumerate(window_list):
        x = build_model_input_from_120s_window(
            w,
            fs=fs,
            seg_sec=seg_sec,
            K=K,
            sample_mode=sample_mode,
            base_seed=base_seed + i,
            bandpass_sos=bandpass_sos,
            normalize=normalize,
        )
        xs.append(x)

    X = np.stack(xs, axis=0)  # [Nw,K,1,L]
    X = torch.from_numpy(X).to(device, non_blocking=True)

    model.eval()
    preds = []
    for st in range(0, X.shape[0], batch_size):
        ed = min(st + batch_size, X.shape[0])
        pred = model(X[st:ed])   # [B,2]
        preds.append(pred.detach().cpu().numpy())

    P = np.concatenate(preds, axis=0)
    return P[:, 0], P[:, 1]


# =========================
# 单个自然小时推理
# =========================

def infer_one_hour(
    model,
    ppg_path: str,
    time_path: str,
    hour_start_ms: int,
    hour_end_ms: int,
    mode="half_hour_sampled",
    channel_idx=2,
    fs=100,
    seg_sec=8,
    K=8,
    sample_mode="uniform",
    base_seed=1234,
    bandpass_sos=None,
    normalize="event_minmax",
    hour_agg="median",
    min_window_coverage=0.80,
):
    arr = load_ppg_memmap(ppg_path)
    sig_all = get_ppg1d(arr, channel_idx=channel_idx)
    t_arr = load_time_memmap(time_path)

    i0 = int(np.searchsorted(t_arr, hour_start_ms, side="left"))
    i1 = int(np.searchsorted(t_arr, hour_end_ms, side="left"))

    sig_hour = np.asarray(sig_all[i0:i1], dtype=np.float32)
    t_hour = np.asarray(t_arr[i0:i1], dtype=np.int64)

    if len(sig_hour) < fs * 120:
        return np.nan, np.nan, 0

    hour_len_sec = int(round((hour_end_ms - hour_start_ms) / 1000.0))
    if hour_len_sec <= 0:
        return np.nan, np.nan, 0

    rel_starts_sec = make_hour_window_starts(
        hour_len_sec=hour_len_sec,
        win_sec=120,
        mode=mode,
    )

    windows = []
    for s_sec in rel_starts_sec:
        st_ms = hour_start_ms + int(s_sec * 1000)
        ed_ms = st_ms + 120 * 1000

        # 不允许超出该自然小时范围
        if ed_ms > hour_end_ms:
            continue

        j0 = int(np.searchsorted(t_hour, st_ms, side="left"))
        j1 = int(np.searchsorted(t_hour, ed_ms, side="left"))

        need = fs * 120
        got = j1 - j0
        cov = got / max(1, need)

        if cov < min_window_coverage:
            continue

        w = sig_hour[j0:j1]
        windows.append(w)

    if len(windows) == 0:
        return np.nan, np.nan, 0

    sbp_w, dbp_w = predict_bp_for_120s_windows(
        model=model,
        window_list=windows,
        device=next(model.parameters()).device,
        fs=fs,
        seg_sec=seg_sec,
        K=K,
        sample_mode=sample_mode,
        base_seed=base_seed,
        bandpass_sos=bandpass_sos,
        normalize=normalize,
        batch_size=16,
    )

    sbp_h = robust_hour_agg(sbp_w, mode=hour_agg)
    dbp_h = robust_hour_agg(dbp_w, mode=hour_agg)

    return sbp_h, dbp_h, len(windows)


# =========================
# 主函数：hourly_gt.csv -> 小时级预测与评估
# =========================

def run_hourly_multiwindow_eval(
    model,
    hourly_gt_csv,
    out_csv,
    mode="half_hour_sampled",       # half_hour_sampled / all_nonoverlap
    channel_idx=2,
    fs=100,
    seg_sec=8,
    K=8,
    sample_mode="uniform",
    base_seed=1234,
    bandpass=True,
    bandpass_lo=0.5,
    bandpass_hi=8.0,
    bandpass_order=3,
    normalize="event_minmax",
    hour_agg="median",
    min_window_coverage=0.80,
):
    """
    直接对 hourly_gt.csv 评估。
    hourly_gt.csv 必须包含：
      id_clean, hour_start_ms, hour_end_ms, y_true_sbp_h, y_true_dbp_h, ppg_path, time_path
    """
    df = pd.read_csv(hourly_gt_csv)

    required = [
        "id_clean", "hour_start_ms", "hour_end_ms",
        "y_true_sbp_h", "y_true_dbp_h",
        "ppg_path", "time_path"
    ]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"hourly_gt.csv missing columns: {miss}")

    df["hour_start_ms"] = df["hour_start_ms"].astype(np.int64)
    df["hour_end_ms"] = df["hour_end_ms"].astype(np.int64)

    sos = build_bandpass(fs=fs, lo=bandpass_lo, hi=bandpass_hi, order=bandpass_order) if bandpass else None

    rows = []
    for _, r in df.iterrows():
        sbp_h, dbp_h, n_valid = infer_one_hour(
            model=model,
            ppg_path=str(r["ppg_path"]),
            time_path=str(r["time_path"]),
            hour_start_ms=int(r["hour_start_ms"]),
            hour_end_ms=int(r["hour_end_ms"]),
            mode=mode,
            channel_idx=channel_idx,
            fs=fs,
            seg_sec=seg_sec,
            K=K,
            sample_mode=sample_mode,
            base_seed=base_seed,
            bandpass_sos=sos,
            normalize=normalize,
            hour_agg=hour_agg,
            min_window_coverage=min_window_coverage,
        )

        rows.append({
            "id_clean": r["id_clean"],
            "hour_start_ms": int(r["hour_start_ms"]),
            "hour_end_ms": int(r["hour_end_ms"]),
            "y_true_sbp_h": float(r["y_true_sbp_h"]),
            "y_true_dbp_h": float(r["y_true_dbp_h"]),
            "y_pred_sbp_h": float(sbp_h) if np.isfinite(sbp_h) else np.nan,
            "y_pred_dbp_h": float(dbp_h) if np.isfinite(dbp_h) else np.nan,
            "n_valid_windows": int(n_valid),
            "exp_mode": mode,
            "hour_agg": hour_agg,
        })

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["y_pred_sbp_h", "y_pred_dbp_h"]).reset_index(drop=True)

    sbp = me_std_mae(out["y_pred_sbp_h"], out["y_true_sbp_h"])
    dbp = me_std_mae(out["y_pred_dbp_h"], out["y_true_dbp_h"])

    print(f"\n===== {mode} | agg={hour_agg} =====")
    print(fmt("SBP", sbp))
    print(fmt("DBP", dbp))
    print(f"Kept hours: {len(out)}")
    print("n_valid_windows desc:")
    print(out["n_valid_windows"].describe())

    out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    return out, sbp, dbp