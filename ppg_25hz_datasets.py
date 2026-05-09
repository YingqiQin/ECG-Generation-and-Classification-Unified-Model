# ppg_25hz_datasets.py
# -*- coding: utf-8 -*-
"""
25Hz-compatible PPG datasets for downstream BP training and SSL pretraining.

Correct workflow:
    1) Slice using original 100Hz indices.
    2) Downsample the sliced window/segment to 25Hz.
    3) Build 8s segments at target_fs=25 => L=200.
    4) Train/evaluate the model using [B, K, 1, 200].
"""

from __future__ import annotations

import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

try:
    from scipy.signal import butter, sosfiltfilt, resample_poly
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _resolve_ppg_path_from_prefix(prefix: str) -> str:
    p = Path(prefix)
    pA = str(p) + "_ppg.npy"
    if Path(pA).exists():
        return pA
    pB = str(p / "ppg.npy")
    if Path(pB).exists():
        return pB
    return pA


def _event_percentile_minmax(win: np.ndarray, p_lo=1.0, p_hi=99.0, eps=1e-6):
    lo = float(np.percentile(win, p_lo))
    hi = float(np.percentile(win, p_hi))
    scale = max(hi - lo, eps)
    return lo, scale


def _make_bandpass_sos(fs: int, lo: float, hi: float, order: int):
    if not _HAS_SCIPY:
        return None
    nyq = 0.5 * float(fs)
    lo_n = max(1e-6, float(lo) / nyq)
    hi_n = min(0.999999, float(hi) / nyq)
    if hi_n <= lo_n:
        return None
    return butter(int(order), [lo_n, hi_n], btype="band", output="sos")


def _safe_sosfiltfilt(x: np.ndarray, sos, order: int) -> np.ndarray:
    if sos is None:
        return x.astype(np.float32, copy=False)
    T = int(len(x))
    if T <= (3 * (2 * int(order) + 1) + 1):
        return x.astype(np.float32, copy=False)
    padlen = min(3 * (2 * int(order) + 1), T - 1)
    return sosfiltfilt(sos, x.astype(np.float64, copy=False), padlen=padlen).astype(np.float32)


def _slice_channel(arr: np.ndarray, start: int, end: int, channel: int) -> np.ndarray:
    if arr.ndim == 1:
        return np.asarray(arr[start:end], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"PPG array must be 1D/2D, got shape={arr.shape}")
    if arr.shape[0] >= 1000 and arr.shape[1] > channel:   # [N,C]
        return np.asarray(arr[start:end, channel], dtype=np.float32)
    if arr.shape[1] >= 1000 and arr.shape[0] > channel:   # [C,N]
        return np.asarray(arr[channel, start:end], dtype=np.float32)
    if arr.shape[1] > channel:
        return np.asarray(arr[start:end, channel], dtype=np.float32)
    return np.asarray(arr[channel, start:end], dtype=np.float32)


def downsample_1d(
    x: np.ndarray,
    source_fs: int = 100,
    target_fs: int = 25,
    mode: str = "stride",
) -> np.ndarray:
    """
    mode:
      - stride: direct extraction x[::factor], matching "直接抽取".
      - polyphase: resample_poly with anti-aliasing.
      - none: no downsampling.
    """
    x = np.asarray(x, dtype=np.float32)
    source_fs = int(source_fs)
    target_fs = int(target_fs)

    if mode == "none" or source_fs == target_fs:
        return x.astype(np.float32, copy=False)

    if source_fs % target_fs != 0:
        if mode == "stride":
            raise ValueError(f"stride requires source_fs % target_fs == 0, got {source_fs}/{target_fs}")
        if not _HAS_SCIPY:
            raise RuntimeError("Non-integer polyphase resampling requires scipy.")
        from math import gcd
        g = gcd(source_fs, target_fs)
        return resample_poly(x, up=target_fs // g, down=source_fs // g).astype(np.float32)

    factor = source_fs // target_fs
    if mode == "stride":
        return x[::factor].astype(np.float32, copy=False)
    if mode == "polyphase":
        if not _HAS_SCIPY:
            raise RuntimeError("polyphase downsampling requires scipy.")
        return resample_poly(x, up=1, down=factor).astype(np.float32)

    raise ValueError(f"Unknown downsample mode={mode}. Use stride/polyphase/none.")


class EventPPG3KSegmentsDatasetResample(Dataset):
    """
    Supervised event-level BP dataset.

    i0/i1 are original source_fs indices, usually 100Hz.
    Output x has target_fs time scale.

    Typical 25Hz setting:
        source_fs=100, target_fs=25, seg_sec=8, K=8
        => x: [K, 1, 200]
    """
    def __init__(
        self,
        csv_path: str,
        source_fs: int = 100,
        target_fs: int = 25,
        seg_sec: int = 8,
        K: int = 8,
        channel_idx: int = 2,
        deterministic: bool = False,
        base_seed: int = 1234,
        sample_mode: str = "random",
        downsample_mode: str = "stride",
        bandpass: bool = True,
        bp_lo: float = 0.5,
        bp_hi: float = 8.0,
        bp_order: int = 3,
        filter_stage: str = "after_downsample",
        normalize: str = "event_minmax",
        cache_files: int = 4,
        min_len_ratio: float = 0.95,
    ):
        self.df = pd.read_csv(csv_path)
        self.csv_path = csv_path

        req = ["id_clean", "i0", "i1", "sbp", "dbp", "sleep", "t_bp_ms"]
        miss = [c for c in req if c not in self.df.columns]
        if miss:
            raise ValueError(f"Missing columns in {csv_path}: {miss}")
        if ("ppg_path" not in self.df.columns) and ("npy_prefix" not in self.df.columns):
            raise ValueError(f"{csv_path} must contain either 'ppg_path' or 'npy_prefix'.")

        self.source_fs = int(source_fs)
        self.target_fs = int(target_fs)
        self.seg_sec = int(seg_sec)
        self.seg_len = int(self.target_fs * self.seg_sec)
        self.K = int(K)
        self.channel_idx = int(channel_idx)
        self.deterministic = bool(deterministic)
        self.base_seed = int(base_seed)
        self.sample_mode = str(sample_mode)
        self.downsample_mode = str(downsample_mode)
        self.normalize = str(normalize).lower().strip()

        self.bandpass = bool(bandpass) and _HAS_SCIPY
        self.bp_lo = float(bp_lo)
        self.bp_hi = float(bp_hi)
        self.bp_order = int(bp_order)
        self.filter_stage = str(filter_stage).lower().strip()
        if self.filter_stage not in ["before_downsample", "after_downsample", "none"]:
            raise ValueError("filter_stage must be before_downsample/after_downsample/none")

        self._sos_source = None
        self._sos_target = None
        if self.bandpass and self.filter_stage == "before_downsample":
            self._sos_source = _make_bandpass_sos(self.source_fs, self.bp_lo, self.bp_hi, self.bp_order)
        if self.bandpass and self.filter_stage == "after_downsample":
            self._sos_target = _make_bandpass_sos(self.target_fs, self.bp_lo, self.bp_hi, self.bp_order)

        self.cache_files = int(cache_files)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.min_len = int(self.seg_len * float(min_len_ratio))

    def __len__(self):
        return len(self.df)

    def _load_ppg_memmap(self, ppg_path: str) -> np.ndarray:
        if ppg_path in self._cache:
            X = self._cache.pop(ppg_path)
            self._cache[ppg_path] = X
            return X
        if not Path(ppg_path).exists():
            raise FileNotFoundError(ppg_path)
        X = np.load(ppg_path, mmap_mode="r")
        self._cache[ppg_path] = X
        while len(self._cache) > self.cache_files:
            self._cache.popitem(last=False)
        return X

    def _get_ppg_path(self, row) -> str:
        if "ppg_path" in self.df.columns and pd.notna(row["ppg_path"]):
            return str(row["ppg_path"])
        return _resolve_ppg_path_from_prefix(str(row["npy_prefix"]))

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        pid = str(r["id_clean"])
        sleep = int(r["sleep"])
        t_bp_ms = int(r["t_bp_ms"])
        i0 = int(r["i0"])
        i1 = int(r["i1"])
        y = np.array([float(r["sbp"]), float(r["dbp"])], dtype=np.float32)

        if self.deterministic:
            seed = (hash((pid, t_bp_ms, self.base_seed)) & 0xFFFFFFFF)
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        ppg_path = self._get_ppg_path(r)
        X = self._load_ppg_memmap(ppg_path)

        win = _slice_channel(X, i0, i1, self.channel_idx)
        source_T = int(win.shape[0])

        if self.bandpass and self.filter_stage == "before_downsample":
            win = _safe_sosfiltfilt(win, self._sos_source, self.bp_order)

        win = downsample_1d(win, self.source_fs, self.target_fs, mode=self.downsample_mode)
        target_T = int(win.shape[0])

        if self.bandpass and self.filter_stage == "after_downsample":
            win = _safe_sosfiltfilt(win, self._sos_target, self.bp_order)
            target_T = int(win.shape[0])

        if target_T < self.min_len:
            pad = self.seg_len - target_T
            if pad > 0:
                win = np.pad(win, (0, pad), mode="edge").astype(np.float32, copy=False)
                target_T = int(win.shape[0])

        if self.normalize == "event_minmax":
            lo, scale = _event_percentile_minmax(win, p_lo=1.0, p_hi=99.0, eps=1e-6)
        elif self.normalize == "none":
            lo, scale = 0.0, 1.0
        else:
            raise ValueError(f"Unknown normalize={self.normalize}. Use event_minmax/none.")

        L = self.seg_len
        T = target_T
        if T <= L:
            starts = [0] * self.K
        else:
            max_start = T - L
            if self.sample_mode == "dense_nonoverlap":
                starts = list(range(0, max_start + 1, L))
                if len(starts) >= self.K:
                    starts = starts[:self.K]
                else:
                    starts = starts + [starts[-1]] * (self.K - len(starts))
            elif self.sample_mode == "uniform":
                starts = [max_start // 2] if self.K == 1 else np.linspace(0, max_start, num=self.K).astype(int).tolist()
            elif self.sample_mode == "random":
                starts = rng.randint(0, max_start + 1, size=self.K).tolist()
            else:
                raise ValueError(f"Unknown sample_mode={self.sample_mode}")

        segs = []
        for s in starts:
            seg = win[s:s + L]
            if len(seg) < L:
                seg = np.pad(seg, (0, L - len(seg)), mode="edge")
            if self.normalize == "event_minmax":
                seg = (seg - lo) / scale
                seg = np.clip(seg, 0.0, 1.0)
            segs.append(seg[None, :].astype(np.float32, copy=False))

        x = np.stack(segs, axis=0)

        meta = {
            "id_clean": pid,
            "sleep": sleep,
            "t_bp_ms": t_bp_ms,
            "idx": int(idx),
            "ppg_path": ppg_path,
            "source_fs": self.source_fs,
            "target_fs": self.target_fs,
            "source_T": source_T,
            "target_T": target_T,
            "downsample_mode": self.downsample_mode,
            "filter_stage": self.filter_stage,
        }
        return torch.from_numpy(x), torch.from_numpy(y), meta


def collate_ksegments(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), metas


@dataclass
class AugCfg:
    amp_scale_p: float = 0.8
    amp_scale_range: tuple = (0.8, 1.2)
    noise_p: float = 0.5
    noise_std: float = 0.01
    time_shift_p: float = 0.5
    time_shift_max: int = 50   # 50 samples at 25Hz = 2 seconds
    mask_p: float = 0.3
    mask_max_frac: float = 0.1
    norm: str = "zscore"


def _norm(x: torch.Tensor, mode: str):
    if mode == "zscore":
        mu = x.mean()
        sd = x.std().clamp_min(1e-6)
        return (x - mu) / sd
    if mode == "minmax":
        mn = x.min()
        mx = x.max()
        return (x - mn) / (mx - mn).clamp_min(1e-6)
    return x


def augment_1d(x: torch.Tensor, cfg: AugCfg):
    x = x.clone()
    if random.random() < cfg.amp_scale_p:
        x = x * random.uniform(*cfg.amp_scale_range)
    if random.random() < cfg.noise_p:
        sd = x.std().clamp_min(1e-6)
        x = x + torch.randn_like(x) * (cfg.noise_std * sd)
    if random.random() < cfg.time_shift_p:
        k = random.randint(-cfg.time_shift_max, cfg.time_shift_max)
        x = torch.roll(x, shifts=k, dims=0)
    if random.random() < cfg.mask_p:
        L = x.numel()
        mlen = max(1, int(L * random.uniform(0.0, cfg.mask_max_frac)))
        st = random.randint(0, L - mlen)
        x[st:st + mlen] = 0.0
    return _norm(x, cfg.norm)


class PPGPretrainPairDatasetResample(Dataset):
    """
    SSL pretraining dataset. start_idx/end_idx are original source_fs indices.

    If source segment is 60s at 100Hz: length=6000.
    With target_fs=25, output length=1500.
    """
    def __init__(
        self,
        manifest_csv: str,
        channel: int = 2,
        source_fs: int = 100,
        target_fs: int = 25,
        downsample_mode: str = "stride",
        aug: AugCfg = AugCfg(),
        bandpass: bool = False,
        bp_lo: float = 0.5,
        bp_hi: float = 8.0,
        bp_order: int = 3,
        filter_stage: str = "after_downsample",
    ):
        self.df = pd.read_csv(manifest_csv)
        self.df["id_clean"] = self.df["id_clean"].astype(str)
        self.channel = int(channel)
        self.source_fs = int(source_fs)
        self.target_fs = int(target_fs)
        self.downsample_mode = str(downsample_mode)
        self.aug = aug
        pids = sorted(self.df["id_clean"].unique().tolist())
        self.pid2i = {p: i for i, p in enumerate(pids)}
        self._cache: Dict[str, np.ndarray] = {}

        self.bandpass = bool(bandpass) and _HAS_SCIPY
        self.bp_lo = float(bp_lo)
        self.bp_hi = float(bp_hi)
        self.bp_order = int(bp_order)
        self.filter_stage = str(filter_stage)
        self._sos_source = None
        self._sos_target = None
        if self.bandpass and self.filter_stage == "before_downsample":
            self._sos_source = _make_bandpass_sos(self.source_fs, self.bp_lo, self.bp_hi, self.bp_order)
        if self.bandpass and self.filter_stage == "after_downsample":
            self._sos_target = _make_bandpass_sos(self.target_fs, self.bp_lo, self.bp_hi, self.bp_order)

    def __len__(self):
        return len(self.df)

    def _load_memmap(self, ppg_path: str):
        if ppg_path in self._cache:
            return self._cache[ppg_path]
        arr = np.load(ppg_path, mmap_mode="r")
        self._cache[ppg_path] = arr
        return arr

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        pid = str(r["id_clean"])
        ppg_path = str(r["ppg_path"])
        start = int(r["start_idx"])
        end = int(r["end_idx"])

        arr = self._load_memmap(ppg_path)
        seg = _slice_channel(arr, start, end, self.channel)

        if self.bandpass and self.filter_stage == "before_downsample":
            seg = _safe_sosfiltfilt(seg, self._sos_source, self.bp_order)

        seg = downsample_1d(seg, self.source_fs, self.target_fs, mode=self.downsample_mode)

        if self.bandpass and self.filter_stage == "after_downsample":
            seg = _safe_sosfiltfilt(seg, self._sos_target, self.bp_order)

        x = torch.from_numpy(seg.astype(np.float32, copy=False))
        v1 = augment_1d(x, self.aug).unsqueeze(0)
        v2 = augment_1d(x, self.aug).unsqueeze(0)
        return v1, v2, self.pid2i[pid]


class PatientWiseBatchSampler(Sampler[List[int]]):
    def __init__(self, df: pd.DataFrame, batch_size: int, seed: int = 1234, drop_last: bool = True):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self.pid_to_indices = defaultdict(list)
        for i, r in df.reset_index(drop=True).iterrows():
            self.pid_to_indices[str(r["id_clean"])].append(i)

        self.pids = list(self.pid_to_indices.keys())
        if len(self.pids) < self.batch_size:
            raise ValueError(f"patients({len(self.pids)}) < batch_size({self.batch_size})")

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        pids = self.pids[:]
        rng.shuffle(pids)
        batch = []
        for pid in pids:
            idx = rng.choice(self.pid_to_indices[pid])
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.pids) // self.batch_size
