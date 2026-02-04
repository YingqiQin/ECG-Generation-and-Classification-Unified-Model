import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_two_col_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "npy_path" in df.columns:
        out = df.copy()
    else:
        out = df.iloc[:, :2].copy()
        out.columns = ["npy_path", "LVEF"]
    out["npy_path"] = out["npy_path"].astype(str)
    return out


def infer_subject_id(npy_path: str) -> str:
    """
    Preferred: parent folder is 7 digits: .../<patient_id>/<file>.npy
    Fallback: first 7-digit number in path.
    """
    p = Path(str(npy_path))
    parent = p.parent.name
    if re.fullmatch(r"\d{7}", parent):
        return parent
    m = re.search(r"(\d{7})", str(p))
    if m:
        return m.group(1)
    return f"subj_{abs(hash(str(p.parent))) % (10**10)}"


def robust_norm_per_segment(x_lc: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    x_lc: [L, C]
    robust z-score per channel via median/MAD
    """
    med = np.median(x_lc, axis=0, keepdims=True)
    mad = np.median(np.abs(x_lc - med), axis=0, keepdims=True)
    return (x_lc - med) / (1.4826 * mad + eps)


class PPGAugment:
    """
    Slight augmentations for contrastive learning.
    Operates on numpy array [L, C].
    """
    def __init__(
        self,
        p_scale: float = 0.9,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        p_noise: float = 0.9,
        noise_std_ratio: float = 0.02,   # relative to per-channel std
        p_shift: float = 0.5,
        max_shift_frac: float = 0.05,    # fraction of segment length (e.g., 0.05*400=20 samples)
        p_mask: float = 0.3,
        mask_frac_range: Tuple[float, float] = (0.02, 0.08),  # mask 2%-8% of samples
    ):
        self.p_scale = p_scale
        self.scale_range = scale_range
        self.p_noise = p_noise
        self.noise_std_ratio = noise_std_ratio
        self.p_shift = p_shift
        self.max_shift_frac = max_shift_frac
        self.p_mask = p_mask
        self.mask_frac_range = mask_frac_range

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # x: [L, C]
        L, C = x.shape

        # random amplitude scale (per-channel)
        if rng.random() < self.p_scale:
            s = rng.uniform(self.scale_range[0], self.scale_range[1], size=(1, C)).astype(np.float32)
            x = x * s

        # additive gaussian noise
        if rng.random() < self.p_noise:
            ch_std = x.std(axis=0, keepdims=True) + 1e-6
            noise = rng.normal(0.0, self.noise_std_ratio, size=(L, C)).astype(np.float32) * ch_std
            x = x + noise

        # small circular time shift
        if rng.random() < self.p_shift:
            max_shift = int(round(self.max_shift_frac * L))
            if max_shift > 0:
                shift = int(rng.integers(-max_shift, max_shift + 1))
                if shift != 0:
                    x = np.roll(x, shift=shift, axis=0)

        # small time masking (set a short span to 0)
        if rng.random() < self.p_mask:
            frac = rng.uniform(self.mask_frac_range[0], self.mask_frac_range[1])
            mlen = max(1, int(round(frac * L)))
            start = int(rng.integers(0, max(1, L - mlen + 1)))
            x = x.copy()
            x[start:start + mlen, :] = 0.0

        return x


class PPGTripletContrastiveDataset(Dataset):
    """
    Produces (x1, x2, xneg) triplets for contrastive pretraining:
      - x1, x2: random segments from same subject
      - xneg: random segment from a different subject

    Each sample is a 4s segment by default:
      seg_len = seg_seconds * fs  (e.g., 4*100=400)

    Returns tensors shaped [C, L] (C=4 channels).
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        fs: int = 100,
        seg_seconds: float = 4.0,
        stride_seconds: Optional[float] = None,   # not used for sampling; kept for future
        mmap: bool = True,
        drop_missing_files: bool = True,
        min_subject_files: int = 1,               # subjects with < this many valid files removed
        n_samples_per_epoch: int = 100000,        # dataset length; controls how many triplets per epoch
        seed: int = 42,
        normalize: str = "robust",                # "none" | "robust"
        augment: Optional[PPGAugment] = None,
        clip_value: float = 5.0,
        return_subject_ids: bool = False,
    ):
        self.fs = int(fs)
        self.seg_len = int(round(seg_seconds * self.fs))
        self.mmap = bool(mmap)
        self.n_samples_per_epoch = int(n_samples_per_epoch)
        self.normalize = normalize
        self.clip_value = float(clip_value)
        self.return_subject_ids = bool(return_subject_ids)

        self.base_seed = int(seed)
        self.augment = augment if augment is not None else PPGAugment()

        df = read_two_col_csv(csv_path)
        df["subject_id"] = df["npy_path"].apply(infer_subject_id)

        if drop_missing_files:
            exists_mask = df["npy_path"].apply(lambda p: os.path.exists(str(p)))
            df = df[exists_mask].reset_index(drop=True)

        # Build file metadata: keep only npy with shape [T,4] and T>=seg_len
        file_meta = []
        for i in range(len(df)):
            path = str(df.loc[i, "npy_path"])
            sid = str(df.loc[i, "subject_id"])
            try:
                arr = np.load(path, mmap_mode="r" if self.mmap else None)
                if arr.ndim != 2 or arr.shape[1] != 4:
                    continue
                T = int(arr.shape[0])
                if T < self.seg_len:
                    continue
            except Exception:
                continue
            file_meta.append((path, sid, T))

        if len(file_meta) == 0:
            raise RuntimeError("No valid npy files found (need shape [T,4] with T >= seg_len).")

        # subject -> list of (path, T)
        subj_map: Dict[str, List[Tuple[str, int]]] = {}
        for path, sid, T in file_meta:
            subj_map.setdefault(sid, []).append((path, T))

        # filter subjects with too few valid files
        subj_ids = [sid for sid, items in subj_map.items() if len(items) >= min_subject_files]
        subj_ids.sort()
        if len(subj_ids) < 2:
            raise RuntimeError("Need at least 2 subjects with valid segments for negative sampling.")

        self.subj_ids = subj_ids
        self.subj_map = {sid: subj_map[sid] for sid in self.subj_ids}

    def __len__(self):
        return self.n_samples_per_epoch

    def _np_load(self, path: str):
        return np.load(path, mmap_mode="r" if self.mmap else None)

    def _sample_segment_from_subject(self, sid: str, rng: np.random.Generator) -> np.ndarray:
        items = self.subj_map[sid]
        # pick a random file within the subject
        path, T = items[int(rng.integers(0, len(items)))]
        # random start such that start+seg_len <= T
        start = int(rng.integers(0, T - self.seg_len + 1))
        arr = self._np_load(path)  # [T,4]
        seg = np.asarray(arr[start:start + self.seg_len, :], dtype=np.float32)  # [L,4]
        return seg

    def _preprocess(self, seg_lc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # normalization first (common for PPG to remove offset/scale), then slight augmentations
        x = seg_lc
        if self.normalize == "robust":
            x = robust_norm_per_segment(x)
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize={self.normalize}")

        x = self.augment(x, rng)

        if self.clip_value > 0:
            x = np.clip(x, -self.clip_value, self.clip_value)

        return x

    def __getitem__(self, idx: int):
        # Make RNG deterministic per index (and per worker) without global state.
        # Worker-specific entropy from torch initial seed:
        worker_seed = torch.initial_seed() % (2**32)
        rng = np.random.default_rng(self.base_seed + worker_seed + int(idx))

        # anchor subject
        sid = self.subj_ids[int(rng.integers(0, len(self.subj_ids)))]

        # negative subject (different)
        sid_neg = sid
        while sid_neg == sid:
            sid_neg = self.subj_ids[int(rng.integers(0, len(self.subj_ids)))]

        # sample segments
        x1 = self._sample_segment_from_subject(sid, rng)
        x2 = self._sample_segment_from_subject(sid, rng)
        xneg = self._sample_segment_from_subject(sid_neg, rng)

        # preprocess / augment
        x1 = self._preprocess(x1, rng)
        x2 = self._preprocess(x2, rng)
        xneg = self._preprocess(xneg, rng)

        # to torch: [C,L]
        x1 = torch.from_numpy(x1).transpose(0, 1).contiguous()
        x2 = torch.from_numpy(x2).transpose(0, 1).contiguous()
        xneg = torch.from_numpy(xneg).transpose(0, 1).contiguous()

        if self.return_subject_ids:
            return x1, x2, xneg, sid, sid_neg
        return x1, x2, xneg
