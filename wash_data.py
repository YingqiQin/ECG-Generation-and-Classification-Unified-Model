import os
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def read_two_col_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    CSV must have:
      - either headers: ['npy_path', 'LVEF']
      - or at least 2 columns (we take the first two)
    """
    df = pd.read_csv(csv_path)
    if "npy_path" in df.columns and "LVEF" in df.columns:
        out = df[["npy_path", "LVEF"]].copy()
    else:
        if df.shape[1] < 2:
            raise ValueError("CSV must have at least 2 columns: npy_path, LVEF")
        out = df.iloc[:, :2].copy()
        out.columns = ["npy_path", "LVEF"]

    out["npy_path"] = out["npy_path"].astype(str)
    out["LVEF"] = pd.to_numeric(out["LVEF"], errors="coerce")
    out = out.dropna(subset=["LVEF"])
    return out


def infer_subject_id(npy_path: str) -> str:
    """
    Tries to infer patient id from path.

    Priority:
      1) parent directory name (common when you saved as .../npy/<patient_id>/file.npy)
      2) any 7-digit number inside the path (fallback)
    """
    p = Path(npy_path)
    parent = p.parent.name
    if re.fullmatch(r"\d{7}", parent):
        return parent

    m = re.search(r"(\d{7})", str(p))
    if m:
        return m.group(1)

    # last resort: stable ID based on parent path (still prevents leakage, but not human-readable)
    return f"subj_{abs(hash(str(p.parent))) % (10**10)}"


def make_subject_splits(
    subjects: List[str],
    ratios: Tuple[float, float, float],
    seed: int,
) -> Dict[str, str]:
    """
    Deterministic subject -> split mapping.
    """
    r_train, r_val, r_test = ratios
    if not (abs((r_train + r_val + r_test) - 1.0) < 1e-6):
        raise ValueError("ratios must sum to 1.0, e.g. (0.7,0.1,0.2)")

    subjects = list(set(subjects))
    subjects.sort()  # stable before shuffling for reproducibility across platforms
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(round(r_train * n))
    n_val = int(round(r_val * n))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    # remaining go to test

    train_sub = set(subjects[:n_train])
    val_sub = set(subjects[n_train:n_train + n_val])
    test_sub = set(subjects[n_train + n_val:])

    mapping = {}
    for s in train_sub:
        mapping[s] = "train"
    for s in val_sub:
        mapping[s] = "val"
    for s in test_sub:
        mapping[s] = "test"
    return mapping


def norm_per_segment(seg_lc: np.ndarray, mode: str, eps: float = 1e-6) -> np.ndarray:
    """
    seg_lc: [L, C]
    mode: "none" | "zscore" | "robust"
    """
    if mode == "none":
        return seg_lc

    if mode == "zscore":
        mu = seg_lc.mean(axis=0, keepdims=True)
        sd = seg_lc.std(axis=0, keepdims=True)
        return (seg_lc - mu) / (sd + eps)

    if mode == "robust":
        med = np.median(seg_lc, axis=0, keepdims=True)
        mad = np.median(np.abs(seg_lc - med), axis=0, keepdims=True)
        return (seg_lc - med) / (1.4826 * mad + eps)

    raise ValueError(f"Unknown normalization mode: {mode}")


class PPGLVEFBinaryDataset(Dataset):
    """
    Full CSV in -> subject-wise split inside -> segment indexing.

    Returns:
      x: FloatTensor [C, L]  (C=4, L=4s*fs)
      y: LongTensor []       (1 if LVEF<=thr else 0)

    Assumptions:
      - each npy is float array of shape [T,4]
      - fs=100 by default
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        split: str = "train",                        # "train" | "val" | "test"
        ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
        seed: int = 42,
        fs: int = 100,
        seg_seconds: float = 4.0,
        threshold: float = 40.0,
        stride_seconds: Optional[float] = None,      # None => non-overlap (drop residual)
        mmap: bool = True,
        normalization: str = "robust",               # "none" | "zscore" | "robust"
        clip_value: float = 5.0,                     # 0 => no clip
        return_meta: bool = False,                   # optionally return (subject_id, path, start)
        drop_missing_files: bool = True,             # skip rows whose npy doesn't exist
    ):
        split = split.lower().strip()
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of: 'train', 'val', 'test'")

        self.split = split
        self.ratios = ratios
        self.seed = int(seed)

        self.fs = int(fs)
        self.seg_len = int(round(seg_seconds * self.fs))
        self.stride_len = self.seg_len if stride_seconds is None else int(round(stride_seconds * self.fs))

        self.threshold = float(threshold)
        self.mmap = bool(mmap)

        self.normalization = normalization
        self.clip_value = float(clip_value)
        self.return_meta = bool(return_meta)

        # Load + clean CSV
        df = read_two_col_csv(csv_path)
        df["subject_id"] = df["npy_path"].apply(infer_subject_id)

        if drop_missing_files:
            exists_mask = df["npy_path"].apply(lambda p: os.path.exists(str(p)))
            df = df[exists_mask].reset_index(drop=True)

        # Binary labels
        df["label"] = (df["LVEF"].astype(float) <= self.threshold).astype(np.int64)

        # Subject-wise split mapping (strict)
        split_map = make_subject_splits(df["subject_id"].tolist(), ratios=self.ratios, seed=self.seed)
        df["split"] = df["subject_id"].map(split_map)

        # Keep only this split
        self.df = df[df["split"] == self.split].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(f"No samples in split='{self.split}'. Check CSV paths and split ratios.")

        # Build global segment index: list of (row_idx, start_sample)
        self.index: List[Tuple[int, int]] = []
        self._build_index()

    def _np_load(self, path: str):
        return np.load(path, mmap_mode="r" if self.mmap else None)

    def _build_index(self):
        skipped = 0
        for i in range(len(self.df)):
            path = str(self.df.loc[i, "npy_path"])
            try:
                arr = self._np_load(path)  # expected [T,4]
                if arr.ndim != 2 or arr.shape[1] != 4:
                    skipped += 1
                    continue
                T = int(arr.shape[0])
            except Exception:
                skipped += 1
                continue

            # non-overlapping by default; drop residual automatically
            for start in range(0, T - self.seg_len + 1, self.stride_len):
                self.index.append((i, start))

        if len(self.index) == 0:
            raise RuntimeError(
                "No segments indexed. Ensure each npy is [T,4] and T >= 4s*fs."
            )
        if skipped > 0:
            print(f"[WARN] {skipped} files skipped in split='{self.split}' due to load/shape issues.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        row_idx, start = self.index[idx]
        row = self.df.iloc[row_idx]

        path = str(row["npy_path"])
        y = int(row["label"])

        arr = self._np_load(path)  # [T,4]
        seg = np.asarray(arr[start:start + self.seg_len, :], dtype=np.float32)  # [L,4]

        # per-segment, per-channel normalization
        seg = norm_per_segment(seg, mode=self.normalization)
        if self.clip_value > 0:
            seg = np.clip(seg, -self.clip_value, self.clip_value)

        # CNN input: [C, L]
        x = torch.from_numpy(seg).transpose(0, 1).contiguous()
        y = torch.tensor(y, dtype=torch.long)

        if self.return_meta:
            return x, y, str(row["subject_id"]), path, int(start)
        return x, y

    def get_subject_ids(self) -> List[str]:
        return sorted(self.df["subject_id"].unique().tolist())


# -----------------------
# Example
# -----------------------
if __name__ == "__main__":
    csv_path = "/path/to/manifest_clean_two_cols.csv"  # columns: npy_path, LVEF

    train_set = PPGLVEFBinaryDataset(csv_path, split="train", seed=42)
    val_set   = PPGLVEFBinaryDataset(csv_path, split="val", seed=42)
    test_set  = PPGLVEFBinaryDataset(csv_path, split="test", seed=42)

    # Sanity: no subject overlap
    tr = set(train_set.get_subject_ids())
    va = set(val_set.get_subject_ids())
    te = set(test_set.get_subject_ids())
    assert len(tr & va) == 0 and len(tr & te) == 0 and len(va & te) == 0, "Subject leakage detected!"
    print("No subject overlap. Good.")
