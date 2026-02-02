import numpy as np
import torch
from torch.utils.data import Dataset

def norm_per_segment_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    x: [L, C]
    """
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)

def norm_per_segment_robust(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    robust z-score via median/MAD, per channel
    x: [L, C]
    """
    med = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - med), axis=0, keepdims=True)
    return (x - med) / (1.4826 * mad + eps)

class PPGLVEFSegmentDataset(Dataset):
    def __init__(
        self,
        df,
        fs: int = 100,
        seg_seconds: float = 4.0,
        threshold: float = 40.0,
        stride_seconds=None,
        mmap: bool = True,
        return_subject: bool = False,
        normalize: str = "robust",  # "none" | "zscore" | "robust"
        clip_value: float = 0.0,    # 0 => no clip, else clip to [-clip_value, clip_value]
    ):
        import os
        from pathlib import Path

        self.df = df.copy()
        if "subject_id" not in self.df.columns:
            self.df["subject_id"] = self.df["npy_path"].apply(lambda p: Path(str(p)).parent.name)

        self.fs = int(fs)
        self.seg_len = int(round(seg_seconds * self.fs))
        self.stride_len = self.seg_len if stride_seconds is None else int(round(stride_seconds * self.fs))
        self.threshold = float(threshold)
        self.mmap = bool(mmap)
        self.return_subject = bool(return_subject)
        self.normalize = normalize
        self.clip_value = float(clip_value)

        self.df["label"] = (self.df["LVEF"].astype(float) <= self.threshold).astype(np.int64)

        self.index = []
        bad_files = 0
        for i in range(len(self.df)):
            path = str(self.df.loc[i, "npy_path"])
            if (not path) or (not os.path.exists(path)):
                bad_files += 1
                continue
            try:
                arr = np.load(path, mmap_mode="r" if self.mmap else None)  # [T,4]
                if arr.ndim != 2 or arr.shape[1] != 4:
                    bad_files += 1
                    continue
                T = int(arr.shape[0])
            except Exception:
                bad_files += 1
                continue

            for start in range(0, T - self.seg_len + 1, self.stride_len):
                self.index.append((i, start))

        if len(self.index) == 0:
            raise RuntimeError("No segments indexed. Check npy paths, shapes [T,4], and lengths.")
        if bad_files > 0:
            print(f"[WARN] {bad_files} files skipped due to missing/invalid npy.")

    def __len__(self):
        return len(self.index)

    def _apply_norm(self, seg: np.ndarray) -> np.ndarray:
        # seg: [L,4]
        if self.normalize == "none":
            out = seg
        elif self.normalize == "zscore":
            out = norm_per_segment_zscore(seg)
        elif self.normalize == "robust":
            out = norm_per_segment_robust(seg)
        else:
            raise ValueError(f"Unknown normalize={self.normalize}")

        if self.clip_value > 0:
            out = np.clip(out, -self.clip_value, self.clip_value)
        return out

    def __getitem__(self, idx: int):
        row_idx, start = self.index[idx]
        row = self.df.iloc[row_idx]

        path = str(row["npy_path"])
        y = int(row["label"])

        arr = np.load(path, mmap_mode="r" if self.mmap else None)  # [T,4]
        seg = np.asarray(arr[start:start + self.seg_len, :], dtype=np.float32)  # [L,4]

        seg = self._apply_norm(seg)

        x = torch.from_numpy(seg).transpose(0, 1).contiguous()  # [4,L]
        y = torch.tensor(y, dtype=torch.long)

        if self.return_subject:
            return x, y, str(row["subject_id"])
        return x, y
