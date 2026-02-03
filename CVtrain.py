import os
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold


def _read_two_col_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
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
    out = out.dropna(subset=["LVEF"]).reset_index(drop=True)
    return out


def _infer_subject_id(npy_path: str) -> str:
    """
    Preferred: parent directory is a 7-digit patient id: .../<patient_id>/<file>.npy
    Fallback: first 7-digit number in path.
    """
    p = Path(str(npy_path))
    parent = p.parent.name
    if re.fullmatch(r"\d{7}", parent):
        return parent

    m = re.search(r"(\d{7})", str(p))
    if m:
        return m.group(1)

    # last resort: stable pseudo-id based on folder path (still prevents leakage)
    return f"subj_{abs(hash(str(p.parent))) % (10**10)}"


def _norm_per_segment(seg_lc: np.ndarray, mode: str, eps: float = 1e-6) -> np.ndarray:
    """
    seg_lc: [L, C]
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


class PPGLVEFSegmentDatasetCV(Dataset):
    """
    One Dataset class that:
      - loads 2-col CSV: npy_path, LVEF
      - creates STRICT patient-wise split internally:
          * hold-out test set (patient-wise, stratified by minority label)
          * remaining patients -> StratifiedKFold for CV train/val
      - slices each npy [T,4] into 4s segments [4, L]

    split: "train" | "val" | "hold-out-test"

    Label convention (binary):
      - positive_if_ge=True => y=1 if LVEF >= thr else 0
      - positive_if_ge=False => y=1 if LVEF <= thr else 0
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        split: str,
        # Task definition
        threshold: float = 40.0,
        positive_if_ge: bool = True,
        # Hold-out test design
        n_test_minority: int = 6,
        test_majority_ratio: int = 10,
        fixed_holdout_subjects: Optional[List[str]] = None,  # reuse across thresholds if you want
        # CV design (on non-test patients)
        n_splits: int = 3,
        fold_id: int = 0,
        seed: int = 42,
        # Signal slicing
        fs: int = 100,
        seg_seconds: float = 4.0,
        stride_seconds: Optional[float] = None,  # None => non-overlap
        # IO & transforms
        mmap: bool = True,
        drop_missing_files: bool = True,
        normalization: str = "robust",  # "none" | "zscore" | "robust"
        clip_value: float = 5.0,        # 0 => no clip
        # Return format
        return_subject: bool = True,
        return_meta: bool = False,      # if True: also returns (path, start)
    ):
        split = split.lower().strip()
        if split not in ("train", "val", "hold-out-test"):
            raise ValueError("split must be one of: 'train', 'val', 'hold-out-test'")

        self.split = split
        self.threshold = float(threshold)
        self.positive_if_ge = bool(positive_if_ge)

        self.n_test_minority = int(n_test_minority)
        self.test_majority_ratio = int(test_majority_ratio)
        self.fixed_holdout_subjects = list(fixed_holdout_subjects) if fixed_holdout_subjects is not None else None

        self.n_splits = int(n_splits)
        self.fold_id = int(fold_id)
        self.seed = int(seed)

        self.fs = int(fs)
        self.seg_len = int(round(seg_seconds * self.fs))
        self.stride_len = self.seg_len if stride_seconds is None else int(round(stride_seconds * self.fs))

        self.mmap = bool(mmap)
        self.drop_missing_files = bool(drop_missing_files)
        self.normalization = normalization
        self.clip_value = float(clip_value)
        self.return_subject = bool(return_subject)
        self.return_meta = bool(return_meta)

        # ---------- load file table ----------
        df = _read_two_col_csv(csv_path)
        df["subject_id"] = df["npy_path"].apply(_infer_subject_id)

        if self.drop_missing_files:
            exists_mask = df["npy_path"].apply(lambda p: os.path.exists(str(p)))
            df = df[exists_mask].reset_index(drop=True)

        # binary label per file
        if self.positive_if_ge:
            df["y"] = (df["LVEF"].astype(float) >= self.threshold).astype(np.int64)
        else:
            df["y"] = (df["LVEF"].astype(float) <= self.threshold).astype(np.int64)

        # patient-level table (you said 1 LVEF per patient, but keep robust)
        pat = df.groupby("subject_id", as_index=False)["y"].max()

        # ---------- build hold-out test subjects ----------
        if self.fixed_holdout_subjects is not None:
            test_subjects = set(self.fixed_holdout_subjects)
        else:
            test_subjects = self._make_holdout_test_subjects(pat)

        # patient pool for CV (exclude test subjects)
        pat_cv = pat[~pat["subject_id"].isin(test_subjects)].reset_index(drop=True)
        if len(pat_cv) == 0:
            raise RuntimeError("After removing hold-out test subjects, no patients left for CV.")

        # Validate CV feasibility: need enough minority patients for n_splits
        counts = pat_cv["y"].value_counts().to_dict()
        n0 = counts.get(0, 0)
        n1 = counts.get(1, 0)
        minority_count = min(n0, n1)
        if minority_count < self.n_splits:
            raise RuntimeError(
                f"Not enough minority patients for {self.n_splits}-fold CV after holdout. "
                f"Counts in CV pool: n0={n0}, n1={n1}. "
                f"Consider reducing n_splits or reducing n_test_minority."
            )

        # ---------- pick subjects for requested split ----------
        if self.split == "hold-out-test":
            selected_subjects = test_subjects
        else:
            tr_sub, va_sub = self._make_cv_fold_subjects(pat_cv, n_splits=self.n_splits, fold_id=self.fold_id)
            selected_subjects = tr_sub if self.split == "train" else va_sub

        # filter file-level df
        self.df = df[df["subject_id"].isin(selected_subjects)].reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError(f"No files found for split='{self.split}'. Check your holdout/CV settings.")

        # ---------- build segment index ----------
        self.index: List[Tuple[int, int]] = []
        self._build_index()

    def _np_load(self, path: str):
        return np.load(path, mmap_mode="r" if self.mmap else None)

    def _make_holdout_test_subjects(self, pat: pd.DataFrame) -> set:
        """
        Deterministically sample test subjects:
          - choose n_test_minority subjects from the minority label
          - choose test_majority_ratio * n_test_minority from the majority label
        """
        rng = np.random.default_rng(self.seed)

        counts = pat["y"].value_counts().to_dict()
        # define minority label (less frequent)
        minority_label = 0 if counts.get(0, 0) < counts.get(1, 0) else 1
        majority_label = 1 - minority_label

        minority_subj = pat.loc[pat["y"] == minority_label, "subject_id"].tolist()
        majority_subj = pat.loc[pat["y"] == majority_label, "subject_id"].tolist()

        if len(minority_subj) < self.n_test_minority:
            raise RuntimeError(
                f"Not enough minority patients for hold-out test: have {len(minority_subj)}, "
                f"need {self.n_test_minority}."
            )

        # stable order before sampling for reproducibility across platforms
        minority_subj = sorted(minority_subj)
        majority_subj = sorted(majority_subj)

        test_minority = rng.choice(minority_subj, size=self.n_test_minority, replace=False).tolist()

        n_test_majority = min(len(majority_subj), self.n_test_minority * self.test_majority_ratio)
        test_majority = rng.choice(majority_subj, size=n_test_majority, replace=False).tolist()

        return set(test_minority) | set(test_majority)

    def _make_cv_fold_subjects(self, pat_cv: pd.DataFrame, n_splits: int, fold_id: int) -> Tuple[set, set]:
        """
        StratifiedKFold on patient level (no leakage).
        """
        if fold_id < 0 or fold_id >= n_splits:
            raise ValueError(f"fold_id must be in [0, {n_splits-1}]")

        subjects = pat_cv["subject_id"].values
        y = pat_cv["y"].values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        splits = list(skf.split(subjects, y))
        tr_idx, va_idx = splits[fold_id]

        tr_sub = set(subjects[tr_idx].tolist())
        va_sub = set(subjects[va_idx].tolist())
        return tr_sub, va_sub

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

            # drop residual automatically
            for start in range(0, T - self.seg_len + 1, self.stride_len):
                self.index.append((i, start))

        if len(self.index) == 0:
            raise RuntimeError(
                f"No segments indexed for split='{self.split}'. "
                f"Check that npy arrays are [T,4] and T >= seg_len={self.seg_len}."
            )
        if skipped > 0:
            print(f"[WARN] {skipped} files skipped in split='{self.split}' due to load/shape issues.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        row_idx, start = self.index[idx]
        row = self.df.iloc[row_idx]

        path = str(row["npy_path"])
        lvef = float(row["LVEF"])

        # binary label
        if self.positive_if_ge:
            y = 1 if lvef >= self.threshold else 0
        else:
            y = 1 if lvef <= self.threshold else 0

        arr = self._np_load(path)  # [T,4]
        seg = np.asarray(arr[start:start + self.seg_len, :], dtype=np.float32)  # [L,4]

        # normalization
        seg = _norm_per_segment(seg, self.normalization)
        if self.clip_value > 0:
            seg = np.clip(seg, -self.clip_value, self.clip_value)

        # CNN input [C,L]
        x = torch.from_numpy(seg).transpose(0, 1).contiguous()
        y = torch.tensor(y, dtype=torch.long)

        if self.return_meta and self.return_subject:
            return x, y, str(row["subject_id"]), path, int(start)
        if self.return_subject:
            return x, y, str(row["subject_id"])
        return x, y

    def get_subject_ids(self) -> List[str]:
        return sorted(self.df["subject_id"].unique().tolist())

    def export_split_subjects(self) -> Dict[str, Union[int, str, List[str]]]:
        """
        Useful for reproducibility/debugging. Call this on any instance.
        """
        return {
            "split": self.split,
            "threshold": self.threshold,
            "positive_if_ge": self.positive_if_ge,
            "seed": self.seed,
            "n_splits": self.n_splits,
            "fold_id": self.fold_id,
            "n_subjects": len(self.get_subject_ids()),
            "subject_ids": self.get_subject_ids(),
        }
