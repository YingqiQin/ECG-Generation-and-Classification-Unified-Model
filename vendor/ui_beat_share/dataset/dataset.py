from __future__ import annotations

import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(parent_dir)
from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from sklearn import preprocessing
from scipy.io import loadmat

import dataset.ecg_preprocess as ep


def _load_mat_variable(path: Path, key: str) -> np.ndarray:
    data = loadmat(str(path))
    if key not in data:
        raise KeyError(f"Variable '{key}' not found in {path.name}")
    return np.asarray(data[key])


def r_peaks_to_mask(
    r_peaks: np.ndarray,
    fs: int,
    signal_length: int,
    mask_sampling_rate: int = 50,
    amplitude: float = 1.0,
    width: int = 3,
) -> np.ndarray:
    r_peaks = np.asarray(r_peaks, dtype=float).ravel()
    mask_length = int(round(signal_length * mask_sampling_rate / fs))
    mask = np.zeros(mask_length, dtype=np.float32)

    if r_peaks.size == 0:
        return mask

    factor = fs / mask_sampling_rate
    peak_indices = np.round(r_peaks / factor).astype(int)
    peak_indices = np.clip(peak_indices, 0, mask_length - 1)

    for idx in peak_indices:
        start = max(idx - width, 0)
        end = min(idx + width + 1, mask_length)
        mask[start:end] += amplitude

    np.clip(mask, 0.0, amplitude, out=mask)

    return mask

def _bandpass_filt(data: np.ndarray, fs: int, lead_num: int) -> np.ndarray:
    data_e = data.copy()
    processed: List[np.ndarray] = []
    for i in range(lead_num):
        tmp = data_e if lead_num == 1 else data_e[:, i]
        tmp = tmp - ep.lowpass_filter(tmp, 0.5, fs)
        tmp = tmp - ep.highpass_filter(tmp, 45, fs)
        tmp = tmp - np.mean(tmp)
        if fs != 200:
            tmp = ep.downsample(tmp, fs, 200)
        tmp = preprocessing.scale(tmp).astype(np.float32)
        processed.append(tmp)
    return np.asarray(processed, dtype=np.float32).T


def preprocess_ecg(signal: np.ndarray, fs: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim == 1:
        cleaned = ep.pp(signal)
        filtered = _bandpass_filt(cleaned, fs, 1)
        return filtered.T

    if signal.ndim == 2:
        samples_first = signal if signal.shape[0] >= signal.shape[1] else signal.T
        lead_num = samples_first.shape[1]
        cleaned = np.zeros_like(samples_first, dtype=np.float32)
        for idx in range(lead_num):
            cleaned[:, idx] = ep.pp(samples_first[:, idx])
        filtered = _bandpass_filt(cleaned, fs, lead_num)
        return filtered.T

    raise ValueError("ECG signal must be either 1D or 2D.")


class CPSC2019Dataset(Dataset):
    def __init__(
        self,
        root: Path,
        fs: int = 500,
        mask_sampling_rate: int = 50,
        peak_amplitude: float = 1.0,
        width: int = 3,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:

        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.ref_dir = self.root / "ref"

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.ref_dir.exists():
            raise FileNotFoundError(f"Reference directory not found: {self.ref_dir}")

        self.fs = fs
        self.mask_sampling_rate = mask_sampling_rate
        self.peak_amplitude = peak_amplitude
        self.width = width
        self.transform = transform

        self.records = sorted(self.data_dir.glob("data_*.mat"))
        if not self.records:
            raise RuntimeError(f"No .mat files found under {self.data_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        data_path = self.records[index]
        record_id = data_path.stem.split("_")[-1]
        ref_path = self.ref_dir / f"R_{record_id}.mat"

        if not ref_path.exists():
            raise FileNotFoundError(f"Reference file missing for record {record_id}")

        raw_signal = _load_mat_variable(data_path, "ecg").astype(np.float32).squeeze()
        signal_length = raw_signal.shape[-1] if raw_signal.ndim > 1 else raw_signal.shape[0]

        r_peaks = _load_mat_variable(ref_path, "R_peak").squeeze()
        r_peaks = np.unique(np.round(r_peaks).astype(int))

        mask = r_peaks_to_mask(
            r_peaks=r_peaks,
            fs=self.fs,
            signal_length=signal_length,
            mask_sampling_rate=self.mask_sampling_rate,
            amplitude=self.peak_amplitude,
            width=self.width,
        )

        processed = preprocess_ecg(raw_signal, self.fs)
        if processed.ndim == 1:
            processed = processed[np.newaxis, :]
        elif processed.shape[0] > processed.shape[1]:
            processed = processed.T
        signal_tensor = torch.from_numpy(processed)
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        if self.transform is not None:
            signal_tensor = self.transform(signal_tensor)

        return {"signal": signal_tensor, "mask": mask_tensor, "record_id": record_id}


def cpsc2019_collate_fn(batch: List[Dict[str, Tensor]]) -> tuple[Tensor, Tensor, List[str]]:
    signals = torch.stack([item["signal"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    record_ids = [item["record_id"] for item in batch]
    return signals, masks, record_ids


__all__ = ["CPSC2019Dataset", "r_peaks_to_mask", "preprocess_ecg", "cpsc2019_collate_fn"]


if __name__ == "__main__":
    dataset_root = Path(__file__).resolve().parents[1] / "data" / "cpsc2019_train"
    dataset = CPSC2019Dataset(root=dataset_root)

    first_item = dataset[0]
    signal_tensor = first_item["signal"]
    mask_tensor = first_item["mask"]

    print(f"Record ID: {first_item['record_id']}")
    print(f"Signal shape: {tuple(signal_tensor.shape)}")
    print(f"Mask shape: {tuple(mask_tensor.shape)}")
    print(f"Mask positives: {(mask_tensor > 0).sum().item()}")
