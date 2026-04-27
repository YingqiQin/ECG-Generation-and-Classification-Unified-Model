from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
STD_LEAD_ORDER = ['I', 'II', 'III', 'avR', 'avL', 'avF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEAD_TO_IDX = {name: i for i, name in enumerate(STD_LEAD_ORDER)}

try:
    from scipy.signal import butter, sosfiltfilt
except Exception as e:
    butter = None
    sosfiltfilt = None

try:
    import wfdb
except Exception:
    wfdb = None


# ----------------------------
# bandpass filter cache
# ----------------------------
class BandpassFilter:
    def __init__(self, low_hz: float = 0.5, high_hz: float = 40.0, order: int = 4):
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order
        self._sos_cache: Dict[int, Any] = {}

    def _get_sos(self, fs: int):
        if butter is None:
            raise ImportError("scipy 未安装或不可用：带通滤波需要 scipy.signal。")
        if fs not in self._sos_cache:
            nyq = 0.5 * fs
            low = self.low_hz / nyq
            high = self.high_hz / nyq
            sos = butter(self.order, [low, high], btype="bandpass", output="sos")
            self._sos_cache[fs] = sos
        return self._sos_cache[fs]

    def apply(self, x: np.ndarray, fs: int) -> np.ndarray:
        """
        x: (C, T)
        """
        sos = self._get_sos(fs)
        x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y = sosfiltfilt(sos, x, axis=-1).astype(np.float32)
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def _sanitize_signal_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if np.isfinite(arr).all():
        return arr.astype(np.float32, copy=False)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def make_patient_split(
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        seed: int = 42,

)-> Dict[str, np.ndarray]:
    patient_ids = df['patient_id'].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))

    train_p = set(patient_ids[:n_train])
    valid_p = set(patient_ids[n_train:n_train+n_valid])
    test_p = set(patient_ids[n_train+n_valid:])

    idx_train = df.index[df['patient_id'].isin(train_p)].to_numpy()
    idx_valid = df.index[df['patient_id'].isin(valid_p)].to_numpy()
    idx_test = df.index[df['patient_id'].isin(test_p)].to_numpy()

    return {'train': idx_train, 'valid': idx_valid, 'test': idx_test}

@dataclass
class ECGRecord:
    npy_path: Path
    signal_format: str
    cls: int
    lvef: float
    fs: float
    n_samples: int
    xml_file: str
    patient_id: str


@dataclass
class ECGTransferRecord:
    signal_path: Path
    signal_format: str
    fs: int
    n_samples: Optional[int]
    record_id: str
    patient_id: str

def patient_id_from_numpy(npy_scr: str) -> str:
    stem = Path(npy_scr).stem
    return stem.split("_", 1)[0] if "_" in stem else stem
def resolve_path(p: str, root: Path) -> Path:
    root = Path(root)
    pp = Path(str(p))

    if pp.is_absolute():
        return pp
    root_posix = root.as_posix().rstrip("/")
    p_posix = pp.as_posix()

    if root.name and (p_posix ==root.name or p_posix.startswith(root.name + "/")):
        pp = Path(*pp.parts[1:])
    return root / pp


def patient_id_from_signal_path(path_str: str) -> str:
    path = Path(str(path_str))
    for part in reversed(path.parts):
        if part.startswith("p") and part[1:].isdigit() and len(part) > 1:
            return part[1:]
    return patient_id_from_numpy(path_str)


def _normalize_wfdb_record_path(path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix in {".hea", ".dat"}:
        return path.with_suffix("")
    return path


def _infer_signal_format(path_str: str, explicit_format: Optional[str] = None) -> str:
    if explicit_format is not None and str(explicit_format).strip():
        fmt = str(explicit_format).strip().lower()
        if fmt not in {"npy", "wfdb", "xml"}:
            raise ValueError("Unsupported signal format '{}'. Use 'npy', 'wfdb', or 'xml'.".format(explicit_format))
        return fmt

    suffix = Path(str(path_str)).suffix.lower()
    if suffix == ".npy":
        return "npy"
    if suffix == ".xml":
        return "xml"
    if suffix in {".hea", ".dat"} or suffix == "":
        return "wfdb"
    raise ValueError("Could not infer signal format from path '{}'.".format(path_str))


def _canonicalize_wfdb_lead_name(name: str) -> str:
    raw = str(name).strip()
    upper = raw.upper()
    mapping = {
        "AVR": "avR",
        "AVL": "avL",
        "AVF": "avF",
        "A VR": "avR",
        "A VL": "avL",
        "A VF": "avF",
    }
    if upper in mapping:
        return mapping[upper]
    return raw


def _load_wfdb_12xT(record_path: Path) -> Tuple[np.ndarray, int]:
    if wfdb is None:
        raise ImportError("wfdb is required to read original MIMIC-IV-ECG files. Install wfdb in the server environment.")

    record = wfdb.rdrecord(str(_normalize_wfdb_record_path(record_path)))
    if getattr(record, "p_signal", None) is not None:
        signal = np.asarray(record.p_signal, dtype=np.float32)
    elif getattr(record, "d_signal", None) is not None:
        signal = np.asarray(record.d_signal, dtype=np.float32)
    else:
        raise ValueError("WFDB record '{}' does not contain p_signal or d_signal.".format(record_path))

    if signal.ndim != 2:
        raise ValueError("Expected WFDB signal to have shape [T, C], got {} for '{}'.".format(signal.shape, record_path))

    sig_names = [_canonicalize_wfdb_lead_name(name) for name in getattr(record, "sig_name", [])]
    if len(sig_names) != signal.shape[1]:
        raise ValueError("WFDB record '{}' has mismatched signal names and channels.".format(record_path))

    lead_to_index = {name: idx for idx, name in enumerate(sig_names)}
    missing = [lead for lead in STD_LEAD_ORDER if lead not in lead_to_index]
    if missing:
        raise ValueError("WFDB record '{}' is missing leads: {}.".format(record_path, ", ".join(missing)))

    ordered = signal[:, [lead_to_index[lead] for lead in STD_LEAD_ORDER]].T
    fs = int(round(float(getattr(record, "fs", 500))))
    return _sanitize_signal_array(ordered), fs


XML_NS = {"h": "urn:hl7-org:v3"}
XML_LEAD_CODE_TO_STD = {
    "MDC_ECG_LEAD_I": "I",
    "MDC_ECG_LEAD_II": "II",
    "MDC_ECG_LEAD_III": "III",
    "MDC_ECG_LEAD_AVR": "avR",
    "MDC_ECG_LEAD_AVL": "avL",
    "MDC_ECG_LEAD_AVF": "avF",
    "MDC_ECG_LEAD_aVR": "avR",
    "MDC_ECG_LEAD_aVL": "avL",
    "MDC_ECG_LEAD_aVF": "avF",
    "MDC_ECG_LEAD_V1": "V1",
    "MDC_ECG_LEAD_V2": "V2",
    "MDC_ECG_LEAD_V3": "V3",
    "MDC_ECG_LEAD_V4": "V4",
    "MDC_ECG_LEAD_V5": "V5",
    "MDC_ECG_LEAD_V6": "V6",
}


def _load_hl7_xml_12xT(xml_path: Path) -> Tuple[np.ndarray, int]:
    root = ET.parse(xml_path).getroot()

    rhythm_series = None
    for series in root.findall(".//h:series", XML_NS):
        code_node = series.find("h:code", XML_NS)
        if code_node is not None and code_node.attrib.get("code") == "RHYTHM":
            rhythm_series = series
            break
    if rhythm_series is None:
        raise ValueError("Could not find RHYTHM series in XML ECG '{}'.".format(xml_path))

    time_increment = None
    lead_data: Dict[str, np.ndarray] = {}
    for seq in rhythm_series.findall(".//h:sequence", XML_NS):
        code_node = seq.find("h:code", XML_NS)
        value_node = seq.find("h:value", XML_NS)
        if code_node is None or value_node is None:
            continue

        increment_node = value_node.find("h:increment", XML_NS)
        if increment_node is not None and time_increment is None:
            try:
                time_increment = float(increment_node.attrib.get("value"))
            except Exception:
                time_increment = None

        digits_node = value_node.find("h:digits", XML_NS)
        if digits_node is None or digits_node.text is None:
            continue

        lead_name = XML_LEAD_CODE_TO_STD.get(code_node.attrib.get("code"))
        if lead_name is None:
            continue

        digits = np.fromstring(digits_node.text.strip(), sep=" ", dtype=np.float32)
        if digits.size == 0:
            continue
        origin_node = value_node.find("h:origin", XML_NS)
        scale_node = value_node.find("h:scale", XML_NS)
        origin = 0.0 if origin_node is None else float(origin_node.attrib.get("value", 0.0))
        scale = 1.0 if scale_node is None else float(scale_node.attrib.get("value", 1.0))
        lead_data[lead_name] = origin + scale * digits

    missing = [lead for lead in STD_LEAD_ORDER if lead not in lead_data]
    if missing:
        raise ValueError("XML ECG '{}' is missing leads: {}.".format(xml_path, ", ".join(missing)))

    lengths = {lead: lead_data[lead].shape[0] for lead in STD_LEAD_ORDER}
    if len(set(lengths.values())) != 1:
        raise ValueError("XML ECG '{}' has inconsistent lead lengths: {}.".format(xml_path, lengths))

    fs = 500
    if time_increment is not None and time_increment > 0:
        fs = int(round(1.0 / time_increment))

    arr = np.stack([lead_data[lead] for lead in STD_LEAD_ORDER], axis=0).astype(np.float32, copy=False)
    arr = _sanitize_signal_array(arr)
    return arr, fs


def _load_signal_12xT(signal_path: Path, signal_format: str, fallback_fs: Optional[int] = None) -> Tuple[np.ndarray, int]:
    if signal_format == "npy":
        arr = np.load(signal_path, allow_pickle=False, mmap_mode=None).astype(np.float32)
        arr = ECGMILDataset._ensure_shape_12xT(arr)
        arr = _sanitize_signal_array(arr)
        fs = int(fallback_fs) if fallback_fs is not None else 500
        return arr, fs
    if signal_format == "wfdb":
        return _load_wfdb_12xT(signal_path)
    if signal_format == "xml":
        return _load_hl7_xml_12xT(signal_path)
    raise ValueError("Unsupported signal format '{}'.".format(signal_format))


def validate_lead_mode(lead_mode: str) -> str:
    if lead_mode == "12":
        return lead_mode
    if lead_mode in LEAD_TO_IDX:
        return lead_mode
    if lead_mode.endswith("_1ch") and lead_mode[:-4] in LEAD_TO_IDX:
        return lead_mode
    valid_modes = ["12"] + STD_LEAD_ORDER + [f"{lead}_1ch" for lead in STD_LEAD_ORDER]
    raise ValueError("Invalid lead mode '{}'. Supported examples: {}.".format(lead_mode, ", ".join(valid_modes[:6]) + ", ..."))


def lead_mode_num_channels(lead_mode: str) -> int:
    lead_mode = validate_lead_mode(lead_mode)
    if lead_mode == "12":
        return 12
    if lead_mode.endswith("_1ch"):
        return 1
    return 12


def select_leads_by_mode(arr: np.ndarray, lead_mode: str) -> np.ndarray:
    lead_mode = validate_lead_mode(lead_mode)
    if lead_mode == "12":
        return arr.astype(np.float32, copy=False)

    single_channel = lead_mode.endswith("_1ch")
    lead_name = lead_mode[:-4] if single_channel else lead_mode
    lead_idx = LEAD_TO_IDX[lead_name]
    selected = arr[lead_idx:lead_idx + 1, :]
    if single_channel:
        return selected.astype(np.float32, copy=False)
    return np.repeat(selected, 12, axis=0).astype(np.float32, copy=False)


def _first_present_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


class ECGMILDataset(Dataset):
    def __init__(self,
                 csv_path: Path,
                 npy_root: Path,
                 seg_sec: float = 10.0,
                 stride_sec: Optional[float] = None,
                 target_fs: int = 500,
                 lead_mode: str = '12',
                 bandpass: bool = True,
                 norm: str = 'zscore',
                 eps: float = 1e-8,
                 deterministic: bool = False,
                 base_seed: int=1234,
                 sample_mode: str= 'random',
                 K: int = 6,
                 return_dict: bool = False,
                 return_seg_quality: bool = False,
                 return_seg_starts: bool = False,
                 ):
        self.csv_path = csv_path
        self.npy_root = npy_root
        self.seg_sec = seg_sec
        self.stride_sec = float(stride_sec) if stride_sec is not None else float(seg_sec)
        self.target_fs = target_fs
        self.lead_mode = lead_mode
        self.bandpass = bandpass
        self.norm = norm
        self.eps = eps
        self.deterministic = deterministic
        self.base_seed = base_seed
        self.sample_mode = sample_mode
        self.K = K
        self.return_dict = return_dict
        self.return_seg_quality = return_seg_quality
        self.return_seg_starts = return_seg_starts

        self.lead_mode = validate_lead_mode(self.lead_mode)
        if self.target_fs != 500:
            raise ValueError(f"Invalid target fs: {self.target_fs}")

        df = pd.read_csv(self.csv_path)
        df.columns = [str(c).strip() for c in df.columns]

        path_col = _first_present_column(df, ["npy_dst", "npy_path", "xml_path", "path", "signal_path", "waveform_path", "file_path"])
        format_col = _first_present_column(df, ["path_format", "signal_format", "waveform_format"])
        patient_src_col = _first_present_column(df, ["npy_src", "npy_source", "patient_src", "subject_id", "patient_id"])

        required_cols = ['class', 'sampling_rate_hz', 'n_samples', 'xml_file', 'LVEF']
        missing = [c for c in required_cols if c not in df.columns]
        if path_col is None:
            raise ValueError("Missing waveform path column. Expected one of: npy_dst, npy_path, xml_path, path, signal_path, waveform_path, file_path.")
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if patient_src_col is not None:
            if patient_src_col in {"subject_id", "patient_id"}:
                df['patient_id'] = df[patient_src_col].astype(str)
            else:
                df['patient_id'] = df[patient_src_col].astype(str).map(patient_id_from_numpy)
        else:
            df['patient_id'] = df[path_col].astype(str).map(patient_id_from_signal_path)

        df['class'] = pd.to_numeric(df['class'], errors='coerce')
        df['sampling_rate_hz'] = pd.to_numeric(df['sampling_rate_hz'], errors='coerce')
        df['n_samples'] = pd.to_numeric(df['n_samples'], errors='coerce')
        df['LVEF'] = pd.to_numeric(df['LVEF'], errors='coerce')

        df = df.dropna(subset=['class', 'sampling_rate_hz', 'n_samples', path_col, 'LVEF']).copy()

        df = df[df['sampling_rate_hz'].isin([500, 1000])].copy()

        self.records: List[ECGRecord] = []
        for _, r in df.iterrows():
            signal_path = resolve_path(str(r[path_col]), self.npy_root)
            signal_format = _infer_signal_format(
                str(r[path_col]),
                explicit_format=None if format_col is None or pd.isna(r[format_col]) else str(r[format_col]),
            )
            if signal_format == "wfdb":
                signal_path = _normalize_wfdb_record_path(signal_path)
            self.records.append(ECGRecord(
                npy_path=signal_path,
                signal_format=signal_format,
                cls=int(r['class']),
                lvef=float(r['LVEF']) if not pd.isna(r['LVEF']) else float('nan'),
                fs=int(r['sampling_rate_hz']),
                n_samples=int(r['n_samples']),
                xml_file=str(r['xml_file']),
                patient_id=str(r['patient_id']),
            ))
        self.seg_len = int(round(self.seg_sec * self.target_fs))  # 10s -> 5000
        self.stride = int(round(self.stride_sec * self.target_fs))

        self.index: List[Tuple[int, int]] = []
        for i, rec in enumerate(self.records):
            n_tgt = self._n_samples_after_resample(rec.n_samples, rec.fs)
            if n_tgt < self.seg_len:
                continue
            # 非重叠/可重叠切片
            for s in range(0, n_tgt - self.seg_len + 1, self.stride):
                self.index.append((i, s))

        self._bp = BandpassFilter(low_hz=0.5, high_hz=40.0, order=4) if self.bandpass else None

    @staticmethod
    def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return x.astype(np.float32, copy=False)
        kernel = np.ones(window, dtype=np.float32) / float(window)
        return np.convolve(x.astype(np.float32, copy=False), kernel, mode="same")

    def _compute_segment_quality(self, seg: np.ndarray) -> np.ndarray:
        x = seg.mean(axis=0).astype(np.float32, copy=False)
        x_std = float(np.std(x) + self.eps)
        diff = np.diff(x, prepend=x[:1])

        flatline_ratio = float(np.mean(np.abs(diff) < max(1e-4, 0.01 * x_std)))

        max_abs = float(np.max(np.abs(x)))
        if max_abs < self.eps:
            clipping_ratio = 1.0
        else:
            clipping_ratio = float(np.mean(np.abs(np.abs(x) - max_abs) < 0.01 * max_abs))

        drift = self._moving_average(x, max(3, int(0.6 * self.target_fs)))
        drift_ratio = float(np.std(drift) / x_std)

        smooth = self._moving_average(x, 5)
        hf = x - smooth
        high_freq_ratio = float(np.std(hf) / x_std)

        features = np.asarray(
            [flatline_ratio, clipping_ratio, drift_ratio, high_freq_ratio],
            dtype=np.float32,
        )
        return np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=0.0)

    @staticmethod
    def _n_samples_after_resample(n: int, fs: int) -> int:
        if fs == 500:
            return n
        if fs == 1000:
            return n // 2
        raise ValueError(f"Unsupported fs={fs}")

    @staticmethod
    def _resample_to_500(x: np.ndarray, fs: int) -> np.ndarray:
        """
        """
        if fs == 500:
            return _sanitize_signal_array(x)
        if fs == 1000:
            return _sanitize_signal_array(x[:, ::2])
        raise ValueError(f"Unsupported fs={fs}")

    @staticmethod
    def _ensure_shape_12xT(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 2:
            raise ValueError(f"Expect 2D npy, got shape={arr.shape}")
        if arr.shape[0] == 12:
            return _sanitize_signal_array(arr)
        if arr.shape[1] == 12:
            return _sanitize_signal_array(arr.T)
        raise ValueError(f"npy shape not compatible with 12 leads: {arr.shape}")

    def __len__(self) -> int:
        return len(self.records)

    def _make_deterministic_rng(self, rec: ECGRecord) -> np.random.RandomState:
        seed_src = f"{rec.cls}|{rec.npy_path}|{self.base_seed}"
        seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
        return np.random.RandomState(seed)

    def _select_leads(self, seg: np.ndarray) -> np.ndarray:
        return select_leads_by_mode(seg, self.lead_mode)

    def _slice_segment(self, arr: np.ndarray, start: int) -> np.ndarray:
        end = start + self.seg_len
        seg = arr[:, start:end]
        if seg.shape[-1] == self.seg_len:
            return seg

        padded = np.zeros((arr.shape[0], self.seg_len), dtype=arr.dtype)
        padded[:, :seg.shape[-1]] = seg
        return padded

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        if self.deterministic:
            rng = self._make_deterministic_rng(rec)
        else:
            rng = np.random
        arr, src_fs = _load_signal_12xT(rec.npy_path, rec.signal_format, fallback_fs=rec.fs)
        arr = _sanitize_signal_array(arr)

        # 先统一到 500Hz（对 1000Hz：直接每 2 点取 1 点）
        arr_500 = self._resample_to_500(arr, src_fs)

        # 带通滤波（在 segment 级别做；轻量且符合你“少预处理”的原则）
        if self._bp is not None:
            arr_500 = self._bp.apply(arr_500, fs=self.target_fs)
        arr_500 = _sanitize_signal_array(arr_500)

        T = arr_500.shape[-1]
        L = self.seg_len

        if T <= L:
            starts = [0] * self.K
        else:
            max_start = T - L
            if self.sample_mode == "dense_nonoverlap":
                starts = list(range(0, max_start + 1, L))  # non-overlap
                if len(starts) >= self.K:
                    starts = starts[:self.K]
                else:
                    starts = starts + [starts[-1]] * (self.K - len(starts))
            elif self.sample_mode == "uniform":
                if self.K == 1:
                    starts = [max_start // 2]
                else:
                    starts = np.linspace(0, max_start, num=self.K).astype(int).tolist()
            elif self.sample_mode == "random":
                starts = rng.randint(0, max_start + 1, size=self.K).tolist()
            else:
                raise ValueError(f"Unsupported sample_mode: {self.sample_mode}")

        segs = []
        seg_quality = []
        for s in starts:
            seg = self._slice_segment(arr_500, s)  # [C, L]
            seg = _sanitize_signal_array(seg)
            seg_quality.append(self._compute_segment_quality(seg))
            if self.norm == "zscore":
                mu = seg.mean(axis=-1, keepdims=True)
                sd = seg.std(axis=-1, keepdims=True)
                seg = (seg - mu) / (sd + self.eps)
            elif self.norm == "none":
                pass
            else:
                raise ValueError("norm must be 'zscore' or 'none'")

            seg = _sanitize_signal_array(seg)
            segs.append(_sanitize_signal_array(self._select_leads(seg)))

        ecg_seg = torch.from_numpy(_sanitize_signal_array(np.stack(segs, axis=0))).float()  # (K, C, T)
        y_class = torch.tensor(rec.cls, dtype=torch.long)      # scalar
        y_lvef = torch.tensor(rec.lvef, dtype=torch.float32)   # scalar (可能为 nan)
        seg_quality_tensor = torch.from_numpy(np.stack(seg_quality, axis=0)).float()
        seg_starts_tensor = torch.tensor(starts, dtype=torch.long)
        seg_mask = torch.ones(self.K, dtype=torch.bool)
        xml_file, patient_id = rec.xml_file, rec.patient_id

        if self.return_dict:
            sample = {
                "signal": ecg_seg,
                "label": y_class,
                "lvef": y_lvef,
                "xml_file": xml_file,
                "patient_id": patient_id,
                "seg_mask": seg_mask,
            }
            if self.return_seg_quality:
                sample["seg_quality"] = seg_quality_tensor
            if self.return_seg_starts:
                sample["seg_starts"] = seg_starts_tensor
            return sample

        output = [ecg_seg, y_class, y_lvef, xml_file, patient_id]
        if self.return_seg_quality:
            output.append(seg_quality_tensor)
        if self.return_seg_starts:
            output.append(seg_starts_tensor)
        return tuple(output)


class ECGLeadTransferDataset(Dataset):
    """
    Dataset for richer-lead -> single-lead transfer pretraining.

    The manifest is intentionally more permissive than the downstream HF dataset.
    Required columns:
      - one of: npy_dst, npy_path, xml_path, path, waveform_path, record_path
      - one of: sampling_rate_hz, fs  (or pass default_fs)

    Optional columns:
      - path_format / signal_format / waveform_format
      - n_samples
      - patient_id
      - xml_file / record_id
      - npy_src (used to derive patient_id when present)
    """

    def __init__(
        self,
        csv_path: Path,
        npy_root: Path,
        clip_sec: float = 10.0,
        target_fs: int = 500,
        teacher_lead_mode: str = "12",
        student_lead_mode: str = "I_1ch",
        bandpass: bool = True,
        norm: str = "zscore",
        eps: float = 1e-8,
        deterministic: bool = False,
        base_seed: int = 1234,
        random_crop: bool = True,
        default_fs: Optional[int] = None,
    ):
        self.csv_path = Path(csv_path)
        self.npy_root = Path(npy_root)
        self.clip_sec = float(clip_sec)
        self.target_fs = int(target_fs)
        self.teacher_lead_mode = validate_lead_mode(teacher_lead_mode)
        self.student_lead_mode = validate_lead_mode(student_lead_mode)
        self.bandpass = bool(bandpass)
        self.norm = norm
        self.eps = float(eps)
        self.deterministic = bool(deterministic)
        self.base_seed = int(base_seed)
        self.random_crop = bool(random_crop)
        self.default_fs = default_fs
        self.clip_len = int(round(self.clip_sec * self.target_fs))

        if self.target_fs != 500:
            raise ValueError("Invalid target fs: {}".format(self.target_fs))

        df = pd.read_csv(self.csv_path)
        df.columns = [str(c).strip() for c in df.columns]

        path_col = _first_present_column(df, ["npy_dst", "npy_path", "xml_path", "path", "signal_path", "waveform_path", "file_path", "record_path"])
        format_col = _first_present_column(df, ["path_format", "signal_format", "waveform_format"])
        fs_col = _first_present_column(df, ["sampling_rate_hz", "fs", "sample_rate", "sampling_frequency"])
        n_samples_col = _first_present_column(df, ["n_samples", "num_samples"])
        record_id_col = _first_present_column(df, ["xml_file", "record_id", "study_id", "ecg_id"])
        patient_id_col = _first_present_column(df, ["patient_id", "subject_id"])
        npy_src_col = _first_present_column(df, ["npy_src", "npy_source"])

        if path_col is None:
            raise ValueError("Transfer manifest must include one of: npy_dst, npy_path, xml_path, path, waveform_path, record_path.")
        if fs_col is None and self.default_fs is None:
            raise ValueError("Transfer manifest must include sampling_rate_hz/fs or pass default_fs.")

        if fs_col is not None:
            df[fs_col] = pd.to_numeric(df[fs_col], errors="coerce")
        if n_samples_col is not None:
            df[n_samples_col] = pd.to_numeric(df[n_samples_col], errors="coerce")

        df = df.dropna(subset=[path_col]).copy()
        if fs_col is not None:
            df = df.dropna(subset=[fs_col]).copy()
            df = df[df[fs_col].isin([500, 1000])].copy()

        self.records: List[ECGTransferRecord] = []
        for _, row in df.iterrows():
            fs_value = int(row[fs_col]) if fs_col is not None else int(self.default_fs)
            n_samples_value = None
            if n_samples_col is not None and not pd.isna(row[n_samples_col]):
                n_samples_value = int(row[n_samples_col])

            if patient_id_col is not None and not pd.isna(row[patient_id_col]):
                patient_id = str(row[patient_id_col])
            elif npy_src_col is not None and not pd.isna(row[npy_src_col]):
                patient_id = patient_id_from_numpy(str(row[npy_src_col]))
            else:
                patient_id = patient_id_from_signal_path(str(row[path_col]))

            if record_id_col is not None and not pd.isna(row[record_id_col]):
                record_id = str(row[record_id_col])
            else:
                record_id = Path(str(row[path_col])).stem

            signal_path = resolve_path(str(row[path_col]), self.npy_root)
            signal_format = _infer_signal_format(
                str(row[path_col]),
                explicit_format=None if format_col is None or pd.isna(row[format_col]) else str(row[format_col]),
            )
            if signal_format == "wfdb":
                signal_path = _normalize_wfdb_record_path(signal_path)

            self.records.append(
                ECGTransferRecord(
                    signal_path=signal_path,
                    signal_format=signal_format,
                    fs=fs_value,
                    n_samples=n_samples_value,
                    record_id=record_id,
                    patient_id=patient_id,
                )
            )

        self._bp = BandpassFilter(low_hz=0.5, high_hz=40.0, order=4) if self.bandpass else None

    def __len__(self) -> int:
        return len(self.records)

    def _make_deterministic_rng(self, rec: ECGTransferRecord) -> np.random.RandomState:
        seed_src = "{}|{}|{}".format(rec.record_id, rec.signal_path, self.base_seed)
        seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
        return np.random.RandomState(seed)

    def _slice_clip(self, arr: np.ndarray, start: int) -> np.ndarray:
        end = start + self.clip_len
        clip = arr[:, start:end]
        if clip.shape[-1] == self.clip_len:
            return clip
        padded = np.zeros((arr.shape[0], self.clip_len), dtype=arr.dtype)
        padded[:, :clip.shape[-1]] = clip
        return padded

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr = _sanitize_signal_array(arr)
        if self.norm == "zscore":
            mu = arr.mean(axis=-1, keepdims=True)
            sd = arr.std(axis=-1, keepdims=True)
            return _sanitize_signal_array((arr - mu) / (sd + self.eps))
        if self.norm == "none":
            return _sanitize_signal_array(arr)
        raise ValueError("norm must be 'zscore' or 'none'")

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        rng = self._make_deterministic_rng(rec) if self.deterministic else np.random

        arr, src_fs = _load_signal_12xT(rec.signal_path, rec.signal_format, fallback_fs=rec.fs)
        arr = _sanitize_signal_array(arr)
        arr_500 = ECGMILDataset._resample_to_500(arr, src_fs)

        if self._bp is not None:
            arr_500 = self._bp.apply(arr_500, fs=self.target_fs)
        arr_500 = _sanitize_signal_array(arr_500)

        T = arr_500.shape[-1]
        if T <= self.clip_len:
            start = 0
        elif self.random_crop:
            start = int(rng.randint(0, T - self.clip_len + 1))
        else:
            start = int((T - self.clip_len) // 2)

        clip = self._slice_clip(arr_500, start)
        clip = _sanitize_signal_array(clip)
        teacher = _sanitize_signal_array(self._normalize(select_leads_by_mode(clip, self.teacher_lead_mode)))
        student = _sanitize_signal_array(self._normalize(select_leads_by_mode(clip, self.student_lead_mode)))

        return {
            "teacher_signal": torch.from_numpy(teacher).float(),
            "student_signal": torch.from_numpy(student).float(),
            "record_id": rec.record_id,
            "patient_id": rec.patient_id,
            "clip_start": torch.tensor(start, dtype=torch.long),
        }
