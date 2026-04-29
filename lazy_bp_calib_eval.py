# lazy_bp_calib_eval.py
# -*- coding: utf-8 -*-

"""
Lazy BP raw inference + calibration evaluation.

Designed for your current loop:
    for x, y, meta in test_loader:
        pred, weight = model(x)

Supports:
    1. Raw event-level prediction saving
    2. Calibration point selection:
        - head
        - tail
        - quantile
        - min_gap
        - random
        - random_min_gap
    3. Calibration methods:
        - bias
        - affine ridge
        - residual bank using raw-pred/time/sleep features
    4. Output:
        - event-level CSV
        - metrics JSON

Minimal dependency:
    numpy, pandas, torch
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


# -------------------------
# Basic helpers
# -------------------------

def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def safe_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def me_std_mae(y_pred: Sequence[float], y_true: Sequence[float]) -> Dict[str, float]:
    y_pred = safe_float_array(y_pred)
    y_true = safe_float_array(y_true)
    err = y_pred - y_true

    return {
        "ME": float(np.mean(err)) if len(err) > 0 else float("nan"),
        "STD": float(np.std(err, ddof=1)) if len(err) > 1 else float("nan"),
        "MAE": float(np.mean(np.abs(err))) if len(err) > 0 else float("nan"),
        "N": int(len(err)),
    }


def add_metric(metrics: Dict[str, Any], name: str, y_pred, y_true) -> None:
    m = me_std_mae(y_pred, y_true)
    for k, v in m.items():
        metrics[f"{name}_{k}"] = v


def stable_softmax(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    score = score - np.max(score)
    exp_score = np.exp(score)
    denom = np.sum(exp_score)
    if denom <= 1e-12:
        return np.ones_like(score) / len(score)
    return exp_score / denom


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


# -------------------------
# Meta parsing
# -------------------------

def _one_meta_value(v: Any, i: int) -> Any:
    """
    Extract one sample value from meta field.
    Handles tensor/list/np/scalar.
    """
    if isinstance(v, torch.Tensor):
        vv = v.detach().cpu()
        if vv.ndim == 0:
            return vv.item()
        return vv[i].item() if vv[i].numel() == 1 else vv[i].numpy()

    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        item = v[i]
        if isinstance(item, np.generic):
            return item.item()
        return item

    if isinstance(v, (list, tuple)):
        return v[i]

    return v


def meta_to_rows(meta: Any, batch_size: int, start_index: int) -> List[Dict[str, Any]]:
    """
    Convert meta to list-of-dict rows.

    Supported meta formats:
        1. dict of batched fields:
            meta = {"id_clean": [...], "t_bp_ms": tensor([...]), "sleep": [...]}
        2. list of dict:
            meta = [{"id_clean": ..., "t_bp_ms": ...}, ...]
        3. None / unsupported:
            creates dummy rows
    """
    rows: List[Dict[str, Any]] = []

    if isinstance(meta, dict):
        for i in range(batch_size):
            row = {}
            for k, v in meta.items():
                row[k] = _one_meta_value(v, i)
            rows.append(row)
        return rows

    if isinstance(meta, (list, tuple)) and len(meta) == batch_size and isinstance(meta[0], dict):
        return [dict(m) for m in meta]

    # fallback
    for i in range(batch_size):
        rows.append({
            "row_id": start_index + i,
            "id_clean": "unknown",
            "t_bp_ms": start_index + i,
            "sleep": 0,
        })

    return rows


def standardize_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to standardize common column names.
    """
    out = df.copy()

    # id
    if "id_clean" not in out.columns:
        for c in ["id", "subject", "subject_id", "pid", "patient_id"]:
            if c in out.columns:
                out["id_clean"] = out[c].astype(str)
                break
    if "id_clean" not in out.columns:
        out["id_clean"] = "unknown"

    # time
    if "t_bp_ms" not in out.columns:
        for c in ["time_ms", "timestamp", "timestamp_ms", "bp_time_ms"]:
            if c in out.columns:
                out["t_bp_ms"] = out[c]
                break
    if "t_bp_ms" not in out.columns:
        out["t_bp_ms"] = np.arange(len(out), dtype=np.int64)

    # sleep
    if "sleep" not in out.columns:
        out["sleep"] = 0

    out["id_clean"] = out["id_clean"].astype(str)
    out["t_bp_ms"] = pd.to_numeric(out["t_bp_ms"], errors="coerce")
    out["sleep"] = pd.to_numeric(out["sleep"], errors="coerce").fillna(0).astype(int)

    return out


# -------------------------
# Raw inference
# -------------------------

@torch.no_grad()
def run_raw_inference(
    model: torch.nn.Module,
    test_loader,
    device: str = "cuda",
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Run raw model inference.

    Accepts model outputs:
        pred, weight
        pred, weight, emb
        dict with keys: pred/y, weight, emb/rep/embedding

    Returns:
        df: event-level dataframe
        emb: optional [N, D] embedding if model provides it
    """
    model.eval()
    model.to(device)

    all_rows: List[Dict[str, Any]] = []
    all_pred: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    all_weight: List[np.ndarray] = []
    all_emb: List[np.ndarray] = []

    start_index = 0

    for batch in test_loader:
        if len(batch) == 3:
            x, y, meta = batch
        elif len(batch) == 2:
            x, y = batch
            meta = None
        else:
            raise ValueError("Expected batch = (x, y, meta) or (x, y).")

        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        out = model(x)

        pred = None
        weight = None
        emb = None

        if isinstance(out, dict):
            pred = out.get("pred", out.get("y", out.get("bp", None)))
            weight = out.get("weight", out.get("attn", out.get("attention", None)))
            emb = out.get("emb", out.get("rep", out.get("embedding", out.get("feat", None))))
        elif isinstance(out, (tuple, list)):
            if len(out) >= 1:
                pred = out[0]
            if len(out) >= 2:
                weight = out[1]
            if len(out) >= 3:
                emb = out[2]
        else:
            pred = out

        if pred is None:
            raise ValueError("Cannot find pred from model output.")

        pred_np = to_numpy(pred)
        y_np = to_numpy(y)

        batch_size = pred_np.shape[0]
        rows = meta_to_rows(meta, batch_size=batch_size, start_index=start_index)

        all_rows.extend(rows)
        all_pred.append(pred_np)
        all_true.append(y_np)

        if weight is not None:
            all_weight.append(to_numpy(weight))
        else:
            all_weight.append(np.full((batch_size, 1), np.nan, dtype=np.float32))

        if emb is not None:
            emb_np = to_numpy(emb)
            if emb_np.ndim > 2:
                emb_np = emb_np.reshape(emb_np.shape[0], -1)
            all_emb.append(emb_np)

        start_index += batch_size

    pred_all = np.concatenate(all_pred, axis=0)
    true_all = np.concatenate(all_true, axis=0)
    weight_all = np.concatenate(all_weight, axis=0)

    df = pd.DataFrame(all_rows)
    df = standardize_required_columns(df)

    df["row_id"] = np.arange(len(df))
    df["y_true_sbp"] = true_all[:, 0]
    df["y_true_dbp"] = true_all[:, 1]
    df["y_pred_sbp_raw"] = pred_all[:, 0]
    df["y_pred_dbp_raw"] = pred_all[:, 1]

    # Store attention weights if available.
    for k in range(weight_all.shape[1]):
        df[f"attn_w{k}"] = weight_all[:, k]

    if len(all_emb) > 0:
        emb_all = np.concatenate(all_emb, axis=0)
    else:
        emb_all = None

    return df, emb_all


# -------------------------
# Calibration point selection
# -------------------------

def _quantile_positions(n: int, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    if n <= k:
        return np.arange(n, dtype=int)
    pos = np.linspace(0, n - 1, k)
    pos = np.round(pos).astype(int)
    # unique while keeping order
    seen = set()
    out = []
    for p in pos:
        if int(p) not in seen:
            out.append(int(p))
            seen.add(int(p))
    # fill if rounding produced duplicate
    for p in range(n):
        if len(out) >= k:
            break
        if p not in seen:
            out.append(p)
            seen.add(p)
    return np.asarray(out[:k], dtype=int)


def _min_gap_greedy_positions(times: np.ndarray, k: int, min_gap_ms: int) -> np.ndarray:
    """
    Greedy earliest selection with minimum time gap.
    Fallback fills by quantile if not enough.
    """
    n = len(times)
    if n <= k:
        return np.arange(n, dtype=int)

    chosen = []
    last_t = None

    for i, t in enumerate(times):
        if last_t is None or int(t) - int(last_t) >= min_gap_ms:
            chosen.append(i)
            last_t = int(t)
        if len(chosen) >= k:
            break

    if len(chosen) < k:
        qpos = _quantile_positions(n, k)
        for p in qpos:
            if int(p) not in chosen:
                chosen.append(int(p))
            if len(chosen) >= k:
                break

    return np.asarray(chosen[:k], dtype=int)


def _random_min_gap_positions(
    times: np.ndarray,
    k: int,
    min_gap_ms: int,
    rng: np.random.Generator,
    max_trials: int = 500,
) -> np.ndarray:
    """
    Randomly select k points satisfying min-gap if possible.
    If impossible, fallback to min_gap greedy.
    """
    n = len(times)
    if n <= k:
        return np.arange(n, dtype=int)

    indices = np.arange(n)

    best = None
    for _ in range(max_trials):
        perm = rng.permutation(indices)
        chosen = []
        chosen_times = []

        for p in perm:
            t = int(times[p])
            ok = True
            for ct in chosen_times:
                if abs(t - ct) < min_gap_ms:
                    ok = False
                    break
            if ok:
                chosen.append(int(p))
                chosen_times.append(t)
            if len(chosen) >= k:
                return np.asarray(sorted(chosen), dtype=int)

        if best is None or len(chosen) > len(best):
            best = chosen

    if best is not None and len(best) >= k:
        return np.asarray(sorted(best[:k]), dtype=int)

    return _min_gap_greedy_positions(times, k, min_gap_ms)


def select_calibration_points(
    df: pd.DataFrame,
    n_calib: int = 7,
    strategy: str = "quantile",
    id_col: str = "id_clean",
    time_col: str = "t_bp_ms",
    min_gap_minutes: float = 30.0,
    seed: int = 42,
    existing_is_calib_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add column:
        is_calib

    Strategies:
        head:
            first n_calib points per subject
        tail:
            last n_calib points per subject
        quantile:
            evenly spread n_calib points across time
        min_gap:
            earliest points with minimum time gap, fallback quantile
        random:
            random n_calib points
        random_min_gap:
            random n_calib points with minimum gap if possible
    """
    out = df.copy()
    out["is_calib"] = False

    if existing_is_calib_col is not None and existing_is_calib_col in out.columns:
        out["is_calib"] = out[existing_is_calib_col].astype(bool)
        return out

    rng = np.random.default_rng(seed)
    min_gap_ms = int(min_gap_minutes * 60 * 1000)

    for _, g in out.groupby(id_col, sort=False):
        g = g.sort_values(time_col)
        idx = g.index.to_numpy()
        n = len(idx)

        if n == 0:
            continue

        k = min(int(n_calib), n)
        times = g[time_col].to_numpy(dtype=np.int64)

        if strategy == "head":
            pos = np.arange(k, dtype=int)

        elif strategy == "tail":
            pos = np.arange(n - k, n, dtype=int)

        elif strategy == "quantile":
            pos = _quantile_positions(n, k)

        elif strategy == "min_gap":
            pos = _min_gap_greedy_positions(times, k, min_gap_ms)

        elif strategy == "random":
            pos = np.sort(rng.choice(np.arange(n), size=k, replace=False))

        elif strategy == "random_min_gap":
            pos = _random_min_gap_positions(times, k, min_gap_ms, rng)

        else:
            raise ValueError(
                f"Unknown strategy={strategy}. "
                f"Use head/tail/quantile/min_gap/random/random_min_gap."
            )

        chosen_idx = idx[pos]
        out.loc[chosen_idx, "is_calib"] = True

    return out


# -------------------------
# Bias calibration
# -------------------------

def apply_bias_calibration(
    df: pd.DataFrame,
    by_sleep: bool = False,
    min_points: int = 1,
) -> pd.DataFrame:
    """
    y_cal = y_raw + mean(y_true - y_raw)
    """
    out = df.copy()
    out["y_pred_sbp_bias"] = out["y_pred_sbp_raw"].astype(float)
    out["y_pred_dbp_bias"] = out["y_pred_dbp_raw"].astype(float)

    calib = out[out["is_calib"].astype(bool)].copy()
    subj_calib = {pid: g for pid, g in calib.groupby("id_clean", sort=False)}

    if by_sleep:
        groups = out.groupby(["id_clean", "sleep"], sort=False)
    else:
        groups = out.groupby(["id_clean"], sort=False)

    for key, g_all in groups:
        if by_sleep:
            pid, slp = key
            g_cal = calib[(calib["id_clean"] == pid) & (calib["sleep"] == slp)]
        else:
            pid = key[0] if isinstance(key, tuple) else key
            g_cal = calib[calib["id_clean"] == pid]

        if len(g_cal) < min_points:
            g_cal = subj_calib.get(pid, pd.DataFrame())

        if len(g_cal) < min_points:
            continue

        sbp_bias = float((g_cal["y_true_sbp"] - g_cal["y_pred_sbp_raw"]).mean())
        dbp_bias = float((g_cal["y_true_dbp"] - g_cal["y_pred_dbp_raw"]).mean())

        idx = g_all.index
        out.loc[idx, "y_pred_sbp_bias"] = out.loc[idx, "y_pred_sbp_raw"].astype(float) + sbp_bias
        out.loc[idx, "y_pred_dbp_bias"] = out.loc[idx, "y_pred_dbp_raw"].astype(float) + dbp_bias

    return out


# -------------------------
# Affine calibration
# -------------------------

def fit_affine_ridge(
    x: Sequence[float],
    y: Sequence[float],
    lam: float = 100.0,
    penalize_intercept: bool = False,
    slope_center: float = 1.0,
) -> Tuple[float, float]:
    """
    Fit y = a*x + b with ridge on slope around slope_center.

    Objective:
        ||Xa - y||^2 + lam * (a - slope_center)^2

    Intercept is not penalized by default.
    """
    x = safe_float_array(x)
    y = safe_float_array(y)

    if len(x) < 2:
        # fallback to bias-only
        b = float(np.mean(y - x)) if len(x) > 0 else 0.0
        return 1.0, b

    X = np.stack([x, np.ones_like(x)], axis=1)  # [N,2]

    # Penalize slope around slope_center by solving shifted ridge.
    # Let a' = a - slope_center, y' = y - slope_center*x
    y_shift = y - slope_center * x
    X_shift = X.copy()
    X_shift[:, 0] = x

    R = np.zeros((2, 2), dtype=np.float64)
    R[0, 0] = lam
    if penalize_intercept:
        R[1, 1] = lam

    try:
        w = np.linalg.solve(X_shift.T @ X_shift + R, X_shift.T @ y_shift)
        a = float(w[0] + slope_center)
        b = float(w[1])
    except np.linalg.LinAlgError:
        b = float(np.mean(y - x))
        a = 1.0

    return a, b


def apply_affine_calibration(
    df: pd.DataFrame,
    by_sleep: bool = False,
    min_points: int = 3,
    lam: float = 100.0,
    penalize_intercept: bool = False,
    slope_clip: Optional[Tuple[float, float]] = (0.65, 1.50),
) -> pd.DataFrame:
    """
    y_cal = a*y_raw + b
    """
    out = df.copy()
    out["y_pred_sbp_aff"] = out["y_pred_sbp_raw"].astype(float)
    out["y_pred_dbp_aff"] = out["y_pred_dbp_raw"].astype(float)

    calib = out[out["is_calib"].astype(bool)].copy()
    subj_calib = {pid: g for pid, g in calib.groupby("id_clean", sort=False)}

    if by_sleep:
        groups = out.groupby(["id_clean", "sleep"], sort=False)
    else:
        groups = out.groupby(["id_clean"], sort=False)

    for key, g_all in groups:
        if by_sleep:
            pid, slp = key
            g_cal = calib[(calib["id_clean"] == pid) & (calib["sleep"] == slp)]
        else:
            pid = key[0] if isinstance(key, tuple) else key
            g_cal = calib[calib["id_clean"] == pid]

        if len(g_cal) < min_points:
            g_cal = subj_calib.get(pid, pd.DataFrame())

        if len(g_cal) < 1:
            continue

        a_s, b_s = fit_affine_ridge(
            g_cal["y_pred_sbp_raw"],
            g_cal["y_true_sbp"],
            lam=lam,
            penalize_intercept=penalize_intercept,
        )
        a_d, b_d = fit_affine_ridge(
            g_cal["y_pred_dbp_raw"],
            g_cal["y_true_dbp"],
            lam=lam,
            penalize_intercept=penalize_intercept,
        )

        if slope_clip is not None:
            lo, hi = slope_clip
            a_s = float(np.clip(a_s, lo, hi))
            a_d = float(np.clip(a_d, lo, hi))

        idx = g_all.index
        out.loc[idx, "y_pred_sbp_aff"] = a_s * out.loc[idx, "y_pred_sbp_raw"].astype(float) + b_s
        out.loc[idx, "y_pred_dbp_aff"] = a_d * out.loc[idx, "y_pred_dbp_raw"].astype(float) + b_d

        out.loc[idx, "aff_sbp_a"] = a_s
        out.loc[idx, "aff_sbp_b"] = b_s
        out.loc[idx, "aff_dbp_a"] = a_d
        out.loc[idx, "aff_dbp_b"] = b_d

    return out


# -------------------------
# Lazy residual bank calibration
# -------------------------

def build_lazy_bank_features(
    df: pd.DataFrame,
    emb: Optional[np.ndarray] = None,
    use_raw_pred: bool = True,
    use_time: bool = False,
    use_sleep: bool = True,
) -> np.ndarray:
    """
    Build query/reference features.

    Lazy mode:
        If emb is None, use [raw_sbp, raw_dbp, sleep] as feature.
        This requires zero model modification.

    Better mode:
        If emb is provided, concatenate emb + optional raw/sleep.
    """
    feats = []

    if emb is not None:
        e = np.asarray(emb, dtype=np.float64)
        if e.ndim > 2:
            e = e.reshape(e.shape[0], -1)
        feats.append(e)

    if use_raw_pred:
        raw = df[["y_pred_sbp_raw", "y_pred_dbp_raw"]].to_numpy(dtype=np.float64)
        # scale BP roughly
        raw = raw / np.array([[120.0, 80.0]], dtype=np.float64)
        feats.append(raw)

    if use_sleep and "sleep" in df.columns:
        slp = df[["sleep"]].to_numpy(dtype=np.float64)
        feats.append(slp)

    if use_time:
        # Usually not recommended as direct feature, time is separately penalized.
        t = df[["t_bp_ms"]].to_numpy(dtype=np.float64)
        t = (t - np.nanmean(t)) / (np.nanstd(t) + 1e-6)
        feats.append(t)

    if len(feats) == 0:
        raise ValueError("No bank features available.")

    feat = np.concatenate(feats, axis=1)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    feat = l2_normalize(feat)
    return feat


def apply_residual_bank_calibration(
    df: pd.DataFrame,
    emb: Optional[np.ndarray] = None,
    by_sleep_fallback: bool = False,
    temperature: float = 0.30,
    time_weight_per_hour: float = 0.20,
    sleep_bonus: float = 0.25,
    uniform_mix: float = 0.20,
    residual_clip: float = 35.0,
    min_points: int = 1,
) -> pd.DataFrame:
    """
    Reference-bank residual calibration.

    For query q:
        residual_j = y_true_j - y_raw_j for calibration point j
        y_cal_q = y_raw_q + sum_j w_qj * residual_j

    Weights use:
        feature cosine similarity
        time distance penalty
        sleep consistency bonus

    This is the lazy version of "reference-conditioned calibration".
    """
    out = df.copy()
    out["y_pred_sbp_bank"] = out["y_pred_sbp_raw"].astype(float)
    out["y_pred_dbp_bank"] = out["y_pred_dbp_raw"].astype(float)
    out["bank_max_w"] = np.nan
    out["bank_eff_n"] = np.nan

    feat = build_lazy_bank_features(out, emb=emb, use_raw_pred=True, use_sleep=True)

    calib_all = out[out["is_calib"].astype(bool)].copy()
    subj_calib_idx = {
        pid: g.index.to_numpy()
        for pid, g in calib_all.groupby("id_clean", sort=False)
    }

    for pid, g_all in out.groupby("id_clean", sort=False):
        idx_all = g_all.index.to_numpy()
        idx_query = g_all.index[~g_all["is_calib"].astype(bool)].to_numpy()

        if len(idx_query) == 0:
            continue

        idx_calib_base = subj_calib_idx.get(pid, np.array([], dtype=int))
        if len(idx_calib_base) < min_points:
            continue

        for qi in idx_query:
            idx_calib = idx_calib_base

            if by_sleep_fallback and "sleep" in out.columns:
                same_sleep = idx_calib_base[
                    out.loc[idx_calib_base, "sleep"].to_numpy()
                    == out.loc[qi, "sleep"]
                ]
                if len(same_sleep) >= min_points:
                    idx_calib = same_sleep

            if len(idx_calib) < min_points:
                continue

            q_feat = feat[qi:qi + 1]
            c_feat = feat[idx_calib]

            sim = (q_feat @ c_feat.T).reshape(-1)

            tq = float(out.loc[qi, "t_bp_ms"])
            tc = out.loc[idx_calib, "t_bp_ms"].to_numpy(dtype=np.float64)
            dt_h = np.abs(tq - tc) / 3600000.0

            score = sim / max(temperature, 1e-6)
            score = score - time_weight_per_hour * dt_h

            if "sleep" in out.columns:
                sq = int(out.loc[qi, "sleep"])
                sc = out.loc[idx_calib, "sleep"].to_numpy(dtype=int)
                score = score + sleep_bonus * (sc == sq).astype(np.float64)

            w = stable_softmax(score)

            if uniform_mix > 0:
                u = np.ones_like(w) / len(w)
                w = (1.0 - uniform_mix) * w + uniform_mix * u

            true_cal = out.loc[idx_calib, ["y_true_sbp", "y_true_dbp"]].to_numpy(dtype=np.float64)
            pred_cal = out.loc[idx_calib, ["y_pred_sbp_raw", "y_pred_dbp_raw"]].to_numpy(dtype=np.float64)
            residual = true_cal - pred_cal

            if residual_clip is not None and residual_clip > 0:
                residual = np.clip(residual, -residual_clip, residual_clip)

            delta = w @ residual

            raw_q = out.loc[qi, ["y_pred_sbp_raw", "y_pred_dbp_raw"]].to_numpy(dtype=np.float64)
            pred_q = raw_q + delta

            out.loc[qi, "y_pred_sbp_bank"] = pred_q[0]
            out.loc[qi, "y_pred_dbp_bank"] = pred_q[1]
            out.loc[qi, "bank_max_w"] = float(np.max(w))
            out.loc[qi, "bank_eff_n"] = float(1.0 / np.sum(w ** 2))

    return out


# -------------------------
# Natural-hour aggregation
# -------------------------

def build_state_aware_natural_hour_groups(
    df_id: pd.DataFrame,
    pred_sbp_col: str,
    pred_dbp_col: str,
    window_minutes: int = 60,
    min_events_per_hour: int = 2,
    separate_sleep: bool = True,
) -> List[Dict[str, Any]]:
    """
    Non-overlapping natural-hour grouping.

    If separate_sleep=True:
        group stops when sleep state changes.
    """
    window_ms = int(window_minutes * 60 * 1000)
    g = df_id.sort_values("t_bp_ms").reset_index(drop=False)
    rows = []

    i = 0
    n = len(g)

    while i < n:
        t0 = int(g.loc[i, "t_bp_ms"])
        sleep0 = int(g.loc[i, "sleep"]) if "sleep" in g.columns else 0
        tend = t0 + window_ms

        j = i
        while j + 1 < n:
            t_next = int(g.loc[j + 1, "t_bp_ms"])
            if t_next > tend:
                break

            if separate_sleep and "sleep" in g.columns:
                sleep_next = int(g.loc[j + 1, "sleep"])
                if sleep_next != sleep0:
                    break

            j += 1

        block = g.loc[i:j].copy()

        if len(block) >= min_events_per_hour:
            rows.append({
                "id_clean": str(block["id_clean"].iloc[0]),
                "sleep": int(sleep0),
                "hour_start_ms": int(t0),
                "hour_end_ms": int(tend),
                "n_event": int(len(block)),
                "y_true_sbp_h": float(block["y_true_sbp"].mean()),
                "y_true_dbp_h": float(block["y_true_dbp"].mean()),
                "y_pred_sbp_h": float(block[pred_sbp_col].mean()),
                "y_pred_dbp_h": float(block[pred_dbp_col].mean()),
            })

        i = j + 1

    return rows


def evaluate_hourly(
    df: pd.DataFrame,
    pred_sbp_col: str,
    pred_dbp_col: str,
    min_events_per_hour: int = 2,
    separate_sleep: bool = True,
    eval_non_calib_only: bool = True,
) -> Dict[str, Any]:
    if eval_non_calib_only:
        d = df[~df["is_calib"].astype(bool)].copy()
    else:
        d = df.copy()

    rows = []
    for _, g in d.groupby("id_clean", sort=False):
        rows.extend(
            build_state_aware_natural_hour_groups(
                g,
                pred_sbp_col=pred_sbp_col,
                pred_dbp_col=pred_dbp_col,
                min_events_per_hour=min_events_per_hour,
                separate_sleep=separate_sleep,
            )
        )

    h = pd.DataFrame(rows)
    metrics = {
        "hourly_N": int(len(h)),
        "hourly_min_events_per_hour": int(min_events_per_hour),
    }

    if len(h) > 0:
        add_metric(metrics, "hourly_sbp", h["y_pred_sbp_h"], h["y_true_sbp_h"])
        add_metric(metrics, "hourly_dbp", h["y_pred_dbp_h"], h["y_true_dbp_h"])

    return metrics


# -------------------------
# Full lazy pipeline
# -------------------------

def evaluate_event_level_all_methods(df: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    eval_df = df[~df["is_calib"].astype(bool)].copy()

    metrics["event_eval_N"] = int(len(eval_df))
    metrics["event_calib_N"] = int(df["is_calib"].sum())
    metrics["subject_N"] = int(df["id_clean"].nunique())

    add_metric(metrics, "raw_sbp", eval_df["y_pred_sbp_raw"], eval_df["y_true_sbp"])
    add_metric(metrics, "raw_dbp", eval_df["y_pred_dbp_raw"], eval_df["y_true_dbp"])

    if "y_pred_sbp_bias" in eval_df.columns:
        add_metric(metrics, "bias_sbp", eval_df["y_pred_sbp_bias"], eval_df["y_true_sbp"])
        add_metric(metrics, "bias_dbp", eval_df["y_pred_dbp_bias"], eval_df["y_true_dbp"])

    if "y_pred_sbp_aff" in eval_df.columns:
        add_metric(metrics, "aff_sbp", eval_df["y_pred_sbp_aff"], eval_df["y_true_sbp"])
        add_metric(metrics, "aff_dbp", eval_df["y_pred_dbp_aff"], eval_df["y_true_dbp"])

    if "y_pred_sbp_bank" in eval_df.columns:
        add_metric(metrics, "bank_sbp", eval_df["y_pred_sbp_bank"], eval_df["y_true_sbp"])
        add_metric(metrics, "bank_dbp", eval_df["y_pred_dbp_bank"], eval_df["y_true_dbp"])

        if "bank_eff_n" in eval_df.columns:
            metrics["bank_eff_n_mean"] = float(np.nanmean(eval_df["bank_eff_n"]))
            metrics["bank_max_w_mean"] = float(np.nanmean(eval_df["bank_max_w"]))

    return metrics


def run_lazy_bp_eval_and_calibration(
    model: torch.nn.Module,
    test_loader,
    device: str = "cuda",
    out_csv: str = "predictions_lazy_calib.csv",
    out_json: str = "metrics_lazy_calib.json",

    # calibration selection
    n_calib: int = 7,
    calib_strategy: str = "quantile",
    min_gap_minutes: float = 30.0,
    seed: int = 42,
    existing_is_calib_col: Optional[str] = None,

    # methods
    run_bias: bool = True,
    run_affine: bool = True,
    run_bank: bool = True,

    # bias / affine config
    by_sleep: bool = False,
    affine_lam: float = 100.0,
    affine_min_points: int = 3,

    # bank config
    bank_temperature: float = 0.30,
    bank_time_weight_per_hour: float = 0.20,
    bank_sleep_bonus: float = 0.25,
    bank_uniform_mix: float = 0.20,
    bank_residual_clip: float = 35.0,

    # hourly eval
    run_hourly: bool = True,
    min_events_per_hour: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    One-call lazy API.
    """
    df, emb = run_raw_inference(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    df = select_calibration_points(
        df,
        n_calib=n_calib,
        strategy=calib_strategy,
        min_gap_minutes=min_gap_minutes,
        seed=seed,
        existing_is_calib_col=existing_is_calib_col,
    )

    if run_bias:
        df = apply_bias_calibration(
            df,
            by_sleep=by_sleep,
            min_points=1,
        )

    if run_affine:
        df = apply_affine_calibration(
            df,
            by_sleep=by_sleep,
            min_points=affine_min_points,
            lam=affine_lam,
            slope_clip=(0.65, 1.50),
        )

    if run_bank:
        df = apply_residual_bank_calibration(
            df,
            emb=emb,
            by_sleep_fallback=False,
            temperature=bank_temperature,
            time_weight_per_hour=bank_time_weight_per_hour,
            sleep_bonus=bank_sleep_bonus,
            uniform_mix=bank_uniform_mix,
            residual_clip=bank_residual_clip,
            min_points=1,
        )

    metrics = evaluate_event_level_all_methods(df)

    metrics["calib_strategy"] = calib_strategy
    metrics["n_calib"] = int(n_calib)
    metrics["min_gap_minutes"] = float(min_gap_minutes)
    metrics["affine_lam"] = float(affine_lam)
    metrics["bank_temperature"] = float(bank_temperature)
    metrics["bank_time_weight_per_hour"] = float(bank_time_weight_per_hour)
    metrics["bank_sleep_bonus"] = float(bank_sleep_bonus)
    metrics["bank_uniform_mix"] = float(bank_uniform_mix)
    metrics["bank_residual_clip"] = float(bank_residual_clip)
    metrics["has_embedding"] = emb is not None

    if run_hourly:
        hourly_methods = {
            "raw": ("y_pred_sbp_raw", "y_pred_dbp_raw"),
        }
        if run_bias:
            hourly_methods["bias"] = ("y_pred_sbp_bias", "y_pred_dbp_bias")
        if run_affine:
            hourly_methods["aff"] = ("y_pred_sbp_aff", "y_pred_dbp_aff")
        if run_bank:
            hourly_methods["bank"] = ("y_pred_sbp_bank", "y_pred_dbp_bank")

        for name, (sbp_col, dbp_col) in hourly_methods.items():
            hm = evaluate_hourly(
                df,
                pred_sbp_col=sbp_col,
                pred_dbp_col=dbp_col,
                min_events_per_hour=min_events_per_hour,
                separate_sleep=True,
                eval_non_calib_only=True,
            )
            for k, v in hm.items():
                metrics[f"{name}_{k}"] = v

    df.to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    return df, metrics