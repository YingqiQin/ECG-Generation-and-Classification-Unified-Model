# lazy_bp_product_eval.py
# -*- coding: utf-8 -*-
"""
Lazy PPG-BP product scenario evaluation.

For your current loop:
    for x, y, meta in test_loader:
        pred, weight = model(x)

Use patterns:
    1) summary, dfs = run_lazy_product_eval(model, test_loader, device="cuda")
    2) summary, dfs = run_product_eval_from_saved_raw("raw_predictions.csv", emb_path=None)

The file supports:
    - raw event-level inference
    - product scenarios: 4+3, 2+2, 1+1, single calibration, day-only, day/night quota
    - calibration point selection: head/tail/quantile/min_gap/random/random_min_gap
    - calibration methods: bias / affine ridge / residual bank
    - evaluation: event / natural-hour / 24h macro mean/std
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # allows pure CSV post-processing without torch installed
    torch = None


# =============================================================================
# 0. Basic utilities
# =============================================================================

def _ensure_dir(path: str | Path) -> None:
    p = Path(path)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)


def to_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def safe_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def stable_softmax(score: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    if len(score) == 0:
        return score
    score = score - np.nanmax(score)
    exp_score = np.exp(np.nan_to_num(score, nan=-1e9))
    denom = float(np.sum(exp_score))
    if denom <= 1e-12 or not np.isfinite(denom):
        return np.ones_like(score, dtype=np.float64) / max(len(score), 1)
    return exp_score / denom


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def me_std_mae(y_pred: Sequence[float], y_true: Sequence[float]) -> Dict[str, float]:
    y_pred = safe_float_array(y_pred)
    y_true = safe_float_array(y_true)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    err = y_pred - y_true
    if len(err) == 0:
        return {"ME": np.nan, "STD": np.nan, "MAE": np.nan, "N": 0}
    return {
        "ME": float(np.mean(err)),
        "STD": float(np.std(err, ddof=1)) if len(err) > 1 else np.nan,
        "MAE": float(np.mean(np.abs(err))),
        "N": int(len(err)),
    }


def _prefix_metrics(dst: Dict[str, Any], prefix: str, stats: Dict[str, Any]) -> None:
    for k, v in stats.items():
        dst[f"{prefix}_{k}"] = v


def add_bp_metrics(
    dst: Dict[str, Any],
    prefix: str,
    df: pd.DataFrame,
    pred_sbp_col: str,
    pred_dbp_col: str,
    true_sbp_col: str = "y_true_sbp",
    true_dbp_col: str = "y_true_dbp",
) -> None:
    if pred_sbp_col not in df.columns or pred_dbp_col not in df.columns:
        return
    _prefix_metrics(dst, f"{prefix}_sbp", me_std_mae(df[pred_sbp_col], df[true_sbp_col]))
    _prefix_metrics(dst, f"{prefix}_dbp", me_std_mae(df[pred_dbp_col], df[true_dbp_col]))


# =============================================================================
# 1. Product scenario abstraction
# =============================================================================

@dataclass
class ProductScenario:
    """
    Product-level calibration and evaluation definition.

    sleep convention:
        sleep = 0 -> day
        sleep = 1 -> night/sleep
    """
    name: str

    # Calibration budget.
    calib_total: int = 7
    support_n: int = 0
    update_n: int = 0

    # Point selection.
    calib_strategy: str = "quantile"  # head/tail/quantile/min_gap/random/random_min_gap
    min_gap_minutes: float = 30.0
    seed: int = 42

    # Calibration pool.
    calib_sleep: str = "all"  # all/day/night
    eval_sleep: str = "all"   # all/day/night
    sleep_quota: Optional[Dict[int, int]] = None
    # Example: {0: 4, 1: 3} -> force 4 day calibration points and 3 night points.

    # Evaluation target.
    eval_mode: str = "both"  # event/hourly/macro24/both/all
    min_events_per_hour: int = 2
    macro_window_hours: int = 24
    min_events_per_macro: int = 6

    # Evaluation fairness.
    exclude_calib_from_eval: bool = True
    separate_sleep_in_hourly: bool = True


def default_product_scenarios(seed: int = 42) -> List[ProductScenario]:
    """Ready-to-use presets for common product settings."""
    return [
        ProductScenario(
            name="4p3_all_hourly",
            calib_total=7,
            support_n=4,
            update_n=3,
            calib_strategy="quantile",
            min_gap_minutes=30,
            seed=seed,
            calib_sleep="all",
            eval_sleep="all",
            eval_mode="hourly",
            min_events_per_hour=2,
        ),
        ProductScenario(
            name="4p3_daynight_hourly",
            calib_total=7,
            support_n=4,
            update_n=3,
            calib_strategy="min_gap",
            min_gap_minutes=30,
            seed=seed,
            sleep_quota={0: 4, 1: 3},
            eval_sleep="all",
            eval_mode="hourly",
            min_events_per_hour=2,
        ),
        ProductScenario(
            name="2p2_macro24",
            calib_total=4,
            support_n=2,
            update_n=2,
            calib_strategy="quantile",
            min_gap_minutes=120,
            seed=seed,
            calib_sleep="all",
            eval_sleep="all",
            eval_mode="macro24",
            macro_window_hours=24,
            min_events_per_macro=6,
        ),
        ProductScenario(
            name="day_1p1_hourly",
            calib_total=2,
            support_n=1,
            update_n=1,
            calib_strategy="min_gap",
            min_gap_minutes=60,
            seed=seed,
            calib_sleep="day",
            eval_sleep="day",
            eval_mode="hourly",
            min_events_per_hour=2,
        ),
        ProductScenario(
            name="one_calib_macro24",
            calib_total=1,
            support_n=1,
            update_n=0,
            calib_strategy="head",
            min_gap_minutes=0,
            seed=seed,
            calib_sleep="all",
            eval_sleep="all",
            eval_mode="macro24",
            macro_window_hours=24,
            min_events_per_macro=6,
        ),
        ProductScenario(
            name="day_one_calib_hourly",
            calib_total=1,
            support_n=1,
            update_n=0,
            calib_strategy="head",
            min_gap_minutes=0,
            seed=seed,
            calib_sleep="day",
            eval_sleep="day",
            eval_mode="hourly",
            min_events_per_hour=2,
        ),
    ]


# =============================================================================
# 2. Meta parsing and raw inference
# =============================================================================

def _one_meta_value(v: Any, i: int) -> Any:
    if torch is not None and isinstance(v, torch.Tensor):
        vv = v.detach().cpu()
        if vv.ndim == 0:
            return vv.item()
        item = vv[i]
        if item.numel() == 1:
            return item.item()
        return item.numpy()
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
    Supported meta formats:
        dict of batched fields OR list[dict].
    Fallback creates dummy id/time/sleep fields.
    """
    if isinstance(meta, dict):
        rows = []
        for i in range(batch_size):
            row = {}
            for k, v in meta.items():
                row[k] = _one_meta_value(v, i)
            rows.append(row)
        return rows

    if isinstance(meta, (list, tuple)) and len(meta) == batch_size and isinstance(meta[0], dict):
        return [dict(m) for m in meta]

    return [
        {"row_id": start_index + i, "id_clean": "unknown", "t_bp_ms": start_index + i, "sleep": 0}
        for i in range(batch_size)
    ]


def standardize_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "id_clean" not in out.columns:
        for c in ["id", "subject", "subject_id", "pid", "patient_id"]:
            if c in out.columns:
                out["id_clean"] = out[c].astype(str)
                break
    if "id_clean" not in out.columns:
        out["id_clean"] = "unknown"

    if "t_bp_ms" not in out.columns:
        for c in ["time_ms", "timestamp_ms", "bp_time_ms", "t_ms", "timestamp"]:
            if c in out.columns:
                out["t_bp_ms"] = out[c]
                break
    if "t_bp_ms" not in out.columns:
        out["t_bp_ms"] = np.arange(len(out), dtype=np.int64)

    if "sleep" not in out.columns:
        out["sleep"] = 0

    out["id_clean"] = out["id_clean"].astype(str)
    out["t_bp_ms"] = pd.to_numeric(out["t_bp_ms"], errors="coerce").fillna(0).astype(np.int64)
    out["sleep"] = pd.to_numeric(out["sleep"], errors="coerce").fillna(0).astype(int)
    return out


def _parse_model_output(out: Any) -> Tuple[Any, Optional[Any], Optional[Any]]:
    pred, weight, emb = None, None, None
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
        raise ValueError("Cannot parse prediction from model output.")
    return pred, weight, emb


def run_raw_inference(
    model: Any,
    test_loader: Iterable,
    device: str = "cuda",
    save_csv: Optional[str] = None,
    save_emb: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Run model once and return event-level raw prediction dataframe.

    Current model output:
        pred, weight = model(x)
    Optional better output:
        pred, weight, emb = model(x)
    """
    if torch is None:
        raise ImportError("PyTorch is required for run_raw_inference(model, test_loader).")

    model.eval()
    model.to(device)

    all_rows, all_pred, all_true, all_weight, all_emb = [], [], [], [], []
    start_index = 0

    with torch.no_grad():
        for batch in test_loader:
            if not isinstance(batch, (tuple, list)):
                raise ValueError("Expected batch to be (x, y, meta) or (x, y).")
            if len(batch) == 3:
                x, y, meta = batch
            elif len(batch) == 2:
                x, y = batch
                meta = None
            else:
                raise ValueError("Expected batch to be (x, y, meta) or (x, y).")

            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()
            out = model(x)
            pred, weight, emb = _parse_model_output(out)

            pred_np = to_numpy(pred)
            y_np = to_numpy(y)
            bs = int(pred_np.shape[0])

            all_rows.extend(meta_to_rows(meta, batch_size=bs, start_index=start_index))
            all_pred.append(pred_np)
            all_true.append(y_np)

            if weight is not None:
                w_np = to_numpy(weight)
                if w_np.ndim == 1:
                    w_np = w_np[:, None]
                all_weight.append(w_np)
            else:
                all_weight.append(np.full((bs, 1), np.nan, dtype=np.float32))

            if emb is not None:
                e_np = to_numpy(emb)
                if e_np.ndim > 2:
                    e_np = e_np.reshape(e_np.shape[0], -1)
                all_emb.append(e_np)

            start_index += bs

    pred_all = np.concatenate(all_pred, axis=0)
    true_all = np.concatenate(all_true, axis=0)
    weight_all = np.concatenate(all_weight, axis=0)

    df = pd.DataFrame(all_rows)
    df = standardize_required_columns(df)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df["y_true_sbp"] = true_all[:, 0].astype(float)
    df["y_true_dbp"] = true_all[:, 1].astype(float)
    df["y_pred_sbp_raw"] = pred_all[:, 0].astype(float)
    df["y_pred_dbp_raw"] = pred_all[:, 1].astype(float)

    for k in range(weight_all.shape[1]):
        df[f"attn_w{k}"] = weight_all[:, k].astype(float)

    emb_all = np.concatenate(all_emb, axis=0).astype(np.float32) if all_emb else None

    if save_csv is not None:
        _ensure_dir(save_csv)
        df.to_csv(save_csv, index=False)
    if save_emb is not None and emb_all is not None:
        _ensure_dir(save_emb)
        np.save(save_emb, emb_all)

    return df, emb_all


def load_raw_predictions(csv_path: str | Path, emb_path: Optional[str | Path] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    df = pd.read_csv(csv_path)
    df = standardize_required_columns(df)
    required = ["y_true_sbp", "y_true_dbp", "y_pred_sbp_raw", "y_pred_dbp_raw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Raw prediction CSV missing required columns: {missing}")
    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df), dtype=np.int64)

    emb = None
    if emb_path is not None and Path(emb_path).exists():
        emb = np.load(emb_path)
        if len(emb) != len(df):
            raise ValueError(f"Embedding rows {len(emb)} != CSV rows {len(df)}")
    return df, emb


# =============================================================================
# 3. Calibration point selection
# =============================================================================

def _sleep_filter(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    mode = str(mode).lower()
    if mode == "all":
        return df
    if mode == "day":
        return df[df["sleep"].astype(int) == 0]
    if mode in ["night", "sleep"]:
        return df[df["sleep"].astype(int) == 1]
    raise ValueError(f"Unknown sleep mode: {mode}")


def _quantile_positions(n: int, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=int)
    if n <= k:
        return np.arange(n, dtype=int)
    raw = np.round(np.linspace(0, n - 1, k)).astype(int)
    seen, pos = set(), []
    for p in raw:
        p = int(p)
        if p not in seen:
            pos.append(p)
            seen.add(p)
    for p in range(n):
        if len(pos) >= k:
            break
        if p not in seen:
            pos.append(p)
            seen.add(p)
    return np.asarray(pos[:k], dtype=int)


def _min_gap_greedy_positions(times: np.ndarray, k: int, min_gap_ms: int) -> np.ndarray:
    n = len(times)
    if k <= 0:
        return np.array([], dtype=int)
    if n <= k:
        return np.arange(n, dtype=int)
    chosen, last_t = [], None
    for i, t in enumerate(times):
        t = int(t)
        if last_t is None or t - last_t >= min_gap_ms:
            chosen.append(i)
            last_t = t
        if len(chosen) >= k:
            break
    if len(chosen) < k:
        for p in _quantile_positions(n, k):
            p = int(p)
            if p not in chosen:
                chosen.append(p)
            if len(chosen) >= k:
                break
    return np.asarray(sorted(chosen[:k]), dtype=int)


def _random_min_gap_positions(times: np.ndarray, k: int, min_gap_ms: int, rng: np.random.Generator, max_trials: int = 500) -> np.ndarray:
    n = len(times)
    if k <= 0:
        return np.array([], dtype=int)
    if n <= k:
        return np.arange(n, dtype=int)
    idx = np.arange(n)
    best: List[int] = []
    for _ in range(max_trials):
        perm = rng.permutation(idx)
        chosen, chosen_times = [], []
        for p in perm:
            p = int(p)
            t = int(times[p])
            if all(abs(t - ct) >= min_gap_ms for ct in chosen_times):
                chosen.append(p)
                chosen_times.append(t)
            if len(chosen) >= k:
                return np.asarray(sorted(chosen), dtype=int)
        if len(chosen) > len(best):
            best = chosen
    if len(best) >= k:
        return np.asarray(sorted(best[:k]), dtype=int)
    return _min_gap_greedy_positions(times, k, min_gap_ms)


def select_positions_by_strategy(times: np.ndarray, k: int, strategy: str, min_gap_minutes: float = 30.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    strategy = str(strategy).lower()
    n = len(times)
    k = min(int(k), n)
    rng = rng or np.random.default_rng(42)
    if k <= 0 or n <= 0:
        return np.array([], dtype=int)
    if strategy == "head":
        return np.arange(k, dtype=int)
    if strategy == "tail":
        return np.arange(n - k, n, dtype=int)
    if strategy == "quantile":
        return _quantile_positions(n, k)
    if strategy == "min_gap":
        return _min_gap_greedy_positions(np.asarray(times, dtype=np.int64), k, int(min_gap_minutes * 60 * 1000))
    if strategy == "random":
        return np.sort(rng.choice(np.arange(n), size=k, replace=False)).astype(int)
    if strategy == "random_min_gap":
        return _random_min_gap_positions(np.asarray(times, dtype=np.int64), k, int(min_gap_minutes * 60 * 1000), rng)
    raise ValueError("Unknown strategy. Use head/tail/quantile/min_gap/random/random_min_gap.")


def select_calibration_points_for_scenario(df: pd.DataFrame, scenario: ProductScenario) -> pd.DataFrame:
    out = df.copy()
    out["is_calib"] = False
    out["calib_phase"] = "none"
    rng = np.random.default_rng(scenario.seed)

    for _, g_subj in out.groupby("id_clean", sort=False):
        g_subj = g_subj.sort_values("t_bp_ms")
        chosen_indices: List[int] = []

        if scenario.sleep_quota is not None:
            for sleep_value, quota in scenario.sleep_quota.items():
                quota = int(quota)
                if quota <= 0:
                    continue
                g_pool = g_subj[g_subj["sleep"].astype(int) == int(sleep_value)]
                if len(g_pool) == 0:
                    continue
                idx = g_pool.index.to_numpy()
                pos = select_positions_by_strategy(
                    g_pool["t_bp_ms"].to_numpy(dtype=np.int64),
                    k=min(quota, len(g_pool)),
                    strategy=scenario.calib_strategy,
                    min_gap_minutes=scenario.min_gap_minutes,
                    rng=rng,
                )
                chosen_indices.extend(idx[pos].tolist())
        else:
            g_pool = _sleep_filter(g_subj, scenario.calib_sleep)
            if len(g_pool) > 0:
                idx = g_pool.index.to_numpy()
                pos = select_positions_by_strategy(
                    g_pool["t_bp_ms"].to_numpy(dtype=np.int64),
                    k=min(scenario.calib_total, len(g_pool)),
                    strategy=scenario.calib_strategy,
                    min_gap_minutes=scenario.min_gap_minutes,
                    rng=rng,
                )
                chosen_indices.extend(idx[pos].tolist())

        if not chosen_indices:
            continue

        chosen = out.loc[chosen_indices].sort_values("t_bp_ms").index.to_list()[: int(scenario.calib_total)]
        out.loc[chosen, "is_calib"] = True

        if scenario.support_n > 0 or scenario.update_n > 0:
            support_idx = chosen[: int(scenario.support_n)]
            update_idx = chosen[int(scenario.support_n): int(scenario.support_n + scenario.update_n)]
            extra_idx = chosen[int(scenario.support_n + scenario.update_n):]
            out.loc[support_idx, "calib_phase"] = "support"
            out.loc[update_idx, "calib_phase"] = "update"
            out.loc[extra_idx, "calib_phase"] = "calib"
        else:
            out.loc[chosen, "calib_phase"] = "calib"
    return out


# =============================================================================
# 4. Calibration methods
# =============================================================================

def apply_bias_calibration(df: pd.DataFrame, by_sleep: bool = False, min_points: int = 1) -> pd.DataFrame:
    out = df.copy()
    out["y_pred_sbp_bias"] = out["y_pred_sbp_raw"].astype(float)
    out["y_pred_dbp_bias"] = out["y_pred_dbp_raw"].astype(float)
    calib = out[out["is_calib"].astype(bool)].copy()
    subj_pool = {pid: g for pid, g in calib.groupby("id_clean", sort=False)}
    group_cols = ["id_clean", "sleep"] if by_sleep else ["id_clean"]

    for key, g_all in out.groupby(group_cols, sort=False):
        if by_sleep:
            pid, slp = key
            g_cal = calib[(calib["id_clean"] == pid) & (calib["sleep"] == slp)]
        else:
            pid = key[0] if isinstance(key, tuple) else key
            g_cal = calib[calib["id_clean"] == pid]
        if len(g_cal) < min_points:
            g_cal = subj_pool.get(pid, pd.DataFrame())
        if len(g_cal) < min_points:
            continue
        sbp_bias = float((g_cal["y_true_sbp"] - g_cal["y_pred_sbp_raw"]).mean())
        dbp_bias = float((g_cal["y_true_dbp"] - g_cal["y_pred_dbp_raw"]).mean())
        idx = g_all.index
        out.loc[idx, "y_pred_sbp_bias"] = out.loc[idx, "y_pred_sbp_raw"].astype(float) + sbp_bias
        out.loc[idx, "y_pred_dbp_bias"] = out.loc[idx, "y_pred_dbp_raw"].astype(float) + dbp_bias
        out.loc[idx, "bias_sbp_b"] = sbp_bias
        out.loc[idx, "bias_dbp_b"] = dbp_bias
    return out


def fit_affine_ridge(x: Sequence[float], y: Sequence[float], lam: float = 100.0, slope_center: float = 1.0, penalize_intercept: bool = False) -> Tuple[float, float]:
    x = safe_float_array(x)
    y = safe_float_array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return 1.0, 0.0
    if len(x) < 2:
        return 1.0, float(np.mean(y - x))

    y_shift = y - slope_center * x
    X = np.stack([x, np.ones_like(x)], axis=1)
    R = np.zeros((2, 2), dtype=np.float64)
    R[0, 0] = float(lam)
    if penalize_intercept:
        R[1, 1] = float(lam)
    try:
        w = np.linalg.solve(X.T @ X + R, X.T @ y_shift)
        a = float(w[0] + slope_center)
        b = float(w[1])
    except np.linalg.LinAlgError:
        a, b = 1.0, float(np.mean(y - x))
    return a, b


def apply_affine_calibration(df: pd.DataFrame, by_sleep: bool = False, min_points: int = 3, lam: float = 100.0, slope_clip: Optional[Tuple[float, float]] = (0.65, 1.50), penalize_intercept: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["y_pred_sbp_aff"] = out["y_pred_sbp_raw"].astype(float)
    out["y_pred_dbp_aff"] = out["y_pred_dbp_raw"].astype(float)
    calib = out[out["is_calib"].astype(bool)].copy()
    subj_pool = {pid: g for pid, g in calib.groupby("id_clean", sort=False)}
    group_cols = ["id_clean", "sleep"] if by_sleep else ["id_clean"]

    for key, g_all in out.groupby(group_cols, sort=False):
        if by_sleep:
            pid, slp = key
            g_cal = calib[(calib["id_clean"] == pid) & (calib["sleep"] == slp)]
        else:
            pid = key[0] if isinstance(key, tuple) else key
            g_cal = calib[calib["id_clean"] == pid]
        if len(g_cal) < min_points:
            g_cal = subj_pool.get(pid, pd.DataFrame())
        if len(g_cal) == 0:
            continue

        a_s, b_s = fit_affine_ridge(g_cal["y_pred_sbp_raw"], g_cal["y_true_sbp"], lam=lam, penalize_intercept=penalize_intercept)
        a_d, b_d = fit_affine_ridge(g_cal["y_pred_dbp_raw"], g_cal["y_true_dbp"], lam=lam, penalize_intercept=penalize_intercept)
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


def build_bank_features(df: pd.DataFrame, emb: Optional[np.ndarray] = None, use_raw_pred: bool = True, use_sleep: bool = True) -> np.ndarray:
    feats: List[np.ndarray] = []
    if emb is not None:
        e = np.asarray(emb, dtype=np.float64)
        if e.ndim > 2:
            e = e.reshape(e.shape[0], -1)
        feats.append(e)
    if use_raw_pred:
        raw = df[["y_pred_sbp_raw", "y_pred_dbp_raw"]].to_numpy(dtype=np.float64)
        raw = raw / np.array([[120.0, 80.0]], dtype=np.float64)
        feats.append(raw)
    if use_sleep and "sleep" in df.columns:
        feats.append(df[["sleep"]].to_numpy(dtype=np.float64))
    if not feats:
        raise ValueError("No feature source for residual bank.")
    x = np.concatenate(feats, axis=1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return l2_normalize(x)


def apply_residual_bank_calibration(df: pd.DataFrame, emb: Optional[np.ndarray] = None, temperature: float = 0.30, time_weight_per_hour: float = 0.20, sleep_bonus: float = 0.25, uniform_mix: float = 0.20, residual_clip: float = 35.0, by_sleep_fallback: bool = False, min_points: int = 1) -> pd.DataFrame:
    out = df.copy()
    out["y_pred_sbp_bank"] = out["y_pred_sbp_raw"].astype(float)
    out["y_pred_dbp_bank"] = out["y_pred_dbp_raw"].astype(float)
    out["bank_max_w"] = np.nan
    out["bank_eff_n"] = np.nan

    feat = build_bank_features(out, emb=emb, use_raw_pred=True, use_sleep=True)
    calib = out[out["is_calib"].astype(bool)].copy()
    subj_calib_idx = {pid: g.index.to_numpy() for pid, g in calib.groupby("id_clean", sort=False)}

    for pid, g_all in out.groupby("id_clean", sort=False):
        idx_query = g_all.index[~g_all["is_calib"].astype(bool)].to_numpy()
        idx_calib_base = subj_calib_idx.get(pid, np.array([], dtype=int))
        if len(idx_query) == 0 or len(idx_calib_base) < min_points:
            continue

        for qi in idx_query:
            idx_calib = idx_calib_base
            if by_sleep_fallback and "sleep" in out.columns:
                same_sleep = idx_calib_base[out.loc[idx_calib_base, "sleep"].to_numpy(dtype=int) == int(out.loc[qi, "sleep"])]
                if len(same_sleep) >= min_points:
                    idx_calib = same_sleep
            if len(idx_calib) < min_points:
                continue

            sim = (feat[qi:qi + 1] @ feat[idx_calib].T).reshape(-1)
            tq = float(out.loc[qi, "t_bp_ms"])
            tc = out.loc[idx_calib, "t_bp_ms"].to_numpy(dtype=np.float64)
            dt_h = np.abs(tq - tc) / 3600000.0
            score = sim / max(float(temperature), 1e-6) - float(time_weight_per_hour) * dt_h
            if "sleep" in out.columns:
                sq = int(out.loc[qi, "sleep"])
                sc = out.loc[idx_calib, "sleep"].to_numpy(dtype=int)
                score = score + float(sleep_bonus) * (sc == sq).astype(np.float64)
            w = stable_softmax(score)
            if uniform_mix > 0:
                u = np.ones_like(w, dtype=np.float64) / len(w)
                w = (1.0 - float(uniform_mix)) * w + float(uniform_mix) * u

            true_cal = out.loc[idx_calib, ["y_true_sbp", "y_true_dbp"]].to_numpy(dtype=np.float64)
            pred_cal = out.loc[idx_calib, ["y_pred_sbp_raw", "y_pred_dbp_raw"]].to_numpy(dtype=np.float64)
            residual = true_cal - pred_cal
            if residual_clip is not None and residual_clip > 0:
                residual = np.clip(residual, -float(residual_clip), float(residual_clip))
            delta = w @ residual
            raw_q = out.loc[qi, ["y_pred_sbp_raw", "y_pred_dbp_raw"]].to_numpy(dtype=np.float64)
            pred_q = raw_q + delta
            out.loc[qi, "y_pred_sbp_bank"] = pred_q[0]
            out.loc[qi, "y_pred_dbp_bank"] = pred_q[1]
            out.loc[qi, "bank_max_w"] = float(np.max(w))
            out.loc[qi, "bank_eff_n"] = float(1.0 / np.sum(w ** 2))
    return out


# =============================================================================
# 5. Evaluation
# =============================================================================

def get_eval_df(df: pd.DataFrame, scenario: ProductScenario) -> pd.DataFrame:
    d = df.copy()
    if scenario.exclude_calib_from_eval and "is_calib" in d.columns:
        d = d[~d["is_calib"].astype(bool)].copy()
    return _sleep_filter(d, scenario.eval_sleep)


def method_columns_available(df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    methods = {"raw": ("y_pred_sbp_raw", "y_pred_dbp_raw")}
    if "y_pred_sbp_bias" in df.columns:
        methods["bias"] = ("y_pred_sbp_bias", "y_pred_dbp_bias")
    if "y_pred_sbp_aff" in df.columns:
        methods["aff"] = ("y_pred_sbp_aff", "y_pred_dbp_aff")
    if "y_pred_sbp_bank" in df.columns:
        methods["bank"] = ("y_pred_sbp_bank", "y_pred_dbp_bank")
    return methods


def evaluate_event_level(df: pd.DataFrame, scenario: ProductScenario, method_cols: Dict[str, Tuple[str, str]]) -> Dict[str, Any]:
    d = get_eval_df(df, scenario)
    metrics: Dict[str, Any] = {"event_N": int(len(d))}
    for method, (sbp_col, dbp_col) in method_cols.items():
        add_bp_metrics(metrics, f"{method}_event", d, sbp_col, dbp_col)
    return metrics


def build_natural_hour_rows(df_id: pd.DataFrame, pred_sbp_col: str, pred_dbp_col: str, window_minutes: int = 60, min_events_per_hour: int = 2, separate_sleep: bool = True) -> List[Dict[str, Any]]:
    window_ms = int(window_minutes * 60 * 1000)
    g = df_id.sort_values("t_bp_ms").reset_index(drop=False)
    rows: List[Dict[str, Any]] = []
    i, n = 0, len(g)
    while i < n:
        t0 = int(g.loc[i, "t_bp_ms"])
        tend = t0 + window_ms
        sleep0 = int(g.loc[i, "sleep"]) if "sleep" in g.columns else 0
        j = i
        while j + 1 < n:
            t_next = int(g.loc[j + 1, "t_bp_ms"])
            if t_next > tend:
                break
            if separate_sleep and "sleep" in g.columns and int(g.loc[j + 1, "sleep"]) != sleep0:
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


def evaluate_hourly(df: pd.DataFrame, scenario: ProductScenario, pred_sbp_col: str, pred_dbp_col: str) -> Dict[str, Any]:
    d = get_eval_df(df, scenario)
    rows: List[Dict[str, Any]] = []
    for _, g in d.groupby("id_clean", sort=False):
        rows.extend(build_natural_hour_rows(g, pred_sbp_col, pred_dbp_col, 60, scenario.min_events_per_hour, scenario.separate_sleep_in_hourly))
    h = pd.DataFrame(rows)
    metrics: Dict[str, Any] = {"hourly_N": int(len(h)), "hourly_min_events_per_hour": int(scenario.min_events_per_hour)}
    if len(h) > 0:
        _prefix_metrics(metrics, "hourly_sbp", me_std_mae(h["y_pred_sbp_h"], h["y_true_sbp_h"]))
        _prefix_metrics(metrics, "hourly_dbp", me_std_mae(h["y_pred_dbp_h"], h["y_true_dbp_h"]))
    return metrics


def build_macro_window_rows(df_id: pd.DataFrame, pred_sbp_col: str, pred_dbp_col: str, window_hours: int = 24, min_events_per_window: int = 6) -> List[Dict[str, Any]]:
    window_ms = int(window_hours * 3600 * 1000)
    g = df_id.sort_values("t_bp_ms").reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    i, n = 0, len(g)
    while i < n:
        t0 = int(g.loc[i, "t_bp_ms"])
        tend = t0 + window_ms
        j = i
        while j + 1 < n and int(g.loc[j + 1, "t_bp_ms"]) <= tend:
            j += 1
        block = g.loc[i:j].copy()
        if len(block) >= min_events_per_window:
            rows.append({
                "id_clean": str(block["id_clean"].iloc[0]),
                "window_start_ms": int(t0),
                "window_end_ms": int(tend),
                "n_event": int(len(block)),
                "true_sbp_mean": float(block["y_true_sbp"].mean()),
                "pred_sbp_mean": float(block[pred_sbp_col].mean()),
                "true_dbp_mean": float(block["y_true_dbp"].mean()),
                "pred_dbp_mean": float(block[pred_dbp_col].mean()),
                "true_sbp_std": float(block["y_true_sbp"].std(ddof=1)) if len(block) > 1 else np.nan,
                "pred_sbp_std": float(block[pred_sbp_col].std(ddof=1)) if len(block) > 1 else np.nan,
                "true_dbp_std": float(block["y_true_dbp"].std(ddof=1)) if len(block) > 1 else np.nan,
                "pred_dbp_std": float(block[pred_dbp_col].std(ddof=1)) if len(block) > 1 else np.nan,
            })
        i = j + 1
    return rows


def evaluate_macro24(df: pd.DataFrame, scenario: ProductScenario, pred_sbp_col: str, pred_dbp_col: str) -> Dict[str, Any]:
    d = get_eval_df(df, scenario)
    rows: List[Dict[str, Any]] = []
    for _, g in d.groupby("id_clean", sort=False):
        rows.extend(build_macro_window_rows(g, pred_sbp_col, pred_dbp_col, scenario.macro_window_hours, scenario.min_events_per_macro))
    m = pd.DataFrame(rows)
    metrics: Dict[str, Any] = {"macro_N": int(len(m)), "macro_window_hours": int(scenario.macro_window_hours), "macro_min_events": int(scenario.min_events_per_macro)}
    if len(m) == 0:
        return metrics
    _prefix_metrics(metrics, "macro_sbp_mean", me_std_mae(m["pred_sbp_mean"], m["true_sbp_mean"]))
    _prefix_metrics(metrics, "macro_dbp_mean", me_std_mae(m["pred_dbp_mean"], m["true_dbp_mean"]))
    _prefix_metrics(metrics, "macro_sbp_std", me_std_mae(m["pred_sbp_std"], m["true_sbp_std"]))
    _prefix_metrics(metrics, "macro_dbp_std", me_std_mae(m["pred_dbp_std"], m["true_dbp_std"]))
    return metrics


# =============================================================================
# 6. Scenario runners
# =============================================================================

def apply_all_calibrations(df: pd.DataFrame, emb: Optional[np.ndarray] = None, run_bias: bool = True, run_affine: bool = True, run_bank: bool = True, by_sleep_calibration: bool = False, affine_lam: float = 100.0, affine_min_points: int = 3, bank_temperature: float = 0.30, bank_time_weight_per_hour: float = 0.20, bank_sleep_bonus: float = 0.25, bank_uniform_mix: float = 0.20, bank_residual_clip: float = 35.0) -> pd.DataFrame:
    out = df.copy()
    if run_bias:
        out = apply_bias_calibration(out, by_sleep=by_sleep_calibration, min_points=1)
    if run_affine:
        out = apply_affine_calibration(out, by_sleep=by_sleep_calibration, min_points=affine_min_points, lam=affine_lam, slope_clip=(0.65, 1.50))
    if run_bank:
        out = apply_residual_bank_calibration(out, emb=emb, temperature=bank_temperature, time_weight_per_hour=bank_time_weight_per_hour, sleep_bonus=bank_sleep_bonus, uniform_mix=bank_uniform_mix, residual_clip=bank_residual_clip, by_sleep_fallback=False, min_points=1)
    return out


def evaluate_scenario_methods(df: pd.DataFrame, scenario: ProductScenario) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    method_cols = method_columns_available(df)
    metrics["scenario"] = scenario.name
    for k, v in asdict(scenario).items():
        metrics[f"scenario_{k}"] = json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v
    metrics["n_rows"] = int(len(df))
    metrics["n_subjects"] = int(df["id_clean"].nunique())
    metrics["n_calib_rows"] = int(df["is_calib"].sum()) if "is_calib" in df.columns else 0
    metrics["n_eval_rows"] = int(len(get_eval_df(df, scenario)))

    if scenario.eval_mode in ["event", "both", "all"]:
        metrics.update(evaluate_event_level(df, scenario, method_cols))
    if scenario.eval_mode in ["hourly", "both", "all"]:
        for method, (sbp_col, dbp_col) in method_cols.items():
            hm = evaluate_hourly(df, scenario, sbp_col, dbp_col)
            for k, v in hm.items():
                metrics[f"{method}_{k}"] = v
    if scenario.eval_mode in ["macro24", "both", "all"]:
        for method, (sbp_col, dbp_col) in method_cols.items():
            mm = evaluate_macro24(df, scenario, sbp_col, dbp_col)
            for k, v in mm.items():
                metrics[f"{method}_{k}"] = v

    if "bank_eff_n" in df.columns:
        d = get_eval_df(df, scenario)
        metrics["bank_eff_n_mean"] = float(np.nanmean(d["bank_eff_n"])) if len(d) else np.nan
        metrics["bank_max_w_mean"] = float(np.nanmean(d["bank_max_w"])) if len(d) else np.nan
    return metrics


def run_one_scenario(df_raw: pd.DataFrame, scenario: ProductScenario, emb: Optional[np.ndarray] = None, run_bias: bool = True, run_affine: bool = True, run_bank: bool = True, by_sleep_calibration: bool = False, affine_lam: float = 100.0, affine_min_points: int = 3, bank_temperature: float = 0.30, bank_time_weight_per_hour: float = 0.20, bank_sleep_bonus: float = 0.25, bank_uniform_mix: float = 0.20, bank_residual_clip: float = 35.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = select_calibration_points_for_scenario(df_raw, scenario)
    df = apply_all_calibrations(df, emb=emb, run_bias=run_bias, run_affine=run_affine, run_bank=run_bank, by_sleep_calibration=by_sleep_calibration, affine_lam=affine_lam, affine_min_points=affine_min_points, bank_temperature=bank_temperature, bank_time_weight_per_hour=bank_time_weight_per_hour, bank_sleep_bonus=bank_sleep_bonus, bank_uniform_mix=bank_uniform_mix, bank_residual_clip=bank_residual_clip)
    metrics = evaluate_scenario_methods(df, scenario)
    metrics.update({
        "run_bias": bool(run_bias),
        "run_affine": bool(run_affine),
        "run_bank": bool(run_bank),
        "by_sleep_calibration": bool(by_sleep_calibration),
        "affine_lam": float(affine_lam),
        "affine_min_points": int(affine_min_points),
        "bank_temperature": float(bank_temperature),
        "bank_time_weight_per_hour": float(bank_time_weight_per_hour),
        "bank_sleep_bonus": float(bank_sleep_bonus),
        "bank_uniform_mix": float(bank_uniform_mix),
        "bank_residual_clip": float(bank_residual_clip),
        "has_embedding": emb is not None,
    })
    return df, metrics


def run_product_scenarios_from_raw(df_raw: pd.DataFrame, emb: Optional[np.ndarray] = None, scenarios: Optional[List[ProductScenario]] = None, out_dir: str | Path = "product_eval_outputs", save_scenario_csv: bool = True, run_bias: bool = True, run_affine: bool = True, run_bank: bool = True, by_sleep_calibration: bool = False, affine_lam: float = 100.0, affine_min_points: int = 3, bank_temperature: float = 0.30, bank_time_weight_per_hour: float = 0.20, bank_sleep_bonus: float = 0.25, bank_uniform_mix: float = 0.20, bank_residual_clip: float = 35.0) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    scenarios = scenarios or default_product_scenarios()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: List[Dict[str, Any]] = []
    scenario_dfs: Dict[str, pd.DataFrame] = {}

    for sc in scenarios:
        df_sc, metrics_sc = run_one_scenario(
            df_raw=df_raw,
            scenario=sc,
            emb=emb,
            run_bias=run_bias,
            run_affine=run_affine,
            run_bank=run_bank,
            by_sleep_calibration=by_sleep_calibration,
            affine_lam=affine_lam,
            affine_min_points=affine_min_points,
            bank_temperature=bank_temperature,
            bank_time_weight_per_hour=bank_time_weight_per_hour,
            bank_sleep_bonus=bank_sleep_bonus,
            bank_uniform_mix=bank_uniform_mix,
            bank_residual_clip=bank_residual_clip,
        )
        scenario_dfs[sc.name] = df_sc
        all_metrics.append(metrics_sc)
        if save_scenario_csv:
            df_sc.to_csv(out_dir / f"predictions_{sc.name}.csv", index=False)
        with open(out_dir / f"metrics_{sc.name}.json", "w", encoding="utf-8") as f:
            json.dump(metrics_sc, f, indent=2, ensure_ascii=False)

    summary_df = pd.DataFrame(all_metrics)
    summary_df.to_csv(out_dir / "scenario_metrics_summary.csv", index=False)
    return summary_df, scenario_dfs


def run_lazy_product_eval(model: Any, test_loader: Iterable, device: str = "cuda", scenarios: Optional[List[ProductScenario]] = None, out_dir: str | Path = "product_eval_outputs", raw_csv_name: str = "raw_predictions.csv", raw_emb_name: str = "raw_embeddings.npy", save_scenario_csv: bool = True, **kwargs) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_raw, emb = run_raw_inference(model=model, test_loader=test_loader, device=device, save_csv=str(out_dir / raw_csv_name), save_emb=str(out_dir / raw_emb_name))
    return run_product_scenarios_from_raw(df_raw=df_raw, emb=emb, scenarios=scenarios, out_dir=out_dir, save_scenario_csv=save_scenario_csv, **kwargs)


def run_product_eval_from_saved_raw(raw_csv: str | Path, emb_path: Optional[str | Path] = None, scenarios: Optional[List[ProductScenario]] = None, out_dir: str | Path = "product_eval_outputs", save_scenario_csv: bool = True, **kwargs) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    df_raw, emb = load_raw_predictions(raw_csv, emb_path)
    return run_product_scenarios_from_raw(df_raw=df_raw, emb=emb, scenarios=scenarios, out_dir=out_dir, save_scenario_csv=save_scenario_csv, **kwargs)


# =============================================================================
# 7. Optional CLI for post-hoc CSV mode
# =============================================================================

def _parse_scenario_names(names: Optional[str], seed: int = 42) -> List[ProductScenario]:
    all_scenarios = {s.name: s for s in default_product_scenarios(seed=seed)}
    if names is None or names.strip() == "" or names.strip().lower() == "all":
        return list(all_scenarios.values())
    selected = []
    for name in names.split(","):
        name = name.strip()
        if name not in all_scenarios:
            raise ValueError(f"Unknown scenario '{name}'. Available: {list(all_scenarios)}")
        selected.append(all_scenarios[name])
    return selected


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Post-hoc product scenario evaluation from raw prediction CSV.")
    p.add_argument("--raw_csv", type=str, required=True)
    p.add_argument("--emb_path", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="product_eval_outputs")
    p.add_argument("--scenarios", type=str, default="all", help="Comma-separated scenario names, or 'all'.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_bias", action="store_true")
    p.add_argument("--no_affine", action="store_true")
    p.add_argument("--no_bank", action="store_true")
    p.add_argument("--by_sleep_calibration", action="store_true")
    p.add_argument("--affine_lam", type=float, default=100.0)
    p.add_argument("--affine_min_points", type=int, default=3)
    p.add_argument("--bank_temperature", type=float, default=0.30)
    p.add_argument("--bank_time_weight_per_hour", type=float, default=0.20)
    p.add_argument("--bank_sleep_bonus", type=float, default=0.25)
    p.add_argument("--bank_uniform_mix", type=float, default=0.20)
    p.add_argument("--bank_residual_clip", type=float, default=35.0)
    args = p.parse_args()

    scenarios = _parse_scenario_names(args.scenarios, seed=args.seed)
    summary, _ = run_product_eval_from_saved_raw(
        raw_csv=args.raw_csv,
        emb_path=args.emb_path,
        scenarios=scenarios,
        out_dir=args.out_dir,
        run_bias=not args.no_bias,
        run_affine=not args.no_affine,
        run_bank=not args.no_bank,
        by_sleep_calibration=args.by_sleep_calibration,
        affine_lam=args.affine_lam,
        affine_min_points=args.affine_min_points,
        bank_temperature=args.bank_temperature,
        bank_time_weight_per_hour=args.bank_time_weight_per_hour,
        bank_sleep_bonus=args.bank_sleep_bonus,
        bank_uniform_mix=args.bank_uniform_mix,
        bank_residual_clip=args.bank_residual_clip,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
