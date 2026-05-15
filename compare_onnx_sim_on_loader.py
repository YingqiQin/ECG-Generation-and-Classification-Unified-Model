#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare original ONNX vs onnxsim-simplified ONNX on an existing test_loader.

Use this by importing it into your current evaluation script:

from compare_onnx_sim_on_loader import compare_onnx_pair_on_test_loader

summary = compare_onnx_pair_on_test_loader(
    test_loader=test_loader,
    original_onnx_path="bp_25hz_b1_k160_l200_2d_attention_fp32.onnx",
    simplified_onnx_path="bp_25hz_b1_k160_l200_2d_attention_fp32_sim.onnx",
    out_dir="onnx_sim_compare",
    providers=["CPUExecutionProvider"],
)

It reports:
  1) original ONNX vs simplified ONNX prediction difference
  2) original ONNX BP metrics vs y
  3) simplified ONNX BP metrics vs y
  4) optional per-sample CSV

If ONNX input is fixed [1,160,1,200], but your DataLoader batch size is >1,
the function automatically splits the batch and runs ONNX one sample at a time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

try:
    import onnxruntime as ort
except Exception as e:
    raise ImportError("Please install onnxruntime first: pip install onnxruntime") from e


def _as_numpy_x(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _as_numpy_y(y: Any) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        return y.detach().cpu().numpy().astype(np.float32)
    return np.asarray(y, dtype=np.float32)


def _make_session(onnx_path: str | Path, providers: Optional[Sequence[str]] = None) -> ort.InferenceSession:
    if providers is None:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=list(providers))


def _get_io_info(sess: ort.InferenceSession) -> Dict[str, Any]:
    return {
        "inputs": [{"name": x.name, "shape": x.shape, "type": x.type} for x in sess.get_inputs()],
        "outputs": [{"name": y.name, "shape": y.shape, "type": y.type} for y in sess.get_outputs()],
    }


def _run_onnx(sess: ort.InferenceSession, x_np: np.ndarray, output_index: int = 0) -> np.ndarray:
    input_name = sess.get_inputs()[0].name
    outs = sess.run(None, {input_name: x_np.astype(np.float32, copy=False)})
    return np.asarray(outs[output_index], dtype=np.float32)


def _extract_meta_value(meta: Any, key: str, sample_idx: int = 0, default: Any = None) -> Any:
    if meta is None:
        return default

    if isinstance(meta, dict):
        if key not in meta:
            return default
        v = meta[key]
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                return v.item()
            if len(v) > sample_idx:
                vv = v[sample_idx]
                return vv.item() if hasattr(vv, "item") else vv
            return default
        if isinstance(v, (list, tuple)):
            return v[sample_idx] if len(v) > sample_idx else default
        return v

    if isinstance(meta, (list, tuple)):
        if len(meta) > sample_idx and isinstance(meta[sample_idx], dict):
            return meta[sample_idx].get(key, default)

    return default


def _metric_errors(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true

    out = {}
    for j, name in enumerate(["sbp", "dbp"]):
        e = err[:, j]
        out[f"{prefix}_{name}_ME"] = float(np.mean(e))
        out[f"{prefix}_{name}_STD"] = float(np.std(e, ddof=1)) if len(e) > 1 else 0.0
        out[f"{prefix}_{name}_MAE"] = float(np.mean(np.abs(e)))
        out[f"{prefix}_{name}_RMSE"] = float(np.sqrt(np.mean(e ** 2)))
    return out


def _diff_metrics(y0: np.ndarray, y1: np.ndarray, prefix: str = "sim_minus_orig") -> Dict[str, float]:
    y0 = np.asarray(y0, dtype=np.float64)
    y1 = np.asarray(y1, dtype=np.float64)
    diff = y1 - y0
    abs_diff = np.abs(diff)

    out = {}
    for j, name in enumerate(["sbp", "dbp"]):
        d = diff[:, j]
        a = abs_diff[:, j]
        out[f"{prefix}_{name}_mean_diff"] = float(np.mean(d))
        out[f"{prefix}_{name}_std_diff"] = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        out[f"{prefix}_{name}_mean_abs_diff"] = float(np.mean(a))
        out[f"{prefix}_{name}_median_abs_diff"] = float(np.median(a))
        out[f"{prefix}_{name}_p95_abs_diff"] = float(np.percentile(a, 95))
        out[f"{prefix}_{name}_p99_abs_diff"] = float(np.percentile(a, 99))
        out[f"{prefix}_{name}_max_abs_diff"] = float(np.max(a))
    out[f"{prefix}_all_mean_abs_diff"] = float(np.mean(abs_diff))
    out[f"{prefix}_all_max_abs_diff"] = float(np.max(abs_diff))
    return out


def _iter_samples_from_batch(x: Any, y: Any, meta: Any):
    x_np = _as_numpy_x(x)
    y_np = _as_numpy_y(y)

    if x_np.ndim != 4:
        raise ValueError(f"Expected x to be [B,K,C,L], got shape={x_np.shape}")

    if y_np.ndim == 1:
        y_np = y_np.reshape(1, -1)
    if y_np.ndim != 2 or y_np.shape[1] < 2:
        raise ValueError(f"Expected y to be [B,2], got shape={y_np.shape}")

    B = int(x_np.shape[0])
    for i in range(B):
        x_i = x_np[i:i + 1].astype(np.float32, copy=False)
        y_i = y_np[i:i + 1, :2].astype(np.float32, copy=False)
        yield x_i, y_i, i


def compare_onnx_pair_on_test_loader(
    test_loader,
    original_onnx_path: str | Path,
    simplified_onnx_path: str | Path,
    out_dir: str | Path = "onnx_sim_compare",
    providers: Optional[Sequence[str]] = None,
    output_index: int = 0,
    max_batches: Optional[int] = None,
    save_csv: bool = True,
    verbose_every: int = 100,
) -> Dict[str, Any]:
    """
    Compare original ONNX and simplified ONNX on a PyTorch test_loader.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sess_orig = _make_session(original_onnx_path, providers=providers)
    sess_sim = _make_session(simplified_onnx_path, providers=providers)

    io_info = {
        "original": _get_io_info(sess_orig),
        "simplified": _get_io_info(sess_sim),
    }

    rows: List[Dict[str, Any]] = []
    y_true_all, y_orig_all, y_sim_all = [], [], []

    sample_counter = 0
    for batch_idx, batch in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if len(batch) == 3:
            x, y, meta = batch
        elif len(batch) == 2:
            x, y = batch
            meta = None
        else:
            raise ValueError("Expected test_loader batch to be (x,y,meta) or (x,y).")

        for x_i, y_i, sample_idx in _iter_samples_from_batch(x, y, meta):
            y_orig = _run_onnx(sess_orig, x_i, output_index=output_index).reshape(1, -1)[:, :2]
            y_sim = _run_onnx(sess_sim, x_i, output_index=output_index).reshape(1, -1)[:, :2]

            y_true_all.append(y_i)
            y_orig_all.append(y_orig)
            y_sim_all.append(y_sim)

            row = {
                "sample_idx_global": sample_counter,
                "batch_idx": batch_idx,
                "sample_idx_in_batch": sample_idx,
                "y_true_sbp": float(y_i[0, 0]),
                "y_true_dbp": float(y_i[0, 1]),
                "orig_pred_sbp": float(y_orig[0, 0]),
                "orig_pred_dbp": float(y_orig[0, 1]),
                "sim_pred_sbp": float(y_sim[0, 0]),
                "sim_pred_dbp": float(y_sim[0, 1]),
                "diff_sim_minus_orig_sbp": float(y_sim[0, 0] - y_orig[0, 0]),
                "diff_sim_minus_orig_dbp": float(y_sim[0, 1] - y_orig[0, 1]),
                "abs_diff_sbp": float(abs(y_sim[0, 0] - y_orig[0, 0])),
                "abs_diff_dbp": float(abs(y_sim[0, 1] - y_orig[0, 1])),
            }

            for key in ["id_clean", "id", "sleep", "t_bp_ms", "idx", "datetime", "ABPM_SBP", "ABPM_DBP", "is_calib"]:
                val = _extract_meta_value(meta, key, sample_idx=sample_idx, default=None)
                if val is not None:
                    row[key] = val

            rows.append(row)
            sample_counter += 1

        if verbose_every and (batch_idx + 1) % verbose_every == 0:
            print(f"[compare] processed batches={batch_idx + 1}, samples={sample_counter}")

    if not rows:
        raise RuntimeError("No samples were processed.")

    y_true = np.concatenate(y_true_all, axis=0)
    y_orig = np.concatenate(y_orig_all, axis=0)
    y_sim = np.concatenate(y_sim_all, axis=0)

    summary: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "original_onnx_path": str(original_onnx_path),
        "simplified_onnx_path": str(simplified_onnx_path),
        "providers": list(providers) if providers is not None else ["CPUExecutionProvider"],
        "io_info": io_info,
    }

    summary.update(_diff_metrics(y_orig, y_sim, prefix="sim_minus_orig"))
    summary.update(_metric_errors(y_true, y_orig, prefix="orig_vs_true"))
    summary.update(_metric_errors(y_true, y_sim, prefix="sim_vs_true"))

    df = pd.DataFrame(rows)
    if save_csv:
        df.to_csv(out_dir / "predictions_compare.csv", index=False)

    with open(out_dir / "compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    flat_summary = {k: v for k, v in summary.items() if not isinstance(v, (dict, list))}
    pd.DataFrame([flat_summary]).to_csv(out_dir / "compare_summary.csv", index=False)

    print("\n=== ONNX original vs simplified summary ===")
    print(pd.DataFrame([flat_summary]).to_string(index=False))
    print(f"\nSaved to: {out_dir}")

    return summary


def _run_torch_model(model: torch.nn.Module, x_np: np.ndarray, device: str = "cuda", output_index: int = 0) -> np.ndarray:
    model = model.to(device).eval()
    x = torch.from_numpy(x_np.astype(np.float32, copy=False)).to(device)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[output_index]
    if isinstance(out, dict):
        out = out.get("pred", list(out.values())[output_index])
    return out.detach().cpu().numpy().astype(np.float32)


def compare_torch_and_onnx_on_test_loader(
    model: torch.nn.Module,
    test_loader,
    onnx_path: str | Path,
    out_dir: str | Path = "torch_onnx_compare",
    providers: Optional[Sequence[str]] = None,
    device: str = "cuda",
    output_index: int = 0,
    max_batches: Optional[int] = None,
    save_csv: bool = True,
) -> Dict[str, Any]:
    """
    Optional: compare PyTorch deploy model output with one ONNX model.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = _make_session(onnx_path, providers=providers)
    rows, y_true_all, y_pt_all, y_onnx_all = [], [], [], []

    sample_counter = 0
    for batch_idx, batch in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if len(batch) == 3:
            x, y, meta = batch
        elif len(batch) == 2:
            x, y = batch
            meta = None
        else:
            raise ValueError("Expected batch to be (x,y,meta) or (x,y).")

        for x_i, y_i, sample_idx in _iter_samples_from_batch(x, y, meta):
            y_pt = _run_torch_model(model, x_i, device=device, output_index=output_index).reshape(1, -1)[:, :2]
            y_onnx = _run_onnx(sess, x_i, output_index=output_index).reshape(1, -1)[:, :2]

            y_true_all.append(y_i)
            y_pt_all.append(y_pt)
            y_onnx_all.append(y_onnx)

            rows.append({
                "sample_idx_global": sample_counter,
                "batch_idx": batch_idx,
                "sample_idx_in_batch": sample_idx,
                "y_true_sbp": float(y_i[0, 0]),
                "y_true_dbp": float(y_i[0, 1]),
                "torch_pred_sbp": float(y_pt[0, 0]),
                "torch_pred_dbp": float(y_pt[0, 1]),
                "onnx_pred_sbp": float(y_onnx[0, 0]),
                "onnx_pred_dbp": float(y_onnx[0, 1]),
                "diff_onnx_minus_torch_sbp": float(y_onnx[0, 0] - y_pt[0, 0]),
                "diff_onnx_minus_torch_dbp": float(y_onnx[0, 1] - y_pt[0, 1]),
                "abs_diff_sbp": float(abs(y_onnx[0, 0] - y_pt[0, 0])),
                "abs_diff_dbp": float(abs(y_onnx[0, 1] - y_pt[0, 1])),
            })
            sample_counter += 1

    if not rows:
        raise RuntimeError("No samples were processed.")

    y_true = np.concatenate(y_true_all, axis=0)
    y_pt = np.concatenate(y_pt_all, axis=0)
    y_onnx = np.concatenate(y_onnx_all, axis=0)

    summary: Dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "onnx_path": str(onnx_path),
        "providers": list(providers) if providers is not None else ["CPUExecutionProvider"],
        "io_info": _get_io_info(sess),
    }
    summary.update(_diff_metrics(y_pt, y_onnx, prefix="onnx_minus_torch"))
    summary.update(_metric_errors(y_true, y_pt, prefix="torch_vs_true"))
    summary.update(_metric_errors(y_true, y_onnx, prefix="onnx_vs_true"))

    df = pd.DataFrame(rows)
    if save_csv:
        df.to_csv(out_dir / "torch_onnx_predictions_compare.csv", index=False)

    with open(out_dir / "torch_onnx_compare_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    flat_summary = {k: v for k, v in summary.items() if not isinstance(v, (dict, list))}
    pd.DataFrame([flat_summary]).to_csv(out_dir / "torch_onnx_compare_summary.csv", index=False)

    print("\n=== PyTorch vs ONNX summary ===")
    print(pd.DataFrame([flat_summary]).to_string(index=False))
    print(f"\nSaved to: {out_dir}")
    return summary


if __name__ == "__main__":
    print(
        "Import this file into your existing eval script.\n"
        "Example:\n"
        "from compare_onnx_sim_on_loader import compare_onnx_pair_on_test_loader\n"
        "summary = compare_onnx_pair_on_test_loader(test_loader, 'orig.onnx', 'sim.onnx')\n"
    )
