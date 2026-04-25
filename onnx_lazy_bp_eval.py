
"""
Super-lazy BP inference helper: Torch -> FP32 ONNX -> INT8 ONNX -> Evaluation.

Expected batch format:
    x, y, meta = batch

Expected shapes:
    x: [B, K, 1, L]
    y: [B, 2], columns [SBP, DBP]

Typical usage in your inference.py:

    from onnx_lazy_bp_eval import run_lazy_torch_onnx_int8_eval

    results = run_lazy_torch_onnx_int8_eval(
        model=model,
        test_loader=test_dataloader,
        device="cuda",
        work_dir="onnx_lazy_eval",
        fp32_onnx_name="model_fp32.onnx",
        int8_onnx_name="model_int8.onnx",
        do_int8=True,
        max_calib_batches=32,
    )

This function will:
    1. Evaluate PyTorch FP32 model
    2. Export FP32 ONNX if missing
    3. Evaluate FP32 ONNX
    4. Static-quantize FP32 ONNX to INT8 ONNX if missing
    5. Evaluate INT8 ONNX
    6. Save prediction CSVs and metric JSONs
    7. Print output-level diffs
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import onnxruntime as ort
except Exception:
    ort = None


# -----------------------------
# Metrics
# -----------------------------

def me_std_mae(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    err = y_pred - y_true
    return {
        "ME": float(np.mean(err)),
        "STD": float(np.std(err, ddof=1)) if len(err) > 1 else float("nan"),
        "MAE": float(np.mean(np.abs(err))),
    }


def bp_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, Dict[str, float]]:
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    if preds.ndim != 2 or preds.shape[1] < 2:
        raise ValueError(f"Expected preds shape [N, >=2], got {preds.shape}")
    if targets.ndim != 2 or targets.shape[1] < 2:
        raise ValueError(f"Expected targets shape [N, >=2], got {targets.shape}")

    return {
        "SBP": me_std_mae(preds[:, 0], targets[:, 0]),
        "DBP": me_std_mae(preds[:, 1], targets[:, 1]),
    }


def print_bp_report(report: Dict[str, Dict[str, float]], title: str = "BP Evaluation") -> None:
    print(f"\n===== {title} =====")
    for name in ["SBP", "DBP"]:
        r = report[name]
        print(f"{name}: ME={r['ME']:+.3f}, STD={r['STD']:.3f}, MAE={r['MAE']:.3f}")


def model_size_mb(path: str | Path) -> float:
    path = Path(path)
    return path.stat().st_size / 1024.0 / 1024.0


# -----------------------------
# Batch / tensor helpers
# -----------------------------

def _unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """
    Expected:
        x, y, meta = batch
    """
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        return batch[0], batch[1], batch[2]
    raise ValueError(
        "Expected each batch to be (x, y, meta). "
        f"Got type={type(batch)}, repr={repr(batch)[:200]}"
    )


def _get_example_input(loader: Iterable, device: str = "cpu") -> torch.Tensor:
    batch = next(iter(loader))
    x, y, meta = _unpack_batch(batch)
    return x.to(device)


def _to_numpy_x(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32, copy=False)


def _to_numpy_y(y: torch.Tensor) -> np.ndarray:
    return y.detach().cpu().numpy().astype(np.float32, copy=False)


def _extract_main_pred_from_torch_output(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Model output must be Tensor or tuple/list whose first item is Tensor, got {type(output)}")
    return output


def _extract_main_pred_from_onnx_outputs(outputs: List[np.ndarray], output_index: int = 0) -> np.ndarray:
    if not isinstance(outputs, list) or len(outputs) == 0:
        raise RuntimeError("ONNX Runtime returned no outputs.")

    pred = outputs[output_index]
    if pred.ndim > 2:
        pred = pred.reshape(pred.shape[0], -1)

    return pred.astype(np.float32, copy=False)


class FirstOutputWrapper(torch.nn.Module):
    """
    Wrap a model so ONNX export only keeps the main prediction output.
    This is useful when your model returns (pred, aux) or similar.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


# -----------------------------
# PyTorch evaluation
# -----------------------------

def evaluate_torch_bp(
    test_loader: Iterable,
    model: torch.nn.Module,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            x, y, meta = _unpack_batch(batch)
            x = x.to(device, non_blocking=True)

            output = model(x)
            pred = _extract_main_pred_from_torch_output(output)

            preds.append(pred.detach().cpu().numpy().astype(np.float32, copy=False))
            targets.append(_to_numpy_y(y))

    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


# -----------------------------
# ONNX export / runtime evaluation
# -----------------------------

def export_torch_to_onnx_lazy(
    model: torch.nn.Module,
    example_x: torch.Tensor,
    output_onnx_path: str | Path,
    input_name: str = "input",
    output_name: str = "output",
    opset_version: int = 17,
    dynamic_batch: bool = True,
    device: str = "cuda",
    force: bool = False,
    use_dynamo_export: bool = False,
) -> Path:
    """
    Export PyTorch model to ONNX.
    By default, uses the classic torch.onnx.export path for compatibility.
    If your PyTorch version supports and prefers dynamo=True, set use_dynamo_export=True.
    """
    output_onnx_path = Path(output_onnx_path)
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if output_onnx_path.exists() and not force:
        print(f"[SKIP] FP32 ONNX already exists: {output_onnx_path} ({model_size_mb(output_onnx_path):.3f} MB)")
        return output_onnx_path

    model.eval()
    wrapped = FirstOutputWrapper(model).to(device).eval()
    example_x = example_x.to(device).float()

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            input_name: {0: "batch"},
            output_name: {0: "batch"},
        }

    print(f"[EXPORT] Exporting FP32 ONNX to: {output_onnx_path}")
    print(f"[EXPORT] Example input shape: {tuple(example_x.shape)}")

    with torch.no_grad():
        export_kwargs = dict(
            args=(example_x,),
            f=str(output_onnx_path),
            input_names=[input_name],
            output_names=[output_name],
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )

        if use_dynamo_export:
            try:
                torch.onnx.export(
                    wrapped,
                    **export_kwargs,
                    dynamo=True,
                )
            except TypeError:
                print("[WARN] torch.onnx.export(dynamo=True) not supported. Falling back to classic export.")
                torch.onnx.export(wrapped, **export_kwargs)
        else:
            torch.onnx.export(wrapped, **export_kwargs)

    print(f"[DONE] FP32 ONNX: {output_onnx_path} ({model_size_mb(output_onnx_path):.3f} MB)")
    return output_onnx_path


def create_onnx_session(
    onnx_path: str | Path,
    providers: Optional[List[str]] = None,
    intra_op_num_threads: Optional[int] = None,
    print_io: bool = True,
) -> Any:
    if ort is None:
        raise ImportError("onnxruntime is not installed. Please run: pip install onnxruntime")

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")

    if providers is None:
        providers = ["CPUExecutionProvider"]

    sess_options = ort.SessionOptions()
    if intra_op_num_threads is not None:
        sess_options.intra_op_num_threads = int(intra_op_num_threads)

    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=providers,
    )

    if print_io:
        print("\n===== ONNX model IO =====")
        print(f"Model: {onnx_path}")
        print(f"Size : {model_size_mb(onnx_path):.3f} MB")
        for i in sess.get_inputs():
            print("INPUT :", i.name, i.shape, i.type)
        for o in sess.get_outputs():
            print("OUTPUT:", o.name, o.shape, o.type)

    return sess


def evaluate_onnx_bp(
    test_loader: Iterable,
    onnx_path: str | Path,
    providers: Optional[List[str]] = None,
    output_index: int = 0,
    intra_op_num_threads: Optional[int] = None,
    print_io: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    sess = create_onnx_session(
        onnx_path,
        providers=providers,
        intra_op_num_threads=intra_op_num_threads,
        print_io=print_io,
    )

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for batch in test_loader:
        x, y, meta = _unpack_batch(batch)

        # x shape is expected to be [B, K, 1, L].
        # Even for INT8 ONNX, input is usually still float32.
        x_np = _to_numpy_x(x)
        y_np = _to_numpy_y(y)

        outputs = sess.run(output_names, {input_name: x_np})
        pred_np = _extract_main_pred_from_onnx_outputs(outputs, output_index=output_index)

        preds.append(pred_np)
        targets.append(y_np)

    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


# -----------------------------
# Static INT8 quantization
# -----------------------------

class TorchLoaderCalibrationDataReader:
    """
    ONNX Runtime CalibrationDataReader-compatible object.

    It reads x from batches formatted as:
        x, y, meta = batch

    x is passed to ONNX as float32. Quantization calibration observes activation ranges.
    """
    def __init__(
        self,
        loader: Iterable,
        input_name: str,
        max_batches: int = 32,
    ):
        try:
            from onnxruntime.quantization import CalibrationDataReader
        except Exception as e:
            raise ImportError(
                "onnxruntime quantization tools are unavailable. "
                "Please install/upgrade onnxruntime: pip install -U onnxruntime"
            ) from e

        # Dynamically inherit is not needed; quantize_static only requires get_next().
        self.loader = loader
        self.input_name = input_name
        self.max_batches = int(max_batches)
        self._iter = iter(loader)
        self._count = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._count >= self.max_batches:
            return None

        try:
            batch = next(self._iter)
        except StopIteration:
            return None

        x, y, meta = _unpack_batch(batch)
        x_np = _to_numpy_x(x)

        self._count += 1
        return {self.input_name: x_np}


def quantize_onnx_static_int8_lazy(
    fp32_onnx_path: str | Path,
    int8_onnx_path: str | Path,
    calib_loader: Iterable,
    input_name: str = "input",
    max_calib_batches: int = 32,
    force: bool = False,
    quant_format: str = "QDQ",
    activation_type: str = "QInt8",
    weight_type: str = "QInt8",
    per_channel: bool = False,
    reduce_range: bool = False,
) -> Path:
    fp32_onnx_path = Path(fp32_onnx_path)
    int8_onnx_path = Path(int8_onnx_path)
    int8_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not fp32_onnx_path.exists():
        raise FileNotFoundError(f"FP32 ONNX does not exist: {fp32_onnx_path}")

    if int8_onnx_path.exists() and not force:
        print(f"[SKIP] INT8 ONNX already exists: {int8_onnx_path} ({model_size_mb(int8_onnx_path):.3f} MB)")
        return int8_onnx_path

    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except Exception as e:
        raise ImportError(
            "onnxruntime quantization tools are unavailable. "
            "Please install/upgrade onnxruntime: pip install -U onnxruntime"
        ) from e

    qformat = QuantFormat.QDQ if quant_format.upper() == "QDQ" else QuantFormat.QOperator

    act_type = QuantType.QInt8 if activation_type.lower() in ["qint8", "s8", "int8"] else QuantType.QUInt8
    wgt_type = QuantType.QInt8 if weight_type.lower() in ["qint8", "s8", "int8"] else QuantType.QUInt8

    print(f"[QUANT] Static INT8 quantization")
    print(f"[QUANT] Input : {fp32_onnx_path} ({model_size_mb(fp32_onnx_path):.3f} MB)")
    print(f"[QUANT] Output: {int8_onnx_path}")
    print(f"[QUANT] format={quant_format}, activation={activation_type}, weight={weight_type}, max_calib_batches={max_calib_batches}")

    reader = TorchLoaderCalibrationDataReader(
        loader=calib_loader,
        input_name=input_name,
        max_batches=max_calib_batches,
    )

    quantize_static(
        model_input=str(fp32_onnx_path),
        model_output=str(int8_onnx_path),
        calibration_data_reader=reader,
        quant_format=qformat,
        activation_type=act_type,
        weight_type=wgt_type,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=per_channel,
        reduce_range=reduce_range,
    )

    print(f"[DONE] INT8 ONNX: {int8_onnx_path} ({model_size_mb(int8_onnx_path):.3f} MB)")
    return int8_onnx_path


# -----------------------------
# Saving / comparison helpers
# -----------------------------

def _safe_meta_value(v: Any, idx: int) -> Any:
    if torch.is_tensor(v):
        v = v.detach().cpu().numpy()

    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        if len(v) > idx:
            item = v[idx]
            return item.item() if np.ndim(item) == 0 else item

    if isinstance(v, (list, tuple)):
        if len(v) > idx:
            return v[idx]

    return v


def collect_prediction_rows(
    test_loader: Iterable,
    preds: np.ndarray,
    targets: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    cursor = 0

    for batch in test_loader:
        x, y, meta = _unpack_batch(batch)
        bs = y.shape[0]

        for i in range(bs):
            row: Dict[str, Any] = {
                "y_true_sbp": float(targets[cursor + i, 0]),
                "y_true_dbp": float(targets[cursor + i, 1]),
                "y_pred_sbp": float(preds[cursor + i, 0]),
                "y_pred_dbp": float(preds[cursor + i, 1]),
            }

            if isinstance(meta, dict):
                for k, v in meta.items():
                    try:
                        val = _safe_meta_value(v, i)
                        if isinstance(val, np.generic):
                            val = val.item()
                        row[str(k)] = val
                    except Exception:
                        pass

            rows.append(row)

        cursor += bs

    return pd.DataFrame(rows)


def save_eval_outputs(
    name: str,
    preds: np.ndarray,
    targets: np.ndarray,
    report: Dict[str, Dict[str, float]],
    test_loader: Iterable,
    out_dir: str | Path,
    save_rows: bool = True,
) -> Optional[pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"metrics_{name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    rows = None
    if save_rows:
        rows = collect_prediction_rows(test_loader, preds, targets)
        rows.to_csv(out_dir / f"predictions_{name}.csv", index=False)

    return rows


def compare_prediction_arrays(
    preds_ref: np.ndarray,
    preds_test: np.ndarray,
    name_ref: str = "torch_fp32",
    name_test: str = "onnx_int8",
) -> Dict[str, float]:
    preds_ref = np.asarray(preds_ref, dtype=np.float32)
    preds_test = np.asarray(preds_test, dtype=np.float32)

    if preds_ref.shape != preds_test.shape:
        raise ValueError(f"Prediction shape mismatch: {preds_ref.shape} vs {preds_test.shape}")

    diff = np.abs(preds_ref - preds_test)

    report = {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "p95_abs_diff": float(np.percentile(diff, 95)),
        "sbp_mean_abs_diff": float(diff[:, 0].mean()),
        "dbp_mean_abs_diff": float(diff[:, 1].mean()),
    }

    print(f"\n===== {name_ref} vs {name_test} output diff =====")
    for k, v in report.items():
        print(f"{k}: {v:.6f}")

    return report


# -----------------------------
# Super-lazy all-in-one entry point
# -----------------------------

def run_lazy_torch_onnx_int8_eval(
    model: torch.nn.Module,
    test_loader: Iterable,
    device: str = "cuda",
    work_dir: str | Path = "onnx_lazy_eval",
    fp32_onnx_name: str = "model_fp32.onnx",
    int8_onnx_name: str = "model_int8.onnx",
    do_int8: bool = True,
    calib_loader: Optional[Iterable] = None,
    max_calib_batches: int = 32,
    force_export: bool = False,
    force_quant: bool = False,
    input_name: str = "input",
    output_name: str = "output",
    output_index: int = 0,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    providers: Optional[List[str]] = None,
    intra_op_num_threads: Optional[int] = None,
    save_rows: bool = True,
    use_dynamo_export: bool = False,
    quant_format: str = "QDQ",
    activation_type: str = "QInt8",
    weight_type: str = "QInt8",
    per_channel: bool = False,
    reduce_range: bool = False,
) -> Dict[str, Any]:
    """
    One-call function for your case.

    It assumes:
        test_loader returns x, y, meta
        x is [B, K, 1, L]
        y is [B, 2]

    Returns a dictionary containing predictions, metrics, model paths, and diffs.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    fp32_onnx_path = work_dir / fp32_onnx_name
    int8_onnx_path = work_dir / int8_onnx_name

    if providers is None:
        providers = ["CPUExecutionProvider"]

    print("\n==============================")
    print(" Lazy Torch -> ONNX -> INT8 BP Eval")
    print("==============================")
    print(f"work_dir: {work_dir}")

    # 0) Example input from loader
    example_x = _get_example_input(test_loader, device=device)

    # 1) Torch FP32 eval
    print("\n[1/5] Evaluating PyTorch FP32 model...")
    preds_torch, targets = evaluate_torch_bp(
        test_loader=test_loader,
        model=model,
        device=device,
    )
    report_torch = bp_metrics(preds_torch, targets)
    print_bp_report(report_torch, "PyTorch FP32")
    save_eval_outputs("torch_fp32", preds_torch, targets, report_torch, test_loader, work_dir, save_rows=save_rows)

    # 2) Export FP32 ONNX
    print("\n[2/5] Exporting FP32 ONNX if needed...")
    export_torch_to_onnx_lazy(
        model=model,
        example_x=example_x,
        output_onnx_path=fp32_onnx_path,
        input_name=input_name,
        output_name=output_name,
        opset_version=opset_version,
        dynamic_batch=dynamic_batch,
        device=device,
        force=force_export,
        use_dynamo_export=use_dynamo_export,
    )

    # 3) FP32 ONNX eval
    print("\n[3/5] Evaluating FP32 ONNX model...")
    preds_onnx_fp32, targets_onnx = evaluate_onnx_bp(
        test_loader=test_loader,
        onnx_path=fp32_onnx_path,
        providers=providers,
        output_index=output_index,
        intra_op_num_threads=intra_op_num_threads,
        print_io=True,
    )
    report_onnx_fp32 = bp_metrics(preds_onnx_fp32, targets_onnx)
    print_bp_report(report_onnx_fp32, "ONNX FP32")
    save_eval_outputs("onnx_fp32", preds_onnx_fp32, targets_onnx, report_onnx_fp32, test_loader, work_dir, save_rows=save_rows)

    diffs: Dict[str, Dict[str, float]] = {}
    diffs["torch_vs_onnx_fp32"] = compare_prediction_arrays(
        preds_torch,
        preds_onnx_fp32,
        name_ref="torch_fp32",
        name_test="onnx_fp32",
    )

    # 4) INT8 quantization
    preds_onnx_int8 = None
    report_onnx_int8 = None

    if do_int8:
        print("\n[4/5] Creating INT8 ONNX if needed...")
        if calib_loader is None:
            print("[INFO] calib_loader is None, using test_loader for calibration subset.")
            print("[INFO] For formal evaluation, use a separate representative calibration loader if possible.")
            calib_loader = test_loader

        quantize_onnx_static_int8_lazy(
            fp32_onnx_path=fp32_onnx_path,
            int8_onnx_path=int8_onnx_path,
            calib_loader=calib_loader,
            input_name=input_name,
            max_calib_batches=max_calib_batches,
            force=force_quant,
            quant_format=quant_format,
            activation_type=activation_type,
            weight_type=weight_type,
            per_channel=per_channel,
            reduce_range=reduce_range,
        )

        # 5) INT8 ONNX eval
        print("\n[5/5] Evaluating INT8 ONNX model...")
        preds_onnx_int8, targets_int8 = evaluate_onnx_bp(
            test_loader=test_loader,
            onnx_path=int8_onnx_path,
            providers=providers,
            output_index=output_index,
            intra_op_num_threads=intra_op_num_threads,
            print_io=True,
        )
        report_onnx_int8 = bp_metrics(preds_onnx_int8, targets_int8)
        print_bp_report(report_onnx_int8, "ONNX INT8")
        save_eval_outputs("onnx_int8", preds_onnx_int8, targets_int8, report_onnx_int8, test_loader, work_dir, save_rows=save_rows)

        diffs["torch_vs_onnx_int8"] = compare_prediction_arrays(
            preds_torch,
            preds_onnx_int8,
            name_ref="torch_fp32",
            name_test="onnx_int8",
        )
        diffs["onnx_fp32_vs_onnx_int8"] = compare_prediction_arrays(
            preds_onnx_fp32,
            preds_onnx_int8,
            name_ref="onnx_fp32",
            name_test="onnx_int8",
        )
    else:
        print("\n[4/5] Skipping INT8 quantization.")
        print("[5/5] Skipping INT8 evaluation.")

    # Save summary
    summary = {
        "paths": {
            "work_dir": str(work_dir),
            "fp32_onnx": str(fp32_onnx_path),
            "int8_onnx": str(int8_onnx_path) if do_int8 else None,
        },
        "sizes_mb": {
            "fp32_onnx": model_size_mb(fp32_onnx_path) if fp32_onnx_path.exists() else None,
            "int8_onnx": model_size_mb(int8_onnx_path) if int8_onnx_path.exists() else None,
        },
        "metrics": {
            "torch_fp32": report_torch,
            "onnx_fp32": report_onnx_fp32,
            "onnx_int8": report_onnx_int8,
        },
        "diffs": diffs,
    }

    with open(work_dir / "lazy_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== Model sizes =====")
    print(f"ONNX FP32: {summary['sizes_mb']['fp32_onnx']:.3f} MB")
    if summary["sizes_mb"]["int8_onnx"] is not None:
        print(f"ONNX INT8: {summary['sizes_mb']['int8_onnx']:.3f} MB")

    print(f"\nSaved summary: {work_dir / 'lazy_eval_summary.json'}")

    return {
        "summary": summary,
        "preds": {
            "torch_fp32": preds_torch,
            "onnx_fp32": preds_onnx_fp32,
            "onnx_int8": preds_onnx_int8,
        },
        "targets": targets,
        "paths": {
            "work_dir": work_dir,
            "fp32_onnx": fp32_onnx_path,
            "int8_onnx": int8_onnx_path,
        },
    }
