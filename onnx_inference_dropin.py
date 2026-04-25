import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import onnxruntime as ort
except Exception:
    ort = None


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


def create_onnx_session(
    onnx_path: str,
    providers: Optional[List[str]] = None,
    intra_op_num_threads: Optional[int] = None,
) -> Any:
    if ort is None:
        raise ImportError("onnxruntime is not installed. Please run: pip install onnxruntime")
    if providers is None:
        providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    if intra_op_num_threads is not None:
        sess_options.intra_op_num_threads = int(intra_op_num_threads)
    sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)

    print("\n===== ONNX model IO =====")
    for i in sess.get_inputs():
        print("INPUT :", i.name, i.shape, i.type)
    for o in sess.get_outputs():
        print("OUTPUT:", o.name, o.shape, o.type)

    return sess


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
            x, y, meta = batch
            x = x.to(device, non_blocking=True)
            output = model(x)
            pred = _extract_main_pred_from_torch_output(output)
            preds.append(pred.detach().cpu().numpy().astype(np.float32, copy=False))
            targets.append(_to_numpy_y(y))

    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def evaluate_onnx_bp(
    test_loader: Iterable,
    onnx_path: str,
    providers: Optional[List[str]] = None,
    output_index: int = 0,
    intra_op_num_threads: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    sess = create_onnx_session(
        onnx_path,
        providers=providers,
        intra_op_num_threads=intra_op_num_threads,
    )
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for batch in test_loader:
        x, y, meta = batch
        # x is expected to be [B, K, 1, L].
        # Even for INT8 ONNX, input is usually still float32.
        x_np = _to_numpy_x(x)
        y_np = _to_numpy_y(y)
        outputs = sess.run(output_names, {input_name: x_np})
        pred_np = _extract_main_pred_from_onnx_outputs(outputs, output_index=output_index)
        preds.append(pred_np)
        targets.append(y_np)

    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


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
        x, y, meta = batch
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


def evaluate_bp_backend(
    test_loader: Iterable,
    backend: str,
    torch_model: Optional[torch.nn.Module] = None,
    onnx_path: Optional[str] = None,
    device: str = "cuda",
    providers: Optional[List[str]] = None,
    output_index: int = 0,
    save_csv: Optional[str] = None,
    save_json: Optional[str] = None,
    collect_meta: bool = True,
    intra_op_num_threads: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]], Optional[pd.DataFrame]]:
    backend = backend.lower().strip()

    if backend == "torch":
        if torch_model is None:
            raise ValueError("torch_model must be provided when backend='torch'.")
        preds, targets = evaluate_torch_bp(test_loader=test_loader, model=torch_model, device=device)
    elif backend == "onnx":
        if onnx_path is None:
            raise ValueError("onnx_path must be provided when backend='onnx'.")
        preds, targets = evaluate_onnx_bp(
            test_loader=test_loader,
            onnx_path=onnx_path,
            providers=providers,
            output_index=output_index,
            intra_op_num_threads=intra_op_num_threads,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'torch' or 'onnx'.")

    report = bp_metrics(preds, targets)
    print_bp_report(report, title=f"{backend.upper()} BP Evaluation")

    rows = None
    if collect_meta or save_csv is not None:
        rows = collect_prediction_rows(test_loader, preds, targets)

    if save_csv is not None:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(save_csv, index=False)
        print(f"Saved predictions: {save_csv}")

    if save_json is not None:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved report: {save_json}")

    return preds, targets, report, rows


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
