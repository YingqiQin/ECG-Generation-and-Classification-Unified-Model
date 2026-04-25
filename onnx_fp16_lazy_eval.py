from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from onnx_lazy_bp_eval import (
    _get_example_input,
    bp_metrics,
    compare_prediction_arrays,
    evaluate_onnx_bp,
    evaluate_torch_bp,
    export_torch_to_onnx_lazy,
    model_size_mb,
    print_bp_report,
    save_eval_outputs,
)


def convert_onnx_to_fp16_lazy(
    fp32_onnx_path: str | Path,
    fp16_onnx_path: str | Path,
    *,
    force: bool = False,
    keep_io_types: bool = True,
    disable_shape_infer: bool = False,
    min_positive_val: float = 1e-7,
    max_finite_val: float = 1e4,
    op_block_list: Optional[List[str]] = None,
    node_block_list: Optional[List[str]] = None,
) -> Path:
    """
    Convert FP32 ONNX to FP16 ONNX.

    keep_io_types=True keeps model inputs/outputs as float32, which is usually
    the most convenient choice for your existing dataloader and evaluation code.
    """
    try:
        import onnx
        from onnxconverter_common import float16
    except Exception as e:
        raise ImportError(
            "FP16 conversion requires onnx and onnxconverter-common. "
            "Install with: pip install onnx onnxconverter-common"
        ) from e

    fp32_onnx_path = Path(fp32_onnx_path)
    fp16_onnx_path = Path(fp16_onnx_path)
    fp16_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not fp32_onnx_path.exists():
        raise FileNotFoundError(f"FP32 ONNX does not exist: {fp32_onnx_path}")

    if fp16_onnx_path.exists() and not force:
        print(f"[SKIP] FP16 ONNX already exists: {fp16_onnx_path} ({model_size_mb(fp16_onnx_path):.3f} MB)")
        return fp16_onnx_path

    print("[FP16] Converting FP32 ONNX to FP16")
    print(f"[FP16] Input : {fp32_onnx_path} ({model_size_mb(fp32_onnx_path):.3f} MB)")
    print(f"[FP16] Output: {fp16_onnx_path}")
    print(f"[FP16] keep_io_types={keep_io_types}")

    model = onnx.load(str(fp32_onnx_path))
    model_fp16 = float16.convert_float_to_float16(
        model,
        min_positive_val=min_positive_val,
        max_finite_val=max_finite_val,
        keep_io_types=keep_io_types,
        disable_shape_infer=disable_shape_infer,
        op_block_list=op_block_list,
        node_block_list=node_block_list,
    )
    onnx.save(model_fp16, str(fp16_onnx_path))

    print(f"[DONE] FP16 ONNX: {fp16_onnx_path} ({model_size_mb(fp16_onnx_path):.3f} MB)")
    return fp16_onnx_path


def run_lazy_torch_onnx_fp16_eval(
    model: torch.nn.Module,
    test_loader: Iterable,
    *,
    device: str = "cuda",
    work_dir: str | Path = "onnx_fp16_eval",
    fp32_onnx_name: str = "model_fp32.onnx",
    fp16_onnx_name: str = "model_fp16.onnx",
    force_export: bool = False,
    force_fp16: bool = False,
    input_name: str = "input",
    output_name: str = "output",
    output_index: int = 0,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    providers: Optional[List[str]] = None,
    intra_op_num_threads: Optional[int] = None,
    save_rows: bool = True,
    use_dynamo_export: bool = False,
    keep_io_types: bool = True,
    disable_shape_infer: bool = False,
    min_positive_val: float = 1e-7,
    max_finite_val: float = 1e4,
    op_block_list: Optional[List[str]] = None,
    node_block_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    One-call FP16 pipeline.

    It does:
        1. PyTorch FP32 evaluation
        2. Export FP32 ONNX if needed
        3. Evaluate FP32 ONNX
        4. Convert FP32 ONNX to FP16 ONNX if needed
        5. Evaluate FP16 ONNX
        6. Compare outputs and save reports
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    fp32_onnx_path = work_dir / fp32_onnx_name
    fp16_onnx_path = work_dir / fp16_onnx_name

    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print("\n==============================")
    print(" Lazy Torch -> ONNX FP16 BP Eval")
    print("==============================")
    print(f"work_dir: {work_dir}")

    example_x = _get_example_input(test_loader, device=device)

    print("\n[1/5] Evaluating PyTorch FP32 model...")
    preds_torch, targets = evaluate_torch_bp(
        test_loader=test_loader,
        model=model,
        device=device,
    )
    report_torch = bp_metrics(preds_torch, targets)
    print_bp_report(report_torch, "PyTorch FP32")
    save_eval_outputs("torch_fp32", preds_torch, targets, report_torch, test_loader, work_dir, save_rows=save_rows)

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

    print("\n[4/5] Creating FP16 ONNX if needed...")
    convert_onnx_to_fp16_lazy(
        fp32_onnx_path=fp32_onnx_path,
        fp16_onnx_path=fp16_onnx_path,
        force=force_fp16,
        keep_io_types=keep_io_types,
        disable_shape_infer=disable_shape_infer,
        min_positive_val=min_positive_val,
        max_finite_val=max_finite_val,
        op_block_list=op_block_list,
        node_block_list=node_block_list,
    )

    print("\n[5/5] Evaluating FP16 ONNX model...")
    preds_onnx_fp16, targets_fp16 = evaluate_onnx_bp(
        test_loader=test_loader,
        onnx_path=fp16_onnx_path,
        providers=providers,
        output_index=output_index,
        intra_op_num_threads=intra_op_num_threads,
        print_io=True,
    )
    report_onnx_fp16 = bp_metrics(preds_onnx_fp16, targets_fp16)
    print_bp_report(report_onnx_fp16, "ONNX FP16")
    save_eval_outputs("onnx_fp16", preds_onnx_fp16, targets_fp16, report_onnx_fp16, test_loader, work_dir, save_rows=save_rows)

    diffs: Dict[str, Dict[str, float]] = {}
    diffs["torch_vs_onnx_fp32"] = compare_prediction_arrays(
        preds_torch,
        preds_onnx_fp32,
        name_ref="torch_fp32",
        name_test="onnx_fp32",
    )
    diffs["torch_vs_onnx_fp16"] = compare_prediction_arrays(
        preds_torch,
        preds_onnx_fp16,
        name_ref="torch_fp32",
        name_test="onnx_fp16",
    )
    diffs["onnx_fp32_vs_onnx_fp16"] = compare_prediction_arrays(
        preds_onnx_fp32,
        preds_onnx_fp16,
        name_ref="onnx_fp32",
        name_test="onnx_fp16",
    )

    summary = {
        "paths": {
            "work_dir": str(work_dir),
            "fp32_onnx": str(fp32_onnx_path),
            "fp16_onnx": str(fp16_onnx_path),
        },
        "sizes_mb": {
            "fp32_onnx": model_size_mb(fp32_onnx_path) if fp32_onnx_path.exists() else None,
            "fp16_onnx": model_size_mb(fp16_onnx_path) if fp16_onnx_path.exists() else None,
        },
        "metrics": {
            "torch_fp32": report_torch,
            "onnx_fp32": report_onnx_fp32,
            "onnx_fp16": report_onnx_fp16,
        },
        "diffs": diffs,
    }

    with open(work_dir / "fp16_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== Model sizes =====")
    print(f"ONNX FP32: {summary['sizes_mb']['fp32_onnx']:.3f} MB")
    print(f"ONNX FP16: {summary['sizes_mb']['fp16_onnx']:.3f} MB")
    print(f"\nSaved summary: {work_dir / 'fp16_eval_summary.json'}")

    return {
        "summary": summary,
        "preds": {
            "torch_fp32": preds_torch,
            "onnx_fp32": preds_onnx_fp32,
            "onnx_fp16": preds_onnx_fp16,
        },
        "targets": targets,
        "paths": {
            "work_dir": work_dir,
            "fp32_onnx": fp32_onnx_path,
            "fp16_onnx": fp16_onnx_path,
        },
    }
