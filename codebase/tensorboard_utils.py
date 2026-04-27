from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


def create_summary_writer(
    log_dir: Path,
    enabled: bool = True,
    flush_secs: int = 30,
):
    if not enabled:
        return None, None

    writer_cls = None
    backend = None
    try:
        from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter

        writer_cls = TorchSummaryWriter
        backend = "torch.utils.tensorboard"
    except Exception:
        try:
            from tensorboardX import SummaryWriter as TensorboardXSummaryWriter

            writer_cls = TensorboardXSummaryWriter
            backend = "tensorboardX"
        except Exception:
            return None, None

    log_dir.mkdir(parents=True, exist_ok=True)
    return writer_cls(log_dir=str(log_dir), flush_secs=flush_secs), backend


def _is_scalar_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.isfinite(float(value))
    return False


def _sanitize_tag(tag: str) -> str:
    return str(tag).replace(" ", "_")


def add_scalar_dict(writer, metrics: Dict[str, Any], step: int, prefix: str = "") -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        if _is_scalar_number(value):
            tag = "{}/{}".format(prefix, _sanitize_tag(key)) if prefix else _sanitize_tag(key)
            writer.add_scalar(tag, float(value), step)


def add_per_class_metrics(
    writer,
    per_class_metrics: Iterable[Dict[str, Any]],
    step: int,
    prefix: str = "per_class",
) -> None:
    if writer is None:
        return
    for class_metrics in per_class_metrics:
        class_name = class_metrics.get("class_name", class_metrics.get("class_idx", "unknown"))
        class_prefix = "{}/{}".format(prefix, _sanitize_tag(str(class_name)))
        add_scalar_dict(writer, class_metrics, step=step, prefix=class_prefix)


def add_confusion_matrix_text(writer, matrix: Any, step: int, tag: str = "val/confusion_matrix") -> None:
    if writer is None or matrix is None:
        return
    matrix_arr = np.asarray(matrix)
    writer.add_text(_sanitize_tag(tag), np.array2string(matrix_arr, separator=", "), step)


def flush_and_close(writer) -> None:
    if writer is None:
        return
    writer.flush()
    writer.close()
