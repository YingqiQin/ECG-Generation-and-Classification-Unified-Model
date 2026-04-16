from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn


def extract_state_dict(state: Any) -> dict[str, torch.Tensor]:
    if isinstance(state, nn.Module):
        state = state.state_dict()
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a valid state dict")

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_checkpoint_state_dict(
    ckpt_path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    state = load_checkpoint_payload(ckpt_path=ckpt_path, device=device)
    return extract_state_dict(state)


def load_checkpoint_payload(
    ckpt_path: str | Path,
    device: torch.device | str = "cpu",
) -> Any:
    return torch.load(Path(ckpt_path), map_location=device)


def extract_embedded_config(state: Any) -> dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    inference_config = state.get("inference_config")
    if isinstance(inference_config, dict):
        return copy.deepcopy(inference_config)
    config = state.get("config")
    if isinstance(config, dict):
        return copy.deepcopy(config)
    return None


def build_upperarm_inference_bundle(
    *,
    model: nn.Module,
    config: dict[str, Any],
    epoch: int | None = None,
    val_loss: float | None = None,
    val_pcc: float | None = None,
    source_checkpoint_name: str | None = None,
) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    reconstruct_cfg = config.get("reconstruct", {})
    inference_data_keys = [
        "dataset_type",
        "file_glob",
        "segment_group_regex",
        "segment_offset_regex",
        "npz_timestamp_key",
        "npz_sampling_rate_key",
        "npz_start_time_key",
        "npz_signal_matrix_key",
        "npz_channel_names_key",
        "input_channel",
        "target_channels",
        "segment_length",
        "segment_stride",
        "segment_policy",
        "padding_mode",
        "apply_filter",
        "normalize_mode",
        "fallback_fs",
        "target_fs",
    ]
    inference_config = {
        "seed": config.get("seed", 42),
        "data": {
            key: copy.deepcopy(data_cfg[key])
            for key in inference_data_keys
            if key in data_cfg
        },
        "model": copy.deepcopy(config.get("model", {})),
        "reconstruct": {
            "corr_method": reconstruct_cfg.get("corr_method", "pearson"),
            "visual_filter_mode": reconstruct_cfg.get("visual_filter_mode", "recon_only"),
            "enable_lag_metrics": reconstruct_cfg.get("enable_lag_metrics", True),
            "lag_search_window_ms": reconstruct_cfg.get("lag_search_window_ms", 150.0),
            "dtw_max_points": reconstruct_cfg.get("dtw_max_points", 2000),
            "rpeak_tolerance_ms": reconstruct_cfg.get("rpeak_tolerance_ms", 120.0),
        },
    }
    return {
        "bundle_type": "mcma_upperarm_inference",
        "bundle_version": 1,
        "model": extract_state_dict(model),
        "inference_config": inference_config,
        "metadata": {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_pcc": val_pcc,
            "source_checkpoint_name": source_checkpoint_name,
        },
    }


def load_shape_matched_checkpoint(
    model: nn.Module,
    ckpt_path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    state_dict = load_checkpoint_state_dict(ckpt_path=ckpt_path, device=device)
    model_state = model.state_dict()

    matched: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    unexpected: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        matched[key] = value

    missing = model.load_state_dict(matched, strict=False)
    return {
        "matched_keys": sorted(matched.keys()),
        "matched_count": len(matched),
        "missing_keys": list(missing.missing_keys),
        "unexpected_keys": sorted(unexpected),
        "skipped_shape_keys": sorted(skipped_shape),
    }
