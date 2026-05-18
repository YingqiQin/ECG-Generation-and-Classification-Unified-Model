# deploy_friendly_bp_model_2d_nchw_k160_no_expand.py
# -*- coding: utf-8 -*-
"""
2D-Conv deploy-friendly BP model with direct NCHW event input.

Deployment input:
    x: [1, 1, 160, 200]
       N=1, C=1, H=160, W=200

Semantic meaning:
    H=160 is the fixed K=160 8-second PPG segments from one BP event/window.
    W=200 is 25Hz * 8s.

Model output:
    pred: [1, 2]
          one event-level SBP/DBP prediction.

Why this version exists
-----------------------
The operator team prefers a standard NCHW input:
    [1,1,160,200]

This version avoids exposing N=160 as the ONNX input batch dimension.
It processes all 160 segments as the H dimension and applies Conv2d kernels
of size (1,k), so the convolution only runs along the temporal W dimension.
The H=160 segment dimension is preserved until segment aggregation.

Equivalence to original logic
-----------------------------
Original model:
    input [1,160,1,200]
    -> reshape [160,1,200]
    -> Conv1d encoder independently per segment
    -> [1,160,D]
    -> attention pooling over K
    -> [1,2]

This model:
    input [1,1,160,200]
    -> Conv2d(kernel=(1,k)) preserving H=160
    -> encoder output [1,D,160,1]
    -> reshape/transpose to [1,160,D]
    -> attention pooling over K
    -> [1,2]

Since kernel height is always 1, segments are not mixed inside the encoder.
Segment mixing only happens in the attention pooling stage, same as before.

Checkpoint loading
------------------
Original Conv1d weights:
    [out, in/groups, k]
Converted Conv2d weights:
    [out, in/groups, 1, k]

Original encoder final Linear weights:
    [out_features, 1280]
Converted final Conv2d-1x1 weights:
    [out_features, 1280, 1, 1]

load_1d_checkpoint_into_nchw_2d_model() handles both conversions.

No-Expand deployment note
-------------------------
The operator team does not support ONNX Expand, but allows implicit broadcast.
Therefore, this version intentionally avoids explicit tensor expansion in SE and
attention pooling. Broadcast is left implicit in Mul operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


# =============================================================================
# Utilities
# =============================================================================

def conv_out_len(L: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    return int((L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def load_state_dict_from_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    return strip_module_prefix(sd)


def convert_1d_state_dict_to_nchw_2d(
    source_sd: Dict[str, torch.Tensor],
    target_model: nn.Module,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convert original 1D checkpoint weights to NCHW 2D deployment model.

    Handles:
    1) Conv1d -> Conv2d:
        [out, in/groups, k] -> [out, in/groups, 1, k]

    2) Linear -> Conv2d 1x1:
        [out, in] -> [out, in, 1, 1]

    BN / LayerNorm / Linear head params are reused when shapes match.
    """
    target_sd = target_model.state_dict()
    converted = {}
    skipped = []

    for k, v in source_sd.items():
        if k not in target_sd:
            skipped.append((k, "not_in_target", tuple(v.shape)))
            continue

        tgt = target_sd[k]

        # Same shape: BN, LayerNorm, attention scorer Linear, final head Linear, etc.
        if tuple(v.shape) == tuple(tgt.shape):
            converted[k] = v
            continue

        # Conv1d -> Conv2d with height=1.
        if (
            v.ndim == 3
            and tgt.ndim == 4
            and v.shape[0] == tgt.shape[0]
            and v.shape[1] == tgt.shape[1]
            and v.shape[2] == tgt.shape[3]
            and tgt.shape[2] == 1
        ):
            converted[k] = v.unsqueeze(2)
            continue

        # Linear -> Conv2d 1x1.
        if (
            v.ndim == 2
            and tgt.ndim == 4
            and v.shape[0] == tgt.shape[0]
            and v.shape[1] == tgt.shape[1]
            and tgt.shape[2] == 1
            and tgt.shape[3] == 1
        ):
            converted[k] = v.unsqueeze(2).unsqueeze(3)
            continue

        skipped.append((k, f"shape_mismatch_source={tuple(v.shape)}_target={tuple(tgt.shape)}", tuple(v.shape)))

    if verbose:
        print(f"[convert] source keys={len(source_sd)}, converted/kept={len(converted)}, skipped={len(skipped)}")
        if skipped:
            print("[convert] first skipped keys:")
            for item in skipped[:40]:
                print("  ", item)

    return converted


def load_1d_checkpoint_into_nchw_2d_model(
    model_2d: nn.Module,
    ckpt_path: str,
    strict: bool = False,
    verbose: bool = True,
):
    source_sd = load_state_dict_from_checkpoint(ckpt_path)
    converted_sd = convert_1d_state_dict_to_nchw_2d(source_sd, model_2d, verbose=verbose)
    msg = model_2d.load_state_dict(converted_sd, strict=strict)
    if verbose:
        print("[load]", msg)
    return msg


# =============================================================================
# 2D blocks
# =============================================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitationNCHWFixedPool(nn.Module):
    """
    SE for [1, C, K, W].

    Pool only over temporal W:
        AvgPool2d(kernel=(1,W)) -> [1,C,K,1]

    This preserves per-segment SE behavior:
        each H/K segment gets its own channel gate.
    """
    def __init__(self, in_channels: int, reduced_dim: int, fixed_w: int):
        super().__init__()
        self.fixed_w = int(fixed_w)
        if self.fixed_w <= 0:
            raise ValueError(f"fixed_w must be positive, got {fixed_w}")

        self.se = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, self.fixed_w), stride=(1, self.fixed_w)),
            nn.Conv2d(in_channels, reduced_dim, kernel_size=(1, 1)),
            Swish(),
            nn.Conv2d(reduced_dim, in_channels, kernel_size=(1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x)          # [1,C,K,1]
        # Do NOT explicitly expand the tensor here.
        # Deployment backend does not support ONNX Expand, but allows implicit broadcast.
        return x * scale            # implicit broadcast: [1,C,K,W] * [1,C,K,1]


class MBConv2DNCHWDeploy(nn.Module):
    """
    2D equivalent of original MBConv1D, but segment dimension K is H.

    Input/output layout:
        [1, C, K, W]

    All kernels are (1,k), so K/H is preserved.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_w: int,
        expansion: int = 6,
        stride: int = 1,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        self.stride = int(stride)
        self.use_residual = (self.stride == 1 and in_channels == out_channels)

        hidden_dim = int(in_channels * expansion)
        reduced_dim = max(1, int(in_channels * se_ratio))

        layers = []
        w = int(input_w)

        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish(),
            ])

        layers.extend([
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=(1, 3),
                stride=(1, self.stride),
                padding=(0, 1),
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            Swish(),
        ])
        w = conv_out_len(w, kernel_size=3, stride=self.stride, padding=1)

        layers.append(SqueezeExcitationNCHWFixedPool(hidden_dim, reduced_dim, fixed_w=w))

        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)
        self.output_w = int(w)

    def forward(self, x):
        y = self.conv(x)
        if self.use_residual:
            return x + y
        return y


class EfficientNet2DNCHWDeploy(nn.Module):
    """
    2D NCHW equivalent of EfficientNet1D.

    Input:
        [1, 1, K=160, W=200]
    Output:
        [1, out_features, K=160, 1]

    The final original Linear layer is implemented as Conv2d(1x1) so it can be
    applied independently to each segment position H=K.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 128,
        width_mult: float = 1.0,
        depth_mult: float = 0.5,
        input_len: int = 200,
        block_config=None,
    ):
        super().__init__()

        if block_config is None:
            block_config = [
                (16, 1, 1, 1),
                (24, 2, 2, 6),
                (40, 2, 2, 6),
                (80, 3, 2, 6),
                (112, 3, 1, 6),
                (192, 4, 2, 6),
                (320, 1, 1, 6),
            ]

        self.input_len = int(input_len)

        stem_channels = int(32 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=False),
            nn.BatchNorm2d(stem_channels),
            Swish(),
        )
        w = conv_out_len(self.input_len, kernel_size=3, stride=2, padding=1)

        layers = []
        in_ch = stem_channels

        for out_ch, num_layers, stride, expansion in block_config:
            out_ch = int(out_ch * width_mult)
            num_layers = max(1, int(num_layers * depth_mult))

            block = MBConv2DNCHWDeploy(
                in_channels=in_ch,
                out_channels=out_ch,
                input_w=w,
                expansion=expansion,
                stride=stride,
            )
            layers.append(block)
            w = block.output_w
            in_ch = out_ch

            for _ in range(num_layers - 1):
                block = MBConv2DNCHWDeploy(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    input_w=w,
                    expansion=expansion,
                    stride=1,
                )
                layers.append(block)
                w = block.output_w

        self.blocks = nn.Sequential(*layers)
        self.final_w = int(w)

        head_channels = int(1280 * width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(head_channels),
            Swish(),
            nn.AvgPool2d(kernel_size=(1, self.final_w), stride=(1, self.final_w)),
        )

        # Original EfficientNet1D.fc was Linear(head_channels -> out_features).
        # Here use 1x1 Conv2d so it is applied independently at each H=K segment.
        self.fc = nn.Conv2d(head_channels, out_features, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        # x: [1,1,160,200]
        x = self.stem(x)      # [1,C,160,W']
        x = self.blocks(x)    # [1,C,160,W'']
        x = self.head(x)      # [1,1280,160,1]
        x = self.fc(x)        # [1,D,160,1]
        return x


# =============================================================================
# Segment aggregation
# =============================================================================

class SegmentAttentionPoolingNCHW(nn.Module):
    """
    Attention pooling over H=K segment embeddings.

    Input:
        feat_map: [1, D, K, 1]
    Convert:
        [1,D,K,1] -> [1,K,D]
    Output:
        pooled: [1,D]
        weights: [1,K]
    """
    def __init__(self, d_model: int, fixed_k: int = 160, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.fixed_k = int(fixed_k)
        self.d_model = int(d_model)
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat_map: torch.Tensor):
        # feat_map: [1,D,K,1]
        feats = feat_map.squeeze(-1).transpose(1, 2)          # [1,K,D]
        feats = feats.reshape(1, self.fixed_k, self.d_model)  # enforce fixed shape

        logits = self.scorer(feats).squeeze(-1)               # [1,K]
        weights = torch.softmax(logits, dim=1)                # [1,K]
        w3 = weights.unsqueeze(-1)                            # [1,K,1]
        # Do NOT explicitly expand the tensor here.
        # Deployment backend does not support ONNX Expand, but allows implicit broadcast.
        pooled = torch.sum(feats * w3, dim=1)                 # implicit broadcast -> [1,D]
        return pooled, weights


class SegmentMeanPoolingNCHW(nn.Module):
    """
    Fixed average pooling over H=K.

    Input:
        feat_map: [1,D,K,1]
    Use:
        AvgPool2d(kernel=(K,1)) -> [1,D,1,1]
    """
    def __init__(self, fixed_k: int = 160):
        super().__init__()
        self.fixed_k = int(fixed_k)
        self.pool = nn.AvgPool2d(kernel_size=(self.fixed_k, 1), stride=(self.fixed_k, 1))

    def forward(self, feat_map: torch.Tensor):
        # feat_map: [1,D,K,1]
        x = self.pool(feat_map)       # [1,D,1,1]
        pooled = x.flatten(1)         # [1,D]
        weights = torch.ones(1, self.fixed_k, dtype=feat_map.dtype, device=feat_map.device)
        weights = weights / float(self.fixed_k)
        return pooled, weights


# =============================================================================
# BP wrappers
# =============================================================================

class HourlyBPModelAttenNCHWInput(nn.Module):
    """
    Direct NCHW event input model.

    Input:
        x: [1,1,160,200]
    Output:
        pred: [1,2]
    """
    def __init__(
        self,
        encoder: nn.Module,
        d_embed: int = 128,
        out_dim: int = 2,
        fixed_k: int = 160,
        return_aux: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.pool = SegmentAttentionPoolingNCHW(d_model=d_embed, fixed_k=fixed_k, hidden=128, dropout=0.1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_embed),
            nn.Linear(d_embed, out_dim),
        )
        self.fixed_k = int(fixed_k)
        self.d_embed = int(d_embed)
        self.return_aux = bool(return_aux)

    def forward(self, x):
        feat_map = self.encoder(x)        # [1,D,160,1]
        pooled, w = self.pool(feat_map)   # [1,D], [1,160]
        pred = self.head(pooled)          # [1,2]
        if self.return_aux:
            return pred, w, pooled
        return pred


class HourlyBPModelMeanNCHWInput(nn.Module):
    """
    NCHW input model with fixed mean pooling over H=K.

    More operator-friendly but not equivalent to attention pooling.
    """
    def __init__(
        self,
        encoder: nn.Module,
        d_embed: int = 128,
        out_dim: int = 2,
        fixed_k: int = 160,
        return_aux: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.pool = SegmentMeanPoolingNCHW(fixed_k=fixed_k)
        self.head = nn.Sequential(
            nn.LayerNorm(d_embed),
            nn.Linear(d_embed, out_dim),
        )
        self.fixed_k = int(fixed_k)
        self.d_embed = int(d_embed)
        self.return_aux = bool(return_aux)

    def forward(self, x):
        feat_map = self.encoder(x)        # [1,D,160,1]
        pooled, w = self.pool(feat_map)   # [1,D], [1,160]
        pred = self.head(pooled)          # [1,2]
        if self.return_aux:
            return pred, w, pooled
        return pred


def build_nchw_2d_attention_model(
    out_features: int = 128,
    out_dim: int = 2,
    width_mult: float = 1.0,
    depth_mult: float = 0.5,
    input_len: int = 200,
    fixed_k: int = 160,
) -> HourlyBPModelAttenNCHWInput:
    encoder = EfficientNet2DNCHWDeploy(
        in_channels=1,
        out_features=out_features,
        width_mult=width_mult,
        depth_mult=depth_mult,
        input_len=input_len,
    )
    return HourlyBPModelAttenNCHWInput(
        encoder=encoder,
        d_embed=out_features,
        out_dim=out_dim,
        fixed_k=fixed_k,
        return_aux=False,
    )


def build_nchw_2d_mean_model(
    out_features: int = 128,
    out_dim: int = 2,
    width_mult: float = 1.0,
    depth_mult: float = 0.5,
    input_len: int = 200,
    fixed_k: int = 160,
) -> HourlyBPModelMeanNCHWInput:
    encoder = EfficientNet2DNCHWDeploy(
        in_channels=1,
        out_features=out_features,
        width_mult=width_mult,
        depth_mult=depth_mult,
        input_len=input_len,
    )
    return HourlyBPModelMeanNCHWInput(
        encoder=encoder,
        d_embed=out_features,
        out_dim=out_dim,
        fixed_k=fixed_k,
        return_aux=False,
    )


# =============================================================================
# Input conversion and export
# =============================================================================

def event_input_to_nchw_input(x_event: torch.Tensor) -> torch.Tensor:
    """
    Convert old logical event input to new NCHW deploy input.

    Old:
        [1,160,1,200] = [B,K,C,L]
    New:
        [1,1,160,200] = [B,C,K,L]
    """
    if x_event.ndim != 4:
        raise ValueError(f"Expected [1,160,1,200], got {tuple(x_event.shape)}")
    if x_event.shape[0] != 1:
        raise ValueError("This helper assumes batch_size=1.")
    return x_event.permute(0, 2, 1, 3).contiguous()


def export_fixed_nchw_onnx(
    model: nn.Module,
    onnx_path: str,
    fixed_shape: Tuple[int, int, int, int] = (1, 1, 160, 200),
    device: str = "cuda",
    opset_version: int = 17,
):
    """
    Export fixed-shape ONNX.

    Input shape:
        [1,1,160,200]
    """
    model = model.to(device).eval()
    dummy_x = torch.randn(*fixed_shape, dtype=torch.float32, device=device)

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_x,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        dynamo=False,
    )
    print(f"Saved ONNX: {onnx_path}")
    return str(onnx_path)


def compare_event_wrapper_and_nchw_model(
    old_model: nn.Module,
    nchw_model: nn.Module,
    device: str = "cuda",
):
    """
    Compare:
        old_model([1,160,1,200])
    vs
        nchw_model([1,1,160,200])

    If both wrappers are equivalent and weights match, difference should be ~0.
    """
    old_model = old_model.to(device).eval()
    nchw_model = nchw_model.to(device).eval()

    x_event = torch.randn(1, 160, 1, 200, device=device)
    x_nchw = event_input_to_nchw_input(x_event)

    with torch.no_grad():
        y_old = old_model(x_event)
        y_new = nchw_model(x_nchw)

    if isinstance(y_old, (tuple, list)):
        y_old = y_old[0]
    if isinstance(y_new, (tuple, list)):
        y_new = y_new[0]

    diff = (y_old - y_new).abs().detach().cpu()
    print("old:", y_old.detach().cpu().numpy())
    print("new:", y_new.detach().cpu().numpy())
    print("mean_abs_diff:", float(diff.mean()))
    print("max_abs_diff:", float(diff.max()))
    return diff


if __name__ == "__main__":
    model = build_nchw_2d_attention_model()
    model.eval()
    x = torch.randn(1, 1, 160, 200)
    with torch.no_grad():
        y = model(x)
    print("input:", x.shape)
    print("output:", y.shape)
    print("encoder.final_w:", model.encoder.final_w)
