# deploy_friendly_bp_model.py
# -*- coding: utf-8 -*-
"""
Deployment-friendly variants of your 25Hz PPG BP model.

Target fixed input:
    x: [1, 160, 1, 200]
       B=1, K=160, C=1, L=200

Main changes vs original:
1) Replace AdaptiveAvgPool1d(1) with fixed AvgPool1d(kernel_size=T).
   This avoids ONNX GlobalAveragePool.
2) Make SE/attention Mul shapes explicit via expand_as before Mul.
   This avoids implicit broadcast at Mul inputs.
3) Provide a raw-output wrapper that returns pred only, not (pred, w, pooled).
4) Optional mean-pooling model over K=160 if Softmax/Mul attention is not supported.

This file preserves the 1D Conv model path. If the operator team strictly requires
all Conv/Pool tensors to be NCHW, convert Conv1d to Conv2d separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


def conv1d_out_len(L: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    return int((L + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)


def strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


class Swish(nn.Module):
    """x * sigmoid(x). Mul inputs have the same shape."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitationFixedPool1D(nn.Module):
    """
    AdaptiveAvgPool1d(1) replacement for fixed sequence length.
    Also expands SE scale before multiplication.
    """
    def __init__(self, in_channels: int, reduced_dim: int, fixed_seq_len: int):
        super().__init__()
        self.fixed_seq_len = int(fixed_seq_len)
        if self.fixed_seq_len <= 0:
            raise ValueError(f"fixed_seq_len must be positive, got {fixed_seq_len}")
        self.se = nn.Sequential(
            nn.AvgPool1d(kernel_size=self.fixed_seq_len, stride=self.fixed_seq_len),
            nn.Conv1d(in_channels, reduced_dim, 1),
            Swish(),
            nn.Conv1d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x)             # [N,C,1]
        scale = scale.expand_as(x)     # [N,C,T]
        return x * scale               # same-shape Mul


class MBConv1DDeploy(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_seq_len: int,
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
        seq_len = int(input_seq_len)

        if expansion != 1:
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                Swish(),
            ])

        layers.extend([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=self.stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            Swish(),
        ])
        seq_len = conv1d_out_len(seq_len, kernel_size=3, stride=self.stride, padding=1)

        layers.append(SqueezeExcitationFixedPool1D(hidden_dim, reduced_dim, fixed_seq_len=seq_len))

        layers.extend([
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)
        self.output_seq_len = int(seq_len)

    def forward(self, x):
        y = self.conv(x)
        if self.use_residual:
            return x + y
        return y


class EfficientNet1DDeploy(nn.Module):
    """
    Drop-in replacement for EfficientNet1D for fixed input length.

    Equivalent to:
        EfficientNet1D(in_channels=1, out_features=128, width_mult=1.0, depth_mult=0.5)
    but with fixed pooling for input_len=200.
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
            nn.Conv1d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(stem_channels),
            Swish(),
        )
        seq_len = conv1d_out_len(self.input_len, kernel_size=3, stride=2, padding=1)

        layers = []
        in_ch = stem_channels
        for out_ch, num_layers, stride, expansion in block_config:
            out_ch = int(out_ch * width_mult)
            num_layers = max(1, int(num_layers * depth_mult))

            block = MBConv1DDeploy(in_ch, out_ch, input_seq_len=seq_len, expansion=expansion, stride=stride)
            layers.append(block)
            seq_len = block.output_seq_len
            in_ch = out_ch

            for _ in range(num_layers - 1):
                block = MBConv1DDeploy(in_ch, out_ch, input_seq_len=seq_len, expansion=expansion, stride=1)
                layers.append(block)
                seq_len = block.output_seq_len

        self.blocks = nn.Sequential(*layers)
        self.final_seq_len = int(seq_len)

        head_channels = int(1280 * width_mult)
        self.head = nn.Sequential(
            nn.Conv1d(in_ch, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(head_channels),
            Swish(),
            nn.AvgPool1d(kernel_size=self.final_seq_len, stride=self.final_seq_len),
        )
        self.fc = nn.Linear(head_channels, out_features)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)      # [N,1280,1]
        x = x.flatten(1)
        x = self.fc(x)
        return x


class SegmentAttentionPoolingDeploy(nn.Module):
    """Attention pooling with explicit expand before Mul."""
    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor, mask: Optional[torch.Tensor] = None):
        logits = self.scorer(feats).squeeze(-1)  # [B,K]
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)   # [B,K]
        w3 = weights.unsqueeze(-1).expand_as(feats)  # [B,K,D]
        pooled = torch.sum(feats * w3, dim=1)    # [B,D]
        return pooled, weights


class SegmentMeanPoolingFixedK(nn.Module):
    """AveragePool2d over fixed K. Avoids Softmax and attention Mul."""
    def __init__(self, fixed_k: int = 160):
        super().__init__()
        self.fixed_k = int(fixed_k)
        self.pool = nn.AvgPool2d(kernel_size=(1, self.fixed_k), stride=(1, self.fixed_k))

    def forward(self, feats: torch.Tensor):
        # [B,K,D] -> [B,D,1,K]
        x = feats.transpose(1, 2).unsqueeze(2)
        x = self.pool(x)         # [B,D,1,1]
        pooled = x.flatten(1)    # [B,D]
        weights = torch.ones(feats.shape[0], feats.shape[1], dtype=feats.dtype, device=feats.device)
        weights = weights / float(self.fixed_k)
        return pooled, weights


class HourlyBPModelAttenDeploy(nn.Module):
    """Fixed-shape attention model. Returns pred only by default."""
    def __init__(
        self,
        encoder: nn.Module,
        d_embed: int = 128,
        out_dim: int = 2,
        fixed_b: int = 1,
        fixed_k: int = 160,
        fixed_c: int = 1,
        fixed_l: int = 200,
        return_aux: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.pool = SegmentAttentionPoolingDeploy(d_embed, hidden=128, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(d_embed), nn.Linear(d_embed, out_dim))
        self.fixed_b = int(fixed_b)
        self.fixed_k = int(fixed_k)
        self.fixed_c = int(fixed_c)
        self.fixed_l = int(fixed_l)
        self.return_aux = bool(return_aux)

    def forward(self, x: torch.Tensor):
        x_flat = x.reshape(self.fixed_b * self.fixed_k, self.fixed_c, self.fixed_l)
        feat = self.encoder(x_flat)
        if feat.dim() == 3:
            feat = feat.squeeze(-1)
        feat = feat.reshape(self.fixed_b, self.fixed_k, -1)
        pooled, w = self.pool(feat, mask=None)
        pred = self.head(pooled)
        if self.return_aux:
            return pred, w, pooled
        return pred


class HourlyBPModelMeanDeploy(nn.Module):
    """Deployment-friendly alternative: fixed average pooling over K instead of attention."""
    def __init__(
        self,
        encoder: nn.Module,
        d_embed: int = 128,
        out_dim: int = 2,
        fixed_b: int = 1,
        fixed_k: int = 160,
        fixed_c: int = 1,
        fixed_l: int = 200,
        return_aux: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.pool = SegmentMeanPoolingFixedK(fixed_k=fixed_k)
        self.head = nn.Sequential(nn.LayerNorm(d_embed), nn.Linear(d_embed, out_dim))
        self.fixed_b = int(fixed_b)
        self.fixed_k = int(fixed_k)
        self.fixed_c = int(fixed_c)
        self.fixed_l = int(fixed_l)
        self.return_aux = bool(return_aux)

    def forward(self, x: torch.Tensor):
        x_flat = x.reshape(self.fixed_b * self.fixed_k, self.fixed_c, self.fixed_l)
        feat = self.encoder(x_flat)
        if feat.dim() == 3:
            feat = feat.squeeze(-1)
        feat = feat.reshape(self.fixed_b, self.fixed_k, -1)
        pooled, w = self.pool(feat)
        pred = self.head(pooled)
        if self.return_aux:
            return pred, w, pooled
        return pred


def build_deploy_attention_model(
    out_features: int = 128,
    out_dim: int = 2,
    width_mult: float = 1.0,
    depth_mult: float = 0.5,
    input_len: int = 200,
    fixed_b: int = 1,
    fixed_k: int = 160,
):
    encoder = EfficientNet1DDeploy(
        in_channels=1,
        out_features=out_features,
        width_mult=width_mult,
        depth_mult=depth_mult,
        input_len=input_len,
    )
    return HourlyBPModelAttenDeploy(
        encoder=encoder,
        d_embed=out_features,
        out_dim=out_dim,
        fixed_b=fixed_b,
        fixed_k=fixed_k,
        fixed_c=1,
        fixed_l=input_len,
        return_aux=False,
    )


def build_deploy_mean_model(
    out_features: int = 128,
    out_dim: int = 2,
    width_mult: float = 1.0,
    depth_mult: float = 0.5,
    input_len: int = 200,
    fixed_b: int = 1,
    fixed_k: int = 160,
):
    encoder = EfficientNet1DDeploy(
        in_channels=1,
        out_features=out_features,
        width_mult=width_mult,
        depth_mult=depth_mult,
        input_len=input_len,
    )
    return HourlyBPModelMeanDeploy(
        encoder=encoder,
        d_embed=out_features,
        out_dim=out_dim,
        fixed_b=fixed_b,
        fixed_k=fixed_k,
        fixed_c=1,
        fixed_l=input_len,
        return_aux=False,
    )


def load_checkpoint_flexible(model: nn.Module, ckpt_path: str, strict: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    sd = strip_module_prefix(sd)
    return model.load_state_dict(sd, strict=strict)


def export_fixed_onnx(
    model: nn.Module,
    onnx_path: str,
    fixed_shape: Tuple[int, int, int, int] = (1, 160, 1, 200),
    device: str = "cuda",
    opset_version: int = 17,
):
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
    return str(onnx_path)


if __name__ == "__main__":
    m = build_deploy_attention_model()
    m.eval()
    x = torch.randn(1, 160, 1, 200)
    with torch.no_grad():
        y = m(x)
    print("output:", y.shape)
    print("encoder final_seq_len:", m.encoder.final_seq_len)
