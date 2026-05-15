# deploy_friendly_bp_model_2d.py
# -*- coding: utf-8 -*-
"""
2D-Conv deployment-friendly BP model for fixed 25Hz input.

Target input:
    x: [1, 160, 1, 200]
       semantic = [B, K, C, L]
       B=1, K=160, C=1, L=200

Internal segment encoder input:
    [B*K, C, 1, L] = [160, 1, 1, 200]

Main conversions from original model:
    Conv1d(k, s, p)        -> Conv2d((1,k), (1,s), (0,p))
    BatchNorm1d            -> BatchNorm2d
    AdaptiveAvgPool1d(1)   -> fixed AvgPool2d(kernel_size=(1, fixed_W))
    SE broadcast Mul       -> explicit expand_as before Mul
    input reshape          -> fixed reshape, no dynamic B/K/L parsing

Checkpoint loading:
    Original Conv1d weights are [out, in/groups, k].
    Conv2d weights are [out, in/groups, 1, k].
    load_1d_checkpoint_into_2d_model() automatically unsqueezes dim=2.

Two wrappers:
    HourlyBPModelAtten2DDeploy:
        preserves attention pooling, but still uses Softmax / Expand / Mul / ReduceSum.
    HourlyBPModelMean2DDeploy:
        replaces attention pooling by fixed AvgPool over K=160.
        More operator-friendly, but not numerically equivalent; requires validation/fine-tune.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

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


def convert_1d_state_dict_to_2d(
    source_sd: Dict[str, torch.Tensor],
    target_model: nn.Module,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convert Conv1d weights to Conv2d weights if target key expects 4D.

    Conv1d:
        [out, in/groups, k]
    Conv2d:
        [out, in/groups, 1, k]
    """
    target_sd = target_model.state_dict()
    converted = {}
    skipped = []

    for k, v in source_sd.items():
        if k not in target_sd:
            skipped.append((k, "not_in_target", tuple(v.shape)))
            continue

        tgt = target_sd[k]

        if v.shape == tgt.shape:
            converted[k] = v
            continue

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

        skipped.append((k, f"shape_mismatch_source={tuple(v.shape)}_target={tuple(tgt.shape)}", tuple(v.shape)))

    if verbose:
        print(f"[convert] source keys={len(source_sd)}, converted/kept={len(converted)}, skipped={len(skipped)}")
        if skipped:
            print("[convert] first skipped keys:")
            for item in skipped[:20]:
                print("  ", item)

    return converted


def load_1d_checkpoint_into_2d_model(
    model_2d: nn.Module,
    ckpt_path: str,
    strict: bool = False,
    verbose: bool = True,
):
    source_sd = load_state_dict_from_checkpoint(ckpt_path)
    converted_sd = convert_1d_state_dict_to_2d(source_sd, model_2d, verbose=verbose)
    msg = model_2d.load_state_dict(converted_sd, strict=strict)
    if verbose:
        print("[load]", msg)
    return msg


# =============================================================================
# 2D modules
# =============================================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation2DFixedPool(nn.Module):
    """
    SE for [N, C, 1, W].
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
        scale = self.se(x)          # [N,C,1,1]
        scale = scale.expand_as(x)  # [N,C,1,W]
        return x * scale            # same-shape Mul


class MBConv2DDeploy(nn.Module):
    """
    2D equivalent of original MBConv1D for [N,C,1,W].
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

        layers.append(SqueezeExcitation2DFixedPool(hidden_dim, reduced_dim, fixed_w=w))

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


class EfficientNet2DDeploy(nn.Module):
    """
    2D equivalent of your EfficientNet1D.

    Input:
        [N, 1, 1, 200], where N = B*K = 160

    Output:
        [N, out_features]
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

            block = MBConv2DDeploy(
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
                block = MBConv2DDeploy(
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

        self.fc = nn.Linear(head_channels, out_features)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)      # [N,1280,1,1]
        x = x.flatten(1)      # [N,1280]
        x = self.fc(x)        # [N,D]
        return x


# =============================================================================
# Segment pooling
# =============================================================================

class SegmentAttentionPoolingDeploy(nn.Module):
    """
    Original attention pooling with explicit Expand before Mul.

    Warning:
        This still requires Softmax, Expand, Mul, ReduceSum.
        If any of these are unsupported, use SegmentMeanPoolingFixedK.
    """
    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor):
        logits = self.scorer(feats).squeeze(-1)   # [1,160]
        weights = torch.softmax(logits, dim=1)    # [1,160]
        w3 = weights.unsqueeze(-1)                # [1,160,1]
        w3 = w3.expand_as(feats)                  # [1,160,D]
        pooled = torch.sum(feats * w3, dim=1)     # [1,D]
        return pooled, weights


class SegmentMeanPoolingFixedK(nn.Module):
    """
    Operator-friendly pooling over K.
    """
    def __init__(self, fixed_k: int = 160):
        super().__init__()
        self.fixed_k = int(fixed_k)
        self.pool = nn.AvgPool2d(kernel_size=(1, self.fixed_k), stride=(1, self.fixed_k))

    def forward(self, feats: torch.Tensor):
        x = feats.transpose(1, 2).unsqueeze(2)  # [1,D,1,160]
        x = self.pool(x)                       # [1,D,1,1]
        pooled = x.flatten(1)                  # [1,D]
        weights = torch.ones(feats.shape[0], feats.shape[1], dtype=feats.dtype, device=feats.device)
        weights = weights / float(self.fixed_k)
        return pooled, weights


# =============================================================================
# BP models
# =============================================================================

class HourlyBPModelAtten2DDeploy(nn.Module):
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
        self.head = nn.Sequential(
            nn.LayerNorm(d_embed),
            nn.Linear(d_embed, out_dim),
        )

        self.fixed_b = int(fixed_b)
        self.fixed_k = int(fixed_k)
        self.fixed_c = int(fixed_c)
        self.fixed_l = int(fixed_l)
        self.d_embed = int(d_embed)
        self.return_aux = bool(return_aux)

    def forward(self, x):
        # x: [1,160,1,200]
        x = x.reshape(self.fixed_b * self.fixed_k, self.fixed_c, 1, self.fixed_l)  # [160,1,1,200]
        feat = self.encoder(x)                                                     # [160,D]
        feat = feat.reshape(self.fixed_b, self.fixed_k, self.d_embed)              # [1,160,D]
        pooled, w = self.pool(feat)
        pred = self.head(pooled)
        if self.return_aux:
            return pred, w, pooled
        return pred


class HourlyBPModelMean2DDeploy(nn.Module):
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
        self.head = nn.Sequential(
            nn.LayerNorm(d_embed),
            nn.Linear(d_embed, out_dim),
        )

        self.fixed_b = int(fixed_b)
        self.fixed_k = int(fixed_k)
        self.fixed_c = int(fixed_c)
        self.fixed_l = int(fixed_l)
        self.d_embed = int(d_embed)
        self.return_aux = bool(return_aux)

    def forward(self, x):
        x = x.reshape(self.fixed_b * self.fixed_k, self.fixed_c, 1, self.fixed_l)
        feat = self.encoder(x)
        feat = feat.reshape(self.fixed_b, self.fixed_k, self.d_embed)
        pooled, w = self.pool(feat)
        pred = self.head(pooled)
        if self.return_aux:
            return pred, w, pooled
        return pred


def build_2d_attention_model(
    out_features: int = 128,
    out_dim: int = 2,
    width_mult: float = 1.0,
    depth_mult: float = 0.5,
    input_len: int = 200,
    fixed_b: int = 1,
    fixed_k: int = 160,
) -> HourlyBPModelAtten2DDeploy:
    encoder = EfficientNet2DDeploy(
        in_channels=1,
        out_features=out_features,
        width_mult=width_mult,
        depth_mult=depth_mult,
        input_len=input_len,
    )
    return HourlyBPModelAtten2DDeploy(
        encoder=encoder,
        d_embed=out_features,
        out_dim=out_dim,
        fixed_b=fixed_b,
        fixed_k=fixed_k,
        fixed_c=1,
        fixed_l=input_len,
        return_aux=False,
    )


def build_2d_mean_model(
    out_features: int = 128,
    out_dim: int = 2,
    width_mult: float = 1.0,
    depth_mult: float = 0.5,
    input_len: int = 200,
    fixed_b: int = 1,
    fixed_k: int = 160,
) -> HourlyBPModelMean2DDeploy:
    encoder = EfficientNet2DDeploy(
        in_channels=1,
        out_features=out_features,
        width_mult=width_mult,
        depth_mult=depth_mult,
        input_len=input_len,
    )
    return HourlyBPModelMean2DDeploy(
        encoder=encoder,
        d_embed=out_features,
        out_dim=out_dim,
        fixed_b=fixed_b,
        fixed_k=fixed_k,
        fixed_c=1,
        fixed_l=input_len,
        return_aux=False,
    )


# =============================================================================
# Export / debug
# =============================================================================

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
    print(f"Saved ONNX: {onnx_path}")
    return str(onnx_path)


def compare_pytorch_models(model_a: nn.Module, model_b: nn.Module, device: str = "cuda"):
    model_a = model_a.to(device).eval()
    model_b = model_b.to(device).eval()
    x = torch.randn(1, 160, 1, 200, device=device)

    with torch.no_grad():
        ya = model_a(x)
        yb = model_b(x)

    if isinstance(ya, (tuple, list)):
        ya = ya[0]
    if isinstance(yb, (tuple, list)):
        yb = yb[0]

    diff = (ya - yb).detach().abs().cpu()
    print("model_a:", ya.detach().cpu().numpy())
    print("model_b:", yb.detach().cpu().numpy())
    print("mean_abs_diff:", float(diff.mean()))
    print("max_abs_diff:", float(diff.max()))


if __name__ == "__main__":
    m = build_2d_attention_model()
    m.eval()
    x = torch.randn(1, 160, 1, 200)
    with torch.no_grad():
        y = m(x)
    print("output:", y.shape)
    print("encoder.final_w:", m.encoder.final_w)
