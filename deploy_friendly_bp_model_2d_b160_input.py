# deploy_friendly_bp_model_2d_b160_input.py
# -*- coding: utf-8 -*-
"""
2D-Conv deploy-friendly BP model with direct segment-batch input.

New deployment input:
    x: [160, 1, 1, 200]
       N=160, C=1, H=1, W=200

Semantic meaning:
    N=160 is NOT 160 independent BP events.
    It is the fixed K=160 8-second PPG segments from one BP event/window.

Model output:
    pred: [1, 2]
          one event-level SBP/DBP prediction.

This removes the previous ONNX entry reshape:
    [1,160,1,200] -> [160,1,1,200]

Now the first Conv2d sees standard NCHW directly:
    [160,1,1,200]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


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
    Convert Conv1d checkpoint weights to Conv2d weights.

    Conv1d: [out, in/groups, k]
    Conv2d: [out, in/groups, 1, k]
    """
    target_sd = target_model.state_dict()
    converted = {}
    skipped = []

    for k, v in source_sd.items():
        if k not in target_sd:
            skipped.append((k, "not_in_target", tuple(v.shape)))
            continue

        tgt = target_sd[k]

        if tuple(v.shape) == tuple(tgt.shape):
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
            for item in skipped[:30]:
                print("  ", item)

    return converted


def load_1d_checkpoint_into_b160_2d_model(
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


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcitation2DFixedPool(nn.Module):
    """
    SE block for [N, C, 1, W].
    Uses fixed AvgPool2d instead of AdaptiveAvgPool/GlobalAveragePool.
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
        return x * scale            # same-shape Mul after Expand


class MBConv2DDeploy(nn.Module):
    """
    2D equivalent of the original MBConv1D block for [N,C,1,W].
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
    2D equivalent of EfficientNet1D.

    Input:
        [N, 1, 1, 200]
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


class SegmentAttentionPoolingB160(nn.Module):
    """
    Attention pooling over K=160 segment embeddings.

    Input:
        feats: [160, D]
    Output:
        pooled: [1,D]
        weights: [1,160]
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

    def forward(self, feats: torch.Tensor):
        feats = feats.reshape(1, self.fixed_k, self.d_model)  # [1,160,D]
        logits = self.scorer(feats).squeeze(-1)               # [1,160]
        weights = torch.softmax(logits, dim=1)                # [1,160]
        w3 = weights.unsqueeze(-1)                            # [1,160,1]
        w3 = w3.expand_as(feats)                              # [1,160,D]
        pooled = torch.sum(feats * w3, dim=1)                 # [1,D]
        return pooled, weights


class SegmentMeanPoolingB160(nn.Module):
    """
    Fixed average pooling over K=160.

    Input:
        feats: [160,D]
    """
    def __init__(self, fixed_k: int = 160, d_model: int = 128):
        super().__init__()
        self.fixed_k = int(fixed_k)
        self.d_model = int(d_model)
        self.pool = nn.AvgPool2d(kernel_size=(1, self.fixed_k), stride=(1, self.fixed_k))

    def forward(self, feats: torch.Tensor):
        feats = feats.reshape(1, self.fixed_k, self.d_model)  # [1,160,D]
        x = feats.transpose(1, 2).unsqueeze(2)                # [1,D,1,160]
        x = self.pool(x)                                     # [1,D,1,1]
        pooled = x.flatten(1)                                # [1,D]
        weights = torch.ones(1, self.fixed_k, dtype=feats.dtype, device=feats.device)
        weights = weights / float(self.fixed_k)
        return pooled, weights


class HourlyBPModelAttenB160Input(nn.Module):
    """
    Direct segment-batch input model.

    Input:
        x: [160,1,1,200]
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
        self.pool = SegmentAttentionPoolingB160(d_model=d_embed, fixed_k=fixed_k, hidden=128, dropout=0.1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_embed),
            nn.Linear(d_embed, out_dim),
        )
        self.fixed_k = int(fixed_k)
        self.d_embed = int(d_embed)
        self.return_aux = bool(return_aux)

    def forward(self, x):
        feat = self.encoder(x)             # [160,D]
        pooled, w = self.pool(feat)        # [1,D], [1,160]
        pred = self.head(pooled)           # [1,2]
        if self.return_aux:
            return pred, w, pooled
        return pred


class HourlyBPModelMeanB160Input(nn.Module):
    """
    Direct segment-batch input model with fixed mean pooling.

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
        self.pool = SegmentMeanPoolingB160(fixed_k=fixed_k, d_model=d_embed)
        self.head = nn.Sequential(
            nn.LayerNorm(d_embed),
            nn.Linear(d_embed, out_dim),
        )
        self.fixed_k = int(fixed_k)
        self.d_embed = int(d_embed)
        self.return_aux = bool(return_aux)

    def forward(self, x):
        feat = self.encoder(x)
        pooled, w = self.pool(feat)
        pred = self.head(pooled)
        if self.return_aux:
            return pred, w, pooled
        return pred


def build_b160_2d_attention_model(
    out_features: int = 128,
    out_dim: int = 2,
    width_mult: float = 1.0,
    depth_mult: float = 0.5,
    input_len: int = 200,
    fixed_k: int = 160,
) -> HourlyBPModelAttenB160Input:
    encoder = EfficientNet2DDeploy(
        in_channels=1,
        out_features=out_features,
        width_mult=width_mult,
        depth_mult=depth_mult,
        input_len=input_len,
    )
    return HourlyBPModelAttenB160Input(
        encoder=encoder,
        d_embed=out_features,
        out_dim=out_dim,
        fixed_k=fixed_k,
        return_aux=False,
    )


def build_b160_2d_mean_model(
    out_features: int = 128,
    out_dim: int = 2,
    width_mult: float = 1.0,
    depth_mult: float = 0.5,
    input_len: int = 200,
    fixed_k: int = 160,
) -> HourlyBPModelMeanB160Input:
    encoder = EfficientNet2DDeploy(
        in_channels=1,
        out_features=out_features,
        width_mult=width_mult,
        depth_mult=depth_mult,
        input_len=input_len,
    )
    return HourlyBPModelMeanB160Input(
        encoder=encoder,
        d_embed=out_features,
        out_dim=out_dim,
        fixed_k=fixed_k,
        return_aux=False,
    )


def export_fixed_b160_onnx(
    model: nn.Module,
    onnx_path: str,
    fixed_shape: Tuple[int, int, int, int] = (160, 1, 1, 200),
    device: str = "cuda",
    opset_version: int = 17,
):
    """
    Export fixed-shape ONNX.

    Input shape:
        [160,1,1,200]
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


def event_input_to_b160_input(x_event: torch.Tensor) -> torch.Tensor:
    """
    Convert old logical event input to new deploy input.

    Old:
        x_event: [1,160,1,200]
    New:
        x_b160: [160,1,1,200]
    """
    if x_event.ndim != 4:
        raise ValueError(f"Expected [1,160,1,200], got {tuple(x_event.shape)}")
    if x_event.shape[0] != 1:
        raise ValueError("This helper assumes batch_size=1.")
    return x_event.reshape(160, 1, 1, 200)


def compare_event_wrapper_and_b160_model(
    old_model: nn.Module,
    b160_model: nn.Module,
    device: str = "cuda",
):
    """
    Compare:
        old_model([1,160,1,200])
    vs
        b160_model([160,1,1,200])
    """
    old_model = old_model.to(device).eval()
    b160_model = b160_model.to(device).eval()

    x_event = torch.randn(1, 160, 1, 200, device=device)
    x_b160 = event_input_to_b160_input(x_event)

    with torch.no_grad():
        y_old = old_model(x_event)
        y_new = b160_model(x_b160)

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
    model = build_b160_2d_attention_model()
    model.eval()
    x = torch.randn(160, 1, 1, 200)
    with torch.no_grad():
        y = model(x)
    print("input:", x.shape)
    print("output:", y.shape)
    print("encoder.final_w:", model.encoder.final_w)
