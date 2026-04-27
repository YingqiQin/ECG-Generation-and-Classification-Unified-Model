"""
Net1D backbone adopted from ECGFounder (https://github.com/PKUDigitalHealth/ECGFounder).

The implementation here focuses on feature extraction so it can be reused as the
student (1-lead) and teacher (12-lead) encoders inside SelfMIS and ECG-Mem.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConv1dPadSame(nn.Module):
    """1D convolution with SAME padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dim = x.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        pad_total = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x = F.pad(x, (pad_left, pad_right), mode="constant", value=0.0)
        return self.conv(x)


class MyMaxPool1dPadSame(nn.Module):
    """MaxPool1d with SAME padding."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_total = max(0, self.kernel_size - 1)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x = F.pad(x, (pad_left, pad_right), mode="constant", value=0.0)
        return self.pool(x)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class BasicBlock(nn.Module):
    """Net1D residual block with squeeze-and-excitation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: float,
        kernel_size: int,
        stride: int,
        groups: int,
        downsample: bool,
        is_first_block: bool = False,
        use_bn: bool = True,
        use_do: bool = True,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.downsample = downsample
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.dropout_p = dropout_p

        mid_channels = int(out_channels * ratio)

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act1 = Swish()
        self.drop1 = nn.Dropout(p=dropout_p)
        self.conv1 = MyConv1dPadSame(in_channels, mid_channels, kernel_size=1, stride=1)

        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.act2 = Swish()
        self.drop2 = nn.Dropout(p=dropout_p)
        self.conv2 = MyConv1dPadSame(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride if downsample else 1, groups=groups)

        self.bn3 = nn.BatchNorm1d(mid_channels)
        self.act3 = Swish()
        self.drop3 = nn.Dropout(p=dropout_p)
        self.conv3 = MyConv1dPadSame(mid_channels, out_channels, kernel_size=1, stride=1)

        self.se_fc1 = nn.Linear(out_channels, out_channels // 2)
        self.se_fc2 = nn.Linear(out_channels // 2, out_channels)
        self.se_activation = Swish()

        if downsample:
            pool_kernel = stride if stride > 1 else 1
            self.pool = MyMaxPool1dPadSame(kernel_size=pool_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.act1(out)
            if self.use_do:
                out = self.drop1(out)
        out = self.conv1(out)

        if self.use_bn:
            out = self.bn2(out)
        out = self.act2(out)
        if self.use_do:
            out = self.drop2(out)
        out = self.conv2(out)

        if self.use_bn:
            out = self.bn3(out)
        out = self.act3(out)
        if self.use_do:
            out = self.drop3(out)
        out = self.conv3(out)

        se = out.mean(-1)
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = self.se_fc2(se)
        se = torch.sigmoid(se)
        out = torch.einsum("abc,ab->abc", out, se)

        if self.downsample:
            identity = self.pool(identity)

        if identity.shape[1] != self.out_channels:
            diff = self.out_channels - identity.shape[1]
            pad_left = diff // 2
            pad_right = diff - pad_left
            identity = F.pad(identity, (0, 0, pad_left, pad_right))

        out = out + identity
        return out


class BasicStage(nn.Module):
    """Sequence of BasicBlocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ratio: float,
        kernel_size: int,
        stride: int,
        groups: int,
        num_blocks: int,
        stage_idx: int,
        use_bn: bool,
        use_do: bool,
    ):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            downsample = i == 0
            stride_i = stride if downsample else 1
            block_in_channels = in_channels if i == 0 else out_channels
            blocks.append(
                BasicBlock(
                    in_channels=block_in_channels,
                    out_channels=out_channels,
                    ratio=ratio,
                    kernel_size=kernel_size,
                    stride=stride_i,
                    groups=groups,
                    downsample=downsample,
                    is_first_block=stage_idx == 0 and i == 0,
                    use_bn=use_bn,
                    use_do=use_do,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


@dataclass
class Net1DConfig:
    base_filters: int = 64
    ratio: float = 1.0
    filter_list: Tuple[int, ...] = (64, 160, 160, 400, 400, 1024, 1024)
    m_blocks_list: Tuple[int, ...] = (2, 2, 2, 3, 3, 4, 4)
    kernel_size: int = 16
    stride: int = 2
    groups_width: int = 16
    use_bn: bool = False
    use_do: bool = False


NET1D_PRESETS: Dict[str, Net1DConfig] = {
    "ecgfounder_large": Net1DConfig(),
}


class Net1D(nn.Module):
    """Feature extractor backbone with optional classification head."""

    def __init__(
        self,
        in_channels: int,
        config: Net1DConfig,
        embedding_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.embedding_dim = embedding_dim or config.filter_list[-1]
        self.num_classes = num_classes

        self.first_conv = MyConv1dPadSame(in_channels, config.base_filters, kernel_size=config.kernel_size, stride=2)
        self.first_bn = nn.BatchNorm1d(config.base_filters)
        self.first_act = Swish()

        stages = []
        in_c = config.base_filters
        for idx, out_c in enumerate(config.filter_list):
            groups = max(1, out_c // config.groups_width)
            stage = BasicStage(
                in_channels=in_c,
                out_channels=out_c,
                ratio=config.ratio,
                kernel_size=config.kernel_size,
                stride=config.stride,
                groups=groups,
                num_blocks=config.m_blocks_list[idx],
                stage_idx=idx,
                use_bn=config.use_bn,
                use_do=config.use_do,
            )
            stages.append(stage)
            in_c = out_c
        self.stages = nn.Sequential(*stages)

        final_dim = config.filter_list[-1]
        if final_dim == self.embedding_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(final_dim, self.embedding_dim)

        self.classifier = nn.Linear(self.embedding_dim, num_classes) if num_classes is not None else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.first_conv(x)
        if self.config.use_bn:
            out = self.first_bn(out)
        out = self.first_act(out)
        out = self.stages(out)
        out = out.mean(dim=-1)
        embedding = self.projection(out)
        return embedding

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        embedding = self.forward_features(x)
        if return_embedding or self.classifier is None:
            return embedding
        logits = self.classifier(embedding)
        return logits


def build_net1d_backbone(
    in_channels: int,
    *,
    preset: str = "ecgfounder_large",
    embedding_dim: Optional[int] = None,
    num_classes: Optional[int] = None,
    use_bn: Optional[bool] = None,
    use_do: Optional[bool] = None,
) -> Net1D:
    """Factory that builds a Net1D backbone with optional overrides."""
    if preset not in NET1D_PRESETS:
        raise ValueError(f"Unknown Net1D preset: {preset}")
    config = NET1D_PRESETS[preset]
    if use_bn is not None or use_do is not None:
        config = Net1DConfig(
            base_filters=config.base_filters,
            ratio=config.ratio,
            filter_list=config.filter_list,
            m_blocks_list=config.m_blocks_list,
            kernel_size=config.kernel_size,
            stride=config.stride,
            groups_width=config.groups_width,
            use_bn=config.use_bn if use_bn is None else use_bn,
            use_do=config.use_do if use_do is None else use_do,
        )
    return Net1D(in_channels=in_channels, config=config, embedding_dim=embedding_dim, num_classes=num_classes)

def remap_keys(sd:dict) -> dict:
    new_sd = {}
    for k, v in sd.items():
        nk = k

        nk = nk.replace('stage_list', 'stages')
        nk = nk.replace('block_list', 'blocks')

        new_sd[nk] = v
    return new_sd


def _extract_state_dict(checkpoint):
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "net", "model", "encoder", "student_encoder"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    if isinstance(state_dict, dict) and state_dict:
        first_key = next(iter(state_dict.keys()))
        if isinstance(first_key, str) and first_key.startswith("module."):
            state_dict = OrderedDict((k.replace("module.", "", 1), v) for k, v in state_dict.items())
    return remap_keys(state_dict)


def _adapt_conv1d_input_weight(weight: torch.Tensor, target_in_channels: int) -> torch.Tensor:
    source_in_channels = int(weight.shape[1])
    if source_in_channels == target_in_channels:
        return weight
    if target_in_channels == 1:
        return weight.mean(dim=1, keepdim=True)
    if source_in_channels == 1:
        return weight.repeat(1, target_in_channels, 1) / float(target_in_channels)
    mean_weight = weight.mean(dim=1, keepdim=True)
    return mean_weight.repeat(1, target_in_channels, 1)

def load_net1d_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool = False) -> Tuple[Iterable[str], Iterable[str]]:
    """Load weights into a Net1D module from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    result = model.load_state_dict(state_dict, strict=strict)
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])
    return missing, unexpected


def load_net1d_checkpoint_flexible(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    adapt_input_channels: bool = True,
) -> Tuple[Iterable[str], Iterable[str], Iterable[str]]:
    """
    Load a Net1D checkpoint while tolerating first-conv input-channel mismatch.

    This is used for teacher -> student transfer where a pretrained 12-lead
    checkpoint initializes a true single-lead student encoder.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = _extract_state_dict(checkpoint)
    target_state = model.state_dict()

    filtered_state = OrderedDict()
    unexpected = []
    skipped_shape = []
    adapted = []

    for key, value in source_state.items():
        if key not in target_state:
            unexpected.append(key)
            continue

        target_value = target_state[key]
        if not torch.is_tensor(value):
            continue

        if value.shape == target_value.shape:
            filtered_state[key] = value.to(dtype=target_value.dtype)
            continue

        if (
            adapt_input_channels
            and key == "first_conv.conv.weight"
            and value.ndim == 3
            and target_value.ndim == 3
            and value.shape[0] == target_value.shape[0]
            and value.shape[2] == target_value.shape[2]
        ):
            filtered_state[key] = _adapt_conv1d_input_weight(value, target_value.shape[1]).to(dtype=target_value.dtype)
            adapted.append(key)
            continue

        skipped_shape.append(key)

    result = model.load_state_dict(filtered_state, strict=False)
    missing = list(getattr(result, "missing_keys", [])) + skipped_shape
    unexpected = list(getattr(result, "unexpected_keys", [])) + unexpected

    if strict and (missing or unexpected):
        raise RuntimeError(
            "Flexible checkpoint loading failed with missing keys {} and unexpected keys {}.".format(
                missing, unexpected
            )
        )

    return missing, unexpected, adapted
def load_ckpt_compat(model, ckpt_path, map_location="cpu", strict=True):
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    sd = ckpt['model'] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if len(sd)>0 and next(iter(sd.keys())).startswith("module"):
        new_sd = OrderedDict()
        for k, v in sd.items():
            new_sd[k.replace("module.", "", 1)] = v
        sd = new_sd
    sd = remap_keys(sd)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    return missing, unexpected
__all__ = [
    "Net1D",
    "Net1DConfig",
    "NET1D_PRESETS",
    "build_net1d_backbone",
    "load_net1d_checkpoint",
    "load_net1d_checkpoint_flexible",
]
