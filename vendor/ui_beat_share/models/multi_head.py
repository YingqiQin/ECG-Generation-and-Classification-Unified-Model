from typing import List, Sequence, Tuple, Union

import torch
from torch import Tensor, nn


torch.set_default_dtype(torch.float32)


TensorOrTensors = Union[Tensor, Sequence[Tensor]]


def sampling(args: TensorOrTensors) -> Tensor:
    """
    Reparameterisation trick for Gaussian latent variables.

    Parameters
    ----------
    args: tuple(Tensor, Tensor)
        Mean and logarithmic variance tensors with identical shapes.

    Returns
    -------
    Tensor
        Sampled latent tensor.
    """
    z_mean, z_logvar = args
    epsilon = torch.randn_like(z_mean)
    return z_mean + torch.exp(0.5 * z_logvar) * epsilon


def _compute_same_padding(kernel_size: int, dilation: int) -> int:
    effective_kernel = (kernel_size - 1) * dilation + 1
    return (effective_kernel - 1) // 2


class SeparableConv1d(nn.Module):
    """
    Depthwise separable 1D convolution implemented with a depthwise followed
    by pointwise convolution, mirroring tf.keras.layers.SeparableConv1D.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = _compute_same_padding(kernel_size, dilation)
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SepConv1DBlock(nn.Module):
    """
    Separable convolution followed by InstanceNorm and ReLU activation.
    Mirrors the behaviour of the TensorFlow helper in the legacy code.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = SeparableConv1d(in_channels, out_channels, kernel_size, stride, dilation)
        self.norm = nn.InstanceNorm1d(out_channels, affine=True)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block operating on 1D feature maps.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, _ = x.shape
        squeeze = self.pool(x).view(batch, channels)
        excitation = self.fc1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.tanh(excitation).view(batch, channels, 1)
        return x * excitation


class SEResidualBlock(nn.Module):
    """
    Residual block composed of separable convolutions, SE attention and
    instance normalisation, emulating the TensorFlow implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        reduction: int = 4,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.sep1 = SepConv1DBlock(in_channels, out_channels, kernel_size, stride, dilation)
        self.extra_norm = nn.InstanceNorm1d(out_channels, affine=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.sep2 = SepConv1DBlock(out_channels, out_channels, kernel_size, 1, dilation)
        self.se = SEBlock(out_channels, reduction)
        self.activation = nn.ReLU(inplace=True)
        self.out_norm = nn.InstanceNorm1d(out_channels, affine=True)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        out = self.sep1(x)
        out = self.extra_norm(out)
        out = self.dropout(out)
        out = self.sep2(out)
        out = self.se(out)
        out = out + identity
        out = self.activation(out)
        out = self.out_norm(out)
        return out


class Encoder4Peak(nn.Module):
    """
    Encoder used for feature extraction of primary peaks (input: batch x 1 x 2000).
    Returns features with shape (batch, 500, 128) to match the TensorFlow API.
    """

    def __init__(self) -> None:
        super().__init__()
        self.block1 = SEResidualBlock(1, 8, kernel_size=5, stride=1, dilation=1)
        self.block2 = SEResidualBlock(8, 16, kernel_size=7, stride=1, dilation=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.block3 = SEResidualBlock(16, 32, kernel_size=11, stride=1, dilation=4)
        self.block4 = SEResidualBlock(32, 64, kernel_size=11, stride=1, dilation=4)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.block5 = SEResidualBlock(64, 128, kernel_size=11, stride=1, dilation=4)

    def forward(self, x: Tensor) -> Tensor:
        out = self.block1(x)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        return out.transpose(1, 2)


class Encoder4Beat(Encoder4Peak):
    """
    Encoder for beat morphology decomposition. Shares architecture with Encoder4Peak.
    """

    pass


def _apply_time_instance_norm(x: Tensor, norm: nn.InstanceNorm1d) -> Tensor:
    """
    Applies InstanceNorm1d to a tensor whose last dimension represents channels.
    """
    x_perm = x.permute(0, 2, 1)
    x_perm = norm(x_perm)
    return x_perm.permute(0, 2, 1)


class PhiPeak(nn.Module):
    """
    Projection heads for Primary Peak branch. Each dense stack operates per time step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pos_branch = nn.Sequential(
            nn.Linear(128, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 128),
        )
        self.neg_branch = nn.Sequential(
            nn.Linear(128, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 128),
        )
        self.norm_pos = nn.InstanceNorm1d(128, affine=True)
        self.norm_neg = nn.InstanceNorm1d(128, affine=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        feat_pos = self.pos_branch(x)
        feat_neg = self.neg_branch(x)
        feat_pos = _apply_time_instance_norm(feat_pos, self.norm_pos)
        feat_neg = _apply_time_instance_norm(feat_neg, self.norm_neg)
        return feat_pos, feat_neg


class _PhiBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 128),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class PhiPB(nn.Module):
    """
    Projection heads for PB branch producing four morphology-specific outputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.branches = nn.ModuleList(_PhiBranch() for _ in range(4))
        self.norms = nn.ModuleList(nn.InstanceNorm1d(128, affine=True) for _ in range(4))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        outputs: List[Tensor] = []
        for branch, norm in zip(self.branches, self.norms):
            feat = branch(x)
            feat = _apply_time_instance_norm(feat, norm)
            outputs.append(feat)
        return tuple(outputs)  # type: ignore[return-value]


class Decoder4Peak(nn.Module):
    """
    Time-distributed decoder predicting binary outputs per time step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(128, 32)
        self.activation = nn.ReLU(inplace=True)
        self.out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        out = self.hidden(x)
        out = self.activation(out)
        out = self.out(out)
        out = self.sigmoid(out)
        return out


class Decoder4PB(nn.Module):
    """
    Time-distributed decoder outputting class probabilities per time step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(128, 32)
        self.activation = nn.ReLU(inplace=True)
        self.out = nn.Linear(32, 4)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.hidden(x)
        out = self.activation(out)
        out = self.out(out)
        out = self.softmax(out)
        return out


def encoder4qrs() -> Encoder4Peak:
    return Encoder4Peak()


def encoder4pb() -> Encoder4Beat:
    return Encoder4Beat()


def phi_qrs() -> PhiPeak:
    return PhiPeak()


def phi_pb() -> PhiPB:
    return PhiPB()


def decoder4qrs() -> Decoder4Peak:
    return Decoder4Peak()


def decoder4pb() -> Decoder4PB:
    return Decoder4PB()


def _ensure_sequence(
    input_shape: Union[Sequence[int], Sequence[Sequence[int]]], device: Union[str, torch.device]
) -> Tuple[List[Tensor], List[Tuple[int, ...]]]:
    if len(input_shape) == 0:
        raise ValueError("input_shape must not be empty.")
    if isinstance(input_shape[0], int):  # type: ignore[index]
        shapes = [tuple(input_shape)]  # type: ignore[arg-type]
    else:
        shapes = [tuple(shape) for shape in input_shape]  # type: ignore[assignment]
    inputs = [torch.zeros(shape, device=device) for shape in shapes]
    return inputs, shapes  # type: ignore[return-value]


def visualize_model(
    model: nn.Module,
    input_shape: Union[Sequence[int], Sequence[Sequence[int]]],
    device: Union[str, torch.device] = "cpu",
) -> str:
    """
    Generates a lightweight, dependency-free summary of a model.

    Parameters
    ----------
    model:
        The torch.nn.Module to be inspected. The module is temporarily moved
        to the provided device for the forward pass.
    input_shape:
        Either a single iterable representing the dummy input shape
        (including batch dimension) or a sequence of shapes for multi-input
        models.
    device:
        Device to place the dummy tensors on. Defaults to "cpu".

    Returns
    -------
    str
        Human-readable summary string containing per-layer input/output
        shapes and parameter counts.
    """

    model = model.to(device)
    model.eval()

    dummy_inputs, _ = _ensure_sequence(input_shape, device)

    summary_rows: List[Tuple[str, List[Tuple[int, ...]], List[Tuple[int, ...]], int]] = []
    hooks = []

    def make_hook(name: str):
        def hook(module: nn.Module, inputs: Tuple[Tensor, ...], outputs: TensorOrTensors) -> None:
            # Avoid recording nested modules twice.
            if any(module.children()):
                return
            out_tensors = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
            in_shapes = [tuple(t.shape) for t in inputs]
            out_shapes = [tuple(t.shape) for t in out_tensors]
            params = sum(p.numel() for p in module.parameters(recurse=False))
            summary_rows.append((name, in_shapes, out_shapes, params))

        return hook

    for name, module in model.named_modules():
        if name == "":
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        if len(dummy_inputs) == 1:
            model(dummy_inputs[0])
        else:
            model(*dummy_inputs)

    for hook in hooks:
        hook.remove()

    total_params = sum(row[3] for row in summary_rows)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    header = "Module (name)        Input Shape(s)              Output Shape(s)             Param #"
    lines = [header, "-" * len(header)]

    for idx, (name, in_shapes, out_shapes, params) in enumerate(summary_rows):
        in_repr = ", ".join(str(shape) for shape in in_shapes)
        out_repr = ", ".join(str(shape) for shape in out_shapes)
        lines.append(f"{idx:03d} {name:<18} {in_repr:<26} {out_repr:<26} {params}")

    lines.append("-" * len(header))
    lines.append(f"Total params: {total_params}")
    lines.append(f"Trainable params: {trainable_params}")
    lines.append(f"Non-trainable params: {non_trainable_params}")

    summary = "\n".join(lines)
    return summary


__all__ = [
    "sampling",
    "Encoder4QRS",
    "Encoder4PB",
    "PhiQRS",
    "PhiPB",
    "Decoder4QRS",
    "Decoder4PB",
    "encoder4qrs",
    "encoder4pb",
    "phi_qrs",
    "phi_pb",
    "decoder4qrs",
    "decoder4pb",
    "visualize_model",
]


if __name__ == "__main__":
    encoder = encoder4qrs()
    summary = visualize_model(encoder, (1, 1, 2000))
    print(summary)
