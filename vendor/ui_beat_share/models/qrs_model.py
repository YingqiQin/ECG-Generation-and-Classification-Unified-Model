from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class QRSModel(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        projection_head: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection_head = projection_head

    def forward(
        self,
        x: Tensor,
        return_projection: bool = True,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass through encoder, decoder, and optionally the projection head.
        Returns decoder logits and, if requested, the projection outputs.
        """
        z = self.encoder(x)
        logits = self.decoder(z)
        if return_projection:
            z_p, z_n = self.projection_head(z)
            logits_p = self.decoder(z_p)
            logits_n = self.decoder(z_n)
            logits = torch.stack([logits, logits_p, logits_n], dim=1)

        return logits


__all__ = ["QRSModel"]
