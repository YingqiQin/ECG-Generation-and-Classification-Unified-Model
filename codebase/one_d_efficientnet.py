from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None:
        mask = mask.to(dtype=torch.bool)
        logits = logits.masked_fill(~mask, -1e9)

    weights = torch.softmax(logits, dim=1)

    if mask is not None:
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    return weights


def _gather_topk_features(
    feats: torch.Tensor,
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_segments, feat_dim = feats.shape
    k = max(1, min(int(topk), num_segments))

    if mask is None:
        valid_mask = torch.ones_like(scores, dtype=torch.bool)
    else:
        valid_mask = mask.to(dtype=torch.bool)

    masked_scores = scores.masked_fill(~valid_mask, -1e9)
    topk_indices = torch.topk(masked_scores, k=k, dim=1).indices

    gather_idx = topk_indices.unsqueeze(-1).expand(batch_size, k, feat_dim)
    topk_feats = torch.gather(feats, dim=1, index=gather_idx)
    topk_valid = torch.gather(valid_mask, dim=1, index=topk_indices)

    denom = topk_valid.to(topk_feats.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (topk_feats * topk_valid.unsqueeze(-1).to(topk_feats.dtype)).sum(dim=1) / denom
    return pooled, topk_indices, topk_valid


class SegmentAttentionPool(nn.Module):
    """
    Standard segment attention pooling.

    Inputs:
      feats: [B, K, D]
      mask:  [B, K] optional
    Outputs:
      pooled: [B, D]
      weights: [B, K]
      aux: dict with raw pooling scores
    """

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seg_quality: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        del seg_quality

        logits = self.scorer(feats).squeeze(-1)
        weights = _masked_softmax(logits, mask=mask)
        pooled = torch.sum(feats * weights.unsqueeze(-1), dim=1)
        aux = {
            "evidence_logits": logits,
            "combined_logits": logits,
        }
        return pooled, weights, aux


class QualityCalibratedAttentionPool(nn.Module):
    """
    Attention pooling where per-segment signal-quality features modulate evidence logits.
    """

    def __init__(
        self,
        d_model: int,
        quality_dim: int = 4,
        hidden: int = 128,
        quality_hidden: int = 32,
        dropout: float = 0.0,
        quality_alpha: float = 1.0,
    ):
        super().__init__()
        self.quality_alpha = float(quality_alpha)
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.quality_net = nn.Sequential(
            nn.Linear(quality_dim, quality_hidden),
            nn.ReLU(),
            nn.Linear(quality_hidden, 1),
        )

    def forward(
        self,
        feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seg_quality: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        evidence_logits = self.scorer(feats).squeeze(-1)

        if seg_quality is None:
            quality_logits = torch.zeros_like(evidence_logits)
        else:
            if seg_quality.ndim == 2:
                seg_quality = seg_quality.unsqueeze(-1)
            quality_logits = self.quality_net(seg_quality.to(feats.dtype)).squeeze(-1)

        combined_logits = evidence_logits + self.quality_alpha * quality_logits
        weights = _masked_softmax(combined_logits, mask=mask)
        pooled = torch.sum(feats * weights.unsqueeze(-1), dim=1)
        aux = {
            "evidence_logits": evidence_logits,
            "quality_logits": quality_logits,
            "combined_logits": combined_logits,
        }
        return pooled, weights, aux


class HybridAttentionTopKPool(QualityCalibratedAttentionPool):
    """
    Quality-calibrated attention plus a top-k evidence path.
    """

    def __init__(
        self,
        d_model: int,
        quality_dim: int = 4,
        hidden: int = 128,
        quality_hidden: int = 32,
        dropout: float = 0.0,
        quality_alpha: float = 1.0,
        topk: int = 4,
        mix_beta: float = 0.5,
    ):
        super().__init__(
            d_model=d_model,
            quality_dim=quality_dim,
            hidden=hidden,
            quality_hidden=quality_hidden,
            dropout=dropout,
            quality_alpha=quality_alpha,
        )
        self.topk = int(topk)
        self.mix_beta = float(mix_beta)

    def forward(
        self,
        feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seg_quality: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        attn_pooled, weights, aux = super().forward(feats, mask=mask, seg_quality=seg_quality)
        topk_pooled, topk_indices, topk_valid = _gather_topk_features(
            feats=feats,
            scores=aux["combined_logits"],
            mask=mask,
            topk=self.topk,
        )
        pooled = self.mix_beta * attn_pooled + (1.0 - self.mix_beta) * topk_pooled
        aux["topk_indices"] = topk_indices
        aux["topk_valid"] = topk_valid
        aux["attn_pooled"] = attn_pooled
        aux["topk_pooled"] = topk_pooled
        return pooled, weights, aux


class ECGModel_Attn(nn.Module):
    """
    MIL ECG classification model.

    Input:
      x: [B, K, C, L]
      mask: [B, K] optional
      seg_quality: [B, K, Q] or [B, K] optional
    Output:
      pred: [B, num_classes]
      weights: [B, K]
      aux: optional diagnostics for pooling
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_embed: int = 512,
        out_dim: int = 2,
        pool_type: str = "attention",
        pool_hidden: int = 128,
        dropout: float = 0.1,
        quality_dim: int = 4,
        quality_hidden: int = 32,
        quality_alpha: float = 1.0,
        topk: int = 4,
        mix_beta: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.pool_type = pool_type

        if pool_type == "attention":
            self.pool = SegmentAttentionPool(d_model=d_embed, hidden=pool_hidden, dropout=dropout)
        elif pool_type == "quality_attention":
            self.pool = QualityCalibratedAttentionPool(
                d_model=d_embed,
                quality_dim=quality_dim,
                hidden=pool_hidden,
                quality_hidden=quality_hidden,
                dropout=dropout,
                quality_alpha=quality_alpha,
            )
        elif pool_type == "hybrid":
            self.pool = HybridAttentionTopKPool(
                d_model=d_embed,
                quality_dim=quality_dim,
                hidden=pool_hidden,
                quality_hidden=quality_hidden,
                dropout=dropout,
                quality_alpha=quality_alpha,
                topk=topk,
                mix_beta=mix_beta,
            )
        else:
            raise ValueError("pool_type must be one of: attention, quality_attention, hybrid")

        self.head = nn.Sequential(
            nn.LayerNorm(d_embed),
            nn.Linear(d_embed, out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        seg_quality: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        batch_size, num_segments, channels, length = x.shape
        x_flat = x.view(batch_size * num_segments, channels, length)

        feat = self.encoder(x_flat)
        if feat.ndim == 3:
            feat = feat.squeeze(-1)

        feat = feat.view(batch_size, num_segments, -1)
        pooled, weights, aux = self.pool(feat, mask=mask, seg_quality=seg_quality)
        pred = self.head(pooled)

        if return_aux:
            aux = dict(aux)
            aux["segment_features"] = feat
            aux["pooled_features"] = pooled
            return pred, weights, aux
        return pred, weights
