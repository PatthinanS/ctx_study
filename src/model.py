from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


# ---------------------------------------------------------------------------
# VAD regression model
# ---------------------------------------------------------------------------

class VADHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 3)

    def forward(self, cls_repr: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(cls_repr))


class VADModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(cfg["backbone"])
        self.head = VADHead(self.backbone.config.hidden_size, cfg.get("dropout", 0.1))
        # RoBERTa and XLM-R have type_vocab_size == 1 (no segment embeddings).
        self._use_token_type_ids = self.backbone.config.type_vocab_size > 1

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and self._use_token_type_ids:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**kwargs)
        # pooler_output matches the notebook's two-arg BERT call that returns pooled.
        # Falls back to CLS token if backbone has no pooler (e.g., some XLM-R configs).
        if outputs.pooler_output is not None:
            cls_repr = outputs.pooler_output
        else:
            cls_repr = outputs.last_hidden_state[:, 0, :]

        return self.head(cls_repr)

    def freeze_encoder(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

# # EMDLoss and CombinedLoss carried over from prior pipeline — not used in this study (MSE only).
# class EMDLoss(nn.Module):
#     """EMD-based loss for continuous VAD regression.
#
#     mode="bins": per-sample CDF-L1 over soft bin assignments.
#     mode="batch_wasserstein": sort-based W1 per dimension over the batch.
#     """
#
#     def __init__(
#         self,
#         n_bins: int = 10,
#         vad_range: tuple[float, float] = (1.0, 5.0),
#         mode: str = "batch_wasserstein",
#     ) -> None:
#         super().__init__()
#         self.mode = mode
#         if mode == "bins":
#             lo, hi = vad_range
#             edges = torch.linspace(lo, hi, n_bins + 1)
#             midpoints = (edges[:-1] + edges[1:]) / 2
#             self.register_buffer("bin_midpoints", midpoints)
#
#     def _soft_assign(self, values: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
#         dists = -((values.unsqueeze(1) - self.bin_midpoints.unsqueeze(0)) ** 2) / temperature
#         return torch.softmax(dists, dim=1)
#
#     def _bin_emd(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         loss = torch.tensor(0.0, device=pred.device)
#         for d in range(3):
#             p_dist = self._soft_assign(pred[:, d])
#             t_dist = self._soft_assign(target[:, d])
#             cdf_diff = torch.cumsum(p_dist, dim=1) - torch.cumsum(t_dist, dim=1)
#             loss = loss + cdf_diff.abs().mean()
#         return loss / 3
#
#     def _batch_wasserstein(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         loss = torch.tensor(0.0, device=pred.device)
#         for d in range(3):
#             sorted_pred = torch.sort(pred[:, d]).values
#             sorted_tgt = torch.sort(target[:, d]).values
#             loss = loss + (sorted_pred - sorted_tgt).abs().mean()
#         return loss / 3
#
#     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         if self.mode == "bins":
#             return self._bin_emd(pred, target)
#         return self._batch_wasserstein(pred, target)
#
#
# class CombinedLoss(nn.Module):
#     def __init__(
#         self,
#         mse_weight: float = 1.0,
#         emd_weight: float = 0.1,
#         emd_loss=None,
#     ) -> None:
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.mse_weight = mse_weight
#         self.emd_weight = emd_weight
#         self.emd = emd_loss or EMDLoss(mode="batch_wasserstein")
#
#     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return self.mse_weight * self.mse(pred, target) + self.emd_weight * self.emd(pred, target)


def get_loss_fn(cfg: dict) -> nn.Module:
    return nn.MSELoss()


# ---------------------------------------------------------------------------
# C3: Speaker-stratified dual-stream model
# ---------------------------------------------------------------------------

class VADModelDualStream(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(cfg["backbone"])
        H = self.backbone.config.hidden_size
        self.fusion = nn.Linear(3 * H, H)
        self.head = VADHead(H, cfg.get("dropout", 0.1))

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        same_input_ids: torch.Tensor,
        same_attention_mask: torch.Tensor,
        same_valid: torch.Tensor,
        cross_input_ids: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        cross_valid: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target = self._encode(input_ids, attention_mask)
        same   = self._encode(same_input_ids, same_attention_mask) * same_valid.unsqueeze(1).float()
        cross  = self._encode(cross_input_ids, cross_attention_mask) * cross_valid.unsqueeze(1).float()
        fused  = self.fusion(torch.cat([target, same, cross], dim=-1))
        return self.head(fused)

    def freeze_encoder(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True


def get_model(cfg: dict) -> nn.Module:
    if cfg["context"]["strategy"] == "dual_stream":
        return VADModelDualStream(cfg)
    return VADModel(cfg)
