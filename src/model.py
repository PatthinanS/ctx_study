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

class EMDLoss(nn.Module):
    """EMD-based loss for continuous VAD regression.

    mode="bins": per-sample CDF-L1 over soft bin assignments.
    mode="batch_wasserstein": sort-based W1 per dimension over the batch.
    """

    def __init__(
        self,
        n_bins: int = 10,
        vad_range: tuple[float, float] = (1.0, 5.0),
        mode: str = "batch_wasserstein",
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "bins":
            lo, hi = vad_range
            edges = torch.linspace(lo, hi, n_bins + 1)
            midpoints = (edges[:-1] + edges[1:]) / 2
            self.register_buffer("bin_midpoints", midpoints)

    def _soft_assign(self, values: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        # values: (B,), bin_midpoints: (n_bins,)
        dists = -((values.unsqueeze(1) - self.bin_midpoints.unsqueeze(0)) ** 2) / temperature
        return torch.softmax(dists, dim=1)

    def _bin_emd(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        for d in range(3):
            p_dist = self._soft_assign(pred[:, d])
            t_dist = self._soft_assign(target[:, d])
            cdf_diff = torch.cumsum(p_dist, dim=1) - torch.cumsum(t_dist, dim=1)
            loss = loss + cdf_diff.abs().mean()
        return loss / 3

    def _batch_wasserstein(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        for d in range(3):
            sorted_pred = torch.sort(pred[:, d]).values
            sorted_tgt = torch.sort(target[:, d]).values
            loss = loss + (sorted_pred - sorted_tgt).abs().mean()
        return loss / 3

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "bins":
            return self._bin_emd(pred, target)
        return self._batch_wasserstein(pred, target)


class CombinedLoss(nn.Module):
    def __init__(
        self,
        mse_weight: float = 1.0,
        emd_weight: float = 0.1,
        emd_loss: EMDLoss | None = None,
    ) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.emd_weight = emd_weight
        self.emd = emd_loss or EMDLoss(mode="batch_wasserstein")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse_weight * self.mse(pred, target) + self.emd_weight * self.emd(pred, target)


def get_loss_fn(cfg: dict) -> nn.Module:
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "mse")
    if name == "mse":
        return nn.MSELoss()
    if name == "emd_bins":
        return EMDLoss(
            n_bins=loss_cfg.get("emd_n_bins", 10),
            vad_range=tuple(loss_cfg.get("emd_vad_range", [1.0, 5.0])),
            mode="bins",
        )
    if name == "emd_batch":
        return EMDLoss(mode="batch_wasserstein")
    if name == "combined":
        emd = EMDLoss(mode="batch_wasserstein")
        return CombinedLoss(
            mse_weight=loss_cfg.get("mse_weight", 1.0),
            emd_weight=loss_cfg.get("emd_weight", 0.1),
            emd_loss=emd,
        )
    raise ValueError(f"Unknown loss '{name}'. Choose: mse, emd_bins, emd_batch, combined.")
