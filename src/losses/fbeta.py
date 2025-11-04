from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


class FBetaLoss(nn.Module):
    """
    Differentiable FBeta loss for binary classification.

    Computes a smooth approximation of the FBeta score using probabilities
    (via sigmoid over logits) without hard thresholding, and returns
    1 - FBeta as the loss.

    Args:
        beta: Weight of recall in the combined score. beta=1.0 -> F1.
        eps: Small epsilon to avoid division by zero.
        reduce: Reduction over the batch: "mean", "sum", or "none".
    """

    def __init__(self, beta: float = 1.0, eps: float = 1e-7, reduce: str = "mean") -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be > 0")
        if reduce not in {"mean", "sum", "none"}:
            raise ValueError('reduce must be one of {"mean", "sum", "none"}')

        self.beta = float(beta)
        self.eps = float(eps)
        self.reduce = reduce

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits: Raw model outputs of shape (N,) or (N, 1).
            targets: Binary targets of shape (N,) with values in {0, 1}.
        Returns:
            Loss tensor (scalar if reduce != "none").
        """
        # Ensure shapes are compatible
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)

        targets = targets.float()

        # Convert logits to probabilities smoothly
        probs = torch.sigmoid(logits)

        # True positives, false positives, false negatives (soft counts)
        true_positives = (probs * targets).sum(dim=0)
        false_positives = (probs * (1.0 - targets)).sum(dim=0)
        false_negatives = ((1.0 - probs) * targets).sum(dim=0)

        precision = true_positives / (true_positives + false_positives + self.eps)
        recall = true_positives / (true_positives + false_negatives + self.eps)

        beta2 = self.beta * self.beta
        fbeta = (1.0 + beta2) * (precision * recall) / (beta2 * precision + recall + self.eps)

        # Convert to loss
        loss = 1.0 - fbeta

        if self.reduce == "mean":
            return loss.mean() if loss.dim() > 0 else loss
        if self.reduce == "sum":
            return loss.sum() if loss.dim() > 0 else loss
        return loss


