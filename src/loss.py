"""
loss.py — Focal Loss with label smoothing and per-class inverse-frequency weights.

Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    https://arxiv.org/abs/1708.02002

Why Focal Loss?
    Standard Cross-Entropy treats every example equally. In a 3× imbalanced
    dataset (69 Physical-Damage vs 207 Bird-drop), CE is dominated by easy
    majority-class samples, causing the model to under-optimize for rare faults.

    Focal Loss modulates with (1 - p_t)^γ:
      • A correctly classified sample at p_t=0.9 contributes (0.1)² = 0.01 to the loss.
      • A hard/rare sample at p_t=0.2 contributes (0.8)² = 0.64.
    With γ=2.0, gradient is concentrated on uncertain and misclassified examples.

    Combined with WeightedRandomSampler (dataset.py) = double imbalance correction:
      one at the data-sampling level, one at the gradient level.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FOCAL_GAMMA, FOCAL_SMOOTH


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    Parameters
    ----------
    alpha    : per-class weight tensor (normalized inverse-frequency weights)
    gamma    : focusing parameter (default 2.0). γ=0 reduces to standard CrossEntropy.
    smoothing: label smoothing epsilon (default 0.05). Prevents overconfidence
               and improves AUC calibration.
    """

    def __init__(
        self,
        alpha=None,
        gamma: float = FOCAL_GAMMA,
        smoothing: float = FOCAL_SMOOTH,
    ):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-sample CE loss (no reduction) with label smoothing
        ce = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.smoothing,
        )
        # Focal modulation: down-weight easy examples
        pt           = torch.exp(-ce)
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


def build_criterion(class_counts: dict, num_classes: int, device: torch.device) -> FocalLoss:
    """
    Build a FocalLoss criterion with normalized inverse-frequency class weights.

    Weights are normalized so they average to 1 — keeps loss magnitude stable
    relative to an unweighted baseline.

    Physical-Damage receives weight ≈ 1.83 (3× higher than Bird-drop),
    ensuring the model is penalized heavily for missing the rarest fault class.
    """
    raw_weights = torch.tensor(
        [1.0 / class_counts[i] for i in range(num_classes)],
        dtype=torch.float32,
    )
    # Normalize: weights average to 1
    class_weights = (raw_weights / raw_weights.mean()).to(device)

    print("Focal Loss class weights (normalized):")
    return FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA, smoothing=FOCAL_SMOOTH)
