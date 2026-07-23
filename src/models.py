"""
models.py — Three ImageNet-pretrained backbone builders.

Each builder:
  1. Loads pretrained weights
  2. Freezes the entire backbone (Phase 1: head-only training)
  3. Replaces the classifier head with a custom regularized head
  4. The unfreezing for Phase 2 is handled in train.py → build_optimizer()

Architectures chosen for maximum ensemble complementarity:
  • EfficientNet-B3 — compound-scaled MobileNet; fine-grained local textures
  • ConvNeXt-Tiny   — modernized pure CNN with transformer-style design; spatial anomalies
  • ResNet50        — classic deep residual; stable hierarchical baseline
"""

import torch.nn as nn
from torchvision import models


def build_efficientnet_b3(num_classes: int) -> nn.Module:
    """
    EfficientNet-B3 (ImageNet-1K pretrained).

    Custom head:
        Dropout(0.40) → Linear(1536→512) → SiLU → BatchNorm1d(512)
        → Dropout(0.30) → Linear(512→num_classes)

    Dual-dropout + SiLU (matches EfficientNet's original design) +
    BatchNorm1d after the first linear for training stability.

    Phase 2 unfreezes: features[5:] + classifier (last 3 compound blocks).
    Trainable params in Phase 2: ~10.7 M
    """
    m = models.efficientnet_b3(weights="IMAGENET1K_V1")
    for p in m.parameters():          # freeze entire backbone
        p.requires_grad = False

    in_f = m.classifier[1].in_features   # 1536 for B3
    m.classifier = nn.Sequential(
        nn.Dropout(0.40),
        nn.Linear(in_f, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.30),
        nn.Linear(512, num_classes),
    )
    for p in m.classifier.parameters():  # classifier trainable from start
        p.requires_grad = True

    return m


def build_convnext_tiny(num_classes: int) -> nn.Module:
    """
    ConvNeXt-Tiny (ImageNet-1K pretrained).

    Custom head:
        m.classifier[2] = Linear(768→384) → GELU → Dropout(0.40) → Linear(384→num_classes)

    Keeps ConvNeXt's LayerNorm2d (m.classifier[0]) and Flatten (m.classifier[1])
    intact. Large 7×7 depthwise kernels and global context make this backbone
    strong for structural / spatial anomalies (cracks, electrical patterns).

    Phase 2 unfreezes: features[4:] + classifier (last 2 ConvNeXt stages).
    """
    m = models.convnext_tiny(weights="IMAGENET1K_V1")
    for p in m.parameters():
        p.requires_grad = False

    in_f = m.classifier[2].in_features   # 768 for Tiny
    m.classifier[2] = nn.Sequential(
        nn.Linear(in_f, 384),
        nn.GELU(),
        nn.Dropout(0.40),
        nn.Linear(384, num_classes),
    )
    for p in m.classifier.parameters():
        p.requires_grad = True

    return m


def build_resnet50(num_classes: int) -> nn.Module:
    """
    ResNet50 (ImageNet-1K pretrained).

    Custom head:
        Linear(2048→512) → ReLU → BatchNorm1d(512) → Dropout(0.45)
        → Linear(512→num_classes)

    Skip connections solve the vanishing gradient problem. Classic
    bottleneck blocks learn hierarchical features from low-level edges
    to high-level semantics — a stable complementary baseline.

    Phase 2 unfreezes: layer3 + layer4 + fc (last two residual groups).
    """
    m = models.resnet50(weights="IMAGENET1K_V1")
    for p in m.parameters():
        p.requires_grad = False

    in_f = m.fc.in_features   # 2048 for ResNet50
    m.fc = nn.Sequential(
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.45),
        nn.Linear(512, num_classes),
    )
    for p in m.fc.parameters():
        p.requires_grad = True

    return m
