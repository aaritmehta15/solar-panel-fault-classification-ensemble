"""
config.py — Single source of truth for all hyperparameters.

Every value here matches exactly what was used in the research notebook
(solar_fault_classifier_tri_ensemble_v2.ipynb) to guarantee reproducibility.
Modify only here — all other modules import from this file.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# DATA_DIR must point to an ImageFolder-compatible directory with 6 class subfolders:
#   Bird-drop / Clean / Dusty / Electrical-damage / Physical-Damage / Snow-Covered
DATA_DIR   = os.environ.get("DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./DL_IA2_SolarFault_TriEnsemble_Outputs")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Image & Batch ─────────────────────────────────────────────────────────────
IMG_SIZE    = 224          # Standard ImageNet input; compatible with all three backbones
BATCH_SIZE  = 16           # Fits 3 large models on a single T4 GPU without OOM
NUM_WORKERS = 2

# ── Train/Val/Test fractions ──────────────────────────────────────────────────
VAL_FRAC  = 0.15           # 15% validation — never used for hyperparameter selection
TEST_FRAC = 0.15           # 15% test     — opened EXACTLY ONCE for final reporting

# ── Training schedule ─────────────────────────────────────────────────────────
PHASE1_EPOCHS = 12         # Head-only: sufficient for head convergence on frozen features
PHASE2_EPOCHS = 28         # Discriminative fine-tune: with early stopping (patience=7)
PATIENCE      = 7          # Early-stopping patience in epochs

# ── Learning rates ────────────────────────────────────────────────────────────
LR_HEAD     = 2e-3         # High LR for randomly-initialized head
LR_BACKBONE = 4e-5         # 50× smaller — prevents catastrophic forgetting of ImageNet weights
WEIGHT_DECAY = 1e-4        # AdamW regularization

# ── Loss function ─────────────────────────────────────────────────────────────
FOCAL_GAMMA  = 2.0         # Focal Loss focusing parameter; 0 = standard CrossEntropy
FOCAL_SMOOTH = 0.05        # Label smoothing epsilon — prevents overconfidence

# ── Mixup ─────────────────────────────────────────────────────────────────────
MIXUP_ALPHA = 0.3          # Beta distribution parameter
MIXUP_PROB  = 0.5          # 50% of batches receive Mixup augmentation

# ── Test-Time Augmentation ────────────────────────────────────────────────────
TTA_N = 5                  # 5 geometric passes: original + HFlip + VFlip + Rot(+10°) + Rot(-10°)

# ── ImageNet normalization ────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
