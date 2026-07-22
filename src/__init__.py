"""
Solar Panel Fault Classification — Modular Package
===================================================
Tri-Model Heterogeneous Ensemble: EfficientNet-B3 + ConvNeXt-Tiny + ResNet50
Test Accuracy: 93.98% | Macro AUC: 0.9915 | Macro F1: 0.9351
"""

__version__ = "2.0.0"
__authors__  = ["Aarit Mehta", "Devanshu Desai"]
__license__  = "MIT"

from .config   import DATA_DIR, OUTPUT_DIR, SEED, IMG_SIZE, BATCH_SIZE
from .utils    import set_seed, get_device, ensure_output_dir, save_metadata
from .dataset  import build_splits, build_loaders, get_train_transform, get_val_transform
from .models   import build_efficientnet_b3, build_convnext_tiny, build_resnet50
from .loss     import FocalLoss, build_criterion
from .train    import train_two_phase
from .ensemble import get_probs, tta_probs, grid_search_weights, ensemble_tta_probs
from .gradcam  import GradCAM, get_gradcam_layers, plot_gradcam_grid

__all__ = [
    # config
    "DATA_DIR", "OUTPUT_DIR", "SEED", "IMG_SIZE", "BATCH_SIZE",
    # utils
    "set_seed", "get_device", "ensure_output_dir", "save_metadata",
    # dataset
    "build_splits", "build_loaders", "get_train_transform", "get_val_transform",
    # models
    "build_efficientnet_b3", "build_convnext_tiny", "build_resnet50",
    # loss
    "FocalLoss", "build_criterion",
    # train
    "train_two_phase",
    # ensemble
    "get_probs", "tta_probs", "grid_search_weights", "ensemble_tta_probs",
    # gradcam
    "GradCAM", "get_gradcam_layers", "plot_gradcam_grid",
]
