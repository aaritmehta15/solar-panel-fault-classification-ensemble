"""
train_all.py — End-to-end replication script.

Runs the complete pipeline from raw images to saved checkpoints, evaluation
plots, and metadata JSON — reproducing the research notebook results exactly.

Usage
-----
    python train_all.py --data_dir /path/to/dataset

Dataset structure expected (ImageFolder-compatible):
    data/
    ├── Bird-drop/
    ├── Clean/
    ├── Dusty/
    ├── Electrical-damage/
    ├── Physical-Damage/
    └── Snow-Covered/

Kaggle dataset: https://www.kaggle.com/datasets/pythonafroz/solar-panel-images

Expected results (SEED=42, Google Colab T4 GPU):
    Test Accuracy : 93.98%
    Macro AUC     : 0.9915
    Macro F1      : 0.9351
    Ensemble Weights: EFF=0.20, CNX=0.40, RES=0.40
"""

import argparse
import os
import json

import torch

# ── Resolve imports whether run as script or module ──────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src import config as cfg
from src.utils import set_seed, ensure_output_dir, get_device, save_metadata
from src.dataset import build_splits, build_loaders
from src.models import build_efficientnet_b3, build_convnext_tiny, build_resnet50
from src.loss import build_criterion
from src.train import train_two_phase
from src.ensemble import get_probs, grid_search_weights, ensemble_tta_probs
from src.evaluate import (
    accuracy,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_roc,
    plot_history,
    plot_weight_search,
    save_classification_report,
)
from src.gradcam import get_gradcam_layers, plot_gradcam_grid
from src.config import PHASE1_EPOCHS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solar Panel Fault Classifier — End-to-End Training"
    )
    parser.add_argument(
        "--data_dir",
        default=cfg.DATA_DIR,
        help="Path to ImageFolder dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default=cfg.OUTPUT_DIR,
        help="Directory to save checkpoints and plots (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 0. Setup ──────────────────────────────────────────────────────────────
    set_seed(cfg.SEED)
    ensure_output_dir(args.output_dir)
    device = get_device()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1: Data loading & splitting")
    print("="*60)
    full_dataset, train_idx, val_idx, test_idx, class_names, class_counts = \
        build_splits(args.data_dir)
    num_classes = len(class_names)

    train_loader, val_loader, test_loader, train_dset, val_dset, test_dset = \
        build_loaders(full_dataset, train_idx, val_idx, test_idx, class_counts)

    # ── 2. Loss ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 2: Focal Loss criterion")
    print("="*60)
    criterion = build_criterion(class_counts, num_classes, device)

    # ── 3. Models ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3: Building models")
    print("="*60)
    model_eff = build_efficientnet_b3(num_classes).to(device)
    model_cnx = build_convnext_tiny(num_classes).to(device)
    model_res = build_resnet50(num_classes).to(device)
    print("  ✅ EfficientNet-B3, ConvNeXt-Tiny, ResNet50 ready")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 4: Two-phase training")
    print("="*60)
    hist_eff, best_eff = train_two_phase(
        model_eff, "efficientnet", "EfficientNet-B3",
        train_loader, val_loader, criterion, device, args.output_dir,
    )
    hist_cnx, best_cnx = train_two_phase(
        model_cnx, "convnext", "ConvNeXt-Tiny",
        train_loader, val_loader, criterion, device, args.output_dir,
    )
    hist_res, best_res = train_two_phase(
        model_res, "resnet", "ResNet50",
        train_loader, val_loader, criterion, device, args.output_dir,
    )

    # ── 5. Training curves ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 5: Training curves")
    print("="*60)
    plot_history(hist_eff, "EfficientNet-B3",  PHASE1_EPOCHS, args.output_dir, "steelblue",    "tomato")
    plot_history(hist_cnx, "ConvNeXt-Tiny",    PHASE1_EPOCHS, args.output_dir, "seagreen",     "darkorange")
    plot_history(hist_res, "ResNet50",          PHASE1_EPOCHS, args.output_dir, "mediumpurple", "crimson")

    # ── 6. Ensemble weight search (val set only) ──────────────────────────────
    print("\n" + "="*60)
    print("  STEP 6: Ensemble weight search on val set")
    print("="*60)

    # Reload best checkpoints
    for model, mtype in [(model_eff, "efficientnet"), (model_cnx, "convnext"), (model_res, "resnet")]:
        ckpt = os.path.join(args.output_dir, f"{mtype}_best.pth")
        model.load_state_dict(torch.load(ckpt, map_location=device))

    probs_eff_val, labels_val = get_probs(model_eff, val_loader, device)
    probs_cnx_val, _          = get_probs(model_cnx, val_loader, device)
    probs_res_val, _          = get_probs(model_res, val_loader, device)

    a_eff = accuracy(probs_eff_val, labels_val)
    a_cnx = accuracy(probs_cnx_val, labels_val)
    a_res = accuracy(probs_res_val, labels_val)
    print(f"\nIndividual val accuracies (no TTA):")
    print(f"  EfficientNet-B3 : {a_eff*100:.2f}%")
    print(f"  ConvNeXt-Tiny   : {a_cnx*100:.2f}%")
    print(f"  ResNet50        : {a_res*100:.2f}%")

    best_w, best_val_acc, results = grid_search_weights(
        probs_eff_val, probs_cnx_val, probs_res_val, labels_val
    )
    W_EFF, W_CNX, W_RES = best_w
    plot_weight_search(results, args.output_dir)

    # ── 7. Final test evaluation (TTA ensemble) ───────────────────────────────
    print("\n" + "="*60)
    print("  STEP 7: Final test evaluation (TTA)")
    print("="*60)
    models_dict = {"efficientnet": model_eff, "convnext": model_cnx, "resnet": model_res}
    weights_dict = {"efficientnet": W_EFF, "convnext": W_CNX, "resnet": W_RES}

    test_probs, test_labels = ensemble_tta_probs(models_dict, test_loader, device, weights_dict)
    test_preds = test_probs.argmax(1)
    test_acc   = accuracy(test_probs, test_labels)
    print(f"\n🏆 Test Accuracy (TTA Ensemble): {test_acc*100:.4f}%")

    # ── 8. Evaluation plots ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 8: Evaluation plots")
    print("="*60)
    plot_confusion_matrix(test_preds.numpy(), test_labels.numpy(), class_names, args.output_dir)
    f1s = plot_per_class_f1(test_preds.numpy(), test_labels.numpy(), class_names, args.output_dir)
    macro_auc, per_class_auc = plot_roc(test_probs, test_labels, class_names, num_classes, args.output_dir)
    save_classification_report(
        test_preds.numpy(), test_labels.numpy(), class_names,
        test_acc, best_w, args.output_dir,
    )

    # ── 9. Grad-CAM ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 9: Grad-CAM explainability")
    print("="*60)
    gcam_eff, gcam_cnx, gcam_res = get_gradcam_layers(model_eff, model_cnx, model_res)
    plot_gradcam_grid(
        model_eff, model_cnx, model_res,
        gcam_eff, gcam_cnx, gcam_res,
        test_dset, class_names, device,
        args.output_dir,
        W_EFF, W_CNX, W_RES,
    )

    # ── 10. Save metadata ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 10: Saving metadata")
    print("="*60)
    metadata = {
        "version"           : "2.0",
        "models"            : ["EfficientNet-B3", "ConvNeXt-Tiny", "ResNet50"],
        "split"             : {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "split_fractions"   : {"train": 0.70, "val": cfg.VAL_FRAC, "test": cfg.TEST_FRAC},
        "split_strategy"    : "stratified",
        "img_size"          : cfg.IMG_SIZE,
        "batch_size"        : cfg.BATCH_SIZE,
        "phase1_epochs"     : cfg.PHASE1_EPOCHS,
        "phase2_epochs"     : cfg.PHASE2_EPOCHS,
        "lr_head"           : cfg.LR_HEAD,
        "lr_backbone"       : cfg.LR_BACKBONE,
        "focal_gamma"       : cfg.FOCAL_GAMMA,
        "mixup_alpha"       : cfg.MIXUP_ALPHA,
        "tta_passes"        : cfg.TTA_N,
        "ensemble_weights"  : {"efficientnet_b3": W_EFF, "convnext_tiny": W_CNX, "resnet50": W_RES},
        "individual_val_acc": {
            "efficientnet_b3": round(a_eff * 100, 4),
            "convnext_tiny"  : round(a_cnx * 100, 4),
            "resnet50"       : round(a_res * 100, 4),
        },
        "ensemble_val_acc"  : round(best_val_acc * 100, 4),
        "final_test_acc"    : round(test_acc * 100, 4),
        "macro_auc"         : round(macro_auc, 4),
        "mean_f1"           : round(float(f1s.mean()), 4),
        "per_class_f1"      : {class_names[i]: round(float(f1s[i]), 4) for i in range(num_classes)},
        "per_class_auc"     : {k: round(v, 4) for k, v in per_class_auc.items()},
        "class_counts_train": {class_names[i]: int(class_counts[i]) for i in range(num_classes)},
        "classes"           : class_names,
    }
    save_metadata(os.path.join(args.output_dir, "run_metadata_v2.json"), metadata)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  🏆  FINAL RESULTS (HELD-OUT TEST SET)")
    print(f"{'='*60}")
    print(f"  Accuracy  : {test_acc*100:.4f}%")
    print(f"  Macro AUC : {macro_auc:.4f}")
    print(f"  Macro F1  : {f1s.mean():.4f}")
    print(f"  Weights   : EFF={W_EFF:.2f}  CNX={W_CNX:.2f}  RES={W_RES:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
