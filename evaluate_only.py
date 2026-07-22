"""
evaluate_only.py — Load saved checkpoints and run evaluation only.

Use this if you already have the .pth checkpoint files from a previous run
(e.g., downloaded separately) and want to reproduce only the evaluation
metrics and plots without re-training.

Note: Model checkpoints are NOT stored in this repository because they
exceed GitHub's 100 MB single-file limit:
    convnext_best.pth   : 109.9 MB
    resnet_best.pth     :  96.3 MB
    efficientnet_best.pth: 45.4 MB

To obtain checkpoints: run train_all.py OR contact the authors.

Usage
-----
    python evaluate_only.py --data_dir /path/to/dataset --weights_dir ./DL_IA2_SolarFault_TriEnsemble_Outputs

Expected results (with original checkpoints, SEED=42):
    Test Accuracy : 93.98%
    Macro AUC     : 0.9915
    Macro F1      : 0.9351
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.utils import set_seed, ensure_output_dir, get_device
from src.dataset import build_splits, build_loaders
from src.models import build_efficientnet_b3, build_convnext_tiny, build_resnet50
from src.ensemble import get_probs, grid_search_weights, ensemble_tta_probs
from src.evaluate import (
    accuracy,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_roc,
    save_classification_report,
)
from src.gradcam import get_gradcam_layers, plot_gradcam_grid
from src import config as cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solar Panel Fault Classifier — Evaluation Only (requires .pth checkpoints)"
    )
    parser.add_argument("--data_dir",    default=cfg.DATA_DIR,    help="ImageFolder dataset root")
    parser.add_argument("--weights_dir", default=cfg.OUTPUT_DIR,  help="Directory containing .pth checkpoints")
    parser.add_argument("--output_dir",  default=cfg.OUTPUT_DIR,  help="Directory to save plots")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(cfg.SEED)
    ensure_output_dir(args.output_dir)
    device = get_device()

    # Data
    full_dataset, train_idx, val_idx, test_idx, class_names, class_counts = \
        build_splits(args.data_dir)
    num_classes = len(class_names)
    _, val_loader, test_loader, _, _, test_dset = \
        build_loaders(full_dataset, train_idx, val_idx, test_idx, class_counts)

    # Load models from checkpoints
    def load(model, mtype):
        ckpt = os.path.join(args.weights_dir, f"{mtype}_best.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}\n"
                "Please run train_all.py first to generate checkpoints."
            )
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device).eval()
        return model

    model_eff = load(build_efficientnet_b3(num_classes), "efficientnet")
    model_cnx = load(build_convnext_tiny(num_classes),   "convnext")
    model_res = load(build_resnet50(num_classes),         "resnet")
    print("✅ All checkpoints loaded")

    # Ensemble weights from val set
    probs_eff_val, labels_val = get_probs(model_eff, val_loader, device)
    probs_cnx_val, _          = get_probs(model_cnx, val_loader, device)
    probs_res_val, _          = get_probs(model_res, val_loader, device)
    best_w, best_val_acc, _   = grid_search_weights(
        probs_eff_val, probs_cnx_val, probs_res_val, labels_val
    )
    W_EFF, W_CNX, W_RES = best_w

    # TTA ensemble on test set
    models_dict  = {"efficientnet": model_eff, "convnext": model_cnx, "resnet": model_res}
    weights_dict = {"efficientnet": W_EFF, "convnext": W_CNX, "resnet": W_RES}
    test_probs, test_labels = ensemble_tta_probs(models_dict, test_loader, device, weights_dict)
    test_preds = test_probs.argmax(1)
    test_acc   = accuracy(test_probs, test_labels)
    print(f"\n🏆 Test Accuracy (TTA Ensemble): {test_acc*100:.4f}%")

    # Plots
    plot_confusion_matrix(test_preds.numpy(), test_labels.numpy(), class_names, args.output_dir)
    f1s = plot_per_class_f1(test_preds.numpy(), test_labels.numpy(), class_names, args.output_dir)
    macro_auc, _ = plot_roc(test_probs, test_labels, class_names, num_classes, args.output_dir)
    save_classification_report(
        test_preds.numpy(), test_labels.numpy(), class_names,
        test_acc, best_w, args.output_dir,
    )

    # Grad-CAM
    gcam_eff, gcam_cnx, gcam_res = get_gradcam_layers(model_eff, model_cnx, model_res)
    plot_gradcam_grid(
        model_eff, model_cnx, model_res,
        gcam_eff, gcam_cnx, gcam_res,
        test_dset, class_names, device,
        args.output_dir,
        W_EFF, W_CNX, W_RES,
    )

    print(f"\n{'='*60}")
    print(f"  Accuracy: {test_acc*100:.4f}%  |  AUC: {macro_auc:.4f}  |  F1: {f1s.mean():.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
