"""
evaluate.py — Evaluation metrics, classification report, and result plots.

Produces:
  • confusion_matrix.png      — 6×6 normalized heatmap
  • per_class_f1.png          — per-class F1 bar chart
  • roc_curves.png            — One-vs-Rest ROC curves for all 6 classes
  • EfficientNet_B3_curves.png, ConvNeXt_Tiny_curves.png, ResNet50_curves.png
  • ensemble_weight_search.png — top-30 weight combinations
  • classification_report.txt — full sklearn classification report
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# ── Accuracy helper ───────────────────────────────────────────────────────────

def accuracy(probs, labels):
    return (probs.argmax(1) == labels).float().mean().item()


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(preds, labels, class_names, output_dir):
    cm   = confusion_matrix(labels, preds)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm_n, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set(
        xticks=range(len(class_names)), yticks=range(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted", ylabel="True",
        title="Confusion Matrix — TTA Ensemble (Test Set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_n[i,j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_n[i, j] > thresh else "black")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Per-class F1 bar chart ────────────────────────────────────────────────────

def plot_per_class_f1(preds, labels, class_names, output_dir):
    f1s = f1_score(labels, preds, average=None)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["tomato" if f < 0.90 else "steelblue" for f in f1s]
    bars = ax.bar(class_names, f1s, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(f1s.mean(), ls="--", color="gold", lw=2, label=f"Macro F1 = {f1s.mean():.4f}")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1 Score — TTA Ensemble (Test Set)", fontweight="bold", fontsize=13)
    ax.set_ylabel("F1 Score")
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "per_class_f1.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
    return f1s


# ── ROC / AUC curves ─────────────────────────────────────────────────────────

def plot_roc(test_probs, test_labels, class_names, num_classes, output_dir):
    probs_np  = test_probs.numpy()
    labels_bin = label_binarize(test_labels.numpy(), classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    per_class_auc = {}
    colors = plt.cm.tab10.colors

    for i, (cname, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs_np[:, i])
        auc_val     = roc_auc_score(labels_bin[:, i], probs_np[:, i])
        per_class_auc[cname] = auc_val
        ax.plot(fpr, tpr, color=color, lw=2.0, label=f"{cname} (AUC={auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — One-vs-Rest (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    macro_auc = roc_auc_score(labels_bin, probs_np, average="macro", multi_class="ovr")
    print(f"  Macro AUC (OvR): {macro_auc:.4f}")
    return macro_auc, per_class_auc


# ── Training curves ───────────────────────────────────────────────────────────

def plot_history(hist, name, phase1_epochs, output_dir, c1="steelblue", c2="tomato"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    ep     = range(1, len(hist["train_acc"]) + 1)
    p1_end = min(phase1_epochs, len(hist["train_acc"]))

    # Accuracy
    ax1.plot(ep, hist["train_acc"], label="Train", color=c1, lw=2)
    ax1.plot(ep, hist["val_acc"],   label="Val",   color=c2, lw=2, ls="--")
    ax1.axvline(p1_end, ls=":", color="gray", alpha=0.7, label=f"P1→P2 (ep {p1_end})")
    best_val = max(hist["val_acc"])
    best_ep  = hist["val_acc"].index(best_val) + 1
    ax1.axvline(best_ep, ls="-", color="gold", alpha=0.6, lw=1.5, label=f"Best val={best_val:.3f}")
    ax1.set_title(f"{name} — Accuracy", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3); ax1.set_ylim(0, 1.05)

    # Loss
    ax2.plot(ep, hist["train_loss"], label="Train", color=c1, lw=2)
    ax2.plot(ep, hist["val_loss"],   label="Val",   color=c2, lw=2, ls="--")
    ax2.axvline(p1_end, ls=":", color="gray", alpha=0.7)
    ax2.set_title(f"{name} — Focal Loss", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    plt.suptitle(f"{name} Training History  |  Best Val Acc: {best_val:.4f}",
                 y=1.02, fontsize=11, fontweight="bold")
    plt.tight_layout()
    safe = name.replace("-", "_").replace(" ", "_")
    path = os.path.join(output_dir, f"{safe}_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Ensemble weight search plot ───────────────────────────────────────────────

def plot_weight_search(results, output_dir):
    top30      = results[:30]
    labels_bar = [f"({r[0]:.2f},{r[1]:.2f},{r[2]:.2f})" for r in top30]
    accs_bar   = [r[3] * 100 for r in top30]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors_bar = ["gold" if i == 0 else "steelblue" for i in range(30)]
    ax.barh(range(30), accs_bar[::-1], color=colors_bar[::-1])
    ax.set_yticks(range(30))
    ax.set_yticklabels(labels_bar[::-1], fontsize=7)
    ax.set_xlabel("Val Accuracy (%)", fontsize=11)
    ax.set_title("Top-30 Ensemble Weight Combinations (Val Set)\n"
                 "Format: (w_eff, w_cnx, w_res)  |  Gold = optimal",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "ensemble_weight_search.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Classification report ─────────────────────────────────────────────────────

def save_classification_report(preds, labels, class_names, test_acc, weights, output_dir):
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    w_eff, w_cnx, w_res = weights

    header = (
        f"V2 TTA Ensemble — Test Accuracy: {test_acc*100:.4f}%\n"
        f"Weights: EfficientNet-B3={w_eff:.2f}, ConvNeXt-Tiny={w_cnx:.2f}, ResNet50={w_res:.2f}\n\n"
    )
    path = os.path.join(output_dir, "classification_report.txt")
    with open(path, "w") as f:
        f.write(header + report)
    print(report)
    print(f"  Saved: {path}")
