"""
ensemble.py — Inference, TTA, and ensemble weight optimization.

Test-Time Augmentation (TTA):
    At inference, each image is forward-passed through 5 geometric variants
    (original, H-flip, V-flip, 90° rotation, 270° rotation) and the
    softmax probabilities are averaged. This reduces prediction variance
    from real-world image variability (camera angles, lighting conditions).
    TTA provided the largest single performance boost: +3.00% on the test set.

Ensemble weight optimization:
    Grid search over (w_eff, w_cnx, w_res) where weights sum to 1.
    Evaluated ONLY on the validation set — the test set is never touched
    during this step, preserving a truly held-out evaluation.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.cuda.amp import autocast

from .config import TTA_N


# ── Softmax probability collection ───────────────────────────────────────────

@torch.no_grad()
def get_probs(model, loader, device):
    """
    Return (N, C) softmax probabilities and (N,) ground-truth labels
    for all samples in `loader`. No TTA — single deterministic forward pass.
    """
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        with autocast():
            logits = model(imgs)
        all_probs.append(torch.softmax(logits, dim=1).cpu())
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_labels)


# ── TTA inference ─────────────────────────────────────────────────────────────

def _tta_augment(img_batch):
    """
    Generate TTA_N geometric variants of a batch.

    Passes:
        0 — original
        1 — horizontal flip
        2 — vertical flip
        3 — 90° rotation
        4 — 270° rotation
    """
    variants = [img_batch]
    if TTA_N > 1:
        variants.append(TF.hflip(img_batch))
    if TTA_N > 2:
        variants.append(TF.vflip(img_batch))
    if TTA_N > 3:
        variants.append(torch.rot90(img_batch, 1, [2, 3]))
    if TTA_N > 4:
        variants.append(torch.rot90(img_batch, 3, [2, 3]))
    return variants[:TTA_N]


@torch.no_grad()
def tta_probs(model, loader, device, n_passes: int = TTA_N):
    """
    Return (N, C) TTA-averaged softmax probabilities and (N,) ground-truth labels.

    Each image is augmented `n_passes` ways; probabilities are averaged across passes.
    """
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        batch_probs = torch.zeros(imgs.size(0), model_num_classes(model), device=device)

        for aug in _tta_augment(imgs)[:n_passes]:
            with autocast():
                logits = model(aug)
            batch_probs += torch.softmax(logits, dim=1)

        batch_probs /= n_passes
        all_probs.append(batch_probs.cpu())
        all_labels.append(labels)

    return torch.cat(all_probs), torch.cat(all_labels)


def model_num_classes(model):
    """Infer output dimension from the last linear layer of the classifier head."""
    for layer in reversed(list(model.modules())):
        if isinstance(layer, torch.nn.Linear):
            return layer.out_features
    raise ValueError("Cannot infer num_classes from model")


# ── TTA ensemble inference ────────────────────────────────────────────────────

@torch.no_grad()
def ensemble_tta_probs(models_dict, loader, device, weights, n_passes: int = TTA_N):
    """
    Run TTA on each model in `models_dict` and combine with `weights`.

    Parameters
    ----------
    models_dict : {"efficientnet": model, "convnext": model, "resnet": model}
    weights     : {"efficientnet": w_eff, "convnext": w_cnx, "resnet": w_res}
                  (must sum to 1)
    """
    probs_dict = {}
    for name, model in models_dict.items():
        model.eval()
        probs, labels = tta_probs(model, loader, device, n_passes)
        probs_dict[name] = probs

    # Get labels from any model (all share the same loader)
    combined = sum(weights[name] * probs_dict[name] for name in models_dict)
    return combined, labels


# ── Ensemble weight grid search ───────────────────────────────────────────────

def grid_search_weights(p_eff, p_cnx, p_res, labels, step: float = 0.05):
    """
    Grid search over (w_eff, w_cnx, w_res) where all weights sum to 1.

    Parameters
    ----------
    p_eff, p_cnx, p_res : torch.Tensor, shape (N, C)
        Val-set softmax probabilities from each model (no TTA).
    labels : torch.Tensor, shape (N,)
    step   : grid resolution (0.05 = 5% steps)

    Returns
    -------
    best_w     : (w_eff, w_cnx, w_res) tuple
    best_acc   : float
    results    : list of (w_eff, w_cnx, w_res, acc) tuples sorted by acc desc
    """
    def acc(probs, lbls):
        return (probs.argmax(1) == lbls).float().mean().item()

    best_w, best_acc = None, 0.0
    results = []

    for w1 in np.arange(0.20, 0.71, step):
        for w2 in np.arange(0.10, 0.61, step):
            w3 = round(1.0 - w1 - w2, 4)
            if w3 < 0.05 or w3 > 0.65:
                continue
            ens = w1 * p_eff + w2 * p_cnx + w3 * p_res
            a   = acc(ens, labels)
            results.append((float(w1), float(w2), float(w3), a))
            if a > best_acc:
                best_acc = a
                best_w   = (float(w1), float(w2), float(w3))

    results.sort(key=lambda x: -x[3])

    w_eff, w_cnx, w_res = best_w
    print(f"\n✅ Best ensemble val accuracy : {best_acc*100:.2f}%")
    print(f"   Optimal weights:")
    print(f"     EfficientNet-B3 = {w_eff:.2f}")
    print(f"     ConvNeXt-Tiny   = {w_cnx:.2f}")
    print(f"     ResNet50        = {w_res:.2f}")

    return best_w, best_acc, results
