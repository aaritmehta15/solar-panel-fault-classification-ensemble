"""
train.py — Two-phase discriminative fine-tuning engine.

Phase 1 (head-only, PHASE1_EPOCHS):
    Backbone frozen. Only the randomly-initialized classifier head is trained.
    Rationale: prevents large early-epoch gradients from the new head from
    corrupting pre-trained ImageNet weights (catastrophic forgetting).

Phase 2 (discriminative fine-tune, PHASE2_EPOCHS):
    Upper backbone blocks unfrozen with LR_BACKBONE (50× smaller than LR_HEAD).
    Allows slow domain adaptation while preserving lower-level ImageNet features.
    Uses early stopping with patience=PATIENCE.

Additional techniques:
  • AMP (FP16 forward / FP32 weights)  — ~2× speed on T4 GPU
  • Gradient clipping (max_norm=1.0)   — prevents gradient explosion
  • Linear warmup + cosine decay LR    — prevents training instability
  • Mixup augmentation (50% of batches) — strong regularizer on small datasets
"""

import os
import math
import time

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from .config import (
    PHASE1_EPOCHS, PHASE2_EPOCHS,
    LR_HEAD, LR_BACKBONE, WEIGHT_DECAY,
    MIXUP_ALPHA, MIXUP_PROB, OUTPUT_DIR, PATIENCE,
)


# ── Optimizer ─────────────────────────────────────────────────────────────────

def build_optimizer(model, phase: int, model_type: str) -> optim.AdamW:
    """
    Build an AdamW optimizer with discriminative learning rates.

    Phase 1: Only classifier head parameters (backbone still frozen).
    Phase 2: Unfreeze upper backbone blocks with LR_BACKBONE;
             classifier head keeps LR_HEAD.
    """
    if model_type == "efficientnet":
        head_params     = list(model.classifier.parameters())
        backbone_params = list(model.features[5:].parameters())
    elif model_type == "convnext":
        head_params     = list(model.classifier.parameters())
        backbone_params = list(model.features[4:].parameters())
    else:  # resnet
        head_params     = list(model.fc.parameters())
        backbone_params = (
            list(model.layer3.parameters()) +
            list(model.layer4.parameters())
        )

    if phase == 1:
        return optim.AdamW(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    # Phase 2: discriminative rates
    for p in backbone_params:
        p.requires_grad = True
    return optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params,     "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )


# ── LR Schedule ───────────────────────────────────────────────────────────────

def get_schedule(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup → cosine decay scheduler."""
    def fn(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(max(warmup_epochs, 1))
        prog = (ep - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return optim.lr_scheduler.LambdaLR(optimizer, fn)


# ── Mixup ─────────────────────────────────────────────────────────────────────

def mixup_batch(x, y, alpha: float = MIXUP_ALPHA):
    """
    Mixup: linearly interpolate two random samples within the batch.

    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization",
    ICLR 2018. https://arxiv.org/abs/1710.09412
    """
    lam  = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    """Run one epoch of training or validation. Returns (avg_loss, accuracy)."""
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if train and np.random.rand() < MIXUP_PROB:
                imgs, y_a, y_b, lam = mixup_batch(imgs, labels)
                with autocast():
                    logits = model(imgs)
                    loss   = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                with autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ── Phase runner ──────────────────────────────────────────────────────────────

def run_phase(
    model,
    model_type: str,
    tag: str,
    epochs: int,
    warmup: int,
    phase: int,
    train_loader,
    val_loader,
    criterion,
    device,
    prev_best_acc: float = 0.0,
    output_dir: str = OUTPUT_DIR,
):
    """
    Train for `epochs` epochs with early stopping.
    Saves the best checkpoint only when val accuracy improves on prev_best_acc.

    Returns
    -------
    history    : dict with train_acc, val_acc, train_loss, val_loss lists
    best_acc   : float — best val accuracy achieved
    """
    optimizer  = build_optimizer(model, phase, model_type)
    scheduler  = get_schedule(optimizer, warmup, epochs)
    scaler     = GradScaler()
    save_path  = os.path.join(output_dir, f"{model_type}_best.pth")

    history    = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_acc   = prev_best_acc
    no_improve = 0

    print(f"\n{'─'*60}")
    print(f"  {tag} — Phase {phase} ({epochs} epochs, warmup={warmup})")
    print(f"{'─'*60}")

    for ep in range(epochs):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, scaler, device, train=False)
        scheduler.step()

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        flag = ""
        if va_acc > best_acc:
            best_acc   = va_acc
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            flag = "  ✅ saved"
        else:
            no_improve += 1

        elapsed = time.time() - t0
        print(
            f"  Ep {ep+1:3d}/{epochs}  "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc*100:.2f}%  "
            f"va_loss={va_loss:.4f}  va_acc={va_acc*100:.2f}%  "
            f"[{elapsed:.1f}s]{flag}"
        )

        if no_improve >= PATIENCE:
            print(f"  ⏹  Early stopping at epoch {ep+1} (patience={PATIENCE})")
            break

    print(f"  Best val acc: {best_acc*100:.4f}%  → {save_path}")
    return history, best_acc


# ── Full two-phase pipeline ───────────────────────────────────────────────────

def train_two_phase(
    model,
    model_type: str,
    tag: str,
    train_loader,
    val_loader,
    criterion,
    device,
    output_dir: str = OUTPUT_DIR,
):
    """
    Run Phase 1 (head-only) then Phase 2 (discriminative fine-tune).

    Returns combined history and best val accuracy across both phases.
    """
    # Phase 1: head only
    hist1, best1 = run_phase(
        model, model_type, tag,
        epochs=PHASE1_EPOCHS, warmup=2, phase=1,
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, device=device,
        prev_best_acc=0.0, output_dir=output_dir,
    )

    # Reload best Phase 1 checkpoint before Phase 2
    save_path = os.path.join(output_dir, f"{model_type}_best.pth")
    model.load_state_dict(torch.load(save_path, map_location=device))

    # Phase 2: discriminative fine-tune
    hist2, best2 = run_phase(
        model, model_type, tag,
        epochs=PHASE2_EPOCHS, warmup=3, phase=2,
        train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, device=device,
        prev_best_acc=best1, output_dir=output_dir,
    )

    # Merge histories
    combined = {
        "train_acc":  hist1["train_acc"]  + hist2["train_acc"],
        "val_acc":    hist1["val_acc"]    + hist2["val_acc"],
        "train_loss": hist1["train_loss"] + hist2["train_loss"],
        "val_loss":   hist1["val_loss"]   + hist2["val_loss"],
    }

    return combined, max(best1, best2)
