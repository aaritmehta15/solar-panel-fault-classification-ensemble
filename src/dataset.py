"""
dataset.py — Data loading, stratified splitting, augmentation, and DataLoaders.

Implements the exact pipeline from the research notebook:
  • TransformDataset wrapper (lazy per-split transforms on a single ImageFolder)
  • Double-stratified 70/15/15 split with hard leakage assertions
  • WeightedRandomSampler for class imbalance at the data level
  • Aggressive train augmentation + deterministic val/test transforms
"""

import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from .config import (
    DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    VAL_FRAC, TEST_FRAC, SEED,
    IMAGENET_MEAN, IMAGENET_STD,
)


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """
    Aggressive stochastic augmentation pipeline for the training split.
    Rationale: With ~103 training images per class on average, deep networks
    will overfit without heavy augmentation on a dataset this small.
    """
    return transforms.Compose([
        # Geometric
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0), ratio=(0.80, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.25),   # panels can appear at any orientation
        transforms.RandomRotation(degrees=15),
        # Photometric
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08),
        transforms.RandomGrayscale(p=0.04),
        # Policy-based (samples 2 ops from a large transform pool)
        transforms.RandAugment(num_ops=2, magnitude=8),
        # Tensor conversion
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # Occlusion simulation — teaches robustness to partial panel coverage
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.18), ratio=(0.3, 3.3), value="random"),
    ])


def get_val_transform() -> transforms.Compose:
    """
    Deterministic transform for validation and test splits.
    Only resize + center-crop + normalize — no stochasticity.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ── Dataset Wrapper ───────────────────────────────────────────────────────────

class TransformDataset(torch.utils.data.Dataset):
    """
    Wraps a torchvision ImageFolder + a list of indices with a per-split transform.

    This avoids three separate ImageFolder scans: one ImageFolder loads all raw
    PIL images (transform=None), and three TransformDataset wrappers apply
    train/val/test transforms lazily at __getitem__ time.
    """

    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pil, label = self.dataset[self.indices[idx]]
        return self.transform(pil), label


# ── Splits ────────────────────────────────────────────────────────────────────

def build_splits(data_dir: str = DATA_DIR):
    """
    Load the dataset and compute stratified 70 / 15 / 15 splits.

    Returns
    -------
    full_dataset : ImageFolder
    train_idx, val_idx, test_idx : list[int]
    class_names  : list[str]
    class_counts : Counter
    """
    # Load with transform=None — transforms applied per-split later
    full_dataset = datasets.ImageFolder(data_dir)
    full_dataset.transform = None

    class_names  = full_dataset.classes
    indices      = list(range(len(full_dataset)))
    targets      = [full_dataset.targets[i] for i in indices]

    # Step 1: 70% train / 30% temp (stratified by class label)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(VAL_FRAC + TEST_FRAC),
        stratify=targets,
        random_state=SEED,
    )

    # Step 2: 30% temp → 15% val / 15% test (stratified again)
    temp_labels = [full_dataset.targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_labels,
        random_state=SEED,
    )

    # Hard leakage assertions — will raise immediately if any overlap exists
    assert len(set(train_idx) & set(val_idx))  == 0, "LEAKAGE: train ∩ val"
    assert len(set(train_idx) & set(test_idx)) == 0, "LEAKAGE: train ∩ test"
    assert len(set(val_idx)   & set(test_idx)) == 0, "LEAKAGE: val ∩ test"
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(full_dataset), \
        "Missing samples after split"

    train_labels = [full_dataset.targets[i] for i in train_idx]
    class_counts = Counter(train_labels)

    print("✅ No data leakage confirmed")
    print(f"\nClasses ({len(class_names)}): {class_names}")
    print(f"Total: {len(full_dataset)} | Train: {len(train_idx)} | "
          f"Val: {len(val_idx)} | Test: {len(test_idx)}")

    return full_dataset, train_idx, val_idx, test_idx, class_names, class_counts


# ── DataLoaders ───────────────────────────────────────────────────────────────

def build_loaders(full_dataset, train_idx, val_idx, test_idx, class_counts):
    """
    Build TransformDataset wrappers and DataLoaders.

    WeightedRandomSampler oversamples minority classes (Physical-Damage,
    Electrical-damage) so each epoch sees an approximately balanced class
    distribution — addressing imbalance at the data-sampling level.
    """
    train_transform = get_train_transform()
    val_transform   = get_val_transform()

    train_dset = TransformDataset(full_dataset, train_idx, train_transform)
    val_dset   = TransformDataset(full_dataset, val_idx,   val_transform)
    test_dset  = TransformDataset(full_dataset, test_idx,  val_transform)

    # Inverse-frequency sampling weights
    train_labels = [full_dataset.targets[i] for i in train_idx]
    sample_w     = [1.0 / class_counts[l] for l in train_labels]
    sampler      = WeightedRandomSampler(
        sample_w, num_samples=len(sample_w), replacement=True
    )

    train_loader = DataLoader(
        train_dset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)} | "
          f"Val batches: {len(val_loader)} | "
          f"Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, train_dset, val_dset, test_dset
