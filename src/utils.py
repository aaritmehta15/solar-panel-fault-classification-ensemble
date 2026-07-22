"""
utils.py — Seed setup, output directory management, and metadata saving.
"""

import os
import json
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix all randomness sources for full reproducibility (matches notebook SEED=42 setup)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Seed fixed: {seed}")


def ensure_output_dir(path: str) -> None:
    """Create the output directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)
    print(f"📁 Output directory: {os.path.abspath(path)}")


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU found — running on CPU (training will be slow)")
    return device


def save_metadata(path: str, metadata: dict) -> None:
    """Persist the full experiment metadata as a JSON file."""
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata saved → {path}")
