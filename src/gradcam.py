"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM).

Grad-CAM answers: "Which image regions did the model attend to?"
It computes gradients of the predicted class score w.r.t. the last
convolutional layer's feature maps; a weighted sum produces a heatmap.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", ICCV 2017.
    https://arxiv.org/abs/1610.02391

Target layers:
    EfficientNet-B3 : model.features[-1]
    ConvNeXt-Tiny   : model.features[-1]
    ResNet50        : model.layer4[-1]

Red regions = most attended; blue/cool regions = less important.
Inter-model disagreement on attended regions is DESIRABLE — it confirms
each backbone extracts complementary information, validating ensemble diversity.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from .config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD


# ── GradCAM class ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM via forward and backward hooks on a specified target layer.

    Usage:
        gcam = GradCAM(model, target_layer)
        heatmap = gcam.generate(img_tensor)   # img_tensor: (1, C, H, W) on device
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self._activations = None
        self._gradients   = None

        def fwd(_, __, out):   self._activations = out
        def bwd(_, __, g):     self._gradients   = g[0].float()

        target_layer.register_forward_hook(fwd)
        target_layer.register_full_backward_hook(bwd)

    def generate(self, tensor, class_idx=None):
        """
        Generate a (H', W') Grad-CAM heatmap in [0, 1].

        Parameters
        ----------
        tensor    : (1, C, H, W) on device — MUST use eval mode with batch size 1
        class_idx : class to explain (None = argmax / predicted class)
        """
        self.model.eval()
        self.model.zero_grad()
        out = self.model(tensor.float())

        if class_idx is None:
            class_idx = out.argmax(1).item()
        out[0, class_idx].backward()

        grads   = self._gradients[0]         # (C, h, w)
        acts    = self._activations[0]       # (C, h, w)
        weights = grads.mean(dim=(1, 2))     # global average pool
        cam     = F.relu((weights[:, None, None] * acts).sum(0))
        cam     = cam - cam.min()
        cam     = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


# ── Visualization helpers ─────────────────────────────────────────────────────

def unnorm(tensor):
    """Undo ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).permute(1, 2, 0).numpy().clip(0, 1)


def overlay_cam(img_tensor, cam):
    """Overlay a Grad-CAM heatmap on the original image (55% image / 45% heat)."""
    import matplotlib.cm as cm_module
    img    = unnorm(img_tensor)
    cam_up = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )
    ) / 255.0
    heat = cm_module.jet(cam_up)[:, :, :3]
    return (0.55 * img + 0.45 * heat).clip(0, 1)


# ── Grad-CAM grid ─────────────────────────────────────────────────────────────

def plot_gradcam_grid(
    model_eff, model_cnx, model_res,
    gcam_eff, gcam_cnx, gcam_res,
    test_dset, class_names, device,
    output_dir,
    w_eff, w_cnx, w_res,
):
    """
    Generate a grid: one row per class (correctly classified), columns:
        Original | EfficientNet-B3 CAM | ConvNeXt-Tiny CAM | ResNet50 CAM

    Only images correctly classified by the ensemble are shown.
    """
    from torch.cuda.amp import autocast

    num_classes   = len(class_names)
    class_examples = {}

    model_eff.eval(); model_cnx.eval(); model_res.eval()

    for img_t, label in test_dset:
        label = int(label)
        if label in class_examples:
            continue
        with torch.no_grad():
            with autocast():
                inp   = img_t.unsqueeze(0).to(device)
                p_eff = torch.softmax(model_eff(inp), 1)
                p_cnx = torch.softmax(model_cnx(inp), 1)
                p_res = torch.softmax(model_res(inp), 1)
                ens_p = w_eff * p_eff + w_cnx * p_cnx + w_res * p_res
        if ens_p.argmax(1).item() == label:
            class_examples[label] = img_t
        if len(class_examples) == num_classes:
            break

    if len(class_examples) < num_classes:
        print(f"⚠️  Only found {len(class_examples)}/{num_classes} correctly classified examples.")

    fig, axes = plt.subplots(num_classes, 4, figsize=(16, num_classes * 3.2))
    col_titles = ["Original", "EfficientNet-B3 CAM", "ConvNeXt-Tiny CAM", "ResNet50 CAM"]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontweight="bold", fontsize=10)

    for row, ci in enumerate(sorted(class_examples)):
        img_t = class_examples[ci]
        inp   = img_t.unsqueeze(0).to(device)

        cam_e = gcam_eff.generate(inp.clone())
        cam_c = gcam_cnx.generate(inp.clone())
        cam_r = gcam_res.generate(inp.clone())

        axes[row, 0].imshow(unnorm(img_t))
        axes[row, 1].imshow(overlay_cam(img_t, cam_e))
        axes[row, 2].imshow(overlay_cam(img_t, cam_c))
        axes[row, 3].imshow(overlay_cam(img_t, cam_r))

        for col in range(4):
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(class_names[ci], fontweight="bold", fontsize=9)

    plt.suptitle(
        "Grad-CAM Explainability — One correctly classified example per class\n"
        "Red = high attention | Blue = low attention",
        fontweight="bold", fontsize=11, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "gradcam_v2.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def get_gradcam_layers(model_eff, model_cnx, model_res):
    """Return (gcam_eff, gcam_cnx, gcam_res) using the correct target layers."""
    gcam_eff = GradCAM(model_eff, model_eff.features[-1])
    gcam_cnx = GradCAM(model_cnx, model_cnx.features[-1])
    gcam_res = GradCAM(model_res, model_res.layer4[-1])
    return gcam_eff, gcam_cnx, gcam_res
