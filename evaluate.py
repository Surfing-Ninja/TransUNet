import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import CFG
from dataset import get_dataloaders
from models.mas_transunet import MaSTransUNet
from utils.metrics import SegmentationMetrics, MetricAggregator


# ======================================================================
# Iterative refinement (test-time FAM loop)
# ======================================================================


def iterative_refinement(
    model: nn.Module,
    image_tensor: torch.Tensor,
    initial_prev_mask: torch.Tensor,
    device: str,
    max_iters: int = 10,
) -> tuple[np.ndarray, int]:
    """Run iterative FAM refinement at test time.

    Args:
        model:              MaSTransUNet in eval mode
        image_tensor:       (1, 3, H, W)
        initial_prev_mask:  (1, 1, H, W)
        device:             "cuda" or "cpu"
        max_iters:          maximum refinement iterations

    Returns:
        final_pred:  (H, W) binary uint8 mask
        iters_used:  number of forward passes executed
    """
    image = image_tensor.to(device)
    prev_mask = initial_prev_mask.to(device)
    prev_binary: np.ndarray | None = None

    for it in range(1, max_iters + 1):
        outputs = model(image, prev_mask)
        pred_prob = outputs["pred_mask"]                        # (1, 1, H, W)
        current_binary = (pred_prob.cpu().numpy()[0, 0] >= 0.5).astype(np.uint8)

        # Convergence check
        if prev_binary is not None:
            eps = 1e-6
            inter = float((current_binary * prev_binary).sum())
            union = float(current_binary.sum() + prev_binary.sum())
            dice = (2 * inter + eps) / (union + eps)
            if dice > 0.999:
                return current_binary, it

        # Feed current prediction back as prev_mask for next iteration
        prev_mask = pred_prob.detach()
        prev_binary = current_binary

    return current_binary, max_iters


# ======================================================================
# Save qualitative examples
# ======================================================================


def _denormalize(img_tensor: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalisation for visualisation.  Input: (3, H, W)."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img_tensor * std + mean
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def save_qualitative(
    image_np: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    filename: str,
    output_dir: str,
) -> None:
    """Save a side-by-side PNG: input | prediction | ground truth."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    fig.tight_layout()
    save_path = os.path.join(output_dir, filename.rsplit(".", 1)[0] + ".png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Full evaluation
# ======================================================================


def evaluate_dataset(dataset_name: str, checkpoint_path: str) -> None:
    """Evaluate MaS-TransUNet on the test split of *dataset_name*.

    Prints a metrics table matching Tables II–VI in the paper and saves
    results to CSV + qualitative PNGs.
    """
    config = CFG
    device = config.device

    # ---- Model -----------------------------------------------------------
    model = MaSTransUNet(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- Data ------------------------------------------------------------
    _, test_loader = get_dataloaders(dataset_name, config)

    # ---- Metrics ---------------------------------------------------------
    aggregator = MetricAggregator()
    per_sample_records: list[dict] = []
    qual_dir = os.path.join(config.log_dir, "qualitative", dataset_name)
    num_qual = 5

    print(f"\nEvaluating {dataset_name} ({len(test_loader.dataset)} samples) "
          f"with up to {config.num_refinement_iters} FAM iterations …\n")

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            images = batch["image"]                              # (B, 3, H, W)
            masks_np = batch["mask"].numpy()                     # (B, 1, H, W)
            prev_masks = batch["prev_mask"]                      # (B, 1, H, W)
            filenames = batch["filename"]

            for i in range(images.shape[0]):
                img_single = images[i : i + 1]                   # (1, 3, H, W)
                pm_single = prev_masks[i : i + 1]                # (1, 1, H, W)

                pred_bin, iters_used = iterative_refinement(
                    model, img_single, pm_single, device,
                    max_iters=config.num_refinement_iters,
                )

                gt_bin = (masks_np[i, 0] >= 0.5).astype(np.uint8)
                metrics = SegmentationMetrics.compute(pred_bin, gt_bin)
                metrics["iters_used"] = iters_used
                metrics["filename"] = filenames[i]
                aggregator.update(metrics)
                per_sample_records.append(metrics)

                # Save qualitative examples
                sample_idx = idx * images.shape[0] + i
                if sample_idx < num_qual:
                    img_vis = _denormalize(images[i].numpy())
                    save_qualitative(
                        img_vis, pred_bin, gt_bin, filenames[i], qual_dir,
                    )

    # ---- Print metrics table (Tables II–VI format) -----------------------
    stats = aggregator.mean_std()
    metric_order = [
        ("dice",                       "Dice"),
        ("iou",                        "IoU"),
        ("sensitivity",                "Sensitivity"),
        ("specificity",                "Specificity"),
        ("precision",                  "Precision"),
        ("accuracy",                   "Accuracy"),
        ("mean_hausdorff_distance",    "MHD"),
        ("mae",                        "MAE"),
        ("ssm",                        "SSM"),
        ("enhanced_alignment_measure", "EM"),
    ]

    header = f"{'Metric':<20s}  {'Mean':>8s}  {'Std':>8s}"
    sep = "-" * len(header)
    print(f"\n  Results: {dataset_name} ({aggregator.count} samples)")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")
    for key, label in metric_order:
        if key in stats:
            mean, std = stats[key]
            print(f"  {label:<20s}  {mean:8.4f}  {std:8.4f}")
    print(f"  {sep}\n")

    # ---- Save CSV --------------------------------------------------------
    os.makedirs(config.log_dir, exist_ok=True)
    csv_path = os.path.join(config.log_dir, f"{dataset_name}_results.csv")
    df = pd.DataFrame(per_sample_records)
    df.to_csv(csv_path, index=False)
    print(f"  Per-sample results saved to {csv_path}")
    print(f"  Qualitative examples saved to {qual_dir}/")


# ======================================================================
# CLI
# ======================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MaS-TransUNet on a test dataset with "
                    "iterative FAM refinement."
    )
    parser.add_argument(
        "dataset_name",
        choices=["covid_ct", "kvasir_seg", "isic2018", "mri_glioma"],
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pth file.  Defaults to "
             "{checkpoint_dir}/{dataset_name}_best.pth",
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(CFG.checkpoint_dir, f"{args.dataset_name}_best.pth")

    if not os.path.isfile(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        raise SystemExit(1)

    evaluate_dataset(args.dataset_name, ckpt_path)
