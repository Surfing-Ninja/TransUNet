import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import CFG
from dataset import get_dataloaders, MedicalSegDataset
from losses import MaSLoss
from models import build_model
from utils.metrics import SegmentationMetrics, MetricAggregator
from utils.checkpointing import (
    save_checkpoint,
    load_latest_checkpoint,
    save_epoch_masks,
    load_epoch_masks,
)


# ======================================================================
# Training helpers
# ======================================================================


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: MaSLoss,
    device: str,
    epoch: int,
    dataset: MedicalSegDataset,
) -> float:
    """Run one training epoch.

    Returns:
        Mean training loss over all batches.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        edges = batch["edge"].to(device)
        prev_masks = batch["prev_mask"].to(device)
        filenames = batch["filename"]

        # Forward
        outputs = model(images, prev_masks)

        targets = {"mask": masks, "edge": edges}
        total_loss, _ = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        num_batches += 1

        # Update previous-epoch masks for FAM
        pred_np = outputs["pred_mask"].detach().cpu().numpy()
        for i, fname in enumerate(filenames):
            # Store as uint8 (H, W) for memory efficiency
            mask_hw = (pred_np[i, 0] * 255).astype(np.uint8)
            dataset.update_prev_mask(fname, mask_hw)

    return running_loss / max(num_batches, 1)


def validate(
    model: nn.Module,
    loader,
    aggregator: MetricAggregator,
    device: str,
) -> float:
    """Run validation / test evaluation.

    Returns:
        Mean Dice score across all samples.
    """
    model.eval()
    aggregator.reset()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            prev_masks = batch["prev_mask"].to(device)
            masks_np = batch["mask"].numpy()  # (B, 1, H, W)

            outputs = model(images, prev_masks)
            preds = outputs["pred_mask"].cpu().numpy()  # (B, 1, H, W)

            for i in range(preds.shape[0]):
                pred_bin = (preds[i, 0] >= 0.5).astype(np.uint8)
                gt_bin = (masks_np[i, 0] >= 0.5).astype(np.uint8)
                metrics = SegmentationMetrics.compute(pred_bin, gt_bin)
                aggregator.update(metrics)

    stats = aggregator.mean_std()
    mean_dice = stats["dice"][0] if stats else 0.0
    return mean_dice


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train MaS-TransUNet on a medical segmentation dataset."
    )
    parser.add_argument(
        "dataset_name",
        choices=["tcga_lgg", "covid_ct", "dsb2018", "kvasir_seg", "isic2018"],
        help="Dataset to train on.",
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name

    config = CFG
    device = config.device

    # ---- Model -----------------------------------------------------------
    model = build_model(config)

    # ---- Optimizer / scheduler -------------------------------------------
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=1e-6,
    )

    # ---- Loss ------------------------------------------------------------
    criterion = MaSLoss(config)

    # ---- Data ------------------------------------------------------------
    train_loader, test_loader = get_dataloaders(dataset_name, config)
    train_dataset: MedicalSegDataset = train_loader.dataset

    # ---- TensorBoard -----------------------------------------------------
    log_dir = os.path.join(config.log_dir, dataset_name)
    writer = SummaryWriter(log_dir=log_dir)

    # ---- Resume from checkpoint ------------------------------------------
    start_epoch, best_dice = load_latest_checkpoint(
        model, optimizer, scheduler, dataset_name, config.checkpoint_dir,
    )

    # Restore previous-epoch masks
    prev_masks = load_epoch_masks(dataset_name, config.checkpoint_dir)
    for fname, mask_arr in prev_masks.items():
        train_dataset.update_prev_mask(fname, mask_arr)

    # ---- Metric aggregator -----------------------------------------------
    aggregator = MetricAggregator()

    # ---- Training loop ---------------------------------------------------
    epoch_bar = tqdm(
        range(start_epoch, config.num_epochs),
        desc=f"[{dataset_name}]",
        unit="epoch",
    )

    for epoch in epoch_bar:
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, train_dataset,
        )

        # Validate
        val_dice = validate(model, test_loader, aggregator, device)

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Checkpointing ----------------------------------------------
        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice

        if (epoch + 1) % config.checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_dice,
                dataset_name, config.checkpoint_dir, is_best=is_best,
            )

        if is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_dice,
                dataset_name, config.checkpoint_dir, is_best=True,
            )

        # Save epoch masks for FAM
        save_epoch_masks(train_dataset.prev_masks, dataset_name, config.checkpoint_dir)

        # ---- TensorBoard logging -----------------------------------------
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/dice", val_dice, epoch)
        writer.add_scalar("val/best_dice", best_dice, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        stats = aggregator.mean_std()
        for metric_name, (mean_val, _) in stats.items():
            writer.add_scalar(f"val/{metric_name}", mean_val, epoch)

        # ---- Progress bar ------------------------------------------------
        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            dice=f"{val_dice:.4f}",
            best=f"{best_dice:.4f}",
            lr=f"{current_lr:.1e}",
        )

    writer.close()
    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
