import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.optim as optim
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
from evaluate import evaluate_dataset


# ======================================================================
# Training helpers
# ======================================================================


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: MaSLoss,
    scaler: GradScaler,
    device: str,
    epoch: int,
    dataset: MedicalSegDataset,
    accumulation_steps: int,
    fam_warmup_epochs: int,
) -> float:
    """Run one training epoch.

    Returns:
        Mean training loss over all batches.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    batch_bar = tqdm(
        loader,
        desc=f"train e{epoch + 1}",
        leave=False,
        unit="batch",
    )

    optimizer.zero_grad()

    use_fam_feedback = epoch >= fam_warmup_epochs

    for batch_idx, batch in enumerate(batch_bar, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        edges = batch["edge"].to(device)
        prev_masks = batch["prev_mask"].to(device)
        filenames = batch["filename"]

        # Forward (mixed precision)
        with autocast("cuda", enabled=device.startswith("cuda")):
            outputs = model(images, prev_masks)

        # Loss in float32 for numerical stability
        targets = {"mask": masks, "edge": edges}
        total_loss, _ = criterion(outputs, targets)

        # Backward (gradient accumulation with AMP)
        loss_for_backward = total_loss / accumulation_steps
        scaler.scale(loss_for_backward).backward()

        if batch_idx % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += total_loss.item()
        num_batches += 1
        batch_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        if use_fam_feedback:
            pred_np = torch.sigmoid(outputs["pred_mask"]).detach().cpu().numpy()
            for i, fname in enumerate(filenames):
                mask_hw = (pred_np[i, 0] * 255).astype(np.uint8)
                dataset.update_prev_mask(fname, mask_hw)

    if num_batches > 0 and num_batches % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return running_loss / max(num_batches, 1)


def validate(
    model: nn.Module,
    loader,
    aggregator: MetricAggregator,
    device: str,
    full_metrics: bool = True,
) -> float:
    """Run validation / test evaluation.

    Args:
        full_metrics: If True, compute the full metric suite (Hausdorff,
            SSIM, S-measure, etc.).  If False, compute only Dice for speed.

    Returns:
        Mean Dice score across all samples.
    """
    model.eval()
    aggregator.reset()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            prev_masks = batch["prev_mask"].to(device)
            masks_np = batch["mask"].cpu().numpy().astype(np.float32)  # (B, 1, H, W)

            outputs = model(images, prev_masks)
            preds = torch.sigmoid(outputs["pred_mask"]).cpu().numpy().astype(np.float32)  # (B, 1, H, W)

            for i in range(preds.shape[0]):
                pred_bin = (preds[i, 0] >= 0.5).astype(np.uint8)
                gt_bin = (masks_np[i, 0] >= 0.5).astype(np.uint8)

                if full_metrics:
                    metrics = SegmentationMetrics.compute(pred_bin, gt_bin)
                else:
                    # Fast Dice-only computation
                    eps = 1e-6
                    pred_f = pred_bin.astype(np.float64)
                    gt_f = gt_bin.astype(np.float64)
                    tp = np.sum(pred_f * gt_f)
                    fp = np.sum(pred_f * (1 - gt_f))
                    fn = np.sum((1 - pred_f) * gt_f)
                    dice = (2 * tp) / (2 * tp + fp + fn + eps)
                    metrics = {"dice": float(dice)}

                aggregator.update(metrics)

    stats = aggregator.mean_std()
    mean_dice = stats["dice"][0] if stats else 0.0
    return mean_dice


# ======================================================================
# Main
# ======================================================================


def train_single_dataset(
    dataset_name: str, config, device: str, results_table: dict
) -> float:
    """Train and evaluate a single dataset. Returns best Dice score."""
    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset_name:15s}  |  batch_size={config.batch_size}  num_workers={config.num_workers}")
    print(f"  accumulation_steps={config.accumulation_steps}  effective_batch_size={config.batch_size * config.accumulation_steps}")
    print(f"  fam_warmup_epochs={config.fam_warmup_epochs}")
    print(f"{'='*70}\n")

    # ---- Model -----------------------------------------------------------
    model = build_model(config)

    # torch.compile for free 10-20% speedup (PyTorch 2.0+)
    try:
        if hasattr(torch, "compile"):
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model)
    except Exception:
        pass

    # ---- AMP scaler ------------------------------------------------------
    scaler = GradScaler("cuda", enabled=device.startswith("cuda"))

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
        eta_min=config.eta_min,
    )
    # ---- Loss ------------------------------------------------------------
    criterion = MaSLoss(config)

    # ---- Data ------------------------------------------------------------
    train_loader, test_loader = get_dataloaders(dataset_name, config)
    train_dataset: MedicalSegDataset = train_loader.dataset
    print(
        f"Data ready: train={len(train_dataset)} | test={len(test_loader.dataset)} | "
        f"batches={len(train_loader)} | fast_mode={config.fast_mode}\n"
    )

    # ---- TensorBoard -----------------------------------------------------
    log_dir = os.path.join(config.log_dir, dataset_name)
    class _W:
        def add_scalar(self,*a,**k): pass
        def close(self): pass
    writer = _W()

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
    val_interval = 3

    # ---- Training loop ---------------------------------------------------
    epoch_bar = tqdm(
        range(start_epoch, config.num_epochs),
        desc=f"[{dataset_name}]",
        unit="epoch",
    )

    for epoch in epoch_bar:
        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            epoch,
            train_dataset,
            config.accumulation_steps,
            config.fam_warmup_epochs,
        )

        # Validate every N epochs
        run_validation = (epoch + 1) % val_interval == 0
        val_dice = None
        if run_validation:
            validation_count = (epoch + 1) // val_interval
            full_metrics = (validation_count % 3 == 0)
            val_dice = validate(model, test_loader, aggregator, device, full_metrics)

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Checkpointing ----------------------------------------------
        is_best = run_validation and (val_dice > best_dice)
        if is_best:
            best_dice = val_dice

        if (epoch + 1) % config.checkpoint_interval == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_dice,
                dataset_name, config.checkpoint_dir, is_best=is_best,
            )

        # Save epoch masks for FAM on checkpoint intervals
        if (epoch + 1) % config.checkpoint_interval == 0:
            save_epoch_masks(train_dataset.prev_masks, dataset_name, config.checkpoint_dir)

        # ---- TensorBoard logging -----------------------------------------
        writer.add_scalar("train/loss", train_loss, epoch)
        if run_validation:
            writer.add_scalar("val/dice", val_dice, epoch)
            stats = aggregator.mean_std()
            for metric_name, (mean_val, _) in stats.items():
                writer.add_scalar(f"val/{metric_name}", mean_val, epoch)
        writer.add_scalar("val/best_dice", best_dice, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        # ---- Progress bar ------------------------------------------------
        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            dice=f"{val_dice:.4f}" if run_validation else "-",
            best=f"{best_dice:.4f}",
            lr=f"{current_lr:.1e}",
        )

    writer.close()
    print(f"\n  Training complete. Best Dice: {best_dice:.4f}\n")

    # ---- Auto-evaluation ------------------------------------------------
    best_ckpt = os.path.join(config.checkpoint_dir, f"{dataset_name}_best.pth")
    if os.path.isfile(best_ckpt):
        print(f"  Evaluating {dataset_name} with best checkpoint…\n")
        try:
            evaluate_dataset(dataset_name, best_ckpt)
        except Exception as e:
            print(f"  [Evaluation skipped due to error: {e}]\n")
    else:
        print(f"  [No best checkpoint found at {best_ckpt}; evaluation skipped]\n")

    results_table[dataset_name] = best_dice
    return best_dice


def main():
    parser = argparse.ArgumentParser(
        description="Train MaS-TransUNet on one or more datasets sequentially."
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",
        choices=["mri_glioma", "kvasir_seg", "isic2018", "covid_ct"],
        default=None,
        help="Dataset to train on. If omitted, sequential all-dataset mode is used.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Force sequential training over all datasets.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode (skips heavy train augmentations).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force local mode even if Kaggle is detected.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Absolute path to local datasets root (overrides base_data_dir).",
    )
    args = parser.parse_args()
    dataset_spec = args.dataset_name

    config = CFG
    if args.data_dir is not None and not os.path.isabs(args.data_dir):
        raise ValueError("--data-dir must be an absolute path, e.g. /Users/name/datasets or C:/datasets")
    config.configure_runtime(force_local=args.local, data_dir=args.data_dir)
    config.fast_mode = args.fast
    device = config.device

    if device == "cpu":
        print("[WARNING] Training is running on CPU. This is slow and may hurt convergence; use a CUDA GPU when possible.")

    # Determine which datasets to train on
    all_datasets = ["mri_glioma", "kvasir_seg", "isic2018", "covid_ct"]
    if args.all or dataset_spec is None:
        datasets_to_train = all_datasets
    else:
        datasets_to_train = [dataset_spec]

    # Train each dataset sequentially
    results_table = {}
    for ds_name in datasets_to_train:
        try:
            train_single_dataset(ds_name, config, device, results_table)
        except Exception as e:
            print(f"\n  [ERROR training {ds_name}: {e}]")
            print(f"  [Continuing to next dataset…]\n")
            results_table[ds_name] = None
            continue

    # ---- Print summary table --------------------------------------------
    print(f"\n{'='*70}")
    print(f"  SUMMARY: All Datasets")
    print(f"{'='*70}\n")
    header = f"{'Dataset':<20s}  {'Best Dice':>10s}"
    sep = "-" * len(header)
    print(f"  {header}")
    print(f"  {sep}")
    for ds_name in all_datasets:
        if ds_name in results_table:
            dice = results_table[ds_name]
            if dice is not None:
                print(f"  {ds_name:<20s}  {dice:10.4f}")
            else:
                print(f"  {ds_name:<20s}  {'FAILED':>10s}")
        else:
            print(f"  {ds_name:<20s}  {'SKIPPED':>10s}")
    print(f"  {sep}\n")


if __name__ == "__main__":
    main()
