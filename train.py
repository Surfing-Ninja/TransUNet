import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
import argparse
import gc

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


def _clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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

    optimizer.zero_grad(set_to_none=True)

    use_fam_feedback = epoch >= fam_warmup_epochs

    for batch_idx, batch in enumerate(batch_bar, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        edges = batch["edge"].to(device, non_blocking=True)
        prev_masks = batch["prev_mask"].to(device, non_blocking=True)
        filenames = batch["filename"]

        # Forward in autocast, loss in float32
        with autocast("cuda", enabled=device.startswith("cuda")):
            outputs = model(images, prev_masks)
        targets = {"mask": masks, "edge": edges}
        total_loss, loss_dict = criterion(outputs, targets)

        # Backward (gradient accumulation with AMP)
        loss_for_backward = total_loss / accumulation_steps
        scaler.scale(loss_for_backward).backward()

        if batch_idx % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += total_loss.item()
        num_batches += 1
        batch_bar.set_postfix(loss=f"{total_loss.item():.4f}")

        # Log loss components and AMP scale periodically
        if batch_idx % 50 == 0:
            ld = {k: f"{v.item():.3f}" for k, v in loss_dict.items()}
            scale = f"{scaler.get_scale():.0f}" if scaler.is_enabled() else "off"
            tqdm.write(f"  [e{epoch+1} b{batch_idx}] {ld}  scale={scale}")

        if use_fam_feedback:
            pred_np = (torch.sigmoid(outputs["pred_mask"]) >= 0.5).detach().cpu().numpy()
            for i, fname in enumerate(filenames):
                mask_hw = (pred_np[i, 0] * 255).astype(np.uint8)
                dataset.update_prev_mask(fname, mask_hw)

        # Drop large references promptly to reduce peak retained memory.
        del images, masks, edges, prev_masks, outputs, targets, total_loss, loss_for_backward

    if num_batches > 0 and num_batches % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / max(num_batches, 1)


def validate(
    model: nn.Module,
    loader,
    aggregator: MetricAggregator,
    device: str,
    full_metrics: bool = True,
    fam_refine_iters: int = 1,
    val_dataset: MedicalSegDataset | None = None,
    threshold: float = 0.5,
) -> float:
    """Run validation / test evaluation.

    Args:
        full_metrics: If True, compute the full metric suite (Hausdorff,
            SSIM, S-measure, etc.).  If False, compute only Dice for speed.
        fam_refine_iters: Number of additional feedback refinement passes
            during validation (1 means total 2 forward passes).
        val_dataset: Validation dataset to update with latest predicted
            masks so subsequent epochs do not restart from Otsu masks.

    Returns:
        Mean Dice score across all samples.
    """
    model.eval()
    aggregator.reset()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            prev_masks = batch["prev_mask"].to(device)
            filenames = batch["filename"]
            masks_np = batch["mask"].cpu().numpy().astype(np.float32)  # (B, 1, H, W)

            # Validation feedback refinement to avoid stale Otsu-only prev masks.
            num_passes = max(1, fam_refine_iters + 1)
            for _ in range(num_passes):
                with autocast("cuda", enabled=device.startswith("cuda")):
                    outputs = model(images, prev_masks)
                prev_masks = (torch.sigmoid(outputs["pred_mask"]) >= threshold).float().detach()

            preds = torch.sigmoid(outputs["pred_mask"]).cpu().numpy().astype(np.float32)  # (B, 1, H, W)

            if val_dataset is not None:
                pred_uint8 = (preds[:, 0] >= threshold).astype(np.uint8) * 255
                for i, fname in enumerate(filenames):
                    val_dataset.update_prev_mask(fname, pred_uint8[i])

            for i in range(preds.shape[0]):
                pred_bin = (preds[i, 0] >= threshold).astype(np.uint8)
                gt_bin = (masks_np[i, 0] >= threshold).astype(np.uint8)

                if full_metrics:
                    metrics = SegmentationMetrics.compute(pred_bin, gt_bin)
                else:
                    metrics = SegmentationMetrics.compute(pred_bin, gt_bin)
                    metrics = {"dice": metrics["dice"]}

                aggregator.update(metrics)

    stats = aggregator.mean_std()
    mean_dice = stats["dice"][0] if stats else 0.0
    return mean_dice


# ======================================================================
# Main
# ======================================================================


def train_single_dataset(
    dataset_name: str,
    config,
    device: str,
    results_table: dict,
    overfit_samples: int = 0,
) -> float:
    """Train and evaluate a single dataset. Returns best Dice score."""
    print(f"\n{'='*70}")
    print(f"  DATASET: {dataset_name:15s}  |  batch_size={config.batch_size}  num_workers={config.num_workers}")
    print(f"  accumulation_steps={config.accumulation_steps}  effective_batch_size={config.batch_size * config.accumulation_steps}")
    print(f"  fam_warmup_epochs={config.fam_warmup_epochs}")
    if overfit_samples > 0:
        print(f"  OVERFIT DEBUG MODE: using {overfit_samples} training samples only")
    print(f"{'='*70}\n")

    # ---- Model -----------------------------------------------------------
    model = build_model(config)

    # ---- AMP scaler ------------------------------------------------------
    # init_scale=256 avoids the default 65536 which causes fp16 gradient
    # overflow → GradScaler silently skips optimizer steps with SGD.
    scaler = GradScaler(
        "cuda", enabled=device.startswith("cuda"), init_scale=256,
    )

    # ---- Optimizer / scheduler -------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    warmup_epochs = int(min(max(getattr(config, "warmup_epochs", 0), 0), max(config.num_epochs - 1, 0)))
    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(getattr(config, "warmup_start_factor", 0.1)),
            total_iters=warmup_epochs,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.num_epochs - warmup_epochs, 1),
            eta_min=config.eta_min,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.num_epochs, 1),
            eta_min=config.eta_min,
        )
    # ---- Loss ------------------------------------------------------------
    criterion = MaSLoss(config)

    # ---- Data ------------------------------------------------------------
    train_loader, test_loader = get_dataloaders(
        dataset_name,
        config,
        overfit_samples=overfit_samples,
    )
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

        # Validate every epoch for smoother best-Dice tracking.
        run_validation = True
        val_dice = None
        if run_validation:
            validation_count = (epoch + 1)
            full_metrics = (validation_count % 6 == 0)
            val_dice = validate(
                model,
                test_loader,
                aggregator,
                device,
                full_metrics=full_metrics,
                fam_refine_iters=1,
                val_dataset=None,   # stateless: don't corrupt test prev_masks
                threshold=float(getattr(config, "metric_threshold", 0.5)),
            )

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

    # Explicitly release large objects before moving to the next dataset.
    del model, optimizer, scheduler, criterion, train_loader, test_loader
    _clear_cuda_memory()

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
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="If >0, train/evaluate on only N training images to debug overfitting.",
    )
    args = parser.parse_args()
    dataset_spec = args.dataset_name

    config = CFG
    if args.data_dir is not None and not os.path.isabs(args.data_dir):
        raise ValueError("--data-dir must be an absolute path, e.g. /Users/name/datasets or C:/datasets")
    config.configure_runtime(force_local=args.local, data_dir=args.data_dir)
    config.fast_mode = args.fast
    config.overfit_samples = max(0, int(args.overfit_samples))
    if config.overfit_samples > 0:
        config.foreground_sampling = False
    device = config.device

    if device == "cpu":
        print("[WARNING] Training is running on CPU. This is slow and may hurt convergence; use a CUDA GPU when possible.")
    elif device.startswith("cuda"):
        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(
            f"[CUDA] total_mem={mem_gb:.2f}GB, batch_size={config.batch_size}, "
            f"accumulation_steps={config.accumulation_steps}"
        )
        if alloc_conf:
            print(f"[CUDA] PYTORCH_CUDA_ALLOC_CONF={alloc_conf}")

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
            train_single_dataset(
                ds_name,
                config,
                device,
                results_table,
                overfit_samples=config.overfit_samples,
            )
        except Exception as e:
            print(f"\n  [ERROR training {ds_name}: {e}]")
            print(f"  [Continuing to next dataset…]\n")
            results_table[ds_name] = None
            _clear_cuda_memory()
            continue

    # ---- Print summary table --------------------------------------------
    print(f"\n{'='*70}")
    print(f"  SUMMARY: All Datasets")
    print(f"{'='*70}\n")
    header = f"{'Dataset':<20s}  {'Best Dice':>10s}"
    sep = "-" * len(header)
    print(f"  {header}")
    print(f"  {sep}")
    for ds_name in datasets_to_train:
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
