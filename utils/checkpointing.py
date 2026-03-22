import os
import glob
import json
import re
import shutil

import torch
import numpy as np


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_dice: float,
    dataset_name: str,
    checkpoint_dir: str,
    is_best: bool = False,
) -> str:
    """Save a training checkpoint.

    File is named ``{dataset_name}_{epoch:03d}.pth``.  When *is_best* is
    True an additional copy ``{dataset_name}_best.pth`` is written.

    Returns:
        Path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Delete old periodic checkpoints to save disk space
    import glob as _glob
    for old in _glob.glob(os.path.join(checkpoint_dir, f"{dataset_name}_[0-9]*.pth")):
        os.remove(old)

    ckpt = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch":                epoch,
        "best_dice":            best_dice,
        "dataset_name":         dataset_name,
    }

    filename = f"{dataset_name}_{epoch:03d}.pth"
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(ckpt, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{dataset_name}_best.pth")
        shutil.copy2(filepath, best_path)

    return filepath


def load_latest_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    dataset_name: str,
    checkpoint_dir: str,
) -> tuple[int, float]:
    """Load the most recent checkpoint for *dataset_name*.

    Returns:
        (start_epoch, best_dice) — ``(0, 0.0)`` when no checkpoint is found.
    """
    pattern = os.path.join(checkpoint_dir, f"{dataset_name}_[0-9]*.pth")
    candidates = glob.glob(pattern)

    if not candidates:
        return 0, 0.0

    # Sort by epoch number embedded in the filename
    epoch_re = re.compile(rf"{re.escape(dataset_name)}_(\d+)\.pth$")

    def _epoch_key(path: str) -> int:
        m = epoch_re.search(os.path.basename(path))
        return int(m.group(1)) if m else -1

    candidates.sort(key=_epoch_key)
    latest = candidates[-1]

    ckpt = torch.load(latest, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt["epoch"] + 1
    best_dice = ckpt.get("best_dice", 0.0)

    return start_epoch, best_dice


def save_epoch_masks(
    masks: dict[str, np.ndarray],
    dataset_name: str,
    checkpoint_dir: str,
) -> str:
    """Save previous-epoch predicted masks as a compressed ``.npz`` file.

    Returns:
        Path to the saved file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, f"{dataset_name}_prev_masks.npz")
    np.savez_compressed(filepath, **masks)
    return filepath


def load_epoch_masks(
    dataset_name: str,
    checkpoint_dir: str,
) -> dict[str, np.ndarray]:
    """Load previous-epoch predicted masks.

    Returns:
        dict mapping filenames to numpy mask arrays, or empty dict if the
        file does not exist.
    """
    filepath = os.path.join(checkpoint_dir, f"{dataset_name}_prev_masks.npz")
    if not os.path.isfile(filepath):
        return {}

    data = np.load(filepath)
    return {k: data[k] for k in data.files}
