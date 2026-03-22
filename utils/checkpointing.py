import os, glob, re, shutil
import torch
import numpy as np


def save_checkpoint(model, optimizer, scheduler, epoch, best_dice,
                    dataset_name, checkpoint_dir, is_best=False):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Delete ALL old periodic checkpoints before saving new one
    for old in glob.glob(os.path.join(checkpoint_dir, f"{dataset_name}_[0-9]*.pth")):
        os.remove(old)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice,
        "dataset_name": dataset_name,
    }

    filepath = os.path.join(checkpoint_dir, f"{dataset_name}_{epoch:03d}.pth")
    torch.save(ckpt, filepath)

    if is_best:
        shutil.copy2(filepath, os.path.join(checkpoint_dir, f"{dataset_name}_best.pth"))

    return filepath


def load_latest_checkpoint(model, optimizer, scheduler, dataset_name, checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, f"{dataset_name}_[0-9]*.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        return 0, 0.0

    epoch_re = re.compile(re.escape(dataset_name) + r"_(\d+)\.pth$")
    candidates.sort(key=lambda p: int(m.group(1)) if (m := epoch_re.search(os.path.basename(p))) else -1)
    latest = candidates[-1]

    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    # optimizer and scheduler not saved to reduce file size
    return ckpt["epoch"] + 1, ckpt.get("best_dice", 0.0)


def save_epoch_masks(masks, dataset_name, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, f"{dataset_name}_prev_masks.npz")
    np.savez_compressed(filepath, **masks)
    return filepath


def load_epoch_masks(dataset_name, checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, f"{dataset_name}_prev_masks.npz")
    if not os.path.isfile(filepath):
        return {}
    data = np.load(filepath)
    return {k: data[k] for k in data.files}
