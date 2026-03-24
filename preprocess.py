import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
import os
import errno
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import CFG

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _is_supported_image(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTS)

ALL_DATASETS = ["mri_glioma", "kvasir_seg", "isic2018", "covid_ct"]


def _is_kaggle_input_path(path: Path) -> bool:
    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)
    return resolved == "/kaggle/input" or resolved.startswith("/kaggle/input/")


def generate_edge_maps(masks_dir: str, edges_dir: str) -> int:
    """Apply Canny edge detection to every mask and save to edges_dir.

    Returns the number of edge maps generated.
    """
    masks_path = Path(masks_dir)
    edges_path = Path(edges_dir)
    edges_path.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(
        f for f in masks_path.iterdir()
        if f.is_file() and _is_supported_image(f.name)
    )

    count = 0
    for mf in tqdm(mask_files, desc="  Edge maps", leave=False):
        mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        edge = cv2.Canny(mask, 50, 150)
        out_name = mf.stem + ".png"
        cv2.imwrite(str(edges_path / out_name), edge)
        count += 1

    return count


def filter_low_content_pairs(
    images_dir: str,
    masks_dir: str,
    threshold: float = 0.01,
) -> tuple[list[str], list[str]]:
    """Return (kept, filtered) filename lists.

    A pair is filtered when the foreground fraction in its mask is below
    *threshold*.
    """
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)

    image_files = sorted(
        f.name for f in images_path.iterdir()
        if f.is_file() and _is_supported_image(f.name)
    )

    mask_lookup = {
        f.stem: f.name for f in masks_path.iterdir()
        if f.is_file() and _is_supported_image(f.name)
    }

    kept: list[str] = []
    filtered: list[str] = []

    for img_name in image_files:
        stem = Path(img_name).stem
        mask_name = mask_lookup.get(stem)
        if mask_name is None:
            continue

        mask = cv2.imread(str(masks_path / mask_name), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        fg_fraction = np.count_nonzero(mask) / mask.size
        if fg_fraction < threshold:
            filtered.append(img_name)
        else:
            kept.append(img_name)

    return kept, filtered


def _apply_filtered_pairs(
    images_dir: str,
    masks_dir: str,
    filtered_images: list[str],
) -> int:
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)

    if _is_kaggle_input_path(images_path) or _is_kaggle_input_path(masks_path):
        print(
            "  [preprocess] skipping physical removal of low-content pairs "
            f"(kaggle input is read-only): images={images_dir}, masks={masks_dir}"
        )
        return 0

    if not os.access(images_path, os.W_OK) or not os.access(masks_path, os.W_OK):
        print(
            "  [preprocess] skipping physical removal of low-content pairs "
            f"(read-only path): images={images_dir}, masks={masks_dir}"
        )
        return 0

    removed = 0

    for img_name in filtered_images:
        pair_removed = False
        img_path = images_path / img_name
        if img_path.exists():
            try:
                img_path.unlink()
                pair_removed = True
            except OSError as exc:
                if exc.errno not in {errno.EROFS, errno.EPERM, errno.EACCES}:
                    raise

        stem = Path(img_name).stem
        for ext in IMAGE_EXTS:
            mask_path = masks_path / f"{stem}{ext}"
            if mask_path.exists():
                try:
                    mask_path.unlink()
                    pair_removed = True
                except OSError as exc:
                    if exc.errno not in {errno.EROFS, errno.EPERM, errno.EACCES}:
                        raise
                break

        if pair_removed:
            removed += 1

    return removed


def preprocess_dataset(dataset_name: str) -> dict:
    """Run full preprocessing for a single dataset.

    Returns a summary dict with counts.
    """
    from dataset import _prepare_kaggle_working_split
    try:
        _prepare_kaggle_working_split(dataset_name, CFG)
    except Exception as exc:
        print(f"  [{dataset_name}] warning: failed to prepare kaggle split: {exc}")

    paths = CFG.dataset_paths[dataset_name]
    summary: dict = {"name": dataset_name}

    # --- Filter low-content pairs (train split) --------------------------
    if os.path.isdir(paths["train_images"]) and os.path.isdir(paths["train_masks"]):
        kept, filtered = filter_low_content_pairs(
            paths["train_images"], paths["train_masks"]
        )
        summary["train_total"] = len(kept) + len(filtered)
        summary["train_kept"] = len(kept)
        summary["train_filtered"] = len(filtered)
        if filtered:
            removed = _apply_filtered_pairs(paths["train_images"], paths["train_masks"], filtered)
            summary["train_removed"] = removed
        if filtered:
            print(f"  [{dataset_name}] train: {len(filtered)}/{summary['train_total']} "
                  f"pairs below foreground threshold")
            print(f"  [{dataset_name}] train: removed {summary['train_removed']} low-content pairs")
    else:
        summary["train_total"] = 0
        summary["train_kept"] = 0
        summary["train_filtered"] = 0
        summary["train_removed"] = 0
        print(f"  [{dataset_name}] train directories not found – skipping filter")

    # --- Filter low-content pairs (test split) ----------------------------
    if os.path.isdir(paths["test_images"]) and os.path.isdir(paths["test_masks"]):
        kept_test, filtered_test = filter_low_content_pairs(
            paths["test_images"], paths["test_masks"]
        )
        summary["test_total"] = len(kept_test) + len(filtered_test)
        summary["test_kept"] = len(kept_test)
        summary["test_filtered"] = len(filtered_test)
        if filtered_test:
            removed = _apply_filtered_pairs(paths["test_images"], paths["test_masks"], filtered_test)
            summary["test_removed"] = removed
        if filtered_test:
            print(f"  [{dataset_name}] test:  {len(filtered_test)}/{summary['test_total']} "
                  f"pairs below foreground threshold")
            print(f"  [{dataset_name}] test:  removed {summary['test_removed']} low-content pairs")
    else:
        summary["test_total"] = 0
        summary["test_kept"] = 0
        summary["test_filtered"] = 0
        summary["test_removed"] = 0
        print(f"  [{dataset_name}] test directories not found – skipping filter")

    # --- Generate edge maps (train) ---------------------------------------
    if os.path.isdir(paths["train_masks"]):
        n = generate_edge_maps(paths["train_masks"], paths["train_edges"])
        summary["train_edges_generated"] = n
        print(f"  [{dataset_name}] generated {n} train edge maps")
    else:
        summary["train_edges_generated"] = 0

    # --- Generate edge maps (test) ----------------------------------------
    if os.path.isdir(paths["test_masks"]):
        n = generate_edge_maps(paths["test_masks"], paths["test_edges"])
        summary["test_edges_generated"] = n
        print(f"  [{dataset_name}] generated {n} test edge maps")
    else:
        summary["test_edges_generated"] = 0

    return summary


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess medical image segmentation datasets for MaS-TransUNet."
    )
    parser.add_argument(
        "dataset",
        choices=ALL_DATASETS + ["all"],
        help="Dataset to preprocess, or 'all' for every dataset.",
    )
    args = parser.parse_args()

    targets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    summaries: list[dict] = []
    for ds in tqdm(targets, desc="Datasets"):
        print(f"\nProcessing: {ds}")
        summaries.append(preprocess_dataset(ds))

    # --- Summary ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    for s in summaries:
        print(
            f"  {s['name']:12s} | "
            f"train filtered: {s['train_filtered']:>4d}/{s['train_total']:<5d} | "
            f"test filtered: {s['test_filtered']:>4d}/{s['test_total']:<5d} | "
            f"edge maps: {s['train_edges_generated'] + s['test_edges_generated']}"
        )
    print("=" * 60)
