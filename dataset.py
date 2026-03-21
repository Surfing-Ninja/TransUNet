import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.filters import threshold_otsu
import albumentations as A

from config import Config, CFG

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _is_supported_image(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTS)


def _find_file_by_stem(directory: str, stem: str) -> str | None:
    d = Path(directory)
    if not d.is_dir():
        return None
    for ext in IMAGE_EXTS:
        candidate = d / (stem + ext)
        if candidate.exists():
            return str(candidate)
    return None


def _prepare_kaggle_working_split(dataset_name: str, config: Config) -> None:
    if not config.is_kaggle:
        return
    if dataset_name not in {"covid_ct", "mri_glioma"}:
        return

    source = config.kaggle_source_paths.get(dataset_name, {})
    src_images = source.get("images")
    src_masks = source.get("masks")
    if not src_images or not src_masks:
        raise ValueError(f"Missing Kaggle source paths for {dataset_name}")

    target_root = Path("/kaggle/working") / dataset_name
    train_images_dir = target_root / "train" / "images"
    train_masks_dir = target_root / "train" / "masks"
    test_images_dir = target_root / "test" / "images"
    test_masks_dir = target_root / "test" / "masks"

    for directory in [train_images_dir, train_masks_dir, test_images_dir, test_masks_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train_existing = [f for f in train_images_dir.iterdir() if f.is_file() and _is_supported_image(f.name)]
    test_existing = [f for f in test_images_dir.iterdir() if f.is_file() and _is_supported_image(f.name)]
    if train_existing and test_existing:
        return

    src_img_path = Path(src_images)
    img_files = sorted(
        f for f in src_img_path.iterdir()
        if f.is_file() and _is_supported_image(f.name)
    )

    def _normalize_stem(stem: str) -> str:
        s = stem.lower()
        if dataset_name == "covid_ct":
            s = s.replace("_org", "")
        s = s.replace("_mask", "")
        s = s.replace("_seg", "")
        s = s.replace("segmentation", "")
        return s

    mask_by_stem: dict[str, Path] = {}
    src_mask_path = Path(src_masks)
    for mask_file in src_mask_path.iterdir():
        if not mask_file.is_file() or not _is_supported_image(mask_file.name):
            continue
        mask_by_stem[mask_file.stem.lower()] = mask_file
        mask_by_stem[_normalize_stem(mask_file.stem)] = mask_file

    pairs: list[tuple[Path, Path]] = []
    for image_file in img_files:
        mask_file = _find_file_by_stem(src_masks, image_file.stem)
        if mask_file is None:
            normalized = _normalize_stem(image_file.stem)
            mask_match = mask_by_stem.get(normalized) or mask_by_stem.get(image_file.stem.lower())
            mask_file = str(mask_match) if mask_match is not None else None
        if mask_file is not None:
            pairs.append((image_file, Path(mask_file)))

    if not pairs:
        raise ValueError(
            f"No image/mask pairs found for {dataset_name} in {src_images} and {src_masks}"
        )

    rng = random.Random(42)
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_test = max(1, int(0.1 * n_total)) if n_total > 1 else 0
    n_train = n_total - n_test

    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    def _copy_pairs(target_img_dir: Path, target_msk_dir: Path, selected_pairs: list[tuple[Path, Path]]) -> None:
        for image_src, mask_src in selected_pairs:
            shutil.copy2(image_src, target_img_dir / image_src.name)
            shutil.copy2(mask_src, target_msk_dir / mask_src.name)

    _copy_pairs(train_images_dir, train_masks_dir, train_pairs)
    _copy_pairs(test_images_dir, test_masks_dir, test_pairs)


class MedicalSegDataset(Dataset):
    """Dataset for MaS-TransUNet that serves images, masks, edge maps, and
    the previous-epoch predicted mask required by the FAM module."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        config: Config,
        transform: A.Compose | None = None,
    ):
        assert split in ("train", "test")
        self.dataset_name = dataset_name
        self.split = split
        self.config = config
        self.transform = transform

        paths = config.dataset_paths[dataset_name]
        self.images_dir = paths[f"{split}_images"]
        self.masks_dir = paths[f"{split}_masks"]
        self.edges_dir = paths[f"{split}_edges"]

        if config.is_kaggle:
            self.edges_dir = f"/kaggle/working/edges/{dataset_name}/{split}"
            os.makedirs(self.edges_dir, exist_ok=True)

        self.filenames = sorted(
            f.name
            for f in Path(self.images_dir).iterdir()
            if f.is_file() and _is_supported_image(f.name)
        )

        # Stores the previous epoch's predicted mask (numpy uint8 H×W)
        # keyed by filename.  Populated via update_prev_mask().
        self.prev_masks: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.filenames)

    def _otsu_initial_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Generate a coarse initial mask via Otsu thresholding."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        thresh = threshold_otsu(gray)
        return (gray > thresh).astype(np.uint8) * 255

    def __getitem__(self, index: int) -> dict:
        fname = self.filenames[index]
        stem = Path(fname).stem

        # --- Load image (BGR → RGB) --------------------------------------
        img_path = os.path.join(self.images_dir, fname)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image file: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Load ground-truth mask (grayscale, binarise) -----------------
        mask_path = self._find_file(self.masks_dir, stem)
        if mask_path is None:
            raise ValueError(
                f"Mask not found for '{fname}' in {self.masks_dir}. "
                f"Supported extensions: {', '.join(IMAGE_EXTS)}"
            )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask file: {mask_path}")
        mask = (mask > 127).astype(np.uint8) * 255

        # --- Load edge map (grayscale) ------------------------------------
        edge_path = self._find_file(self.edges_dir, stem)
        if edge_path is not None:
            edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            if edge is None:
                edge = cv2.Canny(mask, 50, 150)
        else:
            edge = cv2.Canny(mask, 50, 150)

        # --- Previous-epoch predicted mask (for FAM) ----------------------
        if fname in self.prev_masks:
            prev_mask = self.prev_masks[fname]
        else:
            prev_mask = self._otsu_initial_mask(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )

        # Ensure all targets match image HxW before Albumentations shape checks.
        image_h, image_w = image.shape[:2]
        if mask.shape[:2] != (image_h, image_w):
            mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        if edge.shape[:2] != (image_h, image_w):
            edge = cv2.resize(edge, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        if prev_mask.shape[:2] != (image_h, image_w):
            prev_mask = cv2.resize(
                prev_mask,
                (image_w, image_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # --- Augmentation / transforms ------------------------------------
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask,
                edge=edge,
                prev_mask=prev_mask,
            )
            image = transformed["image"]       # already HWC float32 after Normalize
            mask = transformed["mask"]
            edge = transformed["edge"]
            prev_mask = transformed["prev_mask"]
        else:
            image = image.astype(np.float32) / 255.0

        # --- Convert to tensors -------------------------------------------
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        else:
            # albumentations + ToTensorV2 not used; Normalize returns ndarray
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()

        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0)
        edge = torch.from_numpy(edge.astype(np.float32) / 255.0).unsqueeze(0)
        prev_mask = torch.from_numpy(
            prev_mask.astype(np.float32) / 255.0
        ).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "edge": edge,
            "prev_mask": prev_mask,
            "filename": fname,
        }

    def update_prev_mask(self, filename: str, prediction: np.ndarray) -> None:
        """Store the model's latest prediction for use in the next epoch."""
        self.prev_masks[filename] = prediction

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _find_file(directory: str, stem: str) -> str | None:
        """Find a file by stem regardless of extension."""
        d = Path(directory)
        if not d.is_dir():
            return None
        for ext in IMAGE_EXTS:
            candidate = d / (stem + ext)
            if candidate.exists():
                return str(candidate)
        return None


# ======================================================================
# Augmentation pipelines
# ======================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(fast_mode: bool = False) -> A.Compose:
    transforms = [
        A.PadIfNeeded(min_height=256, min_width=256),
        A.RandomCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),
    ]

    if not fast_mode:
        transforms.extend(
            [
                A.ElasticTransform(p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.2),
            ]
        )

    transforms.extend(
        [
            A.RandomBrightnessContrast(p=0.4),
            A.RandomGamma(p=0.3),
            A.CLAHE(p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return A.Compose(
        transforms,
        additional_targets={"edge": "mask", "prev_mask": "mask"},
    )


def get_val_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
        additional_targets={"edge": "mask", "prev_mask": "mask"},
    )


# ======================================================================
# Dataloader factory
# ======================================================================


def get_dataloaders(
    dataset_name: str,
    config: Config = CFG,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for the given dataset."""

    _prepare_kaggle_working_split(dataset_name, config)

    train_ds = MedicalSegDataset(
        dataset_name=dataset_name,
        split="train",
        config=config,
        transform=get_train_transforms(config.fast_mode),
    )
    test_ds = MedicalSegDataset(
        dataset_name=dataset_name,
        split="test",
        config=config,
        transform=get_val_transforms(),
    )

    if len(train_ds) == 0:
        raise ValueError(
            f"No training samples found for '{dataset_name}' in {train_ds.images_dir}. "
            f"Supported image extensions: {', '.join(IMAGE_EXTS)}"
        )
    if len(test_ds) == 0:
        raise ValueError(
            f"No test samples found for '{dataset_name}' in {test_ds.images_dir}. "
            f"Supported image extensions: {', '.join(IMAGE_EXTS)}"
        )

    # Use configured num_workers (auto-set based on Colab detection)
    pin_memory = config.device.startswith("cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
