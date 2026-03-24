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


def _to_binary_uint8(mask_like: np.ndarray) -> np.ndarray:
    arr = mask_like.astype(np.float32)
    threshold = 0.5 if arr.max() <= 1.0 else 127.0
    return (arr >= threshold).astype(np.uint8) * 255


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
    src_mask_path = Path(src_masks)

    if not src_img_path.is_dir() or not src_mask_path.is_dir():
        print(
            f"Skipping {dataset_name}: source paths not found. "
            f"images={src_images}, masks={src_masks}"
        )
        return

    img_files = sorted(
        f for f in src_img_path.rglob("*")
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
    for mask_file in src_mask_path.rglob("*"):
        if not mask_file.is_file() or not _is_supported_image(mask_file.name):
            continue
        mask_by_stem[mask_file.stem.lower()] = mask_file
        mask_by_stem[_normalize_stem(mask_file.stem)] = mask_file

    pairs: list[tuple[Path, Path]] = []
    for image_file in img_files:
        if dataset_name == "mri_glioma" and "_mask" in image_file.stem.lower():
            continue

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
        for pair_index, (image_src, mask_src) in enumerate(selected_pairs):
            target_stem = f"{pair_index:05d}"
            shutil.copy2(image_src, target_img_dir / f"{target_stem}{image_src.suffix}")
            shutil.copy2(mask_src, target_msk_dir / f"{target_stem}{mask_src.suffix}")

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

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")
        if not os.path.isdir(self.masks_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.masks_dir}")

        if config.is_kaggle:
            self.edges_dir = f"/kaggle/working/edges/{dataset_name}/{split}"
            os.makedirs(self.edges_dir, exist_ok=True)

        self.filenames = sorted(
            f.name
            for f in Path(self.images_dir).iterdir()
            if f.is_file() and _is_supported_image(f.name)
        )

        train_images_path = os.path.normpath(os.path.abspath(paths["train_images"]))
        test_images_path = os.path.normpath(os.path.abspath(paths["test_images"]))
        if train_images_path == test_images_path:
            shuffled_filenames = self.filenames[:]
            rng = random.Random(42)
            rng.shuffle(shuffled_filenames)

            split_idx = int(0.9 * len(shuffled_filenames))
            if self.split == "train":
                self.filenames = shuffled_filenames[:split_idx]
            else:
                self.filenames = shuffled_filenames[split_idx:]

        paired_filenames: list[str] = []
        missing_mask_filenames: list[str] = []
        for filename in self.filenames:
            if self._find_file(self.masks_dir, Path(filename).stem) is None:
                missing_mask_filenames.append(filename)
            else:
                paired_filenames.append(filename)
        self.filenames = paired_filenames

        if missing_mask_filenames:
            print(
                f"[{self.dataset_name}:{self.split}] skipped {len(missing_mask_filenames)} images without matching masks"
            )

        min_fg_ratio = float(getattr(self.config, "min_foreground_ratio", 0.01))
        apply_filter = bool(getattr(self.config, "apply_low_content_filter", True))
        if apply_filter and self.split == "train":
            filtered_filenames: list[str] = []
            for filename in self.filenames:
                mask_path = self._find_file(self.masks_dir, Path(filename).stem)
                if mask_path is None:
                    continue
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    continue
                fg_ratio = float(np.count_nonzero(mask_img > 127)) / float(mask_img.size)
                if fg_ratio >= min_fg_ratio:
                    filtered_filenames.append(filename)

            removed_count = len(self.filenames) - len(filtered_filenames)
            self.filenames = filtered_filenames
            if removed_count > 0:
                print(
                    f"[{self.dataset_name}:{self.split}] removed {removed_count} low-content masks "
                    f"(fg_ratio < {min_fg_ratio})"
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
            mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
            std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
            image = (image - mean) / std

        mask = _to_binary_uint8(mask)
        edge = _to_binary_uint8(edge)
        prev_mask = _to_binary_uint8(prev_mask)

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

        mask = (mask >= 0.5).float()
        edge = (edge >= 0.5).float()
        prev_mask = (prev_mask >= 0.5).float()

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
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=30,
            p=0.5,
            border_mode=cv2.BORDER_REFLECT_101,
        ),
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
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
