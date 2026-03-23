import os
from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    # Image and training
    image_size: int = 224
    batch_size: int = 2
    num_epochs: int = 100
    learning_rate: float = 0.01
    eta_min: float = 1e-5
    momentum: float = 0.99
    weight_decay: float = 1e-4
    dropout: float = 0.1
    fast_mode: bool = False  # Skip heavy augmentations for faster training

    # Swin Transformer
    window_size: int = 7          # Swin Transformer window size
    num_heads: int = 8            # attention heads
    swin_rstm_depth: int = 6      # number of STBs in RSTM
    swin_bstm_depth: int = 12     # number of STBs in BSTM
    swin_sdm_depth: int = 4       # number of STBs in SDM

    # Loss
    lambda_weight: float = 1.0    # BCE weight in primary loss

    # Test-time refinement
    num_refinement_iters: int = 10  # test-time FAM iterations

    # Checkpointing / logging
    checkpoint_interval: int = 5  # save every 5 epochs
    log_dir: str = "./logs"

    # Data loading
    num_workers: int = field(
        default_factory=lambda: 0
        if ('COLAB_GPU' in os.environ or 'COLAB_RELEASE_TAG' in os.environ or os.path.isdir('/content/drive'))
        else 4
    )

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Environment flag – set True when running on Google Colab
    is_colab: bool = 'COLAB_GPU' in os.environ or 'COLAB_RELEASE_TAG' in os.environ
    is_kaggle: bool = ('KAGGLE_KERNEL_RUN_TYPE' in os.environ) or os.path.isdir('/kaggle/input')

    # ------------------------------------------------------------------
    # Derived paths (resolved after init based on is_colab)
    # ------------------------------------------------------------------
    base_data_dir: str = field(init=False)
    checkpoint_dir: str = field(init=False)
    dataset_paths: dict = field(init=False)
    kaggle_source_paths: dict = field(init=False)

    @staticmethod
    def _find_existing_subdir(root: str, candidates: list[str]) -> str | None:
        for rel in candidates:
            path = os.path.join(root, rel)
            if os.path.isdir(path):
                return path
        return None

    @classmethod
    def _find_first_matching_subdir(cls, root: str, keyword_candidates: list[str]) -> str | None:
        lowered = [k.lower() for k in keyword_candidates]
        if not os.path.isdir(root):
            return None

        for current_root, dirs, _ in os.walk(root):
            for directory in dirs:
                d_low = directory.lower()
                if any(k in d_low for k in lowered):
                    return os.path.join(current_root, directory)
        return None

    @staticmethod
    def _resolve_kaggle_root(preferred: str = "transunet") -> str:
        base = "/kaggle/input"
        preferred_path = os.path.join(base, preferred)
        if os.path.isdir(preferred_path):
            return preferred_path

        if not os.path.isdir(base):
            return preferred_path

        preferred_lower = preferred.lower()
        candidates = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]

        for d in candidates:
            if d.lower() == preferred_lower:
                return os.path.join(base, d)

        for d in candidates:
            d_low = d.lower()
            if "transunet" in d_low or "trans-unet" in d_low:
                return os.path.join(base, d)

        return preferred_path

    def __post_init__(self):
        if self.is_kaggle:
            self.is_colab = False
        # Auto-detect Colab by checking if Drive is mounted
        if os.path.isdir('/content/drive'):
            self.is_colab = True

        # Explicit worker policy:
        # - Colab: 0 (Drive + OpenCV workers can hang)
        # - Kaggle: 4
        # - Local: 4
        self.num_workers = 0 if self.is_colab else 4

        if self.is_colab:
            self.base_data_dir = "/content/drive/MyDrive/datasets"
            self.checkpoint_dir = "/content/drive/MyDrive/mas_transunet_checkpoints"
        elif self.is_kaggle:
            explicit_kaggle_data_root = "/kaggle/input/datasets/mohitkhalote/transunet"
            if not os.path.isdir(explicit_kaggle_data_root):
                raise FileNotFoundError(
                    f"Expected Kaggle dataset at {explicit_kaggle_data_root}. "
                    "Attach that dataset path before running this notebook."
                )
            self.base_data_dir = explicit_kaggle_data_root
            self.checkpoint_dir = "/kaggle/working/checkpoints"
        else:
            self.base_data_dir = "./data"
            self.checkpoint_dir = "./checkpoints"

        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        except OSError:
            pass

        dataset_names = [
            "mri_glioma",
            "kvasir_seg",
            "isic2018",
            "covid_ct",
        ]

        self.dataset_paths = {}
        self.kaggle_source_paths = {}

        if self.is_kaggle:
            kaggle_root = self.base_data_dir
            edges_root = "/kaggle/working/edges"

            covid_images = os.path.join(kaggle_root, "archive (4)", "ct_scans")
            covid_masks = os.path.join(kaggle_root, "archive (4)", "lung_and_infection_mask")
            self.kaggle_source_paths["covid_ct"] = {
                "images": covid_images,
                "masks": covid_masks,
            }

            kvasir_root = os.path.join(kaggle_root, "archive (5)", "kvasir-seg")
            kvasir_images = self._find_existing_subdir(kvasir_root, ["images", "Images"]) or kvasir_root
            kvasir_masks = self._find_existing_subdir(kvasir_root, ["masks", "Masks"]) or kvasir_root

            isic_root = os.path.join(kaggle_root, "archive (6)")
            isic_train_images = os.path.join(isic_root, "ISIC2018_Task1-2_Training_Input")
            isic_train_masks = os.path.join(isic_root, "ISIC2018_Task1_Training_GroundTruth")
            isic_test_images = os.path.join(isic_root, "ISIC2018_Task1-2_Test_Input")
            isic_test_masks = os.path.join(isic_root, "ISIC2018_Task1-2_Validation_Input")

            mri_archive_root = os.path.join(kaggle_root, "archive (7)")
            mri_root = self._find_existing_subdir(
                mri_archive_root,
                [
                    "Brain_MRI",
                    "Brain MRI",
                    "brain_mri",
                    "brain-mri",
                    "BrainMRI",
                ],
            ) or self._find_first_matching_subdir(
                mri_archive_root,
                ["brain", "mri", "glioma"],
            ) or mri_archive_root
            mri_images = self._find_existing_subdir(mri_root, ["images", "Images"]) \
                or self._find_first_matching_subdir(mri_root, ["image", "t1", "flair"]) \
                or mri_root
            mri_masks = self._find_existing_subdir(mri_root, ["masks", "Masks", "mask", "Mask"]) \
                or self._find_first_matching_subdir(mri_root, ["mask", "seg", "label"]) \
                or mri_root
            self.kaggle_source_paths["mri_glioma"] = {
                "images": mri_images,
                "masks": mri_masks,
            }

            self.dataset_paths["mri_glioma"] = {
                "train_images": "/kaggle/working/mri_glioma/train/images",
                "train_masks":  "/kaggle/working/mri_glioma/train/masks",
                "train_edges":  os.path.join(edges_root, "mri_glioma", "train"),
                "test_images":  "/kaggle/working/mri_glioma/test/images",
                "test_masks":   "/kaggle/working/mri_glioma/test/masks",
                "test_edges":   os.path.join(edges_root, "mri_glioma", "test"),
            }
            self.dataset_paths["kvasir_seg"] = {
                "train_images": kvasir_images,
                "train_masks":  kvasir_masks,
                "train_edges":  os.path.join(edges_root, "kvasir_seg", "train"),
                "test_images":  kvasir_images,
                "test_masks":   kvasir_masks,
                "test_edges":   os.path.join(edges_root, "kvasir_seg", "test"),
            }
            self.dataset_paths["isic2018"] = {
                "train_images": isic_train_images,
                "train_masks":  isic_train_masks,
                "train_edges":  os.path.join(edges_root, "isic2018", "train"),
                "test_images":  isic_test_images,
                "test_masks":   isic_test_masks,
                "test_edges":   os.path.join(edges_root, "isic2018", "test"),
            }
            self.dataset_paths["covid_ct"] = {
                "train_images": "/kaggle/working/covid_ct/train/images",
                "train_masks":  "/kaggle/working/covid_ct/train/masks",
                "train_edges":  os.path.join(edges_root, "covid_ct", "train"),
                "test_images":  "/kaggle/working/covid_ct/test/images",
                "test_masks":   "/kaggle/working/covid_ct/test/masks",
                "test_edges":   os.path.join(edges_root, "covid_ct", "test"),
            }
        else:
            for name in dataset_names:
                ds_root = os.path.join(self.base_data_dir, name)
                self.dataset_paths[name] = {
                    "train_images": os.path.join(ds_root, "train", "images"),
                    "train_masks":  os.path.join(ds_root, "train", "masks"),
                    "train_edges":  os.path.join(ds_root, "train", "edges"),
                    "test_images":  os.path.join(ds_root, "test", "images"),
                    "test_masks":   os.path.join(ds_root, "test", "masks"),
                    "test_edges":   os.path.join(ds_root, "test", "edges"),
                }


CFG = Config()
