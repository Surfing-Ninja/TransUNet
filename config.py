import os
import sys
from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    # Image and training
    image_size: int = 224
    batch_size: int = 2
    num_epochs: int = 100
    # Stable default for AdamW. If switching back to SGD, use 1e-2.
    learning_rate: float = 3e-4
    accumulation_steps: int = 4
    fam_warmup_epochs: int = 1
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    eta_min: float = 1e-5
    momentum: float = 0.99
    weight_decay: float = 1e-4
    dropout: float = 0.2
    fast_mode: bool = False  # Skip heavy augmentations for faster training
    local_data_dir: str | None = None
    apply_low_content_filter: bool = True
    min_foreground_ratio: float = 0.02
    foreground_sampling: bool = True
    foreground_sampling_threshold: float = 0.01
    foreground_sampling_boost: float = 4.0
    metric_threshold: float = 0.5

    # Overfit-debug mode
    overfit_samples: int = 0

    # Memory safety
    use_gradient_checkpointing: bool = True
    use_amp: bool = True

    # Swin Transformer
    window_size: int = 7          # Swin Transformer window size
    num_heads: int = 8            # attention heads
    swin_rstm_depth: int = 6      # number of STBs in RSTM
    swin_bstm_depth: int = 12     # number of STBs in BSTM
    swin_sdm_depth: int = 4       # number of STBs in SDM

    # Loss
    lambda_weight: float = 1.0    # BCE weight in primary loss
    boundary_loss_weight: float = 0.15
    ds_loss_weight: float = 0.2

    # Test-time refinement
    num_refinement_iters: int = 10  # test-time FAM iterations
    eval_use_tta: bool = True
    eval_keep_lcc: bool = True

    # Checkpointing / logging
    checkpoint_interval: int = 5  # save every 5 epochs
    log_dir: str = "./logs"

    # Data loading
    num_workers: int = 4

    # Device
    device: str = "cpu"

    # Environment flag – set True when running on Google Colab
    is_colab: bool = False
    is_kaggle: bool = False

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

    @staticmethod
    def _detect_kaggle() -> bool:
        return ('KAGGLE_KERNEL_RUN_TYPE' in os.environ) or os.path.isdir('/kaggle/input')

    @staticmethod
    def _detect_colab() -> bool:
        return (
            'COLAB_GPU' in os.environ
            or 'COLAB_RELEASE_TAG' in os.environ
            or os.path.isdir('/content/drive')
        )

    @staticmethod
    def _resolve_local_base_dir(local_data_dir: str | None) -> str:
        if not local_data_dir:
            return "./data"
        return local_data_dir if os.path.isabs(local_data_dir) else os.path.abspath(local_data_dir)

    @staticmethod
    def _auto_batch_size() -> int:
        if not torch.cuda.is_available():
            return 2
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_memory_gb >= 24:
            return 4
        if total_memory_gb >= 16:
            return 2
        return 1

    def _auto_num_workers(self) -> int:
        if self.is_kaggle:
            return 0
        if sys.platform.startswith("win"):
            return 2
        return 4

    @staticmethod
    def _auto_accumulation_steps(default_steps: int, is_kaggle: bool) -> int:
        if not torch.cuda.is_available():
            return default_steps

        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        # Kaggle T4 class GPUs (~15GB) need higher accumulation since
        # batch_size is forced to 1 for memory safety.
        if is_kaggle and 10 <= total_memory_gb < 16:
            return max(default_steps, 16)

        # L4/V100 class GPUs can usually handle batch_size=2 safely.
        if is_kaggle and 16 <= total_memory_gb < 24:
            return max(default_steps, 8)

        return default_steps

    def _build_dataset_paths(self) -> None:
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
            return

        edges_root = os.path.join(self.base_data_dir, "edges")
        for name in dataset_names:
            ds_root = os.path.join(self.base_data_dir, name)
            self.dataset_paths[name] = {
                "train_images": os.path.join(ds_root, "train", "images"),
                "train_masks":  os.path.join(ds_root, "train", "masks"),
                "train_edges":  os.path.join(edges_root, name, "train"),
                "test_images":  os.path.join(ds_root, "test", "images"),
                "test_masks":   os.path.join(ds_root, "test", "masks"),
                "test_edges":   os.path.join(edges_root, name, "test"),
            }

        # Local fallback: kvasir_seg may be arranged as
        # data/kvasir_seg/kvasir-seg/{images,masks} without train/test folders.
        kvasir_root = os.path.join(self.base_data_dir, "kvasir_seg")
        kvasir_paths = self.dataset_paths["kvasir_seg"]
        if not os.path.isdir(kvasir_paths["train_images"]):
            seg_root = self._find_existing_subdir(
                kvasir_root,
                ["kvasir-seg", "kvasir_seg", "kvasir seg"],
            )
            seg_images = self._find_existing_subdir(seg_root, ["images", "Images"]) if seg_root else None
            seg_masks = self._find_existing_subdir(seg_root, ["masks", "Masks"]) if seg_root else None

            if seg_images and seg_masks:
                self.dataset_paths["kvasir_seg"] = {
                    "train_images": seg_images,
                    "train_masks":  seg_masks,
                    "train_edges":  os.path.join(edges_root, "kvasir_seg", "train"),
                    "test_images":  seg_images,
                    "test_masks":   seg_masks,
                    "test_edges":   os.path.join(edges_root, "kvasir_seg", "test"),
                }

        # Local fallback: mri_glioma may be arranged under a Brain_MRI folder
        # (possibly nested) without train/test folders.
        mri_root = os.path.join(self.base_data_dir, "mri_glioma")
        mri_paths = self.dataset_paths["mri_glioma"]
        if not os.path.isdir(mri_paths["train_images"]):
            brain_mri_root = self._find_existing_subdir(
                mri_root,
                ["Brain_MRI", "Brain MRI", "brain_mri", "brain-mri", "BrainMRI"],
            ) or self._find_first_matching_subdir(mri_root, ["brain", "mri", "glioma"]) or mri_root

            mri_images = self._find_existing_subdir(brain_mri_root, ["images", "Images"]) \
                or self._find_first_matching_subdir(brain_mri_root, ["image", "t1", "flair"]) \
                or brain_mri_root
            mri_masks = self._find_existing_subdir(brain_mri_root, ["masks", "Masks", "mask", "Mask"]) \
                or self._find_first_matching_subdir(brain_mri_root, ["mask", "seg", "label"]) \
                or brain_mri_root

            if os.path.isdir(mri_images) and os.path.isdir(mri_masks):
                shared_edges = os.path.join(edges_root, "mri_glioma")
                self.dataset_paths["mri_glioma"] = {
                    "train_images": mri_images,
                    "train_masks":  mri_masks,
                    "train_edges":  shared_edges,
                    "test_images":  mri_images,
                    "test_masks":   mri_masks,
                    "test_edges":   shared_edges,
                }

    def configure_runtime(self, force_local: bool = False, data_dir: str | None = None) -> None:
        if data_dir is not None:
            self.local_data_dir = data_dir

        kaggle_detected = self._detect_kaggle()
        self.is_kaggle = kaggle_detected and not force_local
        self.is_colab = self._detect_colab() and not self.is_kaggle

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = self._auto_batch_size()
        self.num_workers = self._auto_num_workers()
        self.accumulation_steps = self._auto_accumulation_steps(self.accumulation_steps, self.is_kaggle)

        if self.is_kaggle:
            explicit_kaggle_data_root = "/kaggle/input/datasets/mohitkhalote/transunet"
            self.base_data_dir = explicit_kaggle_data_root
            self.checkpoint_dir = "/kaggle/working/checkpoints"
            self.log_dir = "/kaggle/working/logs"
        else:
            self.base_data_dir = self._resolve_local_base_dir(self.local_data_dir)
            self.checkpoint_dir = "./checkpoints"
            self.log_dir = "./logs"

        for output_dir in [self.checkpoint_dir, self.log_dir]:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError:
                pass

        self._build_dataset_paths()

    def __post_init__(self):
        self.configure_runtime(force_local=False, data_dir=self.local_data_dir)


CFG = Config()
