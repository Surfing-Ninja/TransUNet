import os
from dataclasses import dataclass, field

import torch


@dataclass
class Config:
    # Image and training
    image_size: int = 224
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 0.01
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
    num_workers: int = field(default_factory=lambda: 0 if ('COLAB_GPU' in os.environ or 'COLAB_RELEASE_TAG' in os.environ or os.path.isdir('/content/drive')) else 4)

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Environment flag – set True when running on Google Colab
    is_colab: bool = 'COLAB_GPU' in os.environ or 'COLAB_RELEASE_TAG' in os.environ

    # ------------------------------------------------------------------
    # Derived paths (resolved after init based on is_colab)
    # ------------------------------------------------------------------
    base_data_dir: str = field(init=False)
    checkpoint_dir: str = field(init=False)
    dataset_paths: dict = field(init=False)

    def __post_init__(self):
        # Auto-detect Colab by checking if Drive is mounted
        if os.path.isdir('/content/drive'):
            self.is_colab = True

        if self.is_colab:
            self.base_data_dir = "/content/drive/MyDrive/datasets"
            self.checkpoint_dir = "/content/drive/MyDrive/mas_transunet_checkpoints"
        else:
            self.base_data_dir = "./data"
            self.checkpoint_dir = "./checkpoints"

        dataset_names = [
            "mri_glioma",
            "kvasir_seg",
            "isic2018",
            "covid_ct",
        ]

        self.dataset_paths = {}
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
