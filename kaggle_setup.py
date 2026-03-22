from __future__ import annotations

import importlib
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.py"
KAGGLE_DATASET_ROOT = "/kaggle/input/datasets/mohitkhalote/transunet"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _count_images(directory: str) -> int:
    path = Path(directory)
    if not path.is_dir():
        return 0
    return sum(
        1
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTS
    )


def _check_dependencies() -> None:
    required_modules = {
        "timm": "timm",
        "albumentations": "albumentations",
        "monai": "monai",
        "cv2": "opencv-python",
    }

    missing_modules: list[str] = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing_modules.append(module_name)

    if not missing_modules:
        print("Dependency check: all required packages are installed")
        return

    missing_pip_names = [required_modules[name] for name in missing_modules]
    pip_command = f"pip install {' '.join(missing_pip_names)}"
    print("Dependency check failed: missing required packages:")
    for module_name in missing_modules:
        print(f"- {module_name}")
    print(f"Install them with: {pip_command}")
    raise SystemExit(1)


def _patch_config_file() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.py not found at {CONFIG_PATH}")

    content = CONFIG_PATH.read_text(encoding="utf-8")
    updated = content

    updated = re.sub(
        r'explicit_kaggle_data_root\s*=\s*"[^"]*"',
        f'explicit_kaggle_data_root = "{KAGGLE_DATASET_ROOT}"',
        updated,
        count=1,
    )

    enforce_block = "if self.is_kaggle:\n            self.is_colab = False"
    if enforce_block not in updated:
        updated = updated.replace(
            "def __post_init__(self):\n",
            "def __post_init__(self):\n        if self.is_kaggle:\n            self.is_colab = False\n",
            1,
        )

    if updated != content:
        CONFIG_PATH.write_text(updated, encoding="utf-8")
        print(f"Step 1: updated {CONFIG_PATH.name} for Kaggle root + flags")
    else:
        print(f"Step 1: {CONFIG_PATH.name} already has required Kaggle settings")


def _reload_cfg():
    sys.modules.pop("config", None)
    import config

    importlib.reload(config)
    return config.CFG


def _prepare_required_working_splits(cfg) -> None:
    from dataset import _prepare_kaggle_working_split

    for dataset_name in ("mri_glioma", "covid_ct"):
        print(f"Step 2: preparing Kaggle working split for {dataset_name}")
        try:
            _prepare_kaggle_working_split(dataset_name, cfg)
        except Exception as exc:
            print(
                f"Warning: failed to prepare working split for {dataset_name}: {exc}"
            )


def _run_preprocessing_for_available_datasets(cfg) -> tuple[list[str], list[str], dict]:
    from preprocess import preprocess_dataset

    ready_datasets: list[str] = []
    missing_datasets: list[str] = []
    preprocess_summaries: dict = {}

    for dataset_name, paths in cfg.dataset_paths.items():
        train_images_dir = paths["train_images"]
        if os.path.isdir(train_images_dir):
            ready_datasets.append(dataset_name)
            print(f"Step 3: preprocessing {dataset_name}")
            preprocess_summaries[dataset_name] = preprocess_dataset(dataset_name)
        else:
            missing_datasets.append(dataset_name)
            print(
                f"Step 3: skipping {dataset_name} (missing train_images: {train_images_dir})"
            )

    return ready_datasets, missing_datasets, preprocess_summaries


def _print_summary(cfg, ready_datasets: list[str], missing_datasets: list[str]) -> None:
    print("\n" + "=" * 72)
    print("Step 4: Dataset Readiness Summary")
    print("=" * 72)

    if ready_datasets:
        print("READY DATASETS")
        for dataset_name in ready_datasets:
            paths = cfg.dataset_paths[dataset_name]
            train_count = _count_images(paths["train_images"])
            test_count = _count_images(paths["test_images"])
            print(
                f"- {dataset_name:12s} | train_images={train_count:5d} | "
                f"test_images={test_count:5d}"
            )
    else:
        print("READY DATASETS")
        print("- none")

    print("\nMISSING DATASETS")
    if missing_datasets:
        for dataset_name in missing_datasets:
            print(f"- {dataset_name}")
    else:
        print("- none")

    print("=" * 72)


def main() -> None:
    os.chdir(PROJECT_ROOT)

    print("Running Kaggle setup for TransUNet")
    print(f"Project root: {PROJECT_ROOT}")
    _check_dependencies()

    _patch_config_file()
    cfg = _reload_cfg()

    if cfg.is_kaggle:
        print("Environment: Kaggle detected")
    else:
        print("Environment: Kaggle not detected (script is intended for Kaggle)")

    _prepare_required_working_splits(cfg)
    ready_datasets, missing_datasets, _ = _run_preprocessing_for_available_datasets(cfg)
    _print_summary(cfg, ready_datasets, missing_datasets)

    print("\nSetup complete. You can now run: python train.py <dataset_name>")


if __name__ == "__main__":
    main()