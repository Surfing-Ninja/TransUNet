import importlib
import os
from pathlib import Path

import torch


def _check_cuda() -> None:
    print("CUDA Check")
    print("-" * 60)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"CUDA available: YES")
        print(f"GPU: {props.name}")
        print(f"VRAM: {vram_gb:.2f} GB")
    else:
        print("CUDA available: NO")
        print("GPU: Not detected (training will run on CPU)")
    print()


def _check_dependencies() -> None:
    required = {
        "torch": "torch",
        "torchvision": "torchvision",
        "timm": "timm",
        "einops": "einops",
        "albumentations": "albumentations",
        "cv2": "opencv-python-headless",
        "skimage": "scikit-image",
        "sklearn": "scikit-learn",
        "monai": "monai",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "pandas": "pandas",
        "tqdm": "tqdm",
        "tensorboard": "tensorboard",
        "scipy": "scipy",
        "PIL": "Pillow",
    }

    missing_modules: list[str] = []
    for module_name in required:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing_modules.append(module_name)

    print("Dependency Check")
    print("-" * 60)
    if not missing_modules:
        print("All required packages are installed.")
    else:
        print("Missing packages detected:")
        for module_name in missing_modules:
            print(f"- {module_name} (pip package: {required[module_name]})")
        print("Install command:")
        print(f"pip install {' '.join(required[m] for m in missing_modules)}")
        print("or")
        print("pip install -r requirements.txt")
    print()


def _print_expected_structure() -> None:
    datasets = ["kvasir_seg", "mri_glioma", "isic2018", "covid_ct"]
    splits = ["train", "test"]
    subdirs = ["images", "masks"]

    print("Expected Local Dataset Structure")
    print("-" * 60)
    print("Create these folders under your project root:")
    for dataset_name in datasets:
        for split in splits:
            for subdir in subdirs:
                print(f"data/{dataset_name}/{split}/{subdir}")
    print()


def _print_training_commands() -> None:
    print("Training Commands")
    print("-" * 60)
    print("Single dataset:")
    print("python train.py kvasir_seg --local")
    print()
    print("All datasets sequentially:")
    print("python train.py --all --local")
    print()
    print("If datasets are outside project data/, provide absolute path:")
    print("python train.py kvasir_seg --local --data-dir /absolute/path/to/datasets")
    print("python train.py kvasir_seg --local --data-dir C:/datasets")
    print()


def main() -> None:
    print("Local Setup Check for MaS-TransUNet")
    print("=" * 60)
    print(f"Working directory: {Path.cwd()}")
    print()

    _check_cuda()
    _check_dependencies()
    _print_expected_structure()
    _print_training_commands()


if __name__ == "__main__":
    main()