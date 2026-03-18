# MaS-TransUNet

This repository contains the PyTorch implementation of the **MaS-TransUNet** architecture for medical image segmentation. 

It uses a modified Swin Transformer as the encoder coupled with Channel Attention Modules (CAM), Edge Attention Modules (EAM), and a Feedback Attention Module (FAM) for iterative multi-scale refinement.

## Repository

GitHub: https://github.com/Surfing-Ninja/TransUNet.git

Clone with:
```bash
git clone https://github.com/Surfing-Ninja/TransUNet.git
cd TransUNet
```

## Requirements

The codebase relies on PyTorch and several standard vision libraries. Install them via:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The code currently supports tests on multiple distinct datasets: `tcga_lgg`, `dsb2018`, `kvasir_seg`, `isic2018`, and `covid_ct`. Make sure the datasets are stored inside the `data` directory (or Google Drive if using Colab) like so:

```text
data/
└── tcga_lgg/
    ├── train/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
```

## Running the Code

### 1. Preprocessing
To generate Canny edge maps automatically for a new dataset, run:
```bash
python preprocess.py all
```
*Note: The code dynamically detects datasets available in your `data/` folder.*

### 2. Training
Start the training routine with a selected dataset using:
```bash
python train.py tcga_lgg
```
Hyperparameters (epochs, batch size, learning rates) are configured centrally inside `config.py`.

### 3. Evaluation
After training, evaluate the best checkpoint and construct side-by-side PNG analyses on the test set:
```bash
python evaluate.py tcga_lgg
```

## Google Colab Usage

A dedicated Jupyter Notebook `colab_train.ipynb` is included for effortless training on Google Colab hardware (e.g. T4 GPUs). 

To use it:
1. Upload your dataset zip files (e.g. `archive*.zip`) to `My Drive/datasets/`.
2. Open `colab_train.ipynb` in Google Colab.
3. Run the auto-unzip dataset cell: it scans `My Drive/datasets/`, extracts zips, and organizes data into dataset folders.
4. Re-running the cell skips unchanged zip files automatically (prevents repeated unzipping).
5. Run the Colab mode check cell and confirm datasets show `Found`.
6. Continue with preprocessing, training, and checkpoint verification cells.
