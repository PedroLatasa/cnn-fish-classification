# Fish Classification in High-Seas Fishing
This repository contains code for classifying 9 fish species using a fine-tuned **EfficientNetB1** model in PyTorch, based on the ["A Large Scale Fish Dataset"](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset). Using transfer learning, the model achieves high accuracy for marine biology and fishing applications. You can also check the kaggle notebook we made for execution: (https://www.kaggle.com/code/pedromaralatasa/cnn-fish-classification)


## Citation
Please cite the original dataset:

```bibtex
@inproceedings{ulucan2020large,
  title={A Large-Scale Dataset for Fish Segmentation and Classification},
  author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
  booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages={1--5},
  year={2020},
  organization={IEEE}
}
```

# Project Overview
The pipeline classifies 9 fish species (9,000 images) using EfficientNetB1, with:

- **Data Loading**: Custom `FishDataset` with augmentations.
- **Training**: 12 epochs (classifier) + 6 epochs (fine-tuning), with early stopping.
- **Evaluation**: Metrics (loss, accuracy, AUC) and confusion matrix, saved as plots.
- **Outputs**: Plots (`plots/`), logs (`logs/`), and models (`checkpoints/`).

A Kaggle notebook provides EDA and training: Kaggle Notebook (update with your notebook URL).

# Environment Setup
Clone the repository and set up the Conda environment:

```bash 
git clone https://github.com/PedroLatasa/cnn-fish-classification.git
cd cnn-fish-classification
conda env create -f environment.yml
conda activate fish_classification
```

- Note: Requires Python 3.11 and Conda.

# Data Preparation
1. Install `kagglehub` for dataset download:

```bash
pip install kagglehub
``` 
2. Update `data/loader.py`:

  - Uncomment the `kagglehub` download line:

```bash
path = kagglehub.dataset_download(Config.DATASET_PATH)
```
  - Remove or comment out Kaggle-specific paths.

3. Update `config.py`:

  - Set the local dataset path:

```bash
DATASET_PATH: str = "crowww/a-large-scale-fish-dataset"
DATASET_SUBDIR: str = "Fish_Dataset/Fish_Dataset"
```
  - The dataset downloads to a local cache (managed by `kagglehub`) and is accessible via the `FishDataset` class.

# Execution
Run the training script: 

```bash
python main.py --device cuda
```

---
### Arguments

| Argument   | Description                           |
|------------|---------------------------------------|
| `--device` | `cuda` for GPU, `cpu` for CPU (default: `cuda`) |

---

## Outputs
- **Plots**: Metrics and confusion matrix in `plots/`.
- **Logs**: Training events in `logs/training.log`.
- **Models**: Best and final models in `checkpoints/`.

# Configuration
Edit `config.py` for custom settings:

- NUM_EPOCHS: 12 (classifier training)
- FINE_TUNE_EPOCHS: 6 (fine-tuning)
- BATCH_SIZE: 64
- DISPLAY_PLOTS: True (enables inline plot display)

# Notes
- **Local Setup**:

  - Ensure `kagglehub` is installed (`pip install kagglehub`).
  - Configure Kaggle API token in `~/.kaggle/kaggle.json`.

- **Troubleshooting**:
  - Verify `__init__.py` files in `datasets/`, `models/`, and `utils/`.
  - Check dataset path in `config.py` if loading fails.

- **Kaggle**:
  - Use the linked notebook for GPU-accelerated training.