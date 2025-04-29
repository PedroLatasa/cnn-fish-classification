# Fish Classification in High-Seas Fishing

A project to classify 9 fish species using a fine-tuned EfficientNetB1 model in PyTorch, based on the "A Large Scale Fish Dataset". This project leverages transfer learning to achieve high accuracy in identifying fish species, suitable for applications in high-seas fishing.

## Citation

If you use this dataset or code, please cite the original work:

```
@inproceedings{ulucan2020large,
  title={A Large-Scale Dataset for Fish Segmentation and Classification},
  author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
  booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages={1--5},
  year={2020},
  organization={IEEE}
}
```

## Project Overview

This project uses the "A Large Scale Fish Dataset" to train an EfficientNetB1 model for classifying 9 fish species. The pipeline includes:

- **Data Loading**: Custom dataset loader (`FishDataset`) to handle the fish dataset with augmentations.
- **Model**: Pretrained EfficientNetB1 with a custom classifier, fine-tuned for the task.
- **Training**: Two-phase training (classifier training + fine-tuning) with early stopping.
- **Evaluation**: Metrics (loss, accuracy) plotted for training and validation, with final evaluation on a test set.
- **Execution**: Supports both local environments and Kaggle with GPU acceleration.

The project is configured via `config.py`, which defines dataset paths, hyperparameters (e.g., 15 epochs for initial training, 10 for fine-tuning), and normalization settings.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **PyTorch**: Version 2.5.1 or compatible, with CUDA support for GPU training.
- **Dataset**: The "A Large Scale Fish Dataset" is required (available on Kaggle).
- **Hardware**:
  - **Local**: A GPU (NVIDIA recommended) is optional but speeds up training.
  - **Kaggle**: Use the GPU T4 x2 accelerator for optimal performance.

## Installation

### Option 1: Local Setup

1. **Clone the Repository**:
   ```bash
   git clone -b EfficientNetB1 https://github.com/PedroLatasa/cnn-fish-classification.git
   cd cnn-fish-classification
   ```

2. **Set Up the Conda Environment**:
   - The `environment.yml` file defines all dependencies for a consistent local setup, including PyTorch, torchvision, and other libraries.
   - Install and activate the environment:
     ```bash
     conda env create -f environment.yml
     conda activate fish_classification
     ```

3. **Install Additional Dependencies** (if needed):
   - If you prefer to use `requirements.txt` instead of `environment.yml`, install dependencies with pip:
     ```bash
     pip install -r requirements.txt
     ```
   - Note: `environment.yml` is preferred for local setups as it ensures consistent versions across platforms. `requirements.txt` is a fallback for environments where Conda is not available.

4. **Configure Kaggle API** (for dataset download):
   - Obtain your `kaggle.json` API token from Kaggle (Profile > Account > Create New API Token).
   - Place it in:
     - Linux/Mac: `~/.kaggle/kaggle.json`
     - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
   - Set permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

5. **Download the Dataset**:
   - Download the "A Large Scale Fish Dataset" manually from Kaggle or use the Kaggle API:
     ```bash
     kaggle datasets download -d tarkika/large-fish-dataset
     unzip large-fish-dataset.zip -d data/Fish_Dataset
     ```
   - Update `config.py` to point to the local dataset path, e.g.:
     ```python
     DATASET_PATH: str = "data/Fish_Dataset/Fish_Dataset"
     DATASET_SUBDIR: str = ""
     ```

### Option 2: Kaggle Setup

1. **Create a Kaggle Notebook**:
   - Go to Kaggle and create a new notebook.
   - Enable the GPU accelerator (Settings > Accelerator > GPU T4 x2).

2. **Clone the Repository**:
   - In a notebook cell, clone the repository:
     ```python
     !git clone -b EfficientNetB1 https://github.com/PedroLatasa/cnn-fish-classification.git
     %cd cnn-fish-classification
     ```

3. **Install Dependencies**:
   - Use `requirements.txt` to install dependencies, as Conda is not fully supported in Kaggle:
     ```python
     !pip install -r requirements.txt
     ```
   - Note: `environment.yml` is not used in Kaggle due to the lack of Conda support. `requirements.txt` includes all necessary packages (e.g., torch, torchvision, tqdm) for Kaggle.

4. **Add the Dataset**:
   - In the Kaggle notebook, go to the "Data" tab and add the "A Large Scale Fish Dataset" (search for "large-fish-dataset").
   - The dataset is automatically mounted at `/kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset`.
   - Ensure `config.py` uses the correct path:
     ```python
     DATASET_PATH: str = "/kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset"
     DATASET_SUBDIR: str = ""
     ```

5. **Configure GitHub Token (Optional)**:
   - If your repository is private, add your GitHub personal access token as a Kaggle Secret:
     - Go to Add-ons > Secrets > Add a new secret.
     - Name it `GITHUB_TOKEN` and paste your token.
   - The notebook will access it automatically via `kaggle_secrets`.

## Project Structure

```
cnn-fish-classification/
├── config.py              # Configuration settings (dataset paths, hyperparameters)
├── data/
│   └── loader.py          # Dataset loading logic
├── datasets/
│   ├── __init__.py
│   └── fish_dataset.py    # Custom FishDataset class and transformations
├── models/
│   ├── __init__.py
│   └── efficientnet.py    # EfficientNetB1 model definition
├── utils/
│   ├── __init__.py
│   ├── train_utils.py     # Training and evaluation functions
│   └── plot_utils.py      # Plotting functions for metrics
├── main.py                # Main script for training and evaluation
├── environment.yml        # Conda environment for local setup
├── requirements.txt       # Pip dependencies for Kaggle/local fallback
└── README.md              # Project documentation
```

## Execution

### Local Execution
1. Ensure the dataset is downloaded and `config.py` points to the correct `DATASET_PATH`.
2. Activate the Conda environment:
   ```bash
   conda activate fish_classification
   ```
3. Run the main script:
   ```bash
   python main.py --device cuda
   ```
   - Use `--device cpu` if no GPU is available.
   - The model will train for 15 epochs (classifier) + 10 epochs (fine-tuning), with early stopping after 3 epochs of no improvement.

### Kaggle Execution
1. Use the following notebook setup (example cells):
   ```python
   # Install dependencies
   !pip install -r requirements.txt

   # Clone repository
   %cd /kaggle/working
   !git clone -b EfficientNetB1 https://github.com/PedroLatasa/cnn-fish-classification.git
   %cd cnn-fish-classification

   # Verify dataset
   !ls /kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset

   # Run main script
   import sys
   sys.path.append('/kaggle/working/cnn-fish-classification')
   !python main.py --device cuda
   ```
2. Ensure the dataset is added and the GPU is enabled.
3. The model will save checkpoints to `/kaggle/working/checkpoints` and the final model to `efficientnetb1_final_model.pth`.

## Configuration

Key settings in `config.py`:
- **Dataset**: `/kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset` (Kaggle) or local path.
- **Hyperparameters**:
  - `NUM_EPOCHS: 15` (initial training).
  - `FINE_TUNE_EPOCHS: 10` (fine-tuning).
  - `BATCH_SIZE: 32`.
  - `LEARNING_RATE: 0.001` (initial), `FINE_TUNE_LR: 1e-5` (fine-tuning).
  - `PATIENCE: 3` (early stopping).
- **Device**: Automatically selects `cuda` if GPU is available, else `cpu`.

## Notes

- **Kaggle vs. Local**:
  - In Kaggle, use `requirements.txt` for dependencies and the mounted dataset path. `environment.yml` is not supported.
  - Locally, use `environment.yml` for a consistent Conda environment, with `requirements.txt` as a fallback.
- **Dataset Loading**:
  - The dataset is accessed directly from `/kaggle/input/` in Kaggle, avoiding `kagglehub` downloads.
  - Locally, download the dataset via the Kaggle API or manually.
- **Training**:
  - The model uses transfer learning with EfficientNetB1, pretrained on ImageNet.
  - Training is split into classifier training (15 epochs) and fine-tuning (10 epochs) for optimal performance.
- **Troubleshooting**:
  - Ensure `__init__.py` files are present in `datasets/`, `models/`, and `utils/` for proper module imports.
  - Check dataset path if errors occur during loading.

## License

This project is licensed under the MIT License. See `LICENSE` for details.