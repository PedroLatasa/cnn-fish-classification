# config.py
from typing import Tuple
import os
import torch

class Config:
    """Configuration settings for the fish classification project.

    This class defines all hyperparameters, dataset paths, and model settings
    required for training and evaluating the EfficientNetB1 model on the A Large Scale
    Fish Dataset. All attributes are class-level constants, and the class provides
    a static method to validate their consistency.

    Attributes:
        DATASET_PATH (str): Kaggle dataset identifier for the fish dataset.
        DATASET_SUBDIR (str): Subdirectory within the dataset containing images.
        USE_AUGMENTED (bool): Whether to use the augmented Fish_Dataset (True) or NA_Fish_Dataset (False).
        NUM_CLASSES (int): Number of fish species to classify.
        IMAGE_SIZE (Tuple[int, int]): Target size for input images (height, width).
        TRAIN_SPLIT (float): Proportion of the dataset for training.
        VAL_SPLIT (float): Proportion of the dataset for validation.
        TEST_SPLIT (float): Proportion of the dataset for testing.
        BATCH_SIZE (int): Number of samples per batch in DataLoader.
        NUM_WORKERS (int): Number of subprocesses for data loading (0 for Windows compatibility).
        NUM_EPOCHS (int): Number of training epochs for initial training.
        LEARNING_RATE (float): Learning rate for the optimizer during initial training.
        FINE_TUNE_EPOCHS (int): Number of epochs for fine-tuning.
        FINE_TUNE_LR (float): Learning rate for fine-tuning.
        PATIENCE (int): Number of epochs to wait before early stopping.
        DEVICE (str): Device for model training ('cuda' if available, else 'cpu').
        CHECKPOINT_DIR (str): Directory to save model checkpoints.
        MODEL_SAVE_PATH (str): Path to save the final trained model.
        NORMALIZE_MEAN (Tuple[float, float, float]): Mean values for image normalization (RGB).
        NORMALIZE_STD (Tuple[float, float, float]): Standard deviation values for image normalization (RGB).
    """

    # --- Dataset Settings ---
    # config.py
    DATASET_PATH: str = "/kaggle/input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset"
    DATASET_SUBDIR: str = ""
    USE_AUGMENTED: bool = True  # Use augmented Fish_Dataset (True) or NA_Fish_Dataset (False)
    NUM_CLASSES: int = 9
    IMAGE_SIZE: Tuple[int, int] = (240, 240)  # Matches EfficientNetB1 IMAGENET1K_V2 crop_size
    TRAIN_SPLIT: float = 0.7
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15

    # --- DataLoader Settings ---
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 0  # Set to 0 for Windows to avoid multiprocessing issues

    # --- Training Settings ---
    NUM_EPOCHS: int = 5  # Reduced for initial training
    LEARNING_RATE: float = 0.001
    FINE_TUNE_EPOCHS: int = 5
    FINE_TUNE_LR: float = 1e-5
    PATIENCE: int = 3
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Directories and Files ---
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
    MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "efficientnetb1_final_model.pth")
    
    # --- Normalization (ImageNet standard values for EfficientNetB1 IMAGENET1K_V2) ---
    NORMALIZE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    NORMALIZE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    @staticmethod
    def validate() -> None:
        """Validates the consistency of configuration parameters.

        Ensures that dataset splits sum to 1.0, and other parameters are positive or non-negative.
        Creates the checkpoint directory if it does not exist.

        Raises:
            AssertionError: If TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT != 1.0, or if
                NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, or PATIENCE are invalid.
        """
        assert Config.TRAIN_SPLIT + Config.VAL_SPLIT + Config.TEST_SPLIT == 1.0, \
            "Dataset split proportions must sum to 1.0"
        assert Config.NUM_CLASSES > 0, "NUM_CLASSES must be positive"
        assert Config.BATCH_SIZE > 0, "BATCH_SIZE must be positive"
        assert Config.NUM_EPOCHS > 0, "NUM_EPOCHS must be positive"
        assert Config.FINE_TUNE_EPOCHS > 0, "FINE_TUNE_EPOCHS must be positive"
        assert Config.PATIENCE >= 0, "PATIENCE must be non-negative"
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)