# config.py
from typing import Tuple
import os
import torch

class Config:
    """Configuración para el proyecto de clasificación de peces."""
    
    # --- Dataset ---
    DATASET_PATH: str = "crowww/a-large-scale-fish-dataset"
    DATASET_SUBDIR: str = "Fish_Dataset/Fish_Dataset"
    USE_AUGMENTED: bool = True  # True para Fish_Dataset (aumentado), False para NA_Fish_Dataset
    NUM_CLASSES: int = 9
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    TRAIN_SPLIT: float = 0.7
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15
    
    # --- DataLoader ---
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 0  # 0 para Windows
    
    # --- Entrenamiento ---
    NUM_EPOCHS: int = 10
    LEARNING_RATE: float = 0.001
    PATIENCE: int = 3
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Directorios y Archivos ---
    CHECKPOINT_DIR: str = "checkpoints"
    MODEL_SAVE_PATH: str = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    
    # --- Normalización (valores estándar de ImageNet) ---
    NORMALIZE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    NORMALIZE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    @staticmethod
    def validate():
        """Valida que los parámetros sean consistentes."""
        assert Config.TRAIN_SPLIT + Config.VAL_SPLIT + Config.TEST_SPLIT == 1.0, \
            "Las proporciones de split deben sumar 1.0"
        assert Config.NUM_CLASSES > 0, "NUM_CLASSES debe ser positivo"
        assert Config.BATCH_SIZE > 0, "BATCH_SIZE debe ser positivo"
        assert Config.NUM_EPOCHS > 0, "NUM_EPOCHS debe ser positivo"
        assert Config.PATIENCE >= 0, "PATIENCE debe ser no negativo"
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)