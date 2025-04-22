# utils/plot_utils.py
import matplotlib.pyplot as plt
from config import Config
import os
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """Grafica las pérdidas y precisiones de entrenamiento y validación."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pérdida durante el Entrenamiento")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Precisión durante el Entrenamiento")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.CHECKPOINT_DIR, "training_metrics.png"))
    plt.show()