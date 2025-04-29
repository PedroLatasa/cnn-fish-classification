# utils/plot_utils.py
import matplotlib.pyplot as plt
from typing import List

def plot_metrics(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    val_aurocs: List[float]
) -> None:
    """Plots training and validation metrics (loss, accuracy, and AUC) over epochs.

    Args:
        train_losses (List[float]): List of training losses per epoch.
        val_losses (List[float]): List of validation losses per epoch.
        train_accuracies (List[float]): List of training accuracies per epoch.
        val_accuracies (List[float]): List of validation accuracies per epoch.
        val_aurocs (List[float]): List of validation AUC scores per epoch.

    Returns:
        None: Displays the plots.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot AUC
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_aurocs, 'g-', label='Val AUC')
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()