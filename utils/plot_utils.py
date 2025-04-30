# utls/plot_utils.py
from config import Config
import logging
import matplotlib.pyplot as plt
from typing import List
import os

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
        None: Displays the plots and saves them to /kaggle/working/plots/.
    """
    # Create plots directory
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)

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
    
    # Save the plot
    plot_path = os.path.join(Config.PLOTS_DIR, "metrics_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Metrics plot saved at {plot_path}")
    
    # List files in plots directory
    logging.info("Files in /kaggle/working/plots/:")
    logging.info(os.listdir(Config.PLOTS_DIR))
    
    # Display plot only if configured
    if Config.DISPLAY_PLOTS:
        plt.show()
    plt.close()

