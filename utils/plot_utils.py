# utils/plot_utils.py
import matplotlib.pyplot as plt
from typing import List
from config import Config
import os

def plot_metrics(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float]
) -> None:
    """Plots training and validation losses and accuracies over epochs.

    Generates two subplots: one for loss (training and validation) and one for
    accuracy (training and validation). The plots are saved to a file in the
    checkpoint directory and displayed on the screen.

    Args:
        train_losses (List[float]): List of average training losses per epoch.
        val_losses (List[float]): List of average validation losses per epoch.
        train_accuracies (List[float]): List of training accuracies per epoch (percentage).
        val_accuracies (List[float]): List of validation accuracies per epoch (percentage).

    Returns:
        None: Saves the plot to a file and displays it.
    """
    # Generate epoch numbers for the x-axis
    epochs = range(1, len(train_losses) + 1)

    # Create a figure with two subplots
    plt.figure(figsize=(12, 5))

    # Plot training and validation losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss During Training")
    plt.legend()

    # Plot training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy During Training")
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot to the checkpoint directory
    plt.savefig(os.path.join(Config.CHECKPOINT_DIR, "training_metrics.png"))

    # Display the plot
    plt.show()