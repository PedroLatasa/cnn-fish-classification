# utils/train_utils.py
import os
import torch
from typing import List, Tuple
from tqdm import tqdm
from config import Config

def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int = Config.NUM_EPOCHS
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Trains the model and evaluates it on the validation set for each epoch.

    This function performs training over the specified number of epochs, computing
    loss and accuracy for both training and validation datasets. It includes early stopping
    based on validation loss and uses a progress bar to display training progress.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (str): Device to run the model on ('cuda' or 'cpu').
        num_epochs (int, optional): Number of training epochs. Defaults to Config.NUM_EPOCHS.

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: A tuple containing:
            - train_losses: List of average training losses per epoch.
            - val_losses: List of average validation losses per epoch.
            - train_accuracies: List of training accuracies per epoch (percentage).
            - val_accuracies: List of validation accuracies per epoch (percentage).
    """
    # Initialize lists to store metrics
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop over epochs
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        train_loss: float = 0.0
        train_correct: int = 0
        train_total: int = 0

        # Train on the training dataset
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Move data to the specified device
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate loss and accuracy
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Compute average training loss and accuracy
        train_accuracy = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Set model to evaluation mode
        model.eval()
        val_loss: float = 0.0
        val_correct: int = 0
        val_total: int = 0

        # Evaluate on the validation dataset
        with torch.no_grad():
            for images, labels in val_loader:
                # Move data to the specified device
                images, labels = images.to(device), labels.to(device)
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Accumulate loss and accuracy
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Compute average validation loss and accuracy
        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Print epoch metrics
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break

    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str
) -> None:
    """Evaluates the model on the test dataset and prints the accuracy.

    This function computes the classification accuracy on the test dataset using
    a progress bar to display evaluation progress.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (str): Device to run the model on ('cuda' or 'cpu').

    Returns:
        None: Prints the test accuracy to the console.
    """
    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")))
    model.eval()
    test_correct: int = 0
    test_total: int = 0

    # Evaluate on the test dataset
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            # Move data to the specified device
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Compute and print test accuracy
    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")