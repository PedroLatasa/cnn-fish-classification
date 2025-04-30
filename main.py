# main.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils.logger import setup_logger
from config import Config
from data.loader import load_fish_dataset
from datasets.fish_dataset import FishDataset, get_transforms
from models.efficientnet import EfficientNetB1
from utils.train_utils import train_model, evaluate_model
from utils.plot_utils import plot_metrics
import os

def main():
    # Initialize logger
    logger = setup_logger("/kaggle/working/logs/training.log")
    logger.info("Starting fish classification training pipeline")
    
    # Validate configuration settings
    Config.validate()
    
    # Check and display the device being used
    print(f"Using device: {Config.DEVICE}")
    
    # Show which GPU is using
    if Config.DEVICE == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Load the fish dataset
    image_paths, labels, classes = load_fish_dataset()
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Create dataset
    dataset = FishDataset(
        image_paths=image_paths,
        labels=labels,
        class_to_idx=class_to_idx,
        transform=get_transforms(train=True)
    )
    
    # Split dataset into train, validation, and test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Update transforms for validation and test
    val_dataset.dataset.transform = get_transforms(train=False)
    test_dataset.dataset.transform = get_transforms(train=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Print dataset sizes
    print(f"Training size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Initialize model, criterion, and optimizer
    model = EfficientNetB1(num_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Train classifier-only
    print("Training Phase 1: Classifier only")
    train_losses, val_losses, train_accuracies, val_accuracies, val_aurocs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=Config.DEVICE,
        num_epochs=Config.NUM_EPOCHS
    )
    
    # Fine-tuning
    print("Training Phase 2: Fine-tuning")
    model.unfreeze_layers()
    optimizer = optim.Adam(model.parameters(), lr=Config.FINE_TUNE_LR)
    train_losses_ft, val_losses_ft, train_accuracies_ft, val_accuracies_ft, val_aurocs_ft = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=Config.DEVICE,
        num_epochs=Config.FINE_TUNE_EPOCHS
    )
    
    # Combine metrics
    train_losses.extend(train_losses_ft)
    val_losses.extend(val_losses_ft)
    train_accuracies.extend(train_accuracies_ft)
    val_accuracies.extend(val_accuracies_ft)
    val_aurocs.extend(val_aurocs_ft)
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, val_aurocs)
    
    # Evaluate on test set
    evaluate_model(model, test_loader, Config.DEVICE, classes=classes)
    
    # Save final model
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "efficientnetb1_final_model.pth"))
    print(f"Final model saved at {os.path.join(Config.CHECKPOINT_DIR, 'efficientnetb1_final_model.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=Config.DEVICE, help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    Config.DEVICE = args.device
    main()
    