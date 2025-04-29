# main.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from config import Config
from data.loader import load_fish_dataset
from datasets.fish_dataset import FishDataset, get_transforms
from models.efficientnet import EfficientNetB1
from utils.train_utils import train_model, evaluate_model
from utils.plot_utils import plot_metrics

def main():
    # Validate configuration settings
    Config.validate()
    
    # Check and display the device being used
    print(f"Using device: {Config.DEVICE}")
    
    # Load the fish dataset
    image_paths, labels, classes = load_fish_dataset()
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Verify the number of classes
    if len(classes) != Config.NUM_CLASSES:
        raise ValueError(f"Expected {Config.NUM_CLASSES} classes, but found {len(classes)}")
    
    # Create the dataset with transformations
    dataset = FishDataset(image_paths, labels, class_to_idx, transform=get_transforms(train=True))
    
    # Split the dataset into train, validation, and test sets
    train_size = int(Config.TRAIN_SPLIT * len(dataset))
    val_size = int(Config.VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders for training, validation, and testing
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    print(f"Training size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Initialize the model
    model = EfficientNetB1(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    
    # Set up training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Train the model and collect metrics (Phase 1: Train only classifier)
    print("Training Phase 1: Classifier only")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=Config.DEVICE,
        num_epochs=Config.NUM_EPOCHS
    )
    
    # Fine-tuning: Unfreeze some layers and train with lower learning rate
    print("Training Phase 2: Fine-tuning")
    model.unfreeze_layers()  # Unfreeze top layers for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=Config.FINE_TUNE_LR)
    train_losses_ft, val_losses_ft, train_accuracies_ft, val_accuracies_ft = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=Config.DEVICE,
        num_epochs=Config.FINE_TUNE_EPOCHS
    )
    
    # Combine metrics for plotting
    train_losses.extend(train_losses_ft)
    val_losses.extend(val_losses_ft)
    train_accuracies.extend(train_accuracies_ft)
    val_accuracies.extend(val_accuracies_ft)
    
    # Evaluate the model on the test set
    evaluate_model(model, test_loader, Config.DEVICE)
    
    # Plot training and validation metrics
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Save the final model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"Final model saved at {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()