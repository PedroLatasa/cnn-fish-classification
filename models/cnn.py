# models/cnn.py
import torch
import torch.nn as nn

class FishCNN(nn.Module):
    """A Convolutional Neural Network (CNN) for classifying fish species.

    This model consists of a feature extraction backbone with convolutional and
    pooling layers, followed by a classifier with fully connected layers. It is
    designed for the A Large Scale Fish Dataset with a configurable number of classes.

    Attributes:
        features (nn.Sequential): Convolutional layers for feature extraction.
        classifier (nn.Sequential): Fully connected layers for classification.
    """

    def __init__(self, num_classes: int) -> None:
        """Initializes the FishCNN model.

        Args:
            num_classes (int): Number of fish species to classify.

        Returns:
            None
        """
        super(FishCNN, self).__init__()
        # Define feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input: 3 channels (RGB)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsample by 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Define classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),  # Assuming input image size is 224x224
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # Extract features
        x = self.features(x)
        # Classify
        x = self.classifier(x)
        return x