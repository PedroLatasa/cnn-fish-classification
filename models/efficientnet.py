# models/efficientnet.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

class EfficientNetB1(nn.Module):
    """EfficientNetB1 model for classifying fish species, using a pre-trained backbone.

    This model uses a pre-trained EfficientNetB1 from torchvision, with a modified
    classifier head to match the number of fish species. It supports freezing/unfreezing
    layers for transfer learning and fine-tuning.

    Attributes:
        model (nn.Module): Pre-trained EfficientNetB1 with a custom classifier.
    """

    def __init__(self, num_classes: int) -> None:
        """Initializes the EfficientNetB1 model.

        Args:
            num_classes (int): Number of fish species to classify.

        Returns:
            None
        """
        super(EfficientNetB1, self).__init__()
        # Load pre-trained EfficientNetB1 with IMAGENET1K_V2 weights
        self.model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2, progress=True)
        
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify the classifier head (inspired by Kaggle notebook)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        return self.model(x)
    
    def unfreeze_layers(self) -> None:
        """Unfreezes the top layers for fine-tuning, keeping earlier layers frozen."""
        # Unfreeze the classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        # Unfreeze the last two blocks of the backbone (features[-2:])
        for name, param in self.model.features[-2:].named_parameters():
            param.requires_grad = True