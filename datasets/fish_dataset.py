# datasets/fish_dataset.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Dict, Tuple, Optional
from config import Config

class FishDataset(Dataset):
    """A custom dataset for loading fish images and their labels.

    This dataset loads images from the A Large Scale Fish Dataset, applies optional
    transformations, and returns image-label pairs for training or evaluation.

    Attributes:
        image_paths (List[str]): List of file paths to the images.
        labels (List[str]): List of class names for each image.
        class_to_idx (Dict[str, int]): Mapping of class names to integer indices.
        transform (Optional[transforms.Compose]): Transformations to apply to images.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[str],
        class_to_idx: Dict[str, int],
        transform: Optional[transforms.Compose] = None
    ) -> None:
        """Initializes the FishDataset.

        Args:
            image_paths (List[str]): List of file paths to the images.
            labels (List[str]): List of class names for each image.
            class_to_idx (Dict[str, int]): Mapping of class names to integer indices.
            transform (Optional[transforms.Compose], optional): Transformations to apply. Defaults to None.

        Returns:
            None
        """
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Retrieves an image and its label by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the transformed image tensor and its label index.
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        # Get label
        label = self.labels[idx]
        label_idx = self.class_to_idx[label]
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        return image, label_idx

def get_transforms() -> transforms.Compose:
    """Returns the transformations for the fish dataset.

    The transformations include resizing to the configured image size, converting
    to a tensor, and normalizing with ImageNet mean and standard deviation.

    Returns:
        transforms.Compose: A composition of image transformations.
    """
    return transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),  # Resize to configured size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)  # Normalize
    ])
    