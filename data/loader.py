# data/loader.py
import kagglehub
import os
from typing import List, Tuple
from tqdm import tqdm
from config import Config

def load_fish_dataset() -> Tuple[List[str], List[str], List[str]]:
    """Downloads and loads the fish dataset, returning image paths, labels, and classes.

    This function downloads the A Large Scale Fish Dataset from Kaggle, selects the
    appropriate subdirectory based on Config.USE_AUGMENTED, and collects image paths
    and labels for all valid images.

    Returns:
        Tuple[List[str], List[str], List[str]]: A tuple containing:
            - image_paths: List of file paths to the images.
            - labels: List of class names for each image.
            - classes: List of unique class names.
    """
    # Download the dataset from Kaggle
    # for local execution
    # path = kagglehub.dataset_download(Config.DATASET_PATH) 
    path = Config.DATASET_PATH # for kaggle execution GPU
    print("Path to dataset files:", path)

    # Select subdirectory based on whether augmented dataset is used
    dataset_subdir = Config.DATASET_SUBDIR if Config.USE_AUGMENTED else "NA_Fish_Dataset"
    dataset_path = os.path.join(path, dataset_subdir)

    # Get class directories, excluding GT folders and non-directories
    classes = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and not d.endswith("GT")
    ]
    print("Classes found:", classes)

    # Warn if the number of classes doesn't match the configuration
    if len(classes) != Config.NUM_CLASSES:
        print(f"Warning: Expected {Config.NUM_CLASSES} classes, but found {len(classes)}")

    # Collect image paths and labels
    image_paths: List[str] = []
    labels: List[str] = []

    for cls in classes:
        cls_path = os.path.join(dataset_path, cls, cls)
        # Check if class directory exists
        if not os.path.exists(cls_path):
            print(f"Error: Directory {cls_path} not found")
            continue
        # Load images with progress bar
        for img_name in tqdm(os.listdir(cls_path), desc=f"Loading {cls}"):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(cls_path, img_name)
                image_paths.append(img_path)
                labels.append(cls)

    print(f"Total images loaded: {len(image_paths)}")
    return image_paths, labels, classes