# data/loader.py
import kagglehub
import os
from tqdm import tqdm
from config import Config

def load_fish_dataset():
    """Descarga y carga el dataset de peces, devolviendo rutas de imágenes y etiquetas."""
    # Descarga el dataset
    path = kagglehub.dataset_download(Config.DATASET_PATH)
    print("Path to dataset files:", path)

    # Seleccionar subdirectorio según USE_AUGMENTED
    dataset_subdir = Config.DATASET_SUBDIR if Config.USE_AUGMENTED else "NA_Fish_Dataset"
    dataset_path = os.path.join(path, dataset_subdir)

    # Obtener solo las carpetas de clases (excluyendo GT y archivos como .txt)
    classes = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and not d.endswith("GT")
    ]
    print("Clases encontradas:", classes)

    if len(classes) != Config.NUM_CLASSES:
        print(f"Advertencia: Se esperaban {Config.NUM_CLASSES} clases, pero se encontraron {len(classes)}")

    # Cargar imágenes y etiquetas
    image_paths = []
    labels = []

    for cls in classes:
        cls_path = os.path.join(dataset_path, cls, cls)
        if not os.path.exists(cls_path):
            print(f"Error: No se encontró la carpeta {cls_path}")
            continue
        for img_name in tqdm(os.listdir(cls_path), desc=f"Cargando {cls}"):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(cls_path, img_name)
                image_paths.append(img_path)
                labels.append(cls)

    print(f"Total imágenes cargadas: {len(image_paths)}")
    return image_paths, labels, classes