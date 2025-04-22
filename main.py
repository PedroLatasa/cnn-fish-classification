# main.py
# Trabajo Final Clasificación de Peces en pesca de alta mar
# @inproceedings{ulucan2020large,
#  title={A Large-Scale Dataset for Fish Segmentation and Classification},
#  author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
#  booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
#  pages={1--5},
#  year={2020},
#  organization={IEEE}
# }

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from config import Config
from data.loader import load_fish_dataset
from datasets.fish_dataset import FishDataset, get_transforms
from models.cnn import FishCNN
from utils.train_utils import train_model, evaluate_model
from utils.plot_utils import plot_metrics

def main():
    # Validar configuración
    Config.validate()
    
    # Verificar dispositivo
    print(f"Usando dispositivo: {Config.DEVICE}")
    
    # Cargar dataset
    image_paths, labels, classes = load_fish_dataset()
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Verificar número de clases
    if len(classes) != Config.NUM_CLASSES:
        raise ValueError(f"Se esperaban {Config.NUM_CLASSES} clases, pero se encontraron {len(classes)}")
    
    # Crear dataset
    dataset = FishDataset(image_paths, labels, class_to_idx, transform=get_transforms())
    
    # Dividir dataset
    train_size = int(Config.TRAIN_SPLIT * len(dataset))
    val_size = int(Config.VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Crear DataLoaders
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
    print(f"Tamaño entrenamiento: {len(train_dataset)}, validación: {len(val_dataset)}, prueba: {len(test_dataset)}")
    
    # Configurar modelo
    model = FishCNN(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    
    # Configurar entrenamiento
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Entrenar y obtener métricas
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=Config.DEVICE,
        num_epochs=Config.NUM_EPOCHS
    )
    
    # Evaluar en conjunto de prueba
    evaluate_model(model, test_loader, Config.DEVICE)
    
    # Graficar métricas
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Guardar modelo final
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"Modelo final guardado en {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()