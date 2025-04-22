# Clasificación de Peces en Pesca de Alta Mar

Proyecto para clasificar 9 especies de peces usando una CNN personalizada en PyTorch, basado en el dataset "A Large Scale Fish Dataset".

## Cita
@inproceedings{ulucan2020large,
  title={A Large-Scale Dataset for Fish Segmentation and Classification},
  author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
  booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages={1--5},
  year={2020},
  organization={IEEE}
}

## Instalación
1. Clona el repositorio.
2. Crea un entorno virtual: `python -m venv venv`
3. Activa el entorno: `venv\Scripts\activate` (Windows) o `source venv/bin/activate` (Linux/Mac)
4. Instala dependencias: `pip install -r requirements.txt`
5. Configura `kaggle.json` en `C:\Users\<TuUsuario>\.kaggle\`.

## Ejecución
Ejecuta el script principal:
```bash
python main.py