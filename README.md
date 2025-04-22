# Fish Classification in High-Seas Fishing 

A project to classify 9 fish species using a custom CNN in PyTorch, based on the "A Large Scale Fish Dataset". 

## Citation 

``` markdown
@inproceedings{ulucan2020large, 
  title={A Large-Scale Dataset for Fish Segmentation and Classification}, 
  author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
  booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)}, 
  pages={1--5}, 
  year={2020}, 
  organization={IEEE} 
} 
```

## Installation 

1. Clone the repository: ```bash git clone cd ``` 

2. Create the Conda environment using the `environment.yml` file: ```bash conda env create -f environment.yml ``` 

3. Activate the environment: ```bash conda activate fish_classification ``` 

4. Configure `kaggle.json` in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\\.kaggle\kaggle.json` (Windows). 

## Execution Run the main script: 

```bash python main.py ```