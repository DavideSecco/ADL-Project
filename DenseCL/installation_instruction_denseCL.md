## Preparazione env x denseCL
per prima cosa, creare ambiente virtuale con conda ed attivarlo:
```bash
conda create -n densecl python=3.7 -y
conda activate densecl
```
successivamente, installare PyTorch e torchvision:
```bash
conda install pytorch torchvision -c pytorch
```
a questo punto, clonare la nostra repository ed entrare nella cartella DenseCL:
```bash
git clone https://github.com/DavideSecco/ADL-Project.git
cd ADL-Project/DenseCL
```
ora, installare le dipendenze:
```bash
pip install -v -e .
```

## installazione dataset
- creare le cartelle 'data/imagenet/train' e 'data/imagenet/meta'
- scaricare il dataset (solo train.X1) dal seguente link: https://www.kaggle.com/datasets/ambityga/imagenet100?select=train.X1
- una volta scaricato, spostare il contenuto della cartella 'train.X1' (ovvero le varie sottocartelle) nella cartella 'data/imagenet/train'
- successivamente, runnare il flie 'prepare_imagenet100.py'. questo creerà un file train.txt (se non esiste).
  
```bash
cd DenseCL
# Creazione delle cartelle
mkdir -p data/imagenet/{meta,train}

# Spostarsi nella cartella di destinazione
cd data/imagenet

# Scaricare il dataset da Kaggle 
# https://www.kaggle.com/datasets/ambityga/imagenet100?select=train.X1 
# dentro data/imagenet

# Estrarre lo zip in data/imagenet/train
unzip train.X1.zip -d data/imagenet/train

# Rimuovere 
rmdir train.X1.zip

# Tornare alla root del progetto
cd ../..

# Lanciare lo script di preparazione
python prepare_imagenet100.py
```

la struttura finale della cartella dovrà essere la seguente:
```text
data/
└── imagenet/
    ├── meta/
    │   └── train.txt
    └── train/
        ├── n01440764/
        │   ├── n01440764_18.JPEG
        │   ├── n01440764_36.JPEG
        │   └── ...
        ├── n01443537/
        │   ├── n01443537_0.JPEG
        │   ├── n01443537_11.JPEG
        │   └── ...
        ├── n01484850/
        │   └── ...
        └── ...
```

## Pre-training
una volta installato il dataset, runnare il seguente comando:
```bash
bash tools/dist_train.sh configs/selfsup/densecl/densecl_imagenet_200ep.py 1
```
