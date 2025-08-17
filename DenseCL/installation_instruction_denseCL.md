## installazione denseCL
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
creare la cartella 'data'.
all'interno della cartella 'data', creare la cartella 'imagenet'.
dentro la cartella 'imagenet', creare le cartelle 'meta' e 'train'. 
scaricare il dataset dal seguente link: https://www.kaggle.com/datasets/ambityga/imagenet100?select=train.X1
una volta scaricato, spostare il contenuto della cartella 'train.X1' (ovvero le varie sottocartelle) nella cartella 'data/imagenet/train'.
successivamente, runnare il flie 'prepare_imagenet100.py'. questo creerà un file train.txt (se non esiste).
la struttura finale della cartella dovrà essere la seguente:

data
  imagenet
    meta
      train.txt
    train
      n01440764
        n01440764_18.JPEG
        ...
      ...

## pre-training
una volta installato il dataset, runnare il seguente comando:
```bash
bash tools/dist_train.sh configs/selfsup/densecl/densecl_imagenet_200ep.py 1
```
