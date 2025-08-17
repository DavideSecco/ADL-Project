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
creare la cartella 'data'.<br>
all'interno della cartella 'data', creare la cartella 'imagenet'.<br>
dentro la cartella 'imagenet', creare le cartelle 'meta' e 'train'. <br>
scaricare il dataset (solo train.X1) dal seguente link: https://www.kaggle.com/datasets/ambityga/imagenet100?select=train.X1<br>
una volta scaricato, spostare il contenuto della cartella 'train.X1' (ovvero le varie sottocartelle) nella cartella 'data/imagenet/train'.<br>
successivamente, runnare il flie 'prepare_imagenet100.py'. questo creerà un file train.txt (se non esiste).<br>
la struttura finale della cartella dovrà essere la seguente:

data<br>
--imagenet<br>
----meta<br>
------train.txt<br>
----train<br>
------n01440764<br>
--------n01440764_18.JPEG<br>
--------...<br>
------...<br>

## pre-training
una volta installato il dataset, runnare il seguente comando:
```bash
bash tools/dist_train.sh configs/selfsup/densecl/densecl_imagenet_200ep.py 1
```
