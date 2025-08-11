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
scaricare il dataset dal seguente link: https://drive.google.com/file/d/1XCVx-eyMJfvBgeIqbO9ZXG4qPgm1VliP/view?usp=sharing
successivamente, spostare la cartella appena scaricata in ADL-Project/DenseCL.