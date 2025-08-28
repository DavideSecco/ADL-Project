### Setta nuovo token 
Crea nuovo token in: https://www.kaggle.com/settings --> API --> Create new token
´´´bash
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
´´´
### Scaricare il dataset
´´´bash
kaggle datasets download -d larbii/thermal-images
´´´

### Se comando kaggle sconosciuto 
```bash 
pip install kaggle
``` 
e ripetere il comando di download.

Dopo aver scaricato e scompattato il dataset, la struttura delle cartelle è la seguente:
```text
kaist-cvpr15/
├── annotations-xml-new/ # Annotazioni originali (XML)
├── annotations-xml-new-revised/ # Annotazioni revisionate 
├── imageSets/ # File di testo con suddivisioni train/test 
└── images/ # Immagini (RGB e Termiche), divise in sequenze
   ├── set00/
   ├── set01/
   ├── ...
   └── set11/
```

