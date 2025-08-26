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
Poi decideremo come dividere il dataset tra preTraining e per obj detec. Nel primo caso tutti i file annotations sono inutili.



## Note per far partire
Usare il solito script passando da pretrain_mm.py   e  in --data1 e --data2 inserire il path dove avete collocato le images del dataset! A differnza di sunrbd dove differenziavamo il path per rgb e depth, qui il codice fa da solo dalla ~images/

Esempio se avete questa struttura: 
```text
ADL-Project/
   ├── DeCur
   ├── kaist-cvpr15/ 
      └── images/
   ├── Installation_instruction_kaist
```
Dovrete inserire --data1  ~/ADL-Project/kaist-cvpr15/images e --data2 ~/ADL-Project/kaist-cvpr15/images

### Script di comodo 
Se avete mia stessa configurazione, da eseguire dentro DeCUR per trovare diretto src/pretrain 
```bash
RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
python src/pretrain/pretrain_mm.py \
  --dataset KAIST \
  --method DeCUR \
  --data1 ~/ADL-Project/kaist-cvpr15/images \
  --data2 ~/ADL-Project/kaist-cvpr15/images \
  --mode MODAL1 MODAL2 \
  --backbone resnet50 \
  --batch-size 8 \
  --workers 8 \
  --epochs 5 \
  --checkpoint-dir ./checkpoint/KAIST_test \
  --cos \
  --learning-rate-weights 0.002 \
  --learning-rate-biases 0.00048 \
  --weight-decay 1e-4 \
  --lambd 0.0051 \
  --projector 8192-8192-8192 \
  --print-freq 20
```

