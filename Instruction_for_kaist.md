## scarica i file 
da https://www.kaggle.com/datasets/adlteam/kaist-dataset/data?select=set05.  (1GB)
 Non credo esista comando da bash con kaggle API che permetta di scaricare solo il set05 quindi devi passare dal link a mano.


 Una volta scaricato e unzippato avrai queste cartelle

- La cartella `visible/` contiene le immagini RGB.
- La cartella `lwir/` contiene le immagini termiche corrispondenti.

Struttura cosi, poi lo sposti dove vuoi
```text
ADL-Project/
│
├─ set05/
│   ├─ visible/
│   └─ lwir/
│
├─ kaist_pairing.py 
```

## Se i file non dovessero essere gia' in ordine
```bash
python3 -u kaist_pairing.py --dataset_path ./set05
```
e su --datasetpath metti il path, di default ti ho lasciato cosi nel caso lo metti nella root sotto ~ADL-Project/set05