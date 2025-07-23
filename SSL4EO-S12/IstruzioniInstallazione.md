# Installazione:
Attiva ambiente thesis (valido per davide) e aggiungi
```bash
pip install lmdb
pip install opencv-python
```

# Script da runnare (?)
```bash
python bigearthnet_dataset_seco.py --data_dir=/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A3S1/AdvancedDeepLearning/ADL-Project/ssl4eo-s12_100patches --save_dir=/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A3S1/AdvancedDeepLearning/ADL-Project/SSL_lmdb_dataset --make_lmdb_dataset=True

python bigearthnet_dataset_seco.py --data_dir=/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A3S1/AdvancedDeepLearning/ADL-Project/ssl4eo-s12_100patches --save_dir=/mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A3S1/AdvancedDeepLearning/ADL-Project/SSL_lmdb_dataset --make_lmdb_dataset=True --download=True
```

Problema con il dataset da runnare. Non riesco a:
- [ ] Scricare i file .txt richiesti
        - Ho provato a scaricare quelli qui: https://github.com/zhu-xlab/DINO-MM/tree/main/datasets/BigEarthNet ma non sono accessibili
- [ ] Comunque le cartelle non sembrano avere la stessa struttura del dataset di sample che ho trovato