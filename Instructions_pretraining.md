## KAIST DATASET
Per il download del dataset si rimanda al file installation_instruction_kaist.md.
In questa documentazione si assume che l’utente abbia già scaricato il dataset e lo abbia collocato all’interno di una cartella denominata kaist-cvpr15 (la posizione può essere arbitraria, purché se ne conosca il path completo).
## PreTraining note in generale 
Per tutte le fasi di pretraining è stato scelto di utilizzare file testuali contenenti l’elenco dei sample (in forma di path locale relativo) che definiscono la split da adottare.

Questa scelta consente diversi vantaggi:

- Flessibilità: i file .txt possono essere facilmente modificati o creati ex-novo, permettendo di gestire split personalizzate senza intervenire sul codice.

- Stabilità: la struttura originale del dataset (kaist-cvpr15) rimane invariata; le liste di sample agiscono come “viste” sul dataset reale.

- Portabilità: il dataset non viene tracciato dalla repository Git, mentre le liste .txt sì, garantendo così piena replicabilità degli esperimenti.

Tutti i file di split utilizzati o creati saranno collocati in: ~/ADL-Project/Kaist_txt_lists


## DeCUR Pretraining
Per il pretraining di DeCUR servira': 
> Conoscere il path assoluto del dataset, fino alla cartella /images inclusa, da collocare in --data-root.

Esempio: --data-root ~/ADL-Project/kaist-cvpr15/images


> Conoscere quale file testuale (lista di sample) far partire, da collocare in --list-train 

Esempio: --list-train ~/ADL-Project/Kaist_txt_lists/train-all-01.txt


Lo script da eseguire all'interno di  ~ADL-Project/DeCUR   sara': 
```bash
RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
python src/pretrain/pretrain_mm.py \
  --dataset KAIST \
  --method DeCUR \
  --data-root ~/ADL-Project/kaist-cvpr15/images \
  --list-train ~/ADL-Project/Kaist_txt_lists/train-all-01.txt \
  --mode rgb thermal \
  --backbone resnet50 \
  --batch-size 8 \
  --workers 8 \
  --epochs 5 \
  --checkpoint-dir ./checkpoint/KAIST_from_lists \
  --cos \
  --learning-rate-weights 0.002 \
  --learning-rate-biases 0.00048 \
  --weight-decay 1e-4 \
  --lambd 0.0051 \
  --projector 8192-8192-8192 \
  --print-freq 20
```

NOTA: ricordarsi di attivare prima l'env corretto per DeCUR!

