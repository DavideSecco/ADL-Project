###DeCUR training LOCALE
1) attiva env
```bash
source decurEnv/bin/activate
cd DeCUR
```


Per il training di DeCUR servira': 
> Conoscere il path assoluto del dataset, fino alla cartella /images inclusa, da collocare in --data-root.

Esempio: --data-root ~/ADL-Project/kaist-cvpr15/images


> Conoscere quale file testuale (lista di sample) far partire, da collocare in --list-train 

Esempio: 

--list-train ~/ADL-Project/Kaist_txt_lists/Training_split_25_forSSL.txt

Versione small del training dataset per test rapidi
--list-train ~/ADL-Project/Kaist_txt_lists/Training_onlyOn_Set05.txt    

> Locazione salvataggio + nome file del checkpoint   path relativo dentro DeCUR
--checkpoint-dir  ./checkpoint/OfficialDeCUR_from_lists

> Epoche e batch size controllare valori al bisogno

> Fondamentale presenza righe iniziali per settare running enviroment corretto con una sola GPU e non distribuito

Lo script da eseguire all'interno di  ~ADL-Project/DeCUR  sara': 
```bash
RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
python src/pretrain/pretrain_mm.py \
  --dataset KAIST \
  --method DeCUR \
  --data-root ~/ADL-Project/kaist-cvpr15/images \
  --list-train ~/ADL-Project/Kaist_txt_lists/Training_onlyOn_Set05.txt \
  --mode rgb thermal \
  --backbone resnet50 \
  --batch-size 8 \
  --workers 8 \
  --epochs 1 \
  --checkpoint-dir ./checkpoint/OfficialDeCUR_from_lists \
  --cos \
  --learning-rate-weights 0.002 \
  --learning-rate-biases 0.00048 \
  --weight-decay 1e-4 \
  --lambd 0.0051 \
  --projector 8192-8192-8192 \
  --print-freq 20
```