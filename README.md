# Todo

Quota su karolina: it4ifsusage
du -h --max-depth=1
du -sh * | sort -h 
for d in */; do echo "$d: $(find "$d" -type f | wc -l)"; done


#### Dataset 
- [X] Dataset KAIST:
  - [X] Scaricare su Karolina [Davide]
  - [X] Spltittare su Karolina [Marco/Davide]
 
- [ ] Dataset Sunrgbd [Da splittare? Ci serve ancora?] Splittare su Karolina [Marco/Davide]

#### Modelli

- [ ] Pretraining di Decur [Davide]
  - [ ] Sistema classe KaistDataset, creane una sola? [Marco]
  - [X] Sunrgbd su Karolina [Davide]
  - [X] KAIST su Karolina [Davide]
      - [ ] Completa training di 150 epoche e parla con Carlo [Davide] - Ottieni i checkpoint definitivi
  
- [ ] Far partire pre-training
  - [ ] DenseCL
  - [X] DeCUR [Davide]
  - [ ] DenseDeCUR

#### Conversione checkpoint
to write...

#### ICAFUSION

- [X] Implementare modello multimodale ICAfusion [Marco]
  - [ ] setup della repo in locale [Marco]
  - [X] Setup della repo su karolina [Davide]

  - [ ] controlla score_thr per visualizzazione

  - [ ] Training da fare:
    - [ ] No pretrained weights:
      - [ ] 1 epoca - confronta metriche e loss con i pretrain [Davide]
      - [ ] full epoche - da definire il numero (20-40)
    - [ ] Pretrain weights (da decur)
      - [ ] 1 epoca - confronta metriche e loss con i no pretrain [Davide]
      - [ ] full epoche - da definire il numero (20-40)
    - [ ] Pretrain weights (da dendedecur)
      - [ ] 1 epoca - confronta metriche e loss con i no pretrain
      - [ ] full epoche - da definire il numero



#### Messaggio 
Secondo me vi potete dividere queste tre task, come dataset userei Kaist (o Sunrgb, perchè DSEC non è molto immediato). Dividete il dataset in due parti (50-50 o anche 70-30), una parte la usate solo per il pretraining e un'altra solo per detection (quindi contiente anche il test set questa).
- pretraining con densecl: prendete una resnet qualunque come backbone (resnet50 va benissimo), e fate due pretraining diversi per le due modalità.
- pretraining con decur: come prima solo un solo pretraining multimodale
- integrazione di densecl in decur

Una volta fatte queste cose (in particolare, le prime due), prendiamo un qualunque modello multimodale di object detection (ad esempio GAFF, MLPD o ProbEn, quello che è più facile da integrare, su questa parte vi do una mano io) e fine tuniamo il tutto confrontando i risultati solo con il training dello stesso modello sul solo split di detection (senza pre-training). L'ultimo step poi sarà fare il pretraining multimodale decur+densecl e il fine tuning

kaist/
    visible/ # qui sono da mettere le immagini rgb
        train/ # tutte le immagini di train
        test/ # tutte le immagini di test
    infrared/ # qui sono da mettere le immagini infrared
        train/
        test/
    labels/ # qui sono da mettere le label
        train/
        test/






