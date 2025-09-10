# Todo

Quota su karolina: it4ifsusage
du -h --max-depth=1
du -sh * | sort -h 
for d in */; do echo "$d: $(find "$d" -type f | wc -l)"; done


#### Dataset 
- [ ] Dataset KAIST:
  - [X] Scaricare su Karolina [Davide]
  - [ ] Spltittare su Karolina [Marco/Davide]
- [ ] Dataset Sunrgbd
  - [X] Scaricare su Karolina
  - [ ] Splittare su Karolina [Marco/Davide]

#### Modelli

- [ ] Pretraining di Decur [Davide]
  - [X] Sunrgbd su Karolina [Davide]
    - [X] Sistema cvtorchvision
    - [X] Fatto partire non splittato
    - [X] Training (dataset non splittato) andato a buon fine
      - [X] Ottieni i checkpoint sul non splittato [Davide]
    - [?] Training (dataset splittato) andato a buon fine (Non necessario)
    
  - [X] KAIST in locale [Marco]
    - [ ] Sistema classe KaistDataset, creane una sola?
  - [X] KAIST su Karolina [Davide] 
    - [X] Fatto partire non splittato
    - [X] Training (dataset standard) andato a buon fine [Davide]
      - [X] Ottieni il checkpoint sul non splittato [Davide]
    - [X] Training (dataset .txt) andato a buon fine 
      - [X] Ottenuti i checkpoint
    - [X] Training (dataset .txt, ma solo 25%) andato a buon fine [Davide] - da aspettare che marco prepari i files .txt (?)
      - [X] Ottenuti i checkpoint Definitivi!
    - [X] Possibile sia necessario un cambio di formato checkpoint --> from pth to pt (compatibilità con objdet)

- [X] Pretraing di DenseCL [Daniele]
  - [X] Farlo partire in locale su Sunrdgd
  - [X] Farlo partire in locale su Kaist
  - [X] Farlo partire su Karolina su Kaist
  - [X] KAIST
    - [X] Modalitá 1
    - [X] Modalitá 2
- [X] Integrazione DenseCL e DeCur [Daniele]
  - [X] Farlo partire in locale su Sunrgbd
  - [X] Farlo partire in locale su Kaist
  - [X] Farlo partire su Karolina su Kaist
     
- [ ] Aspetti da considerare prima del pre-training
  - [ ] definire augmentations usate per KAIST
    - [ ] confrontare con quelle di DenseCL: ci sono differenze?possono essere integrate?
  - [ ] trovare dimensione comune embedding DeCUR
  
- [ ] far partire pre-training
  - [ ] DenseCL
  - [ ] DeCUR
  - [ ] DenseDeCUR

#### Evoluzione

- [ ] Implementare modello multimodale (Carlo deve consigliare) [Marco]
  - [ ] setup della repo in locale [Marco]
  - [X] Setup della repo su karolina [Davide]

  - [X] Portare a termine trainig parziale (set parziale di label - no pesi nostri)
    - [X] Start training con i pesi forniti   
    - [X] Traduzione parziale labels from xml to txt format
    - [X] Training completo senza errori 

  - [ ] Portare a termine trainig completo (set completo di label - pesi nostri)
    - [ ] Tastare ICafusion con i files .txt --> Alla fine ho fatto in un altro modo
    - [X] FORSE: modificare come salva i pesi decur
    - [X] Adattamento nomi layer della rete per compatibilità coi nostri pesi pretrainati (Carlo(?)) [Davide]
    - [X] Traduzione di tutte le label kaist  
    - [X] Start training con i pesi da DeCUR


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