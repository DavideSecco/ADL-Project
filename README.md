# Todo

Quota su karolina: it4ifsusage
du -h --max-depth=1
du -sh * | sort -h 


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
    - [] Training (dataset .txt, ma solo 25%) andato a buon fine [Davide] - da aspettare che marco prepari i files .txt (?)
      - [] Ottenuti i checkpoint Definitivi!
    - [ ] Possibile sia necessario un cambio di formato checkpoint --> from pth to pt (compatibilità con objdet)

- [ ] Pretraing di DenseCL [Daniele]
  - [X] Farlo partire in locale su Sunrdgd
  - [ ] Farlo partire in locale su Kaist
  - [ ] Farlo partire su Karolina su Kaist
  - [ ] KAIST
    - [ ] Modalitá 1
    - [ ] Modalitá 2
- [ ] Integrazione DenseCL e DeCur [Daniele]
  - [ ] Farlo partire in locale su Sunrgbd
  - [ ] Farlo partire in locale su Kaist
  - [ ] Farlo partire su Karolina su Kaist

#### Evoluzione

- [ ] Implementare modello multimodale (Carlo deve consigliare) [Marco]
  - [ ] setup della repo in locale [Marco]
  - [X] Setup della repo su karolina [Davide]

  - [ ] Portare a termine trainig parziale (set parziale di label - no pesi nostri)
    - [X] Start training con i pesi forniti   
    - [X] Traduzione parziale labels from xml to txt format
    - [ ] Training completo senza errori 

  - Portare a termine trainig completo (set completo di label - pesi nostri)
    - [ ] FORSE: modificare come salva i pesi decur
    - [ ] Adattamento nomi layer della rete per compatibilità coi nostri pesi pretrainati (Carlo(?)) [Davide]
    - [ ] Traduzione di tutte le label kaist from  
    - [ ] Start training con i pesi da DeCUR