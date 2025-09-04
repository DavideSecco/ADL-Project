# Todo

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
  - [ ] Setup della repo su karolina [Davide] 
