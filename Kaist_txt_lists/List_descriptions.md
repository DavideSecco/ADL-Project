## Introduzione
Ogni lista contiene, riga per riga, il percorso relativo a un singolo campione.

Esempio:
```
set00/V000/I00000
set00/V000/I00001
set00/V000/I00002
...
```

## Informazioni sul dataset completo
Il dataset è composto da circa 95.324 coppie di immagini visible + lwir, suddivise in 12 insiemi denominati set.
I set sono numerati in ordine crescente, dal Set00 fino al Set11.

All’interno di ciascun set, i dati sono ulteriormente organizzati in video, identificati dalla sigla V*. La numerazione dei video parte sempre da V000 e prosegue fino a coprire tutti i video di quel set.
Poiché ogni set è indipendente dagli altri, è possibile trovare lo stesso identificativo di video (ad esempio V000) in più set diversi.

# Descrizione liste

### 1) Trainig_split_50
Questa lista rappresenta circa il 53% delle coppie del dataset originale, per un totale di 50.184 elementi.
I campioni inclusi provengono in maniera contigua dai set Set00 fino a Set05 (incluso).

È importante sottolineare che questa suddivisione deriva direttamente dallo split ufficiale training-test del dataset.

### 2) Test_split_50
Questa lista è complementare alla precedente e contiene il restante 47% delle coppie, pari a 45.140 elementi.
Gli elementi provengono in sequenza dai set Set06 fino a Set11 (incluso), senza alcuna sovrapposizione con la parte di training sopra descritta.

Come già accennato, questo file rappresenta lo split ufficiale per la fase di testing sul dataset ed è utilizzato per garantire confronti standardizzati.

N.B. Nel nostro caso, useremo questa suddivisione per valutare le performance finali durante la fase di object detection.

### 3) Training_split_25_forSSL
Questa lista è stata generata a partire da training_split_50, separando i blocchi di immagini in base all’indice Vxxx.
In particolare, i blocchi vengono assegnati alternativamente a questo file e al file gemello: il primo blocco (V000) confluisce qui, il successivo (V001) nell’altro file, e così via, continuando in modo sequenziale anche al cambio di set.

Esempio: 
```
set00/V001/I00000
set00/V001/I00001
set00/V001/I00002
...
set00/V003/I00000
set00/V003/I00001
...
...
```
In totale, la lista è composta da 23 349 coppie, pari a circa il 50% di training_split_50 e quindi al 25% dell’intero dataset originale.

N.B. Questa lista verrà utilizzata per eseguire il pretraining Self-Supervised dei diversi modelli.

### 4) Training_split_25_forObjDet
Questa lista rappresenta la parte complementare della precedente.
Anch’essa è stata generata dividendo i blocchi Vxxx in modo alternato, continuando in modo sequenziale anche al cambio di set.

Esempio: 
```
set00/V000/I00000
set00/V000/I00001
set00/V000/I00002
...
set00/V002/I00000
set00/V002/I00001
...
...
```
In totale, la lista è composta da 26 835 coppie, pari a circa il 50% di training_split_50 e quindi al 25% dell’intero dataset originale.

N.B. Questa lista verrà utilizzata per eseguire il training di Obj Detection dei diversi modelli.
