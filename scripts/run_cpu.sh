#!/usr/bin/env bash

DATA_DIR="data"
EPOCHS=1
BATCH=64
DIST_URL="tcp://127.0.0.1:29500"

python src/main.py $DATA_DIR \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --dist-url $DIST_URL \
  --epochs $EPOCHS \
  --batch-size $BATCH \
  --print-freq 10
