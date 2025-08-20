#!/usr/bin/env bash
# UNA GPU locale con DDP

DATA_DIR="data"

# Iperparametri
EPOCHS=200             
BATCH=256              
WORKERS=32            
DIST_URL="tcp://127.0.0.1:29500"

python src/main.py "$DATA_DIR" \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --dist-url "$DIST_URL" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  -j "$WORKERS" \
  --aug-plus \
  --cos \
  --use-mixed-precision \
  --print-freq 10
