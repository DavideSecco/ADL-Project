#!/bin/bash
#SBATCH --job-name=pretrain_DeCUR
#SBATCH --account=eu-25-19
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --error=logs/DeCUR_pretrain.err
#SBATCH --output=logs/DeCUR_pretrain.out

echo "[INFO] Starting DeCUR pretraining..."

# Carica moduli necessari (Python, CUDA se richiesto)
module purge
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.7.0

cd DeCUR

echo pwd
pwd

# Attiva virtualenv
source venv/bin/activate

# Verifica che il virtualenv sia attivo
echo "[INFO] VIRTUAL_ENV: $VIRTUAL_ENV"
which python
which pip

# Verifica configurazione
echo "[INFO] Python version:"
python --version
echo "[INFO] CUDA version:"
nvcc --version || echo "[WARN] nvcc not found"

# Imposta variabili per distributed training (anche se single GPU)
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Esegui il pretraining
python src/pretrain/pretrain_mm.py \
    --dataset SSL4EO \
    --method DeCUR \
    --data1 /path/to/modality1 \
    --data2 /path/to/modality2 \
    --mode MODAL1 MODAL2

echo "[INFO] DeCUR pretraining completed!"
