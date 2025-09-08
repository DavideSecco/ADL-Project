#!/bin/bash
#SBATCH --job-name=DeCUR_pretrain_kaist_original
#SBATCH --account=eu-25-19
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --error=logs/DeCUR_pretrain_kaist_original.err
#SBATCH --output=logs/DeCUR_pretrain_kaist_original.out

echo "[INFO] Starting DeCUR pretraining..."

# Carica moduli necessari: Python + PyTorch compatibile A100
# module purge
module load Python/3.9.6-GCCcore-11.2.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

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

# PyTorch info
echo "[INFO] PyTorch version and GPU:"
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0))"

# Imposta variabili per distributed training (anche se single GPU)
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export CUDA_LAUNCH_BLOCKING=1

# Esegui il pretraining
python src/pretrain/pretrain_mm.py \
    --dataset KAIST \
    --method DeCUR \
    --data1 /scratch/project/eu-25-19/kaist-cvpr15/images \
    --data2 /scratch/project/eu-25-19/kaist-cvpr15/images \
    --mode rgb thermal \
    --checkpoint-dir ./checkpoint/KAIST_original \
    --epochs 15 \
    --batch-size 128 \
    --print-freq 2 \

# Epoche: Ridotto perchè per kaist non ne servono molte di più
# 128 <= Batch size < 256 

echo "[INFO] DeCUR pretraining completed!"
