#!/bin/bash
#SBATCH --job-name=pretrain_DeCUR_A100
#SBATCH --account=eu-25-19
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --error=logs/DeCUR_Install.err
#SBATCH --output=logs/DeCUR_Install.out

echo "[INFO] Starting DeCUR pretraining on A100 with CUDA 11.7..."

module purge
module load Python/3.9.6-GCCcore-11.2.0
module load CUDA/11.7.0

cd DeCUR

# Step 1: crea (se serve) virtualenv
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtualenv with Python 3.9.6..."
    python -m venv venv
fi

source venv/bin/activate
which python              # deve essere .../venv/bin/python
python --version

# Step 2: installa PyTorch compatibile con CUDA 11.7 (sm_80)
echo "[INFO] Installing PyTorch + CUDA 11.7 (compatibile A100)..."
pip install --upgrade pip
pip uninstall -y torch torchvision torchaudio
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Step 3: installa dipendenze del progetto
echo "[INFO] Installing project dependencies..."
pip install -r requirements.txt --verbose
pip install numpy<2.0
pip install opencv-contrib-python<4.7
pip install tensorboard
pip install diffdist

# Step 4: debug CUDA
echo "[INFO] Checking PyTorch GPU support..."
python -c "import torch; print('CUDA:', torch.version.cuda); print('Device:', torch.cuda.get_device_name(0)); print('Available:', torch.cuda.is_available()); print('Capability:', torch.cuda.get_device_capability())"

# Step 5: set variabili per training distribuito
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Step 6: esegui pretraining
python /mnt/Volume/Mega/LaureaMagistrale/CorsiSemestre/A3S1/AdvancedDeepLearning/ADL-Project/DeCUR/src/pretrain/pretrain_mm.py \
  --dataset SSL4EO \
  --method DeCUR \
  --data1 /path/to/modality1 \
  --data2 /path/to/modality2 \
  --mode MODAL1 MODAL2

echo "[INFO] DeCUR pretraining completed."
