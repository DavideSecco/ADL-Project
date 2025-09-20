#!/bin/bash
#SBATCH --job-name=ICAFusion_DDP
#SBATCH --account=eu-25-19
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus 2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --error=logs/icafusion_ddp.err
#SBATCH --output=logs/icafusion_ddp.out

set -euo pipefail

echo "[INFO] Starting DDP job"; nvidia-smi || true

source /mnt/proj3/eu-25-19/davide_secco/miniconda3/etc/profile.d/conda.sh
conda activate icafusion

cd /mnt/proj3/eu-25-19/davide_secco/ADL-Project/ICAFusion
pwd

# Env utili per debug/robustezza
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export WORLD_SIZE=2
# Se serve su questo cluster:
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0

WEIGHTS="/mnt/proj3/eu-25-19/davide_secco/ADL-Project/ICAFusion/checkpoint_0014_REWIRED_first636.pth"
CFG="./models/transformer/yolov5_ResNet50_Transfusion_kaist.yaml"
DATA="./data/multispectral/kaist-karolina-scratch-kaist-x-icafusion-v1.yaml"
HYP="data/hyp.scratch.yaml"

EPOCHS=3
TOTAL_BATCH=28        # batch TOTALE: verrà diviso per #GPU
IMG_TRAIN=640
IMG_TEST=640
WORKERS=0
PROJECT="runs/train"
RUN_NAME="icafusion_ddp"


# ⚠️ NON passare --device "0,1" in DDP
set -x
torchrun --nproc_per_node=2 --standalone --tee 3 \
  train.py \
  --weights "${WEIGHTS}" \
  --cfg "${CFG}" \
  --data "${DATA}" \
  --hyp "${HYP}" \
  --epochs "${EPOCHS}" \
  --batch-size "${TOTAL_BATCH}" \
  --img-size "${IMG_TRAIN}" "${IMG_TEST}" \
  --workers "${WORKERS}" \
  --project "${PROJECT}" \
  --name "${RUN_NAME}" \
  --exist-ok \
  --notest
