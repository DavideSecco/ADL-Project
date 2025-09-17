#!/bin/bash
#SBATCH --job-name=pretrain_DeCUR
#SBATCH --account=eu-25-19
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/pretrain_decur.out
#SBATCH --error=logs/pretrain_decur.err


echo "[INFO] starting"



module load Python/3.9.6-GCCcore-11.2.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1


source venv/bin/activate


export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export CUDA_LAUNCH_BLOCKING=1


python /mnt/proj3/eu-25-19/davide_secco/ADL-Project/DenseDeCUR/main.py \
  --dataset KAIST \
  --method DeCUR \
  --densecl_stream thermal \
  --data-root /scratch/project/eu-25-19/kaist-cvpr15/images \
  --list-train /mnt/proj3/eu-25-19/davide_secco/ADL-Project/Kaist_txt_lists/Training_split_25_forSSL.txt \
  --batch-size 128 \
  --epochs 15 \
  --checkpoint-dir ./checkpoint/KAIST_from_lists \
  --cos \
  --learning-rate-weights 0.002 \
  --learning-rate-biases 0.00048 \
  --weight-decay 1e-4 \
  --lambd 0.0051 \
  --projector 8192-8192-8192 \
  --print-freq 20


  echo "[INFO] ending"
