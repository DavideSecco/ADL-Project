#!/bin/bash
#SBATCH --job-name=icafusion_train_pretrain_single_gpu
#SBATCH --account=eu-25-19
#SBATCH --partition=qgpu
#SBATCH --gpus  1
#SBATCH --time=24:00:00
#SBATCH --error=logs/icafusion_train_pretrain_single_gpu.err
#SBATCH --output=logs/icafusion_train_pretrain_single_gpu.out

############ IMPORTANTE ###############################
## AUMENTARE IL NUMERO DI GPU/ALZARE IL NUMERO DI WORKERS 
## FA BLOCCARE IL TRAINING
#######################################################

echo "[INFO] Starting ICAFusion training job..." 

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" 

nvidia-smi

# === Directory del progetto ===
# Cambia questa riga con la root del repo di ICAFusion (o del fork YOLOv5-TransFusion)
cd ICAFusion

echo "[INFO] Working directory:"
pwd

# === Attiva virtualenv ===
# Assumo che tu abbia già creato 
source /mnt/proj3/eu-25-19/davide_secco/miniconda3/etc/profile.d/conda.sh
conda activate icafusion

# === Versioni per debug ===
echo "[INFO] Python version:"
python --version
echo "[INFO] PyTorch & CUDA:"
python - <<'PY'
import torch
print("Torch:", torch.__version__, flush=True)
print("CUDA :", torch.version.cuda, flush=True)
print("GPU  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
PY

# === Parametri di training (personalizzabili) ===
# WEIGHTS="yolov5l.pt"                                                # --weights (DA SETTARE! nostri hanno estensione sbagliata)
WEIGHTS="/mnt/proj3/eu-25-19/davide_secco/ADL-Project/ICAFusion/checkpoint_0014_REWIRED_first636.pth"            # --weights (DA SETTARE! nostri hanno estensione sbagliata)
CFG="./models/transformer/yolov5_ResNet50_Transfusion_kaist.yaml"   # --cfg     (DA CONFERMARE)
DATA="./data/multispectral/kaist-karolina-scratch-kaist-x-icafusion-v1.yaml"   
# DATA="./data/multispectral/kaist-karolina-scratch-small.yaml"                          
HYP="data/hyp.scratch.yaml"                                         # --hyp     (LASCIO DEFAULT, MA SICURO DA CAMBIARE)

EPOCHS=1
BATCH=28                                                           # 16 <= optimal value < 64 (probabilemente 16 è ok) (16 --> 20 min a epoca) (22 --> 18 min a epoca)
IMG_TRAIN=640                                                       # credo dimensione immagini kaist
IMG_TEST=640
WORKERS=0                                                          # dataloader workers; puoi portarlo a 0 se il FS è lento

PROJECT="runs/train"                                                # NON HO CAPITO: credo non importante
RUN_NAME="icafusion_fkaist_pretrained_weights"                              # NON HO CAPITO: credo non importante
ENTITY=""           # --entity (W&B); lascia vuoto per disabilitare
DEVICE=""           # --device (vuoto=auto). Es: "0" oppure "0,1"

# === Variabili DDP (robuste anche a 1 GPU) ===
# export MASTER_ADDR=127.0.0.1
export MASTER_ADDR=$(srun --ntasks=1 hostname | tail -n1)
export MASTER_PORT=29500
export WORLD_SIZE=1
# export RANK=${SLURM_PROCID}
# export LOCAL_RANK=${SLURM_LOCALID}
# unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
export CUDA_LAUNCH_BLOCKING=1

# === Comando di training ===
# Sostituisci "train.py" col path esatto del file principale che definisce l'argparse incollato
TRAIN_SCRIPT="train.py"

CMD=(
  python "$TRAIN_SCRIPT"
  --weights "$WEIGHTS"
  --cfg "$CFG"
  --data "$DATA"
  --hyp "$HYP"
  --epochs "$EPOCHS"
  --batch-size "$BATCH"
  --img-size "$IMG_TRAIN" "$IMG_TEST"
  --workers "$WORKERS"
  --project "$PROJECT"
  --name "$RUN_NAME"
  --exist-ok
  --notest
)

# Opzionali: abilita/disabilita aggiungendo i flag desiderati (scommenta se utili)
# CMD+=(--cache-images)
# CMD+=(--image-weights)
# CMD+=(--rect)
# CMD+=(--noautoanchor)
# CMD+=(--linear-lr)
# CMD+=(--label-smoothing 0.05)
# CMD+=(--save_period 10)
# CMD+=(--upload_dataset)
# CMD+=(--bbox_interval 5)
# CMD+=(--adam)
# CMD+=(--quad)
# CMD+=(--single-cls)

# Device (se specificato)
if [[ -n "$DEVICE" ]]; then
  CMD+=(--device "$DEVICE")
fi

# W&B entity (se usi Weights & Biases)
if [[ -n "$ENTITY" ]]; then
  CMD+=(--entity "$ENTITY")
fi

echo "[INFO] Launch command:"
printf '%q ' "${CMD[@]}"; echo

# === Avvio training ===
"${CMD[@]}"

echo "[INFO] ICAFusion training completed!"
