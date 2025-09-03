#!/bin/bash
#SBATCH --job-name=ICAFusion_train
#SBATCH --account=eu-25-19
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --error=logs/icafusion_train.err
#SBATCH --output=logs/icafusion_train.out

echo "[INFO] Starting ICAFusion training job..."

# === Moduli (A100 compatibile) ===
# module purge
# module load Python/3.9.6-GCCcore-11.2.0
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

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
print("Torch:", torch.__version__)
print("CUDA :", torch.version.cuda)
print("GPU  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

# === Parametri di training (personalizzabili) ===
WEIGHTS="yolov5l.pt"                                                # --weights (DA SETTARE! nostri hanno estensione sbagliata)
CFG="./models/transformer/yolov5_ResNet50_Transfusion_kaist.yaml"   # --cfg     (DA CONFERMARE)
DATA="./data/multispectral/kaist.yaml"                              # --data        
HYP="data/hyp.scratch.yaml"                                         # --hyp     (LASCIO DEFAULT, MA SICURO DA CAMBIARE)

EPOCHS=60
BATCH=128                                                           # totale (per tutte le GPU). Se 1 GPU, è per-GPU.
IMG_WIDTH=640                                                       # credo dimensione immagini kaist
IMG_HEIGHT=512
WORKERS=4                                                           # dataloader workers; puoi portarlo a 0 se il FS è lento

PROJECT="runs/train"                                                # NON HO CAPITO: credo non importante
RUN_NAME="icafusion_fkaist_trasfusion"                              # NON HO CAPITO: credo non importante
ENTITY=""           # --entity (W&B); lascia vuoto per disabilitare
DEVICE=""           # --device (vuoto=auto). Es: "0" oppure "0,1"

# === Variabili DDP (robuste anche a 1 GPU) ===
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=29503
# export WORLD_SIZE=${SLURM_GPUS:-1}
# export RANK=0
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
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
  --img-size "$IMG_WIDTH" "$IMG_HEIGHT"
  --workers "$WORKERS"
  --project "$PROJECT"
  --name "$RUN_NAME"
  --exist-ok
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
