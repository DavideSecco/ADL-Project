#!/bin/bash
# Unico script: sottomette 1 job per modello (con sbatch) e contiene il body del job.
# Log e job-name = RUN_TAG = "<model>_<dataVer>_epochs_<EPOCHS>_%j"

set -euo pipefail

# ===== Parametri GLOBALI (validi per tutti i modelli) =====
export EPOCHS=12
export PER_GPU=28
export IMG_TRAIN=640
export IMG_TEST=640
export WORKERS=0

# Numero GPU e nodi:
export NGPUS=4
export NNODES=1

# Se ti serve il world size lato shell (NON esportarlo, ci pensa torchrun per i processi)
export WORLD_SIZE=$(( NGPUS * NNODES ))

# Config comune
export CFG="./models/transformer/yolov5_ResNet50_Transfusion_kaist.yaml"
# export DATA="./data/multispectral/kaist-karolina-scratch-kaist-x-icafusion-v1.0-small.yaml"
export DATA="./data/multispectral/kaist-karolina-scratch-kaist-x-icafusion-v1.yaml"
export HYP="data/hyp.scratch.yaml"
export PROJECT="runs/train"

# Opzioni invio
PARALLEL=0       # 1 per invio parallelo
MAX_JOBS=4       # limite di invii concorrenti se PARALLEL=1
DRY_RUN=0

# ===== Lista dei MODELLI =====
MODELS=(
  no_pretrained
  # decur_original
  decur
  densecl
  decurdensecl
)

mkdir -p logs

# ===== Estrazione versione dataset (es. v1.0-small / v1.0 / ecc.) =====
DATA_BASENAME=$(basename "$DATA" .yaml)
case "$DATA_BASENAME" in
  *kaist-karolina-scratch-kaist-x-icafusion-v1.0-small*)  DATA_VERSION="v1.0-small" ;;
  *kaist-karolina-scratch-kaist-x-icafusion-v1*)          DATA_VERSION="v1.0-compl" ;;
esac

# variante “pulita” (punti -> underscore, rimuovi prefissi lunghi opzionali)
DATA_VERSION_SHORT=$(echo "$DATA_VERSION" | sed -E 's/icafusion-//g; s/\./_/g')

wait_for_slots() {
  local max=$1
  while (( $(jobs -pr | wc -l) >= max )); do sleep 1; done
}

for MODEL in "${MODELS[@]}"; do
  # RUN_TAG con %j (jobid) per allineare nome log = RUN_TAG
  RUN_TAG_PATTERN="gpu-${WORLD_SIZE}_dataset-${DATA_VERSION_SHORT}_epochs-${EPOCHS}_${MODEL}"

  JOB_NAME="${RUN_TAG_PATTERN}"        # job-name = RUN_TAG
  OUT_PATH="logs/${RUN_TAG_PATTERN}.out"
  ERR_PATH="logs/${RUN_TAG_PATTERN}.err"

  echo "[LAUNCH] sbatch --job-name='${JOB_NAME}' --output='${OUT_PATH}' --error='${ERR_PATH}' (MODEL=${MODEL})"

  if (( DRY_RUN )); then
    continue
  fi

  if (( PARALLEL )); then
    wait_for_slots "$MAX_JOBS"
  fi

  # Esporta variabili globali e specifiche che userà il body del job
  # Nota: inviamo lo script su STDIN (qui-doc). Le option --job-name/--output/--error sono impostate fuori.
  sbatch \
    --partition=qgpu_free \
    --account=eu-25-19 \
    --nodes=${NNODES} \
    --gpus-per-node=${NGPUS} \
    --cpus-per-task=8 \
    --time=18:00:00 \
    --job-name="${JOB_NAME}" \
    --output="${OUT_PATH}" \
    --error="${ERR_PATH}" \
    --export=ALL,MODEL="${MODEL}",EPOCHS="${EPOCHS}",PER_GPU="${PER_GPU}",IMG_TRAIN="${IMG_TRAIN}",IMG_TEST="${IMG_TEST}",WORKERS="${WORKERS}",CFG="${CFG}",DATA="${DATA}",HYP="${HYP}",PROJECT="${PROJECT}",RUN_TAG_PATTERN="${RUN_TAG_PATTERN}" \
    <<'SLURM_BODY'
#!/bin/bash
set -euo pipefail

# ===== Variabili passate dall'esterno =====
: "${MODEL:?MODEL not set}"
: "${RUN_TAG_PATTERN:?RUN_TAG_PATTERN not set}"

# Risolvi RUN_TAG sostituendo %j con SLURM_JOB_ID
RUN_TAG="${RUN_TAG_PATTERN//%j/${SLURM_JOB_ID:-NA}}"

echo "[INFO] MODEL=${MODEL}"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}  SLURM_JOB_NAME=${SLURM_JOB_NAME:-NA}"
echo "[INFO] RUN_TAG=${RUN_TAG}  (log files coincidono con RUN_TAG)"

# ===== GPU info =====
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true

# ===== Repo & env =====
cd ICAFusion
source /mnt/proj3/eu-25-19/davide_secco/miniconda3/etc/profile.d/conda.sh
conda activate icafusion

echo "[INFO] Python version:"; python --version
python - <<'PY'
import torch
print("Torch:", torch.__version__, flush=True)
print("CUDA :", torch.version.cuda, flush=True)
print("GPU  :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
PY

# ===== Env utili torch/NCCL =====
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ===== Deriva da SLURM =====
NGPUS=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}
NNODES=${SLURM_JOB_NUM_NODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
WORLD_SIZE=$(( NGPUS * NNODES ))

# ===== Parametri GLOBALI (con fallback, già esportati) =====
EPOCHS="${EPOCHS:-1}"
PER_GPU="${PER_GPU:-28}"
IMG_TRAIN="${IMG_TRAIN:-640}"
IMG_TEST="${IMG_TEST:-640}"
WORKERS="${WORKERS:-128}"
CFG="${CFG:-./models/transformer/yolov5_ResNet50_Transfusion_kaist.yaml}"
DATA="${DATA:-./data/multispectral/kaist-karolina-scratch-kaist-x-icafusion-v1.0-small.yaml}"
HYP="${HYP:-data/hyp.scratch.yaml}"
PROJECT="${PROJECT:-runs/train}"

TOTAL_BATCH=$(( PER_GPU * WORLD_SIZE ))

# ===== Mappa pesi per modello =====
WEIGHTS=""
case "$MODEL" in
  no_pretrained)
    WEIGHTS=""   # nessun weight
    ;;
  decur_original)
    WEIGHTS="/mnt/proj3/eu-25-19/davide_secco/ADL-Project/ICAFusion/final_checkpoints/icafusion_from_decur_original.pth"
    ;;
  decur)
    WEIGHTS="/mnt/proj3/eu-25-19/davide_secco/ADL-Project/ICAFusion/final_checkpoints/icafusion_from_decur.pth"
    ;;
  densecl)
    WEIGHTS="/mnt/proj3/eu-25-19/davide_secco/ADL-Project/ICAFusion/final_checkpoints/icafusion_from_denseCL.pth"
    ;;
  decurdensecl)
    WEIGHTS="/mnt/proj3/eu-25-19/davide_secco/ADL-Project/ICAFusion/final_checkpoints/icafusion_from_denseDecur.pth"
    ;;
  *)
    echo "[ERROR] Modello non riconosciuto: $MODEL"; exit 2;;
esac

# ===== RUN_NAME (directory di run) =====
RUN_NAME="${RUN_TAG}"

echo "[INFO] NGPUS=$NGPUS NNODES=$NNODES WORLD_SIZE=$WORLD_SIZE TOTAL_BATCH=$TOTAL_BATCH"
echo "[INFO] EPOCHS=$EPOCHS PER_GPU=$PER_GPU IMG=[$IMG_TRAIN,$IMG_TEST] WORKERS=$WORKERS"
echo "[INFO] CFG=$CFG"; echo "[INFO] DATA=$DATA"; echo "[INFO] HYP=$HYP"
echo "[INFO] PROJECT=$PROJECT  RUN_NAME=$RUN_NAME"
echo "[INFO] WEIGHTS='${WEIGHTS:-<none>}'"
echo "[INFO] Log files: logs/${RUN_TAG}.out / logs/${RUN_TAG}.err"

# ===== Costruzione argomenti train.py =====
COMMON_ARGS=(
  --cfg "$CFG"
  --data "$DATA"
  --hyp "$HYP"
  --epochs "$EPOCHS"
  --batch-size "$TOTAL_BATCH"
  --img-size "$IMG_TRAIN" "$IMG_TEST"
  --workers "$WORKERS"
  --project "$PROJECT"
  --name "$RUN_NAME"
  --exist-ok
  --notest
)
if [[ -n "$WEIGHTS" ]]; then
  COMMON_ARGS=( --weights "$WEIGHTS" "${COMMON_ARGS[@]}" )
else
  echo "[INFO] Avvio SENZA --weights (no_pretrained)."
fi

set -x
if [ "$NNODES" -eq 1 ]; then
  torchrun --nproc_per_node="$NGPUS" --standalone --tee 3 \
    train.py "${COMMON_ARGS[@]}"
else
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
  MASTER_PORT=${MASTER_PORT:-29500}
  torchrun \
    --nproc_per_node="$NGPUS" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --tee 3 \
    train.py "${COMMON_ARGS[@]}"
fi
SLURM_BODY

  if (( PARALLEL )); then
    :
  else
    sleep 1
  fi
done

if (( PARALLEL )); then wait; fi
echo "[DONE] Tutti i job sono stati inviati."
