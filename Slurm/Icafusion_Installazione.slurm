#!/bin/bash -l
#SBATCH --job-name=Icafusion_env_setup
#SBATCH --account=eu-25-19
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/icafusion_env_setup.out
#SBATCH --error=logs/icafusion_env_setup.err

# rimuovere env vecchio: conda env remove -n icafusion

# ===== Conda bootstrap =====
CONDA_BASE="/mnt/proj3/eu-25-19/davide_secco/miniconda3"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
else
  # fallback (molto raro)
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook)"
fi

# ===== Crea env se non esiste (con Python) =====
if [ ! -d "$CONDA_BASE/envs/icafusion" ]; then
  echo "[INFO] Creating conda env 'icafusion' with Python..."
  conda create -y -n icafusion python=3.10
else
  echo "[INFO] Conda env 'icafusion' already exists."
  if [ ! -x "$CONDA_BASE/envs/icafusion/bin/python" ]; then
    echo "[INFO] Installing Python into existing env..."
    conda install -y -n icafusion python=3.10
  fi
fi

# ===== Attiva env =====
conda activate icafusion
hash -r

echo "[INFO] Python:"
python --version || { echo "[ERROR] Python non disponibile nell'env"; exit 1; }
echo "[INFO] Pip:"
pip --version || true

# ===== Installazione requirements =====
# (segue le tue istruzioni: cd ICAFusion && pip install -r requirements.txt)
if [ -d "ICAFusion" ]; then
  cd ICAFusion
else
  echo "[WARN] Directory 'ICAFusion' non trovata in $(pwd). Proseguo comunque."
fi

python -m pip install --upgrade pip
if [ -f "requirements.txt" ]; then
  echo "[INFO] Installing requirements from $(pwd)/requirements.txt"
  python -m pip install -r requirements.txt
else
  echo "[WARN] requirements.txt non trovato: salto l'installazione."
fi

# Pacchetti aggiuntivi:
pip install requests, einops, timm

# ===== Report finale =====
echo "[INFO] conda list python:"
conda list python || true

echo "[INFO] Done."
