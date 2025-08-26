#!/bin/bash
#SBATCH --job-name=extract_dataset
#SBATCH --account=eu-25-19
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --error=logs/extract_dataset.err
#SBATCH --output=logs/extract_dataset.out

echo "[INFO] Starting dataset extraction..."

# Carica eventuali moduli di utilità (se richiesto dal cluster)
# module purge
# module load tar
# module load unzip

# Definisci variabili
ARCHIVE=/mnt/proj3/eu-25-19/davide_secco/ADL-Project/kaist-thermal-images.zip
DEST=/mnt/proj3/eu-25-19/davide_secco/ADL-Project/kaist/

# Crea cartella destinazione se non esiste
mkdir -p $DEST

# Estrazione in base al tipo di archivio
if [[ $ARCHIVE == *.zip ]]; then
    echo "[INFO] Extracting ZIP..."
    unzip -oq $ARCHIVE -d $DEST
else
    echo "[ERROR] Unknown archive format: $ARCHIVE"
    exit 1
fi

echo "[INFO] Extraction completed!"
