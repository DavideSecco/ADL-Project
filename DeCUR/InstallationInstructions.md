# Setup
Preliminary on linux: install python3.9
(nel mio caso si da per scontato che pyenv sia installato e configurato nella shell)
```bash
pyenv install 3.9.18
pyenv local 3.9.18      # Setta la versione di python
python --version        # Deve mostrare Python 3.9.18
which python            # Deve puntare a ~/.pyenv/versions/3.9.18/bin/python
```

Create env:
```bash
python -m venv venv 
```

Activate env:
```bash
source venv/bin/activate
python --version # verifica che python <= 3.9
```

Install packages:
Nota: installare pytorch della versione che preferisci, sembra non essere stringente (ma occhio alla versione di CUDA)
```bash 
pip install -r requirements.txt --verbose
pip install "numpy<2.0" # Testato che il modello preferisca numpy 1.x
pip install "opencv-contrib-python<4.7" # Necessario perchè altrimenti non compatibile con numpy 1.x

```

# Download checkpoints:

```bash
mkdir checkpoints
cd checkpoints
wget https://huggingface.co/wangyi111/DeCUR/resolve/main/rn50_rda_ssl4eo-s12_joint_decur_ep100.pth 
wget https://huggingface.co/wangyi111/DeCUR/resolve/main/rn50_rda_geonrw_joint_decur_ep100.pth
```

# Performing Transfer Learning
```bash
cd src/transfer_classification_BE
python linear_BE_resnet.py --backbone resnet50 --mode s1 s2 --pretrained /path/to/pretrained_weights ...
```

Se da errore per execstack:
```bash
yay -S execstack patchelf
patchelf --clear-execstack venv/lib/python3.9/site-packages/torch/lib/libtorch_cpu.so
```

# Preparare dataset:

## SUNRGBD
```bash
git clone https://github.com/chrischoy/SUN_RGBD?tab=readme-ov-file
cd SUN_RGBD
./download_and_extract.sh
```

oppure:
```bash
https://github.com/ankurhanda/sunrgbd-meta-data e scarichi i file
```

devi però anche rinominare i files, altrimenti non parte lo scrpit:
```bash
cd /mnt/proj3/eu-25-19/davide_secco/ADL-Project/SUN_RGBD/depth/train && for f in *.png; do n=$(printf "img-%06d.png" $(basename "$f" .png | sed 's/^0*//')); mv "$f" "$n"; done
```

## Altri Dataset (todo - molto futuro)


# Far partire il Pretrain (da aggiunstare):

## SUNRGBD
```bash
RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 python DeCUR/src/pretrain/pretrain_mm.py --dataset SUNRGBD --method DeCUR --data1 SUN_RGBD/image/train/ --data2 SUN_RGBD/depth/train/ --mode MODAL1 MODAL2
```





