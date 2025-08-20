### Setta nuovo token 
Crea nuovo token in: https://www.kaggle.com/settings --> API --> Create new token
´´´bash
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
´´´
### Scaricare il dataset
´´´bash
kaggle datasets download -d larbii/thermal-images
´´´