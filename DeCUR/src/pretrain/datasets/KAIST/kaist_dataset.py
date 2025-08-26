
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import glob
from PIL import Image

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def random_subset(dataset, frac, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(range(len(dataset)), int(frac * len(dataset)))
    return Subset(dataset, indices)


class KAISTDataset(Dataset):
    def __init__(self, rgb_dir, th_dir, mode=['rgb','thermal'],
                 rgb_transform=None, th_transform=None):
        self.rgb_dir = rgb_dir
        self.th_dir = th_dir
        self.mode = mode

        # prende TUTTE le immagini visibili in setXX/VYYY/visible/*.jpg
        # considera tutto il path per arrivare all'immagine
        # Esempio:
        # set00/V000/visible/I000001.jpg  !=  set05/V000/visible/I000001.jpg  anche se il nome finale dell'immagine è lo stesso
        self.rgb_paths = glob.glob(os.path.join(rgb_dir, 'set*', 'V*', 'visible', '*.jpg'))
        self.rgb_paths.sort()
        self.length = len(self.rgb_paths)

        self.rgb_transform = rgb_transform
        self.th_transform = th_transform

    def __getitem__(self, index):          #alla richiesta della i-esima immagine rgb restituisce anche la termica abbinata convertita su 3 canali
        rgb_path = self.rgb_paths[index]

        # costruisci path termico corrispondente
        th_path = rgb_path.replace('visible', 'lwir').replace(self.rgb_dir, self.th_dir)

       # carica RGB
        rgb = np.array(Image.open(rgb_path).convert('RGB'))
        if self.rgb_transform is not None:
            rgb = self.rgb_transform(rgb)

        # carica TH: 1 canale → replicato a 3 canali
        th = np.array(Image.open(th_path).convert('L').convert('RGB'))
        if self.th_transform is not None:
            th = self.th_transform(th)

        return rgb, th

    def __len__(self):
        return self.length
