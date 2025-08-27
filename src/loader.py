# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class FourCropsTransform:
    """
    Restituisce 4 viste della stessa immagine:
    [y1_1, y1_2, y2_1, y2_2]
    t1 è la pipeline di aug per la 'mod1', t2 per la 'mod2'.
    Se t2 è None, usa t1 anche per mod2.
    """
    def __init__(self, t1, t2=None):
        self.t1 = t1
        self.t2 = t2 if t2 is not None else t1

    def __call__(self, x):
        return [self.t1(x), self.t1(x), self.t2(x), self.t2(x)]