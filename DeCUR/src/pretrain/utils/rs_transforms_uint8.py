import numpy as np
import torch
import random
import cv2

from albumentations.core.transforms_interface import ImageOnlyTransform

class RandomBrightness(ImageOnlyTransform):
    def __init__(self, brightness=0.4, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.brightness = brightness

    def apply(self, img, **params):
        s = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        img = img * s
        return np.clip(img, 0, 255).astype(np.uint8)


class RandomContrast(ImageOnlyTransform):
    def __init__(self, contrast=0.4, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.contrast = contrast

    def apply(self, img, **params):
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        s = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        img = (img - mean) * s + mean
        return np.clip(img, 0, 255).astype(np.uint8)


class ToGray(ImageOnlyTransform):
    def __init__(self, out_channels, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.out_channels = out_channels

    def apply(self, img, **params):
        gray = np.mean(img, axis=-1)
        gray_img = np.stack([gray] * self.out_channels, axis=-1)
        return gray_img.astype(np.uint8)


class GaussianBlur(ImageOnlyTransform):
    def __init__(self, sigma=[0.1, 2.0], always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.sigma = sigma

    def apply(self, img, **params):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return cv2.GaussianBlur(img, (0, 0), sigma)


class Solarize(ImageOnlyTransform):
    def __init__(self, threshold=128, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.threshold = threshold

    def apply(self, img, **params):
        img = img.copy()
        img[img < self.threshold] = 255 - img[img < self.threshold]
        return img.astype(np.uint8)

       
        
class RandomChannelDrop(object):
    """ Random Channel Drop """
    
    def __init__(self, min_n_drop=1, max_n_drop=8):
        self.min_n_drop = min_n_drop
        self.max_n_drop = max_n_drop

    def __call__(self, sample):
        n_channels = random.randint(self.min_n_drop, self.max_n_drop)
        channels = np.random.choice(range(sample.shape[0]), size=n_channels, replace=False)

        for c in channels:
            sample[c, :, :] = 0        
        return sample        
        
        
class RandomSensorDrop_S1S2(object):
    """ Random Channel Drop """
    
    def __init__(self):
        pass

    def __call__(self, sample):
        sensor = np.random.choice([1,2], replace=False)

        if sensor==2:
            sample[:13, :, :] = 0
        elif sensor==1:
            sample[13:,:,:] = 0
        
        return sample
    
class SensorDrop_S1S2(object):
    def __init__(self, sensor):
        self.sensor = sensor
    def __call__(self,sample):
        if self.sensor == 'S1':
            sample[13:,:,:] = 0
        elif self.sensor == 'S2':
            sample[:13,:,:] = 0
        return sample
    
    
class RandomSensorDrop_RGBD(object):
    """ Random Channel Drop """
    
    def __init__(self):
        pass

    def __call__(self, sample):
        sensor = np.random.choice([1,2], replace=False, p=[0.8,0.2])

        if sensor==2:
            sample[:3, :, :] = 0
        elif sensor==1:
            sample[3:,:,:] = 0
        
        return sample
    
class SensorDrop_RGBD(object):
    def __init__(self, sensor):
        self.sensor = sensor
    def __call__(self,sample):
        if self.sensor == 'D':
            sample[3:,:,:] = 0
        elif self.sensor == 'RGB':
            sample[:3,:,:] = 0
        return sample