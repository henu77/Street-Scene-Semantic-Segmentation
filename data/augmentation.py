"""
数据增强模块
"""
import numpy as np
from PIL import Image, ImageFilter
import random
import math
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    """如果图像太小则进行填充"""
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    """组合多个变换"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转"""
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    """随机裁剪"""
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class ColorJitter(T.ColorJitter):
    """颜色抖动"""
    def __call__(self, image, target):
        return super().__call__(image), target


class RandomResizedCrop(T.RandomResizedCrop):
    """随机调整大小并裁剪"""
    def __call__(self, image, target):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        return F.resized_crop(image, i, j, h, w, self.size, self.interpolation), \
               F.resized_crop(target, i, j, h, w, self.size, Image.NEAREST)


class RandomBlur(object):
    """随机模糊"""
    def __init__(self, prob=0.5, radius_range=(0.1, 2.0)):
        self.prob = prob
        self.radius_range = radius_range
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            radius = random.uniform(self.radius_range[0], self.radius_range[1])
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image, target


class RandomBrightnessContrast(object):
    """随机亮度对比度调整"""
    def __init__(self, brightness=0.2, contrast=0.2, prob=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            brightness_factor = random.uniform(1-self.brightness, 1+self.brightness)
            contrast_factor = random.uniform(1-self.contrast, 1+self.contrast)
            
            image = F.adjust_brightness(image, brightness_factor)
            image = F.adjust_contrast(image, contrast_factor)
            
        return image, target


class RandomErasing(object):
    """随机擦除"""
    def __init__(self, probability=0.5, sl=0.02, sh=0.33, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, image, target):
        if random.random() < self.probability:
            img = np.array(image)
            h, w, _ = img.shape
            area = h * w

            for _ in range(1):  # 随机擦除次数
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h_erase = int(round(math.sqrt(target_area * aspect_ratio)))
                w_erase = int(round(math.sqrt(target_area / aspect_ratio)))

                if w_erase < w and h_erase < h:
                    x1 = random.randint(0, w - w_erase)
                    y1 = random.randint(0, h - h_erase)
                    img[y1:y1 + h_erase, x1:x1 + w_erase] = 255

            image = Image.fromarray(img)

        return image, target


def get_augmentation(is_strong=False):
    """获取数据增强策略"""
    if is_strong:
        return Compose([
            RandomHorizontalFlip(),
            RandomResizedCrop(384),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            RandomBlur(prob=0.5, radius_range=(0.1, 2.0)),
            RandomBrightnessContrast(prob=0.5),
            RandomErasing(probability=0.5),
        ])
    else:
        return Compose([
            RandomHorizontalFlip(),
            RandomResizedCrop(384),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
