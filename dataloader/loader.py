import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from . import preprocess 


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, center, right, training, loader=default_loader, dploader= disparity_loader, augment=False):
 
        self.left = left
        self.right = right
        self.center = center
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.augment = augment

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        if self.center is not None:
            center = self.center[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        if self.center is not None:
            center_img = self.loader(center)


        if self.training:  
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            if self.center is not None:
                center_img = center_img.crop((x1, y1, x1 + tw, y1 + th))

            RANDOM_SEED = random.randint(0,2**32)

            processed1 = preprocess.get_transform(augment=self.augment)

            random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            left_img = processed1(left_img)
            random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            right_img = processed1(right_img)
            if self.center is not None:
                random.seed(RANDOM_SEED)
                torch.manual_seed(RANDOM_SEED)
                center_img = processed1(center_img)

            if self.center is not None:
                return left_img, center_img, right_img
            else:
                return left_img, right_img

        else:
            w, h = left_img.size

            crop_h = (h // 32) * 32
            crop_w = (w // 32) * 32
            left_img = left_img.crop((w-crop_w, h-crop_h, w, h))
            right_img = right_img.crop((w-crop_w, h-crop_h, w, h))
            if self.center is not None:
                center_img = center_img.crop((w-crop_w, h-crop_h, w, h))

            processed = preprocess.get_transform(augment=self.augment)  
            left_img       = processed(left_img)
            right_img      = processed(right_img)
            if self.center is not None:
                center_img      = processed(center_img)

            if self.center is not None:
                return left_img, center_img, right_img
            else:
                return left_img, right_img

    def __len__(self):
        return len(self.left)
