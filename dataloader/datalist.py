import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, mode='3frame'):
    left_train = []
    right_train = []
    center_train = []
    if mode == '3frame':
        for subpath in os.listdir(filepath):
            if subpath == 'SyntheData':
                for scene in os.listdir(filepath + subpath):
                    left_train.append(filepath + subpath + '/' + scene + '/view0.png')
                    center_train.append(filepath + subpath + '/' + scene + '/view1.png')
                    right_train.append(filepath + subpath + '/' + scene + '/view2.png')
        return left_train, center_train, right_train

    elif mode == '2frame':
        for subpath in os.listdir(filepath):
            if subpath == 'SyntheData':
                for scene in os.listdir(filepath + subpath):
                    left_train.append(filepath + subpath + '/' + scene + '/im0.png')
                    right_train.append(filepath + subpath + '/' + scene + '/im1.png')
            
        return left_train, right_train



    