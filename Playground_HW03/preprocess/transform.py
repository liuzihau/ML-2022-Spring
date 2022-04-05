import torch
import torchvision.utils
from torchvision import transforms
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.mixup import Mixup
import numpy as np
import cv2


def transform(config):
    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # However, it is also possible to use augmentation in the testing phase.
    # You may use train_tfm to produce a variety of images and then test using ensemble methods
    rescale_and_crop = transforms.Compose([
        RandomResizedCropAndInterpolation(size=128, scale=(0.7, 1.1)),
        transforms.ToTensor()
    ])
    random_augment = transforms.Compose([
        rand_augment_transform(
            config_str='rand-m9-mstd0.5',
            hparams={'img_mean': (124, 116, 104)}
        ),
        transforms.Resize((128, 128)),
        transforms.ToTensor()])
    return {"test": [test_tfm], "train": [test_tfm, rescale_and_crop]+config['random_transform_times']*[random_augment]}
