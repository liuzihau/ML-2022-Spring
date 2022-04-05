import json
import os

import torch
import torchvision.utils
from torchvision import transforms
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.mixup import Mixup
import numpy as np
import cv2
from PIL import Image


def transform(config):
    test_tfm = transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
    ])

    # However, it is also possible to use augmentation in the testing phase.
    # You may use train_tfm to produce a variety of images and then test using ensemble methods
    rescale_and_crop = transforms.Compose([
        RandomResizedCropAndInterpolation(size=192, scale=(0.7, 1.1)),
        transforms.ToTensor()
    ])
    random_augment = transforms.Compose([
        rand_augment_transform(
            config_str='rand-m9-mstd0.5',
            hparams={'img_mean': (124, 116, 104)}
        ),
        transforms.Resize((192, 192)),
        transforms.ToTensor()])
    return {"test": [test_tfm],
            "train": [test_tfm, rescale_and_crop] + config['random_transform_times'] * [random_augment]}


def make_argument(config):
    # However, it is also possible to use augmentation in the testing phase.
    # You may use train_tfm to produce a variety of images and then test using ensemble methods
    rescale_and_crop = transforms.Compose([
        RandomResizedCropAndInterpolation(size=192, scale=(0.7, 1.1)),
    ])
    random_augment = transforms.Compose([
        rand_augment_transform(
            config_str='rand-m9-n3-mstd0.5',
            hparams={'img_mean': (124, 116, 104)}
        )])
    return [rescale_and_crop] + config['random_transform_times'] * [random_augment]


if __name__ == "__main__":
    with open('../config/config.json', 'r', encoding="utf-8") as config_json:
        config = json.loads(config_json.read())
    transform_set = make_argument(config)
    path = "../food11_origin/"
    folders = os.listdir(path)
    for i, trans in enumerate(transform_set):

        dst_path = f"../food11/argument{i:02d}/"
        test_dst_path = f"../food11/test_argument{i:02d}/"

        isExist = os.path.exists(dst_path)
        isExist2 = os.path.exists(test_dst_path)
        if not isExist:
            os.makedirs(dst_path)
            print(f"create : {dst_path}")
        if not isExist2:
            os.makedirs(test_dst_path)
            print(f"create : {test_dst_path}")
        for folder in folders:
            image_list = os.listdir(path + folder)
            for image_name in image_list:
                image_path = path + folder + "/" + image_name
                image = Image.open(image_path)
                image = trans(image)
                if folder == "test":
                    image.save(test_dst_path + image_name)
                else:
                    image.save(dst_path + image_name.split('_')[0] + '_' + folder + image_name.split('_')[1])




