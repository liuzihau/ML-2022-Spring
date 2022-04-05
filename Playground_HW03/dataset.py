import os
import random

from PIL import Image
import numpy as np
import torch
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset


class FoodDataset(Dataset):

    def __init__(self, path, config, tfm, files=None, split=0, mode="train"):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if mode == "train":
            self.files = [x for i, x in enumerate(self.files) if
                          (i % config["train_valid_split"] != split and
                           i % config["train_valid_split"] != split + config["train_valid_split"] // 2)]
        elif mode == "valid":
            self.files = [x for i, x in enumerate(self.files) if
                          (i % config["train_valid_split"] == split or
                           i % config["train_valid_split"] == split + config["train_valid_split"] // 2)]
        else:
            print(f"warning : no train-valid split in {path}!")
        if files is not None:
            self.files = files
        print(f"One {path} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files) * len(self.transform)

    def __getitem__(self, idx):
        transform_idx = int(idx // len(self.files))
        idx = idx % len(self.files)
        figure_name = self.files[idx]
        im = Image.open(figure_name)

        im = self.transform[transform_idx](im)
        # im = self.data[idx]
        try:
            label = int(figure_name.split("\\")[-1].split("_")[0])
            label = torch.from_numpy(np.array(np.eye(11)[label]))
        except Exception as e:
            print(e)
            label = -1  # test has no label
        return im, label


class FoodDatasetMixUp(Dataset):

    def __init__(self, path, config, tfm, files=None, split=0, mode="train"):
        super(FoodDataset).__init__()
        self.path = path
        self.files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")]
        if mode == "train":
            self.files = [x for i, x in enumerate(self.files) if
                          (i % config["train_valid_split"] != split and
                           i % config["train_valid_split"] != split + config["train_valid_split"] // 2)]
            random.shuffle(self.files)
        elif mode == "valid":
            self.files = [x for i, x in enumerate(self.files) if
                          (i % config["train_valid_split"] == split or
                           i % config["train_valid_split"] == split + config["train_valid_split"] // 2)]
        else:
            print(f"warning : no train-valid split in {path}!")

        if files is not None:
            self.files = files
        print(f"One {path} sample", self.files[0])
        self.lam = config['lambda']
        self.transform = tfm

    def __len__(self):
        return len(self.files) * len(self.transform) * self.lam

    def __getitem__(self, idx):
        my_lambda = np.random.beta(0.8, 0.8)
        lam_idx = idx // (len(self.files) * len(self.transform))
        # transform_idx = int((idx // len(self.files)) % self.lam)
        transform_idx = 0
        idx = idx % len(self.files)
        idx2 = random.randint(0, len(self.files) - 1)
        figure_name1, figure_name2 = self.files[idx], self.files[idx2]
        im1 = Image.open(figure_name1)
        im2 = Image.open(figure_name2)
        im1 = self.transform[transform_idx](im1)
        im2 = self.transform[transform_idx](im2)
        im = my_lambda * im1 + (1 - my_lambda) * im2
        # im = self.data[idx]
        try:
            label1 = int(figure_name1.split("\\")[-1].split("_")[0])
            label1 = torch.from_numpy(np.array(np.eye(11)[label1]))
            label2 = int(figure_name2.split("\\")[-1].split("_")[0])
            label2 = torch.from_numpy(np.array(np.eye(11)[label2]))
            label = my_lambda * label1 + (1 - my_lambda) * label2
        except Exception as e:
            print(e)
            label = -1  # test has no label
        return im, label


class FoodDatasetEnsemble(Dataset):

    def __init__(self, path, config, tfm, files=None, split=0, mode="train"):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path[0], x) for x in os.listdir(path[0]) if x.endswith(".jpg")])
        self.ensemble_file = []
        for unit_path in path:
            self.ensemble_file.append(sorted([os.path.join(unit_path, x) for x in os.listdir(unit_path) if x.endswith(".jpg")]))
        if mode == "train":
            self.files = [x for i, x in enumerate(self.files) if
                          (i % config["train_valid_split"] != split and
                           i % config["train_valid_split"] != split + config["train_valid_split"] // 2)]
        elif mode == "valid":
            self.files = [x for i, x in enumerate(self.files) if
                          (i % config["train_valid_split"] == split or
                           i % config["train_valid_split"] == split + config["train_valid_split"] // 2)]
        else:
            print(f"warning : no train-valid split in {path[0]}!")
        if files is not None:
            self.files = files
        print(f"One {path[0]} sample", self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        for file_id, files in enumerate(self.ensemble_file):
            figure_name = files[idx]
            im = Image.open(figure_name)
            im = self.transform[0](im)
            im = torch.unsqueeze(im,0)
            if file_id == 0:
                img_set = im
            else:
                img_set = torch.cat((img_set, im), 0)
        try:
            label = int(figure_name.split("\\")[-1].split("_")[0])
            label = torch.from_numpy(np.array(np.eye(11)[label]))
        except Exception as e:
            label = -1  # test has no label
        return img_set, label
