import os
import glob

import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


# prepare for CrypkoDataset
class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples


def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset


if __name__ == "__main__":
    workspace_dir = '.'
    temp_dataset = get_dataset(os.path.join(workspace_dir, 'faces'))
    images = [temp_dataset[i] for i in range(8)]
    grid_img = torchvision.utils.make_grid(images, nrow=4)
    torchvision.utils.save_image(grid_img,'./sample/grid_example.jpg')
