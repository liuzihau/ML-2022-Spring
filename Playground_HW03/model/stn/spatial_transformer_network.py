import numpy as np
from torch import nn
import torch
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # [3,128,128]-->[32,128,128]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # [32,128,128]-->[32,64,64]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [32,64,64]-->[64,64,64]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # [64,64,64]-->[64,32,32]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [64,32,32]-->[128,32,32]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # [128,32,32]-->[128,16,16]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # [128,16,16]-->[256,16,16]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),  # [256,16,16]-->[256]
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 256)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
