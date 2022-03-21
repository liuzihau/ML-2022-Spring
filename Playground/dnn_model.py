import torch.nn as nn


class DNNModel(nn.Module):
    def __init__(self, input_dim):
        totel_feature = 1
        for i in input_dim:
            totel_feature *= i
        super(DNNModel, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(totel_feature, 256),
            # nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(64, 32),
            # nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Linear(32, 16),
            # nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x
