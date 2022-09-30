from torch import nn
from model.stn.spatial_transformer_network import SpatialTransformer
from model.stn.thin_plate_spline import ThinPlateSpline


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.stn_name = config['stn']
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        if self.stn_name == "Spatial_transformer_network":
            self.stn = SpatialTransformer()
        elif self.stn_name == "Thin_plate_spline":
            self.stn = ThinPlateSpline(16, (128, 128), (128, 128), 3)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * config['input_size']/32 * config['input_size']/32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        if self.stn_name == "Spatial_transformer_network":
            stn_x = self.stn(x)
        elif self.stn_name == "Thin_plate_spline":
            stn_x = self.stn(x)
        else:
            stn_x = x
        out = self.cnn(stn_x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
