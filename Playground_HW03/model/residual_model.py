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
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1, 1),
            nn.BatchNorm2d(64)
        )

        self.conv64x2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )
        self.conv128x2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
        self.conv256x2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
        self.conv512x2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2, 2, 0)
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
        x = self.conv_input(stn_x)  # [3, H, W] --> [64, H, W]
        identity = x
        out = self.conv64x2(x)  # [64, H, W]
        out += identity
        identity = out
        out = self.conv128x2(out)  # [128, H, W]
        out = self.cnn(stn_x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
