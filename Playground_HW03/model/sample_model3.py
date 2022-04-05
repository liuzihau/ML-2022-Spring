from torch import nn
from model.stn.spatial_transformer_network import SpatialTransformer
from model.stn.thin_plate_spline import ThinPlateSpline
from model.convnext2 import ConvNeXt


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
            self.stn = ThinPlateSpline(16, (192, 192), (128, 128), 3)
        self.cnn = ConvNeXt(config)

    def forward(self, x):
        if self.stn_name == "Spatial_transformer_network":
            stn_x = self.stn(x)
        elif self.stn_name == "Thin_plate_spline":
            stn_x = self.stn(x)
        else:
            stn_x = x

        return self.cnn(stn_x)
