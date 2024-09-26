import torch.nn as nn
from collections import OrderedDict
import torch


class UNet2d(nn.Module):
    """"
    2d Unet network
    """

    def __init__(self, in_channels, out_channels, init_features=16):
        super(UNet2d, self).__init__()
        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder1 = UNet2d._block(self.in_channels, self.features, name="enc1", prob=0.2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2d._block(self.features, self.features * 2, name="enc2", prob=0.2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2d._block(self.features * 2, self.features * 4, name="enc3", prob=0.2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2d._block(self.features * 4, self.features * 8, name="enc4", prob=0.2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = UNet2d._block(self.features * 8, self.features * 16, name="bottleneck", prob=0.2)
        self.upconv4 = nn.ConvTranspose2d(self.features * 16, self.features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet2d._block((self.features * 8) * 2, self.features * 8, name="dec4", prob=0.2)
        self.upconv3 = nn.ConvTranspose2d(self.features * 8, self.features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet2d._block((self.features * 4) * 2, self.features * 4, name="dec3", prob=0.2)
        self.upconv2 = nn.ConvTranspose2d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet2d._block((self.features * 2) * 2, self.features * 2, name="dec2", prob=0.2)
        self.upconv1 = nn.ConvTranspose2d(self.features * 2, self.features, kernel_size=2, stride=2)
        self.decoder1 = UNet2d._block(self.features * 2, self.features, name="dec1", prob=0.2)
        self.conv = nn.Conv2d(in_channels=self.features, out_channels=self.out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        out_logit = self.conv(dec1)

        if self.out_channels == 1:
            output = torch.sigmoid(out_logit)
        if self.out_channels > 1:
            output = torch.softmax(out_logit, dim=1)
        return out_logit, output

    @staticmethod
    def _block(in_channels, features, name, prob=0.2):
        return nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False, ),),
            (name + "norm1", nn.GroupNorm(8, features)),
            (name + "drop1", nn.Dropout2d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
            (name + "conv2", nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False, ),),
            (name + "norm2", nn.GroupNorm(8, features)),
            (name + "drop2", nn.Dropout2d(p=prob, inplace=True)),
            (name + "relu2", nn.ReLU(inplace=True)),
        ]))


class LUConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.activation = nn.ReLU(out_channels)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def make_n_conv_layer(in_channels, depth, double_channel=False):
    if double_channel:
        layer1 = LUConv(in_channels, 32 * (2 ** (depth + 1)))
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)))
    else:
        layer1 = LUConv(in_channels, 32 * (2 ** depth))
        layer2 = LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2)

    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channels, depth):
        super(DownTransition, self).__init__()
        self.ops = make_n_conv_layer(in_channels, depth)
        self.pool = nn.MaxPool2d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.pool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.ops = make_n_conv_layer(in_channels + out_channels // 2, depth, double_channel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_channels, n_labels):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv2d(in_channels, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.final_conv(x)
        out = self.sigmoid(x)
        return x, out


class UNet2dlandmarkregression(nn.Module):
    def __init__(self, in_channels=1, n_class=1):
        super(UNet2dlandmarkregression, self).__init__()

        self.down_tr64 = DownTransition(in_channels, 0)
        self.down_tr128 = DownTransition(64, 1)
        self.down_tr256 = DownTransition(128, 2)
        self.down_tr512 = DownTransition(256, 3)

        self.up_tr256 = UpTransition(512, 512, 2)
        self.up_tr128 = UpTransition(256, 256, 1)
        self.up_tr64 = UpTransition(128, 128, 0)
        self.out_tr = OutputTransition(64, n_class)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128, self.skip_out128 = self.down_tr128(self.out64)
        self.out256, self.skip_out256 = self.down_tr256(self.out128)
        self.out512, self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512, self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)
        return self.out


if __name__ == '__main__':
    model = UNet2dlandmarkregression(in_channels=1, n_class=38)
    print(model)
