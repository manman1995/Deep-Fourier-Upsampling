import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np


# try:
#   from model_utils import *
# except:
#   from .model_utils import *

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.apply(self._init_weights)

    def forward(self, x):
        return self.double_conv(x)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# spatial attention
class SpatialGate(nn.Module):
    def __init__(self, in_channels):
        super(SpatialGate, self).__init__()
        self.spatial = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_out = self.spatial(x)
        scale = self.sigmoid(x_out)
        return scale * x


# sobel
class SobelOperator(nn.Module):
    def __init__(self):
        super(SobelOperator, self).__init__()
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv_x.weight[0].data[:, :, :] = torch.FloatTensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]])
        self.conv_y.weight[0].data[:, :, :] = torch.FloatTensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]])

    def forward(self, x):
        G_x = self.conv_x(x)
        G_y = self.conv_y(x)
        grad_mag = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        return grad_mag


class offset_estimator(nn.Sequential):
    def __init__(self, kernel_size, fwhm, in_channel, mid_channel, out_channel) -> None:
        super().__init__()
        model = []
        assert len(kernel_size) == len(fwhm), "length error"
        for i in range(len(kernel_size)):
            if i == 0:
                gaussian_weight = torch.FloatTensor(gaussian_2d(kernel_size[i], fwhm=fwhm[i]))
                gauss_filter = nn.Conv2d(in_channel, mid_channel, kernel_size[i], padding=(kernel_size[i] - 1) // 2,
                                         bias=False)
                gauss_filter.weight[0].data[:, :, :] = gaussian_weight
                model += [gauss_filter, nn.ReLU(inplace=True)]
            elif i == len(kernel_size) - 1:
                gaussian_weight = torch.FloatTensor(gaussian_2d(kernel_size[i], fwhm=fwhm[i]))
                gauss_filter = nn.Conv2d(mid_channel, out_channel, kernel_size[i], padding=(kernel_size[i] - 1) // 2,
                                         bias=False)
                gauss_filter.weight[0].data[:, :, :] = gaussian_weight
                model += [gauss_filter, nn.ReLU(inplace=True)]
            else:
                gaussian_weight = torch.FloatTensor(gaussian_2d(kernel_size[i], fwhm=fwhm[i]))
                gauss_filter = nn.Conv2d(mid_channel, mid_channel, kernel_size[i], padding=(kernel_size[i] - 1) // 2,
                                         bias=False)
                gauss_filter.weight[0].data[:, :, :] = gaussian_weight
                model += [gauss_filter, nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# Channel attention
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


# LBP
def LBP(image):  # b, 3, h, w tensor
    radius = 2
    n_points = 8 * radius
    method = 'uniform'
    gray_img = rgb_to_grayscale(image)  # b, 1, h, w
    gray_img = gray_img.squeeze(1)
    lbf_feature = np.zeros((gray_img.shape[0], gray_img.shape[1], gray_img.shape[2]))
    for i in range(gray_img.shape[0]):
        lbf_feature[i] = feature.local_binary_pattern(gray_img[i], n_points, radius, method)
    return torch.FloatTensor(lbf_feature).unsqueeze(1)


class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel

        def discriminator_block(in_filters, out_filters):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=False)]
            return layers

        self.model = nn.Sequential(
            *discriminator_block(self.in_channel, 4),
            *discriminator_block(4, 4),
            *discriminator_block(4, 4),
            *discriminator_block(4, 4),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(4, 1, 4, padding=1, bias=False)
        )

    def forward(self, x):
        return self.model(x)


class Discriminator_new(nn.Module):
    def __init__(self):
        super().__init__()

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = 3
        for i, out_filters in enumerate([4, 6, 8, 10]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.ZeroPad2d((1, 0, 1, 0)))
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

