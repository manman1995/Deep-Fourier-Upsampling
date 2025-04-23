import torch
import torch.nn as nn
from .do_conv import DOConv2d

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

class BasicDOConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicDOConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ResFFTBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResFFTBlock, self).__init__()
        self.main = nn.Sequential(
            BasicDOConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicDOConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.conv_1 = BasicDOConv(out_channel, out_channel, kernel_size=1, stride=1, relu=True)
        self.conv_2 = BasicDOConv(out_channel, out_channel, kernel_size=1, stride=1, relu=False)
    def forward(self, x):
        x_fft =torch.fft.fft2(x, dim=(-2, -1))
        x_real = x_fft.real

        x_real = self.conv_1(x_real)
        x_real = self.conv_2(x_real)
        x_fft.real = x_real
        x_fft_res = torch.fft.ifft2(x_fft, dim=(-2, -1))

        return self.main(x) + x + x_fft_res.real