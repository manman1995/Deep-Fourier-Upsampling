#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from basicsr.utils.show import feature_show


class freup_pad(nn.Module):
    def __init__(self, channels):
        super(freup_pad, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        amp_fuse = torch.tile(Mag, (2, 2))
        pha_fuse = torch.tile(Pha, (2, 2))

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return self.post(output)


class fresadd(nn.Module):
    def __init__(self, subnet, channels):
        super(fresadd, self).__init__()

        self.opspa = subnet
        self.opfre = freup_pad(channels)

        self.fuse1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.fuse2 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.fuse = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')

        x1 = self.opspa(x1)
        x2 = self.opspa(x2)
        x3 = self.opspa(x3)

        x3f = self.opfre(x3)
        x3s = F.interpolate(x3, size=(x2.size()[2], x2.size()[3]), mode='bilinear')
        x32 = self.fuse1(x3f + x3s)

        x2m = x2 + x32

        x2f = self.opfre(x2m)
        x2s = F.interpolate(x2m, size=(x1.size()[2], x1.size()[3]), mode='bilinear')
        x21 = self.fuse2(x2f + x2s)

        x1m = x1 + x21
        x = self.fuse(x1m)

        return x


class Z_PRENet_Padding(nn.Module):
    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(Z_PRENet_Padding, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv11 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.res_conv1 = fresadd(self.res_conv11, 32)

        self.res_conv21 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.res_conv2 = fresadd(self.res_conv21, 32)

        self.res_conv31 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.res_conv3 = fresadd(self.res_conv31, 32)

        self.res_conv41 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = fresadd(self.res_conv41, 32)

        self.res_conv51 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = fresadd(self.res_conv51, 32)

        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col))
        c = Variable(torch.zeros(batch_size, 32, row, col))

        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()

        x_list = []
        for iter in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            # feature_show(x, 'z_prenet_norain172', iter, 1)

            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            # feature_show(x, 'z_prenet_norain172', iter, 2)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            # feature_show(x, 'z_prenet_norain172', iter, 3)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            # feature_show(x, 'z_prenet_norain172', iter, 4)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list

if __name__ == "__main__":
    pass