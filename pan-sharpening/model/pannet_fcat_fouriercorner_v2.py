#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-11-05 20:47:04
LastEditTime: 2020-12-09 23:12:31
Description: PanNet: A deep network architecture for pan-sharpening (VDSR-based)
2000 epoch, decay 1000 x0.1, batch_size = 128, learning_rate = 1e-2, patch_size = 33, MSE
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F

class freup_Cornerdinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Cornerdinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)  # n c h w
        fft_x = torch.fft.fftshift(fft_x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        Mag = torch.nn.functional.pad(Mag, (W // 2, W // 2, H // 2, H // 2))
        Pha = torch.nn.functional.pad(Pha, (W // 2, W // 2, H // 2, H // 2))

        real = Mag * torch.cos(Pha)
        imag = Mag * torch.sin(Pha)
        out = torch.complex(real, imag)

        out = torch.fft.ifftshift(out)
        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return output

class freup_inter(nn.Module):
    def __init__(self, channels):
        super(freup_inter, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))

        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)
        
        amp_fuse = Mag.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        pha_fuse = Pha.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        
        output = torch.fft.ifft2(out)
        output = torch.abs(output)
        
        crop = torch.zeros_like(x)
        crop[:, :, 0:int(H/2), 0:int(W/2)] = output[:, :, 0:int(H/2), 0:int(W/2)]
        crop[:, :, int(H/2):H, 0:int(W/2)] = output[:, :, int(H*1.5):2*H, 0:int(W/2)]
        crop[:, :, 0:int(H/2), int(W/2):W] = output[:, :, 0:int(H/2), int(W*1.5):2*W]
        crop[:, :, int(H/2):H, int(W/2):W] = output[:, :, int(H*1.5):2*H, int(W*1.5):2*W]
        crop = F.interpolate(crop, (2*H, 2*W))

        return self.post(crop)

class freup_pad(nn.Module):
    def __init__(self, channels):
        super(freup_pad, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))

        self.post = nn.Conv2d(channels,channels,1,1,0)

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
    def __init__(self, channels=32):
        super(fresadd, self).__init__()

        # self.opspa = freup_pad(channels)
        # self.opfre = ConvBlock(48, 32, 5, 1, 2, activation=None, norm=None, bias = False)

        self.opspa = ConvBlock(48, 32, 5, 1, 2, activation=None, norm=None, bias = False)
        self.opfre = freup_pad(channels)

        self.fuse1 = nn.Conv2d(channels, channels,1,1,0)
        self.fuse2 = nn.Conv2d(channels, channels,1,1,0)
        self.fuse = nn.Conv2d(channels, channels,1,1,0)

    def forward(self,x):

        x1 = x
        x2 = F.interpolate(x1,scale_factor=0.5,mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')

        x1 = self.opspa(x1)
        x2 = self.opspa(x2)
        x3 = self.opspa(x3)

        x3f = self.opfre(x3)
        x3s = F.interpolate(x3, size=(x2.size()[2], x2.size()[3]), mode='bilinear')
        x32 = self.fuse1(x3f + x3s)

        x2m = x2 + x32

        x2f = self.opfre(x2m)
        x2s = F.interpolate(x2m,size=(x1.size()[2],x1.size()[3]),mode='bilinear')
        x21 = self.fuse2(x2f + x2s)

        x1m = x1 + x21
        x = self.fuse(x1m)

        return x

class frescat(nn.Module):
    def __init__(self, channels=32):
        super(frescat, self).__init__()

        # self.opspa = freup_pad(channels)
        # self.opfre = ConvBlock(48, 32, 5, 1, 2, activation=None, norm=None, bias = False)

        self.opspa = ConvBlock(48, 32, 5, 1, 2, activation=None, norm=None, bias = False)
        self.opfre = freup_Cornerdinterpolation(channels)

        self.fuse1 = nn.Conv2d(2*channels, channels,1,1,0)
        self.fuse2 = nn.Conv2d(2*channels, channels,1,1,0)
        self.fuse = nn.Conv2d(2*channels, channels,1,1,0)

    def forward(self,x):

        x1 = x
        x2 = F.interpolate(x1,scale_factor=0.5,mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')

        x1 = self.opspa(x1)
        x2 = self.opspa(x2)
        x3 = self.opspa(x3)

        x3f = self.opfre(x3)
        x3s = F.interpolate(x3, size=(x2.size()[2], x2.size()[3]), mode='bilinear')
        x32 = self.fuse1(torch.cat([x3f,x3s],dim=1))

        x2m = x2 + x32

        x2f = self.opfre(x2m)
        x2s = F.interpolate(x2m,size=(x1.size()[2],x1.size()[3]),mode='bilinear')
        x21 = self.fuse2(torch.cat([x2f,x2s],dim=1))

        # x1m = x1 + x21
        x = self.fuse(torch.cat([x1,x21],dim=1))

        return x


class freup_pad_b(nn.Module):
    def __init__(self, channels):
        super(freup_pad_b, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))

        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, x):

        N, C, H, W = x.shape

        # fft_x = torch.fft.fft2(x)
        # mag_x = torch.abs(fft_x)
        # pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(x)
        Pha = self.pha_fuse(x)

        # amp_fuse = torch.tile(Mag, (2, 2))
        # pha_fuse = torch.tile(Pha, (2, 2))

        # real = amp_fuse * torch.cos(pha_fuse)
        # # imag = amp_fuse * torch.sin(pha_fuse)
        # out = torch.complex(real, imag)

        # output = torch.fft.ifft2(out)
        # output = torch.abs(output)
        
        return self.post(F.interpolate(Mag+Pha,scale_factor=2,mode='bilinear'))

class fresadd_b(nn.Module):
    def __init__(self, channels=32):
        super(fresadd_b, self).__init__()

        # self.opspa = freup_pad(channels)
        # self.opfre = ConvBlock(48, 32, 5, 1, 2, activation=None, norm=None, bias = False)

        self.opspa = ConvBlock(48, 32, 5, 1, 2, activation=None, norm=None, bias = False)
        self.opfre = freup_pad_b(channels)

        self.fuse1 = nn.Conv2d(channels, channels,1,1,0)
        self.fuse2 = nn.Conv2d(channels, channels,1,1,0)
        self.fuse = nn.Conv2d(channels, channels,1,1,0)

    def forward(self,x):

        x1 = x
        x2 = F.interpolate(x1,scale_factor=0.5,mode='bilinear')
        x3 = F.interpolate(x1, scale_factor=0.25, mode='bilinear')

        x1 = self.opspa(x1)
        x2 = self.opspa(x2)
        x3 = self.opspa(x3)

        x3f = self.opfre(x3)
        x3s = F.interpolate(x3, size=(x2.size()[2], x2.size()[3]), mode='bilinear')
        x32 = self.fuse1(x3f + x3s)

        x2m = x2 + x32

        x2f = self.opfre(x2m)
        x2s = F.interpolate(x2m,size=(x1.size()[2],x1.size()[3]),mode='bilinear')
        x21 = self.fuse2(x2f + x2s)

        x1m = x1 + x21
        x = self.fuse(x1m)

        return x

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()

        base_filter = 64
        num_channels = 7
        out_channels = 4
        self.args = args
        self.head = ConvBlock(num_channels, 48, 9, 1, 4, activation='relu', norm=None, bias = False)



        # self.body = ConvBlock(48, 32, 5, 1, 2, activation=None, norm=None, bias = False)
        self.body = frescat(32)

        self.output_conv = ConvBlock(32, out_channels, 5, 1, 2, activation=None, norm=None, bias = False)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
        	    #torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
        	    #torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, l_ms, b_ms, x_pan):
        b_ms = F.interpolate(l_ms, scale_factor=self.args['data']['upsacle'], mode='bicubic')
        NDWI = ((l_ms[:, 1, :, :] - l_ms[:, 3, :, :]) / (l_ms[:, 1, :, :] + l_ms[:, 3, :, :])).unsqueeze(1)
        NDWI = F.interpolate(NDWI, scale_factor=self.args['data']['upsacle'], mode='bicubic')
        NDVI = ((l_ms[:, 3, :, :] - l_ms[:, 2, :, :]) / (l_ms[:, 3, :, :] + l_ms[:, 2, :, :])).unsqueeze(1)
        NDVI = F.interpolate(NDVI, scale_factor=self.args['data']['upsacle'], mode='bicubic')
        x_f = torch.cat([b_ms, x_pan, NDVI, NDWI], 1)
        x_f = self.head(x_f)
        x_f = self.body(x_f)
        x_f = self.output_conv(x_f)
        x_f = torch.add(x_f,b_ms)
     
        return x_f
        
