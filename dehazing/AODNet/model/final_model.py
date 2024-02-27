import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import scipy.signal
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


#!/usr/bin/env python
# coding=utf-8
'''
Author: zhou man
Date: 2022-3-6

'''
import os
import torch
import torch.nn as nn
from torchvision.transforms import *
import torch.nn.functional as F
def calculate_d(x, y, M, N):
    term1 = 1
    term2 = torch.exp(1j * torch.tensor(np.pi) * x / M)
    term3 = torch.exp(1j * torch.tensor(np.pi) * y / N)
    term4 = torch.exp(1j * torch.tensor(np.pi) * (x/M + y/N))

    result = term1 + term2 + term3 + term4
    return torch.abs(result) / 4

def get_D_map_optimized(feature):
    B, C, H, W = feature.shape
    d_map = torch.zeros((1, 1, H, W), dtype=torch.float32).cuda()
    
    #Create a grid to store the indices of all (i, j) pairs
    i_indices = torch.arange(H, dtype=torch.float32).reshape(1, 1, H, 1).repeat(1, 1, 1, W).cuda()
    j_indices = torch.arange(W, dtype=torch.float32).reshape(1, 1, 1, W).repeat(1, 1, H, 1).cuda()
    
    # Compute d_map using vectorization operations
    d_map[:, :, :, :] = calculate_d(i_indices, j_indices, H, W)
    
    return d_map

class freup_Areadinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Areadinterpolation, self).__init__()

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

class freup_AreadinterpolationV2(nn.Module):
    def __init__(self, channels):
        super(freup_AreadinterpolationV2, self).__init__()

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
        d_map= get_D_map_optimized(x)
        crop = torch.zeros_like(x)
        crop[:,:,:,:] = output[:,:,:H,:W]
        d_map= get_D_map_optimized(x)
        crop = crop / d_map
        crop = F.interpolate(crop, (2*H, 2*W))

        return self.post(crop)


class freup_Periodicpadding(nn.Module):
    def __init__(self, channels):
        super(freup_Periodicpadding, self).__init__()

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


class freup_Cornerdinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Cornerdinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        # self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)  # n c h w
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        r = x.size(2)  # h
        c = x.size(3)  # w

        I_Mup = torch.zeros((N, C, 2 * H, 2 * W)).cuda()
        I_Pup = torch.zeros((N, C, 2 * H, 2 * W)).cuda()

        if r % 2 == 1:  # odd
            ir1, ir2 = r // 2 + 1, r // 2 + 1
        else:  # even
            ir1, ir2 = r // 2 + 1, r // 2
        if c % 2 == 1:  # odd
            ic1, ic2 = c // 2 + 1, c // 2 + 1
        else:  # even
            ic1, ic2 = c // 2 + 1, c // 2

        I_Mup[:, :, :ir1, :ic1] = Mag[:, :, :ir1, :ic1]
        I_Mup[:, :, :ir1, ic2 + c:] = Mag[:, :, :ir1, ic2:]
        I_Mup[:, :, ir2 + r:, :ic1] = Mag[:, :, ir2:, :ic1]
        I_Mup[:, :, ir2 + r:, ic2 + c:] = Mag[:, :, ir2:, ic2:]

        if r % 2 == 0:  # even
            I_Mup[:, :, ir2, :] = I_Mup[:, :, ir2, :] * 0.5
            I_Mup[:, :, ir2 + r, :] = I_Mup[:, :, ir2 + r, :] * 0.5
        if c % 2 == 0:  # even
            I_Mup[:, :, :, ic2] = I_Mup[:, :, :, ic2] * 0.5
            I_Mup[:, :, :, ic2 + c] = I_Mup[:, :, :, ic2 + c] * 0.5

        I_Pup[:, :, :ir1, :ic1] = Pha[:, :, :ir1, :ic1]
        I_Pup[:, :, :ir1, ic2 + c:] = Pha[:, :, :ir1, ic2:]
        I_Pup[:, :, ir2 + r:, :ic1] = Pha[:, :, ir2:, :ic1]
        I_Pup[:, :, ir2 + r:, ic2 + c:] = Pha[:, :, ir2:, ic2:]

        if r % 2 == 0:  # even
            I_Pup[:, :, ir2, :] = I_Pup[:, :, ir2, :] * 0.5
            I_Pup[:, :, ir2 + r, :] = I_Pup[:, :, ir2 + r, :] * 0.5
        if c % 2 == 0:  # even
            I_Pup[:, :, :, ic2] = I_Pup[:, :, :, ic2] * 0.5
            I_Pup[:, :, :, ic2 + c] = I_Pup[:, :, :, ic2 + c] * 0.5

        real = I_Mup * torch.cos(I_Pup)
        imag = I_Mup * torch.sin(I_Pup)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return output



class freup_Cornerdinterpolation_v2(nn.Module):
    def __init__(self, channels):
        super(freup_Cornerdinterpolation_v2, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)  # n c h w
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







## the plug-and-play operator

class fresadd(nn.Module):
    def __init__(self, channels=32, version=1):
        super(fresadd, self).__init__()
        if version==1:
            self.Fup = freup_Areadinterpolation(channels)
        if version==2:
            self.Fup = freup_AreadinterpolationV2(channels)
        if version==3:
            self.Fup = freup_Periodicpadding(channels)
        if version==4:
            self.Fup = freup_Cornerdinterpolation(channels)

        self.fuse = nn.Conv2d(channels, channels,1,1,0)

    def forward(self,x):

        x1 = x
        
        x2 = F.interpolate(x1,scale_factor=2,mode='bilinear')
      
        x3 = self.Fup(x1)
     


        xm = x2 + x3
        xn = self.fuse(xm)

        return xn




class frescat(nn.Module):
    def __init__(self, channels=32, version=1):
        super(frescat, self).__init__()
        if version==1:
            self.Fup = freup_Areadinterpolation(channels)
        if version==2:
            self.Fup = freup_AreadinterpolationV2(channels)
        if version==3:
            self.Fup = freup_Periodicpadding(channels)
        if version==4:
            self.Fup = freup_Cornerdinterpolation(channels)

        self.fuse = nn.Conv2d(2*channels, channels,1,1,0)

    def forward(self,x):

        x1 = x
        
        x2 = F.interpolate(x1,scale_factor=2,mode='bilinear')
        # x2 =
      
        x3 = self.Fup(x1)
        
        xn = self.fuse(torch.cat([x2,x3],dim=1))
     
      
        return xn





class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)

        self.conv11_1 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)
        self.conv11_2 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)
        self.conv11_3 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1)
        self.conv11_4 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1)
        # self.conv11_5 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)

        self.b = 1

        self.fourierup1 = frescat(channels=3, version=4)   # 需要设置channels和versions的值. 这个模块插入在哪个位置就用哪个位置对应的channels,  version的值有1，2，3，4四个，对应四种傅立叶上采样算法。
        self.fourierup2 = frescat(channels=3, version=4) 
        self.fourierup3 = frescat(channels=6, version=4) 
        self.fourierup4 = frescat(channels=6, version=4) 

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1_f = F.interpolate(x,scale_factor=0.5,mode='bilinear')
        x1_f = self.fourierup1(x1_f)
        x1 = self.conv11_1(torch.cat([x, x1_f], dim=1))

        x2 = F.relu(self.conv2(x1))
        x2_f = F.interpolate(x1,scale_factor=0.5,mode='bilinear')
        x2_f = self.fourierup2(x2_f)
        x2 = self.conv11_2(torch.cat([x1, x2_f], dim=1))

        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))

        x3_f = F.interpolate(cat1,scale_factor=0.5,mode='bilinear')
        x3_f = self.fourierup3(x3_f)
        x3 = self.conv11_3(torch.cat([cat1, x3_f], dim=1))

        cat2 = torch.cat((x2, x3),1)
        x4 = F.relu(self.conv4(cat2))

        x4_f = F.interpolate(cat2,scale_factor=0.5,mode='bilinear')
        x4_f = self.fourierup4(x4_f)
        x4 = self.conv11_4(torch.cat([cat2, x4_f], dim=1))


        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))

        # x5_f = F.interpolate(x5,scale_factor=0.5,mode='bilinear')
        # x5_f = self.fourierup(x5_f)
        # x5 = self.conv11_5(torch.cat([x5, x5_f], dim=1))

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return F.relu(output)

if __name__ == '__main__':
    data = torch.randn(1, 3, 256, 256)
    model = AODNet()
    print(model(data).shape)
