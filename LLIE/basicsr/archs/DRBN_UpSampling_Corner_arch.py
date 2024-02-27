import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer, ConvLReLUNoBN, upsample_and_concat, single_conv, up, outconv
from basicsr.utils.registry import ARCH_REGISTRY

import os
import torch
import torch.nn as nn
from torchvision.transforms import *
import torch.nn.functional as F
import numpy as np

def calculate_d(x, y, M, N):
    term1 = 1
    term2 = torch.exp(1j * torch.tensor(np.pi) * x / M)
    term3 = torch.exp(1j * torch.tensor(np.pi) * y / N)
    term4 = torch.exp(1j * torch.tensor(np.pi) * (x/M + y/N))

    result = term1 + term2 + term3 + term4
    return torch.abs(result) / 4

def get_D_map(feature):
    B, C, H, W = feature.shape
    out = torch.zeros((1, 1, 2*H, 2*W), dtype=torch.float32)

    for i in range(2*H):
        for j in range(2*W):
            if i < H and j < W:
                out[:, :, i, j] = calculate_d(i, j, H, W)
            elif i >= H and j < W:
                out[:, :, i, j] = calculate_d(2*H - i, j, H, W)
            elif i < H and j >= W:
                out[:, :, i, j] = calculate_d(i, 2*W - j, H, W)
            else:
                out[:, :, i, j] = calculate_d(2*H - i, 2*W - j, H, W)
    return out

class freup_Areadinterpolation(nn.Module):
    def __init__(self, channels, channel_out):
        super(freup_Areadinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))

        self.post = nn.Conv2d(channels,channel_out,1,1,0)

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
        d_map= get_D_map(x)
        output = output / d_map
        crop = torch.zeros_like(x)
        crop[:, :, 0:int(H/2), 0:int(W/2)] = output[:, :, 0:int(H/2), 0:int(W/2)]
        crop[:, :, int(H/2):H, 0:int(W/2)] = output[:, :, int(H*1.5):2*H, 0:int(W/2)]
        crop[:, :, 0:int(H/2), int(W/2):W] = output[:, :, 0:int(H/2), int(W*1.5):2*W]
        crop[:, :, int(H/2):H, int(W/2):W] = output[:, :, int(H*1.5):2*H, int(W*1.5):2*W]
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
    def __init__(self, channels, channels_out):
        super(freup_Cornerdinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        self.post = nn.Conv2d(channels,channels_out,1,1,0)

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

        return self.post(output)


class fresadd(nn.Module):
    def __init__(self, channels=32):
        super(fresadd, self).__init__()
        
        self.Fup = freup_Areadinterpolation(channels)

        self.fuse = nn.Conv2d(channels, channels,1,1,0)

    def forward(self,x):

        x1 = x
        
        x2 = F.interpolate(x1,scale_factor=2,mode='bilinear')
      
        x3 = self.Fup(x1)
     
        xm = x2 + x3
        xn = self.fuse(xm)

        return xn


class frescat_area(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(frescat_area, self).__init__()
        
        self.up_f = freup_Cornerdinterpolation(channels_in, channels_out)
        self.up = nn.ConvTranspose2d(channels_in, channels_out, 4, stride=2, padding=1)

        self.fuse = nn.Conv2d(2*channels_out, channels_out,1,1,0)

    def forward(self,x):

        x1 = x
        x2 = self.up(x1)
        x3 = self.up_f(x1)
        
        xn = self.fuse(torch.cat([x2,x3],dim=1))
     
        return xn
    

class frescat_areaV2(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(frescat_areaV2, self).__init__()
        
        self.up_f = freup_AreadinterpolationV2(channels_in, channels_out)
        self.up = nn.ConvTranspose2d(channels_in, channels_out, 4, stride=2, padding=1)

        self.fuse = nn.Conv2d(2*channels_out, channels_out,1,1,0)

    def forward(self,x):

        x1 = x
        x2 = self.up(x1)
        x3 = self.up_f(x1)
        
        xn = self.fuse(torch.cat([x2,x3],dim=1))
     
        return xn

    
class frescat_padding(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(frescat_padding, self).__init__()
        
        self.up_f = freup_Periodicpadding(channels_in, channels_out)
        self.up = nn.ConvTranspose2d(channels_in, channels_out, 4, stride=2, padding=1)

        self.fuse = nn.Conv2d(2*channels_out, channels_out,1,1,0)

    def forward(self,x):

        x1 = x
        x2 = self.up(x1)
        x3 = self.up_f(x1)
        
        xn = self.fuse(torch.cat([x2,x3],dim=1))
     
        return xn


class frescat_corner(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(frescat_corner, self).__init__()
        
        self.up_f = freup_Cornerdinterpolation(channels_in, channels_out)
        self.up = nn.ConvTranspose2d(channels_in, channels_out, 4, stride=2, padding=1)

        self.fuse = nn.Conv2d(2*channels_out, channels_out,1,1,0)

    def forward(self,x):

        x1 = x
        x2 = self.up(x1)
        x3 = self.up_f(x1)
        
        xn = self.fuse(torch.cat([x2,x3],dim=1))
     
        return xn


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        feat1 = self.convs(x)
        feat2 = self.LFF(feat1) + x
        return feat2

class DRBN_BU(nn.Module):
    def __init__(self, n_color):
        super(DRBN_BU, self).__init__()

        G0 = 16
        kSize = 3
        self.D = 6
        G = 8
        C = 4

        self.SFENet1 = nn.Conv2d(n_color*2, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        self.RDBs = nn.ModuleList()

        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = 2*G0, growRate = 2*G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = 2*G0, growRate = 2*G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.UPNet2 = nn.Sequential(*[
                nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.UPNet4 = nn.Sequential(*[
                nn.Conv2d(G0*2, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.Down1 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=2)
        self.Down2 = nn.Conv2d(G0, G0*2, kSize, padding=(kSize-1)//2, stride=2)

        self.Up1 = nn.ConvTranspose2d(G0, G0, kSize+1, stride=2, padding=1)
        # self.Up1 = frescat_corner(G0, G0)
        # self.Up2 = nn.ConvTranspose2d(G0*2, G0, kSize+1, stride=2, padding=1)
        self.Up2 = frescat_corner(G0*2, G0)

        self.Relu = nn.ReLU()
        self.Img_up = nn.Upsample(scale_factor=2, mode='bilinear')

    def part_forward(self, x):
        #
        # Stage 1
        #
        flag = x[0]
        input_x = x[1]

        prev_s1 = x[2]
        prev_s2 = x[3]
        prev_s4 = x[4]

        prev_feat_s1 = x[5]
        prev_feat_s2 = x[6]
        prev_feat_s4 = x[7]

        f_first = self.Relu(self.SFENet1(input_x))
        f_s1  = self.Relu(self.SFENet2(f_first))
        f_s2 = self.Down1(self.RDBs[0](f_s1))
        f_s4 = self.Down2(self.RDBs[1](f_s2))

        if flag == 0:
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4))
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4))
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2))+f_first
        else:
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4)) + prev_feat_s4
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4)) + prev_feat_s2
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2))+f_first + prev_feat_s1

        res4 = self.UPNet4(f_s4)
        res2 = self.UPNet2(f_s2) + self.Img_up(res4)
        res1 = self.UPNet(f_s1) + self.Img_up(res2)

        return res1, res2, res4, f_s1, f_s2, f_s4


    def forward(self, x_input):
        x = x_input

        res1, res2, res4, f_s1, f_s2, f_s4 = self.part_forward(x)

        return res1, res2, res4, f_s1, f_s2, f_s4


@ARCH_REGISTRY.register()
class DRBN_UpSampling_Corner(nn.Module):
    def __init__(self, n_color):
        super(DRBN_UpSampling_Corner, self).__init__()

        self.recur1 = DRBN_BU(n_color)
        self.recur2 = DRBN_BU(n_color)
        self.recur3 = DRBN_BU(n_color)
        self.recur4 = DRBN_BU(n_color)

    def forward(self, x_input):
        x = x_input

        res_g1_s1, res_g1_s2, res_g1_s4, feat_g1_s1, feat_g1_s2, feat_g1_s4 = self.recur1([0, torch.cat((x, x), 1), 0, 0, 0, 0, 0, 0])
        res_g2_s1, res_g2_s2, res_g2_s4, feat_g2_s1, feat_g2_s2, feat_g2_s4 = self.recur2([1, torch.cat((res_g1_s1, x), 1), res_g1_s1, res_g1_s2, res_g1_s4, feat_g1_s1, feat_g1_s2, feat_g1_s4])
        res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4 = self.recur3([1, torch.cat((res_g2_s1, x), 1), res_g2_s1, res_g2_s2, res_g2_s4, feat_g2_s1, feat_g2_s2, feat_g2_s4])
        res_g4_s1, res_g4_s2, res_g4_s4, feat_g4_s1, feat_g4_s2, feat_g4_s4 = self.recur4([1, torch.cat((res_g3_s1, x), 1), res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4])

        return res_g4_s1, res_g4_s2, res_g4_s4
