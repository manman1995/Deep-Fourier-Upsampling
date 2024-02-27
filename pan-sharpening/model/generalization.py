import torch
import torch.nn as nn
from torchvision.transforms import *
import torch.nn.functional as F
# from dirfl_utils.INN import *
# from dirfl_utils.UnetInn_SNR import ColorNet, LowNet
# from dirfl_utils.pasa import Downsample_PASA_group_softmax
from .modules import InvertibleConv1x1
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2
def create_mask(b,c,h,w):
    tensor_size = (1, 1, 128, 128)
    r = torch.tensor(80).cuda()
    # 生成网格点坐标
    x = torch.linspace(0, 128, 128, dtype=torch.int).cuda()
    y = torch.linspace(0, 128, 128, dtype=torch.int).cuda()
    xx, yy = torch.meshgrid(x, y)
    xx = xx.cuda()
    yy = yy.cuda()
    d = torch.sqrt((xx - 64) ** 2 + (yy - 64) ** 2).cuda()  # 在这里将中心点设置为(0,0)
    # 创建输出张量
    out_tensor = torch.where(d>=r.view(1,1,1,1),torch.ones(tensor_size).cuda(),torch.zeros(tensor_size).cuda())
    return out_tensor.cuda()

# def create_mask(batch_size, channels, height, width):
#     # 创建一个全零的mask
#     mask = torch.zeros((batch_size, channels, height, width), dtype=torch.float32)

#     # 计算图像中心
#     center_h = height // 2
#     center_w = width // 2

#     # 计算半径
#     radius = 0.9 * min(center_h, center_w)

#     # 生成网格坐标
#     grid_h, grid_w = torch.meshgrid(torch.arange(height), torch.arange(width))

#     # 计算距离中心的距离
#     distance = torch.sqrt((grid_h - center_h)**2 + (grid_w - center_w)**2)

#     # 将小于半径的部分设置为0，大于等于半径的部分设置为1
#     mask[distance < radius] = 0
#     mask[distance >= radius] = 1

#     return mask

def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


def conv_identify(weight, bias):
    weight.data.zero_()
    if bias is not None:
        bias.data.zero_()
    o, i, h, w = weight.shape
    y = h // 2
    x = w // 2
    for p in range(i):
        for q in range(o):
            if p == q:
                weight.data[q, p, :, :] = 1.0


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if (self.filt_size == 1):
            a = np.array([1., ])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat(self.channels, 1, 1, 1))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Downsample_PASA_group_softmax(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2):
        super(Downsample_PASA_group_softmax, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size // 2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group * kernel_size * kernel_size, kernel_size=kernel_size, stride=1,
                              bias=False)
        self.bn = nn.InstanceNorm2d(group * kernel_size * kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma_o = self.conv(self.pad(x))
        sigma_o = self.bn(sigma_o)
        sigma_o = self.softmax(sigma_o)

        n, c, h, w = sigma_o.shape

        sigma_o = sigma_o.reshape(n, 1, c, h * w)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(
            (n, c, self.kernel_size * self.kernel_size, h * w))

        n, c1, p, q = x.shape
        x = x.permute(1, 0, 2, 3).reshape(self.group, c1 // self.group, n, p, q).permute(2, 0, 1, 3, 4)

        n, c2, p, q = sigma_o.shape
        sigma = sigma_o.permute(2, 0, 1, 3).reshape(
            (p // (self.kernel_size * self.kernel_size), self.kernel_size * self.kernel_size, n, c2, q)).permute(2, 0,
                                                                                                                 3, 1,
                                                                                                                 4)
        sigma_o = sigma_o.reshape(sigma.shape)
        x = torch.sum(x * (sigma), dim=3).reshape(n, c1, h, w)
        return x[:, :, torch.arange(h) % self.stride == 0, :][:, :, :, torch.arange(w) % self.stride == 0]

def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.process = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x


class Refine(nn.Module):
    def __init__(self, n_feat, out_channel):
        super(Refine, self).__init__()
        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
            #  CALayer(n_feat,4),
            CALayer(n_feat, 4),
            CALayer(n_feat, 4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out

class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out
class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_3 = nn.Conv2d(in_size+in_size,out_size,3,1,1)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        # input = torch.cat([x,resi],dim=1)
        # out = self.conv_3(input)
        return x+resi

class HFeatureProcess(nn.Module):
    '''STP: Structural Preservation Mudule '''
    def __init__(self, channels):
        super(HFeatureProcess, self).__init__()
        self.pre_fuse = nn.Sequential(InvBlock(HinResBlock, 2*channels, channels),InvBlock(HinResBlock, 2*channels, channels),InvBlock(HinResBlock, 2*channels, channels),InvBlock(HinResBlock, 2*channels, channels),InvBlock(HinResBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels,channels,1,1,0))
        self.panconv = nn.Conv2d(1, channels, 3, stride=1, padding=1)
        self.msconv = nn.Conv2d(4, channels, 3, stride=1, padding=1)
        self.refine = Refine(channels, 4)
    def forward(self, msh, panh):
        msf = self.msconv(msh)
        panf = self.panconv(panh)
        hp_fused = self.pre_fuse(torch.cat([panf, msf], 1))
        return hp_fused


class SFeatureProcess(nn.Module):
    '''SPP: Spectral Preservation Module'''
    def __init__(self, in_channal, out_channal,nc=32):
        super(SFeatureProcess, self).__init__()
        self.pre_fuse = nn.Sequential(InvBlock(HinResBlock, 2*nc, nc),InvBlock(HinResBlock, 2*nc, nc),InvBlock(HinResBlock, 2*nc, nc),
                                         nn.Conv2d(2*nc,nc,1,1,0))
        self.msphaconv = nn.Conv2d(in_channal, nc, 1, stride=1, padding=0)
        self.panhaconv = nn.Conv2d(1, nc, 1, stride=1, padding=0)
        self.pha_fuse = nn.Sequential(nn.Conv2d(2*nc,2*nc,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(2*nc,2*nc,1,1,0))
        self.amp_fuse = nn.Sequential(nn.Conv2d(2*nc,2*nc,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(2*nc,2*nc,1,1,0))
        self.panamp = nn.Conv2d(1, nc, 1, stride=1, padding=0)
        self.msamp = nn.Conv2d(in_channal, nc, 1, stride=1, padding=0)
        self.mask = create_mask(1,4,128,128)
    def forward(self, ms, pan):
        '''decouples the phase and amplitude of PAN and LRMS'''
        H, W = ms.shape[-2:]
        # DFT MS

        ms_fft = torch.fft.fft2(ms, norm='ortho')
        ms_amp = torch.abs(ms_fft)
        out_amp = ms_amp
        ms_pha = torch.angle(ms_fft)
        pan_fft = torch.fft.fft2(pan, norm='ortho')
        pan_amp = torch.abs(pan_fft)
        pan_pha = torch.angle(pan_fft)
        #pha融合
        ms_pha = self.msphaconv(ms_pha)
        pan_pha = self.panhaconv(pan_pha)
        f_pha = self.pha_fuse(torch.cat([ms_pha, pan_pha], 1))
        
        #ms
        ms_amp*=self.mask
        ms_amp = self.msamp(ms_amp)
        
        #pan
        pan_amp = self.panamp(pan_amp)
        f_amp = self.amp_fuse(torch.cat([ms_amp, pan_amp], 1))

        real = f_amp * torch.cos(f_pha)
        imag = f_amp * torch.sin(f_pha)
        domain_inver = torch.complex(real, imag)
        domain_inver = torch.abs(torch.fft.ifft2(domain_inver, s=(H, W), norm='ortho'))
        out = self.pre_fuse(domain_inver)

        return out,out_amp
def instance_norm(x):
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
    x = (x - mean) / (torch.sqrt(var)+1e-5)
    return x,mean,var


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()

        channels = 4
        ms_channel = num_channels
        pan_channel = 1
        base_filter=32
        self.ms_fliter = Downsample_PASA_group_softmax(ms_channel, 1, 1)
        self.pan_fliter = Downsample_PASA_group_softmax(ms_channel, 1, 1)
        self.convin = nn.Conv2d(pan_channel, ms_channel, 1, 1, padding='same')
        self.convout = nn.Conv2d(ms_channel, pan_channel, 1, 1, padding='same')

        self.STP = HFeatureProcess(base_filter)
        self.SPP = SFeatureProcess(in_channal=channels, out_channal=channels)
        self.refine = Refine(base_filter, 4)
    def forward(self, lms, _, pan):
        Batchnum, _, M, N = pan.shape
        mHR = upsample(lms, M, N)
        # mHR,mean,var = instance_norm(mHR)
        #得到空间域的高频分量
        # ms_high = mHR - self.ms_fliter(mHR)
        # pan_high = pan - self.convout(self.pan_fliter(self.convin(pan)))
        # #高频融合结果
        # h_fused = self.STP(ms_high, pan_high)

        #进行域无关的频域融合（振幅&相位）
        f_fused,ms_amp = self.SPP(mHR, pan)
        fused_uni = self.refine(f_fused)+mHR
        # fused_uni = self.refine(torch.cat([h_fused, f_fused], 1))
        # fused_uni_fft = torch.fft.fft2(fused_uni + 1e-8, norm='backward')
        # fused_uni_pha = torch.angle(fused_uni_fft)

        # real = ms_amp * torch.cos(fused_uni_pha) + 1e-8
        # imag = ms_amp * torch.sin(fused_uni_pha) + 1e-8
        # fuse = torch.complex(real, imag) + 1e-8
        # fuse = torch.abs(torch.fft.ifft2(fuse, s=(M, N), norm='backward'))+mHR
        return fused_uni
        # return (fused_uni)*(torch.sqrt(var)+1e-5) + mean


if __name__ == '__main__':
    pass
    # f = create_mask(1,1,128,128)
    # print(f.size())
    # print(f[0][0][0][0])
    # print(create_mask(1,1,128,128))
    # a = Net(1, 1, 1)
    # x = torch.randn((1, 4, 128, 128))
    # y = torch.randn((1, 1, 128, 128))
    # fuse = a(x, x, y)
    # print(fuse.size())
