# ------------------------------------------------------------------------
# Purpose: Implementation of  Theorem-1 in the paper
# Author: Yiwei Gui
# Date: 2025/4/3
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
'''
derain_nips: Half Instance Normalization Network for Image Restoration

@inproceedings{chen2021derain_nips,
  title={derain_nips: Half Instance Normalization Network for Image Restoration},
  author={Liangyu Chen and Xin Lu and Jie Zhang and Xiaojie Chu and Chengpeng Chen},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2021}
}
'''

import torch
import torch.nn as nn
import numpy as np

def pad(x, kernel_size=3, dilation=1):
    """For stride = 2 or stride = 3"""
    pad_total = dilation * (kernel_size - 1) - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    x_padded = torch.nn.functional.pad(x, pad=(pad_beg, pad_end, pad_beg, pad_end))
    return x_padded, pad_beg, pad_end

def lap_split(img, kernel):
    _, _, _, k_size = kernel.shape
    img_pad, pad_beg, pad_end = pad(img, kernel_size=k_size)
    # print(img.shape)
    low = nn.functional.conv2d(img_pad, kernel, bias=None, stride=2, groups=3)
    # print(low.shape)
    low_upsample = nn.functional.conv_transpose2d(low, kernel*4, bias=None, stride=2,groups=3)
    # print(low_upsample.shape)
    high = img - low_upsample[:,:,pad_beg:-pad_end, pad_beg:-pad_end]
    # print(high.shape)
    return low, high,pad_beg, pad_end

def LaplacianPyramid(img,kernel,n):
    levels = []
    pad_beg_list = []
    pad_end_list = []

    for i in range(n):
        img, high,pad_beg, pad_end = lap_split(img, kernel)
        levels.append(high)
        pad_beg_list.append(pad_beg)
        pad_end_list.append(pad_end)

    levels.append(img)

    return levels[::-1], pad_beg_list[::-1], pad_end_list[::-1]

def GaussianPyramid(img,kernel,n):
    levels = []
    low = img
    for i in range(n):
        _, _, _, k_size = kernel.shape
        low, pad_beg, pad_end = pad(low, kernel_size=k_size)
        low = nn.functional.conv2d(low, kernel,bias=None, stride=2, groups=3)
        levels.append(low)

    return levels[::-1]
# Theorem-1 implementation
class FourierUpTheoremOne(nn.Module):
    def __init__(self):
        super(FourierUpTheoremOne, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        # 做二维 FFT
        fft_x = torch.fft.fft2(x)

        # 周期复制到 2H x 2W，乘以 1/4 缩放
        fft_x_up = fft_x / 4

        fft_x_up = fft_x_up.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W, 1, 1]
        fft_x_up = fft_x_up.repeat(1, 1, 1, 1, 2, 2)     # → 每个点复制到四个象限
        fft_x_up = fft_x_up.permute(0, 1, 4, 2, 5, 3).reshape(B, C, 2*H, 2*W)

        # IFFT 回到空间域
        x_up = torch.fft.ifft2(fft_x_up).real  # 取实部

        return x_up


######## Laplacian and Gaussian Pyramid ########
class LPNet_pad_theory(nn.Module):

    def __init__(self, in_chn=3, num_pyramids=5,num_blocks=5, num_feature=32,relu_slope=0.2):
        super(LPNet_pad_theory, self).__init__()
        self.num_pyramids = num_pyramids
        self.num_blocks = num_blocks
        self.num_feature = num_feature
        self.k = np.float32([.0625, .25, .375, .25, .0625])  # Gaussian kernel for image pyramid
        self.k = np.outer( self.k,  self.k)
        self.kernel = self.k[None, None, :, :]
        # self.kernel = np.repeat(self.kernel, 3, axis=1)
        self.kernel = np.repeat(self.kernel, 3, axis=0)
        self.kernel = torch.tensor(self.kernel).cuda()


        self.subnet_0 = Subnet(num_feature=int((self.num_feature)/16), num_blocks=self.num_blocks)
        self.relu_0 = nn.LeakyReLU(0.2, inplace=False)
        self.fup0 = FourierUpTheoremOne()
        self.fuse_0 = nn.Conv2d(6,3,1,1,0)

        self.subnet_1 = Subnet(num_feature=int((self.num_feature) / 8), num_blocks=self.num_blocks)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=False)
        self.fup1 = FourierUpTheoremOne()
        self.fuse_1 = nn.Conv2d(6,3,1,1,0)

        self.subnet_2 = Subnet(num_feature=int((self.num_feature) / 4), num_blocks=self.num_blocks)
        self.relu_2 = nn.LeakyReLU(0.2, inplace=False)
        self.fup2 = FourierUpTheoremOne()
        self.fuse_2 = nn.Conv2d(6,3,1,1,0)

        self.subnet_3 = Subnet(num_feature=int((self.num_feature) / 2), num_blocks=self.num_blocks)
        self.relu_3 = nn.LeakyReLU(0.2, inplace=False)
        self.fup3= FourierUpTheoremOne()
        self.fuse_3 = nn.Conv2d(6,3,1,1,0)
        
        self.subnet_4 = Subnet(num_feature=int((self.num_feature) / 1), num_blocks=self.num_blocks)
        self.relu_4 = nn.LeakyReLU(0.2, inplace=False)


    def forward(self, images):
        pyramid, pad_beg_list, pad_end_list = LaplacianPyramid(images, self.kernel, (self.num_pyramids - 1))  # rainy Laplacian pyramid

        out_0 = self.subnet_0(pyramid[0])
        out_0 = self.relu_0(out_0)
        # out_0_t = nn.functional.conv_transpose2d(out_0, self.kernel*4, bias=None, stride=2,groups=3) + self.fup0(out_0)
        out_0_t = nn.functional.conv_transpose2d(out_0, self.kernel * 4, bias=None, stride=2, groups=3)
        out_0_t = out_0_t[:, :, pad_beg_list[0]:-pad_end_list[0], pad_beg_list[0]:-pad_end_list[0]]
        out_0_t = self.fuse_0(torch.concat([out_0_t,self.fup0(out_0)],1))

        out_1 = self.subnet_1(pyramid[1])
        out_1 = out_1 + out_0_t
        out_1 = self.relu_1(out_1)
        out_1_t = nn.functional.conv_transpose2d(out_1, self.kernel*4, bias=None, stride=2,groups=3)
        out_1_t = out_1_t[:, :, pad_beg_list[1]:-pad_end_list[1], pad_beg_list[1]:-pad_end_list[1]]
        
        out_1_t = self.fuse_1(torch.concat([out_1_t,self.fup1(out_1)],dim=1))

        out_2 = self.subnet_2(pyramid[2])
        out_2 = out_2 + out_1_t
        out_2 = self.relu_2(out_2)
        out_2_t = nn.functional.conv_transpose2d(out_2, self.kernel*4, bias=None, stride=2,groups=3)
        out_2_t = out_2_t[:, :, pad_beg_list[2]:-pad_end_list[2], pad_beg_list[2]:-pad_end_list[2]]
        out_2_t = self.fuse_2(torch.concat([out_2_t,self.fup2(out_2)],dim=1))

        out_3 = self.subnet_3(pyramid[3])
        out_3 = out_3 + out_2_t
        out_3 = self.relu_3(out_3)
        out_3_t = nn.functional.conv_transpose2d(out_3, self.kernel*4, bias=None, stride=2,groups=3)
        out_3_t = out_3_t[:, :, pad_beg_list[3]:-pad_end_list[3], pad_beg_list[3]:-pad_end_list[3]]
        out_3_t = self.fuse_3(torch.concat([out_3_t,self.fup3(out_3)],dim=1))

        out_4 = self.subnet_4(pyramid[4])
        out_4 = out_4 + out_3_t
        out_4 = self.relu_4(out_4)

        outout_pyramid = []
        outout_pyramid.append(out_0)
        outout_pyramid.append(out_1)
        outout_pyramid.append(out_2)
        outout_pyramid.append(out_3)
        outout_pyramid.append(out_4)
        return outout_pyramid


    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class Subnet(nn.Module):
    def __init__(self,num_feature, num_blocks):
        super(Subnet, self).__init__()
        self.num_blocks = num_blocks
        self.conv_0 = nn.Conv2d(3, num_feature, kernel_size=3, padding=1, bias=True)
        self.relu_0 = nn.LeakyReLU(0.2, inplace=False)

        self.conv_1 = nn.Conv2d(num_feature, num_feature, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=False)

        self.conv_2 = nn.Conv2d(num_feature, num_feature, kernel_size=1, bias=True)
        self.relu_2 = nn.LeakyReLU(0.2, inplace=False)

        self.conv_3 = nn.Conv2d(num_feature, num_feature, kernel_size=3, padding=1, bias=True)
        self.relu_3 = nn.LeakyReLU(0.2, inplace=False)

        self.conv_4 = nn.Conv2d(num_feature, num_feature, kernel_size=3, padding=1, bias=True)
        self.relu_4 = nn.LeakyReLU(0.2, inplace=False)

        self.conv_5= nn.Conv2d(num_feature, num_feature, kernel_size=3, padding=1, bias=True)
        self.relu_5 = nn.LeakyReLU(0.2, inplace=False)

        self.conv_6= nn.Conv2d(num_feature, num_feature, kernel_size=3, padding=1, bias=True)
        self.relu_6 = nn.LeakyReLU(0.2, inplace=False)
        self.conv_7= nn.Conv2d(num_feature, num_feature, kernel_size=3,padding=1,  bias=True)
        self.relu_7 = nn.LeakyReLU(0.2, inplace=False)
        self.conv_8= nn.Conv2d(num_feature, 3, kernel_size=1,  bias=True)


    def forward(self, images):
        out = self.conv_0(images)
        out = self.relu_0(out)

        #  recursive blocks
        for i  in range(self.num_blocks):
            out = self.conv_1(out)
            out = self.relu_1(out)

            out = self.conv_2(out)
            out = self.relu_2(out)

            out = self.conv_3(out)
            out = self.relu_3(out)
            out = self.conv_4(out)
            out = self.relu_4(out)
            out = self.conv_5(out)
            out = self.relu_5(out)
            out = self.conv_6(out)
            out = self.relu_6(out)
            out = self.conv_7(out)
            out = self.relu_7(out)


        out = self.conv_8(out)

        out = out + images

        return out

if __name__ == "__main__":
    pass
