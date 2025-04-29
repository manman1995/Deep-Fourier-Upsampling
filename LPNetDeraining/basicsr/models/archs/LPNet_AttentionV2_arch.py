# ------------------------------------------------------------------------
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
import math

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

# Lightweight FrequencyAttention module
class FrequencyAttention(nn.Module):
    def __init__(self, channels, num_heads=2, dim_head=8, dropout=0.4):
        super(FrequencyAttention, self).__init__()
        inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        # Reduce parameter count by using smaller dimensions
        self.to_q = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(channels, inner_dim, kernel_size=1, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, channels, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w = x.shape # batch,channel,height,width
        
        # Project to query, key, value separately
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape for multi-head attention - ensure consistent dimensions
        q = q.reshape(b, self.num_heads, self.dim_head, h*w).permute(0, 1, 3, 2)  # b, heads, h*w, dim_head
        k = k.reshape(b, self.num_heads, self.dim_head, h*w)  # b, heads, dim_head, h*w
        v = v.reshape(b, self.num_heads, self.dim_head, h*w).permute(0, 1, 3, 2)  # b, heads, h*w, dim_head
        
        # Compute attention scores
        attn = torch.matmul(q, k) * self.scale  # b, heads, h*w, h*w
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # b, heads, h*w, dim_head
        out = out.permute(0, 1, 3, 2).reshape(b, -1, h, w)  # b, heads*dim_head, h, w
        
        # Project back to original dimension
        return self.to_out(out)

# Modified freup_pad with frequency domain self-attention
class freup_pad_attention(nn.Module):
    def __init__(self, channels, num_heads=4, dim_head=8, dropout=0.4, use_attention=False):
        super(freup_pad_attention, self).__init__()
        
        # Flag to optionally disable attention
        self.use_attention = use_attention

        # Amplitude and phase processing
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        
        # Self-attention modules (only used when use_attention is True)
        if use_attention:
            self.amp_attention = FrequencyAttention(channels, num_heads, dim_head, dropout)
            self.pha_attention = FrequencyAttention(channels, num_heads, dim_head, dropout)
            
            # Feature fusion after attention
            self.amp_post = nn.Conv2d(channels, channels, 1, 1, 0)
            self.pha_post = nn.Conv2d(channels, channels, 1, 1, 0)
        
        # Final processing
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        N, C, H, W = x.shape

        # Convert to frequency domain
        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        # Process amplitude and phase
        # Mag = self.amp_fuse(mag_x)
        # Pha = self.pha_fuse(pha_x)
        
        # Apply self-attention in frequency domain if enabled
        if self.use_attention:
            Mag = mag_x
            Pha = pha_x
            Mag_attn = self.amp_attention(Mag)
            Pha_attn = self.pha_attention(Pha)
            
            # Post-process attended features
            Mag = self.amp_post(Mag_attn + Mag)  # Residual connection
            Pha = self.pha_post(Pha_attn + Pha)  # Residual connection
        else:
            # Process amplitude and phase
            Mag = self.amp_fuse(mag_x)
            Pha = self.pha_fuse(pha_x)

        
        # Repeat (tile) in whole block horizontally and vertically
        amp_fuse = torch.tile(Mag, (2, 2))
        pha_fuse = torch.tile(Pha, (2, 2))

        # Convert back to spatial domain
        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        
        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return self.post(output)


class LPNet_pad_attentionV2(nn.Module):

    def __init__(self, in_chn=3, num_pyramids=5,num_blocks=5, num_feature=32,relu_slope=0.2):
        super(LPNet_pad_attentionV2, self).__init__()
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
        self.fup0 = freup_pad_attention(3, num_heads=2, dim_head=8, use_attention=True)
        self.fuse_0 = nn.Conv2d(6,3,1,1,0) # in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0

        self.subnet_1 = Subnet(num_feature=int((self.num_feature) / 8), num_blocks=self.num_blocks)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=False)
        self.fup1 = freup_pad_attention(3, num_heads=2, dim_head=8, use_attention=False)
        self.fuse_1 = nn.Conv2d(6,3,1,1,0)

        self.subnet_2 = Subnet(num_feature=int((self.num_feature) / 4), num_blocks=self.num_blocks)
        self.relu_2 = nn.LeakyReLU(0.2, inplace=False)
        self.fup2 = freup_pad_attention(3, num_heads=2, dim_head=8, use_attention=True)
        self.fuse_2 = nn.Conv2d(6,3,1,1,0)
        self.subnet_3 = Subnet(num_feature=int((self.num_feature) / 2), num_blocks=self.num_blocks)
        self.relu_3 = nn.LeakyReLU(0.2, inplace=False)
        self.fup3 = freup_pad_attention(3, num_heads=2, dim_head=8, use_attention=False)
        self.fuse_3 = nn.Conv2d(6,3,1,1,0)
        self.subnet_4 = Subnet(num_feature=int((self.num_feature) / 1), num_blocks=self.num_blocks)
        self.relu_4 = nn.LeakyReLU(0.2, inplace=False)


    def forward(self, images):
        pyramid, pad_beg_list, pad_end_list = LaplacianPyramid(images, self.kernel, (self.num_pyramids - 1))  # rainy Laplacian pyramid

        out_0 = self.subnet_0(pyramid[0])
        out_0 = self.relu_0(out_0)
        # out_0_t = nn.functional.conv_transpose2d(out_0, self.kernel*4, bias=None, stride=2,groups=3) + self.fup0(out_0)
        out_0_t = nn.functional.conv_transpose2d(out_0, self.kernel * 4, bias=None, stride=2,groups=3)
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
        out_3_t = nn.functional.conv_transpose2d(out_3, self.kernel*4, bias=None, stride=2,groups=3) # normal spatial upsampling
        out_3_t = out_3_t[:, :, pad_beg_list[3]:-pad_end_list[3], pad_beg_list[3]:-pad_end_list[3]]
        out_3_t = out_3_t + self.fup3(out_3) # necessary
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
