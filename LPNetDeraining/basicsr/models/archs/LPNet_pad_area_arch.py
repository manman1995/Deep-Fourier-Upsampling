import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
def get_D_map_optimized(feature):
    B, C, H, W = feature.shape
    d_map = torch.zeros((1, 1, H, W), dtype=torch.float32).to(feature.device)
    
    #Create a grid to store the indices of all (i, j) pairs
    i_indices = torch.arange(H, dtype=torch.float32).reshape(1, 1, H, 1).repeat(1, 1, 1, W).cuda()
    j_indices = torch.arange(W, dtype=torch.float32).reshape(1, 1, 1, W).repeat(1, 1, H, 1).cuda()
    
    # Compute d_map using vectorization operations
    d_map[:, :, :, :] = calculate_d(i_indices, j_indices, H, W)
    
    return d_map
import numpy as np
def calculate_d(x, y, M, N):
    term1 = 1
    term2 = torch.exp(1j * torch.tensor(np.pi) * x / M).cuda()
    term3 = torch.exp(1j * torch.tensor(np.pi) * y / N).cuda()
    term4 = torch.exp(1j * torch.tensor(np.pi) * (x/M + y/N)).cuda()

    result = term1 + term2 + term3 + term4
    return torch.abs(result) / 4

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
    low = nn.functional.conv2d(img_pad, kernel, bias=None, stride=2, groups=3)
    low_upsample = nn.functional.conv_transpose2d(low, kernel*4, bias=None, stride=2,groups=3)
    high = img - low_upsample[:,:,pad_beg:-pad_end, pad_beg:-pad_end]
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

class LPNet_pad_area(nn.Module):
    def __init__(self, in_chn=3, num_pyramids=5, num_blocks=5, num_feature=32, relu_slope=0.2):
        super(LPNet_pad_area, self).__init__()
        self.num_pyramids = num_pyramids
        self.num_blocks = num_blocks
        self.num_feature = num_feature
        self.k = np.float32([.0625, .25, .375, .25, .0625])  # Gaussian kernel for image pyramid
        self.k = np.outer(self.k,  self.k)
        self.kernel = self.k[None, None, :, :]
        self.kernel = np.repeat(self.kernel, 3, axis=0)
        self.kernel = torch.tensor(self.kernel).cuda()

        self.subnet_0 = Subnet(num_feature=int((self.num_feature)/16), num_blocks=self.num_blocks)
        self.relu_0 = nn.LeakyReLU(0.2, inplace=False)
        self.fup0_area = freup_AreadinterpolationV2(3)
        self.fup0_pad = freup_pad(3)
        self.fuse_0 = nn.Conv2d(9, 3, 1, 1, 0)  # 9 channels: 3 from corner + 3 from pad + 3 from original

        self.subnet_1 = Subnet(num_feature=int((self.num_feature) / 8), num_blocks=self.num_blocks)
        self.relu_1 = nn.LeakyReLU(0.2, inplace=False)
        self.fup1_area = freup_AreadinterpolationV2(3)
        self.fup1_pad = freup_pad(3)
        self.fuse_1 = nn.Conv2d(9, 3, 1, 1, 0)

        self.subnet_2 = Subnet(num_feature=int((self.num_feature) / 4), num_blocks=self.num_blocks)
        self.relu_2 = nn.LeakyReLU(0.2, inplace=False)
        self.fup2_area = freup_AreadinterpolationV2(3)
        self.fup2_pad = freup_pad(3)
        self.fuse_2 = nn.Conv2d(9, 3, 1, 1, 0)

        self.subnet_3 = Subnet(num_feature=int((self.num_feature) / 2), num_blocks=self.num_blocks)
        self.relu_3 = nn.LeakyReLU(0.2, inplace=False)
        self.fup3_area = freup_AreadinterpolationV2(3)
        self.fup3_pad = freup_pad(3)
        self.fuse_3 = nn.Conv2d(9, 3, 1, 1, 0)

        self.subnet_4 = Subnet(num_feature=int((self.num_feature) / 1), num_blocks=self.num_blocks)
        self.relu_4 = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, images):
        pyramid, pad_beg_list, pad_end_list = LaplacianPyramid(images, self.kernel, (self.num_pyramids - 1))

        out_0 = self.subnet_0(pyramid[0])
        out_0 = self.relu_0(out_0)
        out_0_t = nn.functional.conv_transpose2d(out_0, self.kernel * 4, bias=None, stride=2,groups=3)
        out_0_t = out_0_t[:, :, pad_beg_list[0]:-pad_end_list[0], pad_beg_list[0]:-pad_end_list[0]]

        area_0 = self.fup0_area(out_0)
        pad_0 = self.fup0_pad(out_0)
        out_0_t = self.fuse_0(torch.cat([out_0_t, area_0, pad_0], dim=1))

        out_1 = self.subnet_1(pyramid[1])
        out_1 = out_1 + out_0_t
        out_1 = self.relu_1(out_1)
        out_1_t = nn.functional.conv_transpose2d(out_1, self.kernel*4, bias=None, stride=2,groups=3)
        out_1_t = out_1_t[:, :, pad_beg_list[1]:-pad_end_list[1], pad_beg_list[1]:-pad_end_list[1]]
        
        area_1 = self.fup1_area(out_1)
        pad_1 = self.fup1_pad(out_1)
        out_1_t = self.fuse_1(torch.cat([out_1_t, area_1, pad_1], dim=1))

        out_2 = self.subnet_2(pyramid[2])
        out_2 = out_2 + out_1_t
        out_2 = self.relu_2(out_2)
        out_2_t = nn.functional.conv_transpose2d(out_2, self.kernel*4, bias=None, stride=2,groups=3)
        out_2_t = out_2_t[:, :, pad_beg_list[2]:-pad_end_list[2], pad_beg_list[2]:-pad_end_list[2]]
        
        area_2 = self.fup2_area(out_2)
        pad_2 = self.fup2_pad(out_2)
        out_2_t = self.fuse_2(torch.cat([out_2_t, area_2, pad_2], dim=1))

        out_3 = self.subnet_3(pyramid[3])
        out_3 = out_3 + out_2_t
        out_3 = self.relu_3(out_3)
        out_3_t = nn.functional.conv_transpose2d(out_3, self.kernel*4, bias=None, stride=2,groups=3)
        out_3_t = out_3_t[:, :, pad_beg_list[3]:-pad_end_list[3], pad_beg_list[3]:-pad_end_list[3]]

        area_3 = self.fup3_area(out_3)
        pad_3 = self.fup3_pad(out_3)
        out_3_t = area_3 + pad_3 + out_3_t
        out_3_t = self.fuse_3(torch.cat([out_3_t, area_3, pad_3], dim=1))

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