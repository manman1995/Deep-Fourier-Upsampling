import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up == nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1),
            )
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_chans * 2, out_chans, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride)


## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias=False):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class Decoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, num, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(Decoder_MDCBlock1, self).__init__()
        self.num_ft = num_ft - 1
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        a = num_filter
        for i in range(self.num_ft):
            b = a + 2 ** (num + i)
            self.down_convs.append(ConvBlock(a, b, kernel_size, stride, padding, bias, activation, norm=None))
            self.up_convs.append(DeconvBlock(b, a, kernel_size, stride, padding, bias, activation, norm=None))
            a = b

    def forward(self, ft_h, ft_l_list):
        ft_fusion = ft_h
        for i in range(len(ft_l_list)):
            ft = ft_fusion
            for j in range(self.num_ft - i):
                ft = self.down_convs[j](ft)
            ft = F.interpolate(ft, size=ft_l_list[i].shape[-2:], mode='bilinear')
            ft = ft - ft_l_list[i]
            for j in range(self.num_ft - i):
                ft = self.up_convs[self.num_ft - i - j - 1](ft)
            ft_fusion = F.interpolate(ft_fusion, size=ft.shape[-2:], mode='bilinear')
            ft_fusion = ft_fusion + ft

        return ft_fusion


class Encoder_MDCBlock1(torch.nn.Module):
    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(Encoder_MDCBlock1, self).__init__()

        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        a = num_filter
        for i in range(self.num_ft):
            b = a - 2 ** (num_ft - i)
            self.up_convs.append(DeconvBlock(a, b, kernel_size, stride, padding, bias, activation, norm=None))
            self.down_convs.append(ConvBlock(b, a, kernel_size, stride, padding, bias, activation, norm=None))
            a = b

    def forward(self, ft_l, ft_h_list):
        ft_fusion = ft_l
        for i in range(len(ft_h_list)):
            ft = ft_fusion
            for j in range(self.num_ft - i):
                ft = self.up_convs[j](ft)
            ft = F.interpolate(ft, size=ft_h_list[i].shape[-2:], mode='bilinear')
            ft = ft - ft_h_list[i]
            for j in range(self.num_ft - i):
                # print(j)
                ft = self.down_convs[self.num_ft - i - j - 1](ft)
            ft_fusion = F.interpolate(ft_fusion, size=ft.shape[-2:], mode='bilinear')
            ft_fusion = ft_fusion + ft

        return ft_fusion


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(channel, channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(channel, channel, 3, stride=1, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                    )
        self.conv_1x1 = nn.Conv2d(channel, channel, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.layers(x)
        short = self.conv_1x1(x)
        out = out + short
        return out


class ResBlock_fft_bench(nn.Module):
    def __init__(self, n_feat):
        super(ResBlock_fft_bench, self).__init__()
        self.main = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
        self.mag = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0)
        self.pha = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)
        mag_out = self.mag(mag)
        mag_res = mag_out - mag
        pooling = torch.nn.functional.adaptive_avg_pool2d(mag_res, (1, 1))
        pooling = torch.nn.functional.softmax(pooling, dim=1)
        pha1 = pha * pooling
        pha1 = self.pha(pha1)
        pha_out = pha1 + pha
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        return self.main(x) + y


class net(nn.Module):
    def __init__(self, num_in_ch=3, base_channel=16, up_mode='upconv', bias=False):
        super().__init__()
        assert up_mode in ('upconv', 'upsample')

        self.layer0 = nn.Conv2d(num_in_ch, base_channel, kernel_size=3, stride=1, padding=1)
        self.layer1 = UNetConvBlock(in_chans=16, out_chans=20)
        self.layer2 = UNetConvBlock(in_chans=20, out_chans=28)
        self.layer3 = UNetConvBlock(in_chans=28, out_chans=44)
        self.layer4 = UNetConvBlock(in_chans=44, out_chans=76)
        self.layer_0 = UNetUpBlock(in_chans=20, out_chans=16, up_mode=up_mode)
        self.layer_1 = UNetUpBlock(in_chans=28, out_chans=20, up_mode=up_mode)
        self.layer_2 = UNetUpBlock(in_chans=44, out_chans=28, up_mode=up_mode)
        self.layer_3 = UNetUpBlock(in_chans=76, out_chans=44, up_mode=up_mode)

        self.layer0_ = nn.Conv2d(num_in_ch, base_channel, kernel_size=3, stride=1, padding=1)
        self.layer1_ = UNetConvBlock(in_chans=16, out_chans=20)
        self.layer2_ = UNetConvBlock(in_chans=20, out_chans=28)
        self.layer3_ = UNetConvBlock(in_chans=28, out_chans=44)
        self.layer4_ = UNetConvBlock(in_chans=44, out_chans=76)
        self.layer_0_ = UNetUpBlock(in_chans=20, out_chans=16, up_mode=up_mode)
        self.layer_1_ = UNetUpBlock(in_chans=28, out_chans=20, up_mode=up_mode)
        self.layer_2_ = UNetUpBlock(in_chans=44, out_chans=28, up_mode=up_mode)
        self.layer_3_ = UNetUpBlock(in_chans=76, out_chans=44, up_mode=up_mode)

        self.last = nn.Conv2d(base_channel, num_in_ch, kernel_size=1)

        self.fft0 = ResBlock_fft_bench(n_feat=base_channel)
        self.fft1 = ResBlock_fft_bench(n_feat=20)
        self.fft2 = ResBlock_fft_bench(n_feat=28)
        self.fft3 = ResBlock_fft_bench(n_feat=44)
        self.fft4 = ResBlock_fft_bench(n_feat=76)
        self.fft_0 = ResBlock_fft_bench(n_feat=base_channel)
        self.fft_1 = ResBlock_fft_bench(n_feat=20)
        self.fft_2 = ResBlock_fft_bench(n_feat=28)
        self.fft_3 = ResBlock_fft_bench(n_feat=44)

        self.res0 = ResBlock(base_channel)
        self.res1 = ResBlock(20)
        self.res2 = ResBlock(28)
        self.res3 = ResBlock(44)
        self.res4 = ResBlock(76)
        self.res_0 = ResBlock(base_channel)
        self.res_1 = ResBlock(20)
        self.res_2 = ResBlock(28)
        self.res_3 = ResBlock(44)

        self.res0_ = ResBlock(base_channel)
        self.res1_ = ResBlock(20)
        self.res2_ = ResBlock(28)
        self.res3_ = ResBlock(44)
        self.res4_ = ResBlock(76)
        self.res_0_ = ResBlock(base_channel)
        self.res_1_ = ResBlock(20)
        self.res_2_ = ResBlock(28)
        self.res_3_ = ResBlock(44)

        self.csff_enc0 = nn.Conv2d(base_channel, base_channel, kernel_size=1, bias=bias)
        self.csff_enc1 = nn.Conv2d(20, 20, kernel_size=1, bias=bias)
        self.csff_enc2 = nn.Conv2d(28, 28, kernel_size=1, bias=bias)
        self.csff_enc3 = nn.Conv2d(44, 44, kernel_size=1, bias=bias)
        self.csff_dec0 = nn.Conv2d(base_channel, base_channel, kernel_size=1, bias=bias)
        self.csff_dec1 = nn.Conv2d(20, 20, kernel_size=1, bias=bias)
        self.csff_dec2 = nn.Conv2d(28, 28, kernel_size=1, bias=bias)
        self.csff_dec3 = nn.Conv2d(44, 44, kernel_size=1, bias=bias)

        self.fusion1 = Encoder_MDCBlock1(20, 2)
        self.fusion2 = Encoder_MDCBlock1(28, 3)
        self.fusion3 = Encoder_MDCBlock1(44, 4)
        self.fusion4 = Encoder_MDCBlock1(76, 5)
        self.fusion_3 = Decoder_MDCBlock1(44, 2, 5)
        self.fusion_2 = Decoder_MDCBlock1(28, 3, 4)
        self.fusion_1 = Decoder_MDCBlock1(20, 4, 3)
        self.fusion_0 = Decoder_MDCBlock1(base_channel, 5, 2)

        self.fusion1_ = Encoder_MDCBlock1(20, 2)
        self.fusion2_ = Encoder_MDCBlock1(28, 3)
        self.fusion3_ = Encoder_MDCBlock1(44, 4)
        self.fusion4_ = Encoder_MDCBlock1(76, 5)
        self.fusion_3_ = Decoder_MDCBlock1(44, 2, 5)
        self.fusion_2_ = Decoder_MDCBlock1(28, 3, 4)
        self.fusion_1_ = Decoder_MDCBlock1(20, 4, 3)
        self.fusion_0_ = Decoder_MDCBlock1(base_channel, 5, 2)

        self.sam = SAM(base_channel, kernel_size=1)
        self.concat = conv(base_channel * 2, base_channel, kernel_size=3)

    def forward(self, x):
        xcopy = x
        blocks = []
        x = self.layer0(x)
        x0 = self.res0(x)
        x0 = self.fft0(x0)
        blocks.append(x0)

        x1 = self.layer1(x0)
        x1 = self.fusion1(x1, blocks)
        x1 = self.res1(x1)
        x1 = self.fft1(x1)
        blocks.append(x1)

        x2 = self.layer2(x1)
        x2 = self.fusion2(x2, blocks)
        x2 = self.res2(x2)
        x2 = self.fft2(x2)
        blocks.append(x2)

        x3 = self.layer3(x2)
        x3 = self.fusion3(x3, blocks)
        x3 = self.res3(x3)
        x3 = self.fft3(x3)
        blocks.append(x3)

        x4 = self.layer4(x3)
        x4 = self.fusion4(x4, blocks)
        x4 = self.res4(x4)
        x4 = self.fft4(x4)

        blocks_up = [x4]
        x_3 = self.layer_3(x4, blocks[-0 - 1])
        x_3 = self.res_3(x_3)
        x_3 = self.fft_3(x_3)
        x_3 = self.fusion_3(x_3, blocks_up)
        blocks_up.append(x_3)

        x_2 = self.layer_2(x_3, blocks[-1 - 1])
        x_2 = self.res_2(x_2)
        x_2 = self.fft_2(x_2)
        x_2 = self.fusion_2(x_2, blocks_up)
        blocks_up.append(x_2)

        x_1 = self.layer_1(x_2, blocks[-2 - 1])
        x_1 = self.res_1(x_1)
        x_1 = self.fft_1(x_1)
        x_1 = self.fusion_1(x_1, blocks_up)
        blocks_up.append(x_1)

        x_0 = self.layer_0(x_1, blocks[-3 - 1])
        x_0 = self.res_0(x_0)
        x_0 = self.fft_0(x_0)
        x_0 = self.fusion_0(x_0, blocks_up)

        x2_samfeats, stage1_output = self.sam(x_0, xcopy)

        blocks1 = []
        y = self.layer0_(xcopy)
        y = self.concat(torch.cat([y, x2_samfeats], 1))

        y0 = self.res0_(y)

        y0 = y0 + self.csff_enc0(x0) + self.csff_dec0(x_0)
        blocks1.append(y0)

        y1 = self.layer1_(y0)
        y1 = self.fusion1_(y1, blocks1)
        y1 = self.res1_(y1)

        y1 = y1 + self.csff_enc1(x1) + self.csff_dec1(x_1)
        blocks1.append(y1)

        y2 = self.layer2_(y1)
        y2 = self.fusion2_(y2, blocks1)
        y2 = self.res2_(y2)
        y2 = y2 + self.csff_enc2(x2) + self.csff_dec2(x_2)
        blocks1.append(y2)

        y3 = self.layer3_(y2)
        y3 = self.fusion3_(y3, blocks1)
        y3 = self.res3_(y3)
        y3 = y3 + self.csff_enc3(x3) + self.csff_dec3(x_3)
        blocks1.append(y3)

        y4 = self.layer4_(y3)
        y4 = self.fusion4_(y4, blocks1)
        y4 = self.res4_(y4)

        blocks1_up = [y4]
        y_3 = self.layer_3_(y4, blocks1[-0 - 1])
        y_3 = self.res_3_(y_3)
        y_3 = self.fusion_3_(y_3, blocks1_up)
        blocks1_up.append(y_3)

        y_2 = self.layer_2_(y_3, blocks1[-1 - 1])
        y_2 = self.res_2_(y_2)
        y_2 = self.fusion_2_(y_2, blocks1_up)
        blocks1_up.append(y_2)

        y_1 = self.layer_1_(y_2, blocks1[-2 - 1])
        y_1 = self.res_1_(y_1)
        y_1 = self.fusion_1_(y_1, blocks1_up)
        blocks1_up.append(y_1)

        y_0 = self.layer_0_(y_1, blocks1[-3 - 1])
        y_0 = self.res_0_(y_0)
        y_0 = self.fusion_0_(y_0, blocks1_up)

        output = self.last(y_0)
        output = torch.clamp(output, 0, 1)
        stage1_output = torch.clamp(stage1_output, 0, 1)

        return output, stage1_output


if __name__ == '__main__':
    data = torch.randn(1, 3, 256, 256)
    model = net()
    res1, res2 = model(data)
    print(res2.shape)
