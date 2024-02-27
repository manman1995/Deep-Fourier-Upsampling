from os import name
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
#from PIL import Image, ImageOps

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d


class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out


class ChannelGCN(nn.Module):
    def __init__(self, planes, ratio=4):
        super(ChannelGCN, self).__init__()
        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)
    
    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, x):
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        return g_out


class DualGCN(nn.Module):
    """
        Feature GCN with coordinate GCN
    """
    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes))
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        # cat or sum, nearly the same results
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out


class DualGCN_parallel(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, stride=1):
        super(DualGCN_parallel, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))
        self.dualgcn = DualGCN(interplanes)
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes + interplanes, interplanes, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(interplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        output = self.conva(x)
        output = self.dualgcn(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class DualGCN_Spatial_fist(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, stride=1):
        super(DualGCN_Spatial_fist, self).__init__()
        self.local = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 3, groups=inplanes, stride=2, padding=1, bias=False),
            BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, inplanes, 3, groups=inplanes, stride=2, padding=1, bias=False),
            BatchNorm2d(inplanes),
            nn.Conv2d(inplanes, inplanes, 3, groups=inplanes, stride=2, padding=1, bias=False),
            BatchNorm2d(inplanes))

        self.gcn_local_attention = SpatialGCN(inplanes)
        
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        
        self.gcn_feature_attention = ChannelGCN(interplanes)

        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(interplanes, interplanes, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(interplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        # spatial part
        local = self.local(x)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # channel part
        CG_part = self.conva(spatial_local_feat)
        CG_part = self.gcn_feature_attention(CG_part)
        CG_part = self.convb(CG_part)

        # output
        output = self.bottleneck(CG_part)

        return output


class DualGCN_Channel_fist(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, stride=1):
        super(DualGCN_Channel_fist, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))
        self.gcn_feature_attention = ChannelGCN(interplanes)
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        self.local = nn.Sequential(
            nn.Conv2d(interplanes, interplanes, 3, groups=interplanes, stride=2, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.Conv2d(interplanes, interplanes, 3, groups=interplanes, stride=2, padding=1, bias=False),
            BatchNorm2d(interplanes),
            nn.Conv2d(interplanes, interplanes, 3, groups=interplanes, stride=2, padding=1, bias=False),
            BatchNorm2d(interplanes))
        self.gcn_local_attention = SpatialGCN(interplanes)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(interplanes, interplanes, kernel_size=3, stride=stride, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(interplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        # channel part
        CG_part = self.conva(x)
        CG_part = self.gcn_feature_attention(CG_part)
        CG_part = self.convb(CG_part)

        # spatial part
        local = self.local(CG_part)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = CG_part * local + CG_part

        # output
        output = self.bottleneck(spatial_local_feat)

        return output


class fu_DualGCN_Spatial_fist(nn.Module):
    def __init__(self, inchannels):
        super(fu_DualGCN_Spatial_fist, self).__init__()
        self.sGCN = SpatialGCN(inchannels)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=1, dilation=1),
            nn.ReLU(inchannels)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=1, dilation=1),
            nn.ReLU(inchannels)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3),
            nn.ReLU(inchannels)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3),
            nn.ReLU(inchannels)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(inchannels*5, inchannels, 1, padding=0),
            nn.ReLU(inchannels)
        )
        self.cGCN = ChannelGCN(inchannels)
    
    def forward(self, x):
        F_sGCN = self.sGCN(x)
        conv1 = self.conv_1(F_sGCN)
        conv2 = self.conv_2(conv1)
        conv3 = self.conv_3(F_sGCN)
        conv4 = self.conv_4(conv3)

        F_DCM = self.conv_5(torch.cat([F_sGCN, conv1, conv2, conv3, conv4], dim=1))
        F_cGCN = self.cGCN(F_DCM)
        F_unit = F_cGCN + x
        return F_unit

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None, pad_model=None):
        super(ConvBlock, self).__init__()

        self.pad_model = pad_model
        self.norm = norm
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(self.output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(self.output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU(init=0.5)
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        
        if self.pad_model == None:   
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, self.padding, bias=self.bias)
        elif self.pad_model == 'reflection':
            self.padding = nn.Sequential(nn.ReflectionPad2d(self.padding))
            self.conv = torch.nn.Conv2d(self.input_size, self.output_size, self.kernel_size, self.stride, 0, bias=self.bias)

    def forward(self, x):
        out = x
        if self.pad_model is not None:
            out = self.padding(out)

        if self.norm is not None:
            out = self.bn(self.conv(out))
        else:
            out = self.conv(out)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class Enhancer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Enhancer, self).__init__()

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

        self.refine1 = nn.Conv2d(in_channels, 20, kernel_size=3, stride=1, padding=1) 
        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  

        self.refine3 = nn.Conv2d(20 + 4, out_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = F.upsample_nearest

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x):
        dehaze = self.relu((self.refine1(x)))
        dehaze = self.relu((self.refine2(dehaze)))
        shape_out = dehaze.data.size()
        
        shape_out = shape_out[2:4]

        print('dehaze:',dehaze.shape)
        x101 = F.avg_pool2d(dehaze, 32) 
        print('x101:',x101.shape)#([1, 20, 4, 4])
        x102 = F.avg_pool2d(dehaze, 16)
        print('x102:',x102.shape) #[1, 20, 8, 8])
        x103 = F.avg_pool2d(dehaze, 8)
        x104 = F.avg_pool2d(dehaze, 4)
        print('x104:',x104.shape)# 20, 32, 32

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        print('up_x1010:',x1010.shape)#[1, 1, 128, 128]
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        print('up_x1020:',x1020.shape)#[1, 1, 128, 128]
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1) 
        print('dehaze:',dehaze.shape) #([1, 24, 128, 128])
        dehaze = self.tanh(self.refine3(dehaze))
        print('dehaze:',dehaze.shape)
        return dehaze


class Net(nn.Module):
    def __init__(self, num_channels): #, base_filter, args
        super(Net, self).__init__()
        inchannel=num_channels
        interplanes=inchannel*6
        self.head = ConvBlock(num_channels, 24, 9, 1, 4, activation='relu', norm=None, bias = False)
        self.enhancer=Enhancer(24,24)
        self.gcn_basic=fu_DualGCN_Spatial_fist(interplanes)
        self.output_conv = ConvBlock(interplanes, inchannel, 5, 1, 2, activation=None, norm=None, bias = False)

    def forward(self,x):
        x=self.head(x)
        print("head_out:",x.shape) #24, 128, 128
        x=self.enhancer(x)
        print("enhancer_out::",x.shape)
        x=self.gcn_basic(x)
        x=self.gcn_basic(x)
        #x=self.gcn_basic(x)
        #x=self.gcn_basic(x)
        print(x.shape) #24, 128, 128
        x=self.output_conv(x)        
        return x


if __name__ == "__main__":

    from torchvision.transforms import Compose, ToTensor
    def transform():
        return Compose([
            ToTensor(),
        ])


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model=Net(4)  #, map_location=torch.device('cpu')
    model.eval()
    #print(model)
    #   ms:torch.Size([4, 128, 128])   pan:([1, 128, 128])
    img=torch.ones((1,4,128,128))
    output_end=model(img)
    print(output_end.shape)
    #print(output_mid.shape)