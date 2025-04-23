import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from .LPNet_AttentionV4_arch import freup_pad_simple_attention_with_conv as FreAtt

# Upsampling_Beginning  ######################################################
# class _SimpleSegmentationModel(nn.Module):
#     def __init__(self, backbone, classifier):
#         super(_SimpleSegmentationModel, self).__init__()
#         self.backbone = backbone
#         self.classifier = classifier
    
#     def forward(self, x):
#         input_shape = x.shape[-2:]
#         features = self.backbone(x)
#         x = self.classifier(features)
#         x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
#         return x

#Fourier UPAttention ######################################################
class _SimpleSegmentationModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone   = backbone
        self.classifier = classifier

        out_ch = self.classifier.classifier[-1].out_channels
        self.fourier_up_final = FourierUp(out_ch, scale_factor=2)
        self.fuse_final = nn.Conv2d(out_ch * 2, out_ch, 1, bias=False)

    def forward(self, x):
        in_size  = x.shape[-2:]
        feats    = self.backbone(x)
        logits_1_4 = self.classifier(feats)             # 1/4 尺度 (DeepLab 已经带有上一步融合)

        # ─── 双线性恢复原尺寸 ───
        
        up_bi = F.interpolate(logits_1_4, size=in_size, mode="bilinear",
                              align_corners=False)
        
        # ─── Fourier‑domain 4× 恢复原尺寸 ───
        #up_fu = self.fourier_up_final(logits_1_4, size=in_size)
        # Fourier Up attention
        up_fu = self.fourier_up_final(logits_1_4, size = in_size)
        # ─── 融合 ───
        logits = self.fuse_final(torch.cat([up_bi, up_fu], dim=1))
        # logits = up_bi
        return logits


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        self.hrnet_flag = hrnet_flag

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            if self.hrnet_flag and name.startswith('transition'): # if using hrnet, you need to take care of transition
                if name == 'transition1': # in transition1, you need to split the module to two streams first
                    x = [trans(x) for trans in module]
                else: # all other transition is just an extra one stream split
                    x.append(module(x[-1]))
            else: # other models (ex:resnet,mobilenet) are convolutions in series.
                x = module(x)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                if name == 'stage4' and self.hrnet_flag: # In HRNetV2, we upsample and concat all outputs streams together
                    output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
                    x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x = torch.cat([x[0], x1, x2, x3], dim=1)
                    out[out_name] = x
                else:
                    out[out_name] = x
        return out

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


class FourierUp(nn.Module):
    def __init__(self, channels: int, scale_factor: int = 2):
        super().__init__()
        assert scale_factor in {2, 4, 8, 16}, "倍率必须是 2,4,8,16 等 2^k"
        self.n_repeat = int(np.log2(scale_factor))
        # 复用同一个 freup_pad 权重，节省显存
        self.fourier2x = freup_pad(channels)
        self.fourier4x = freup_pad(channels)

    def forward(self, x: torch.Tensor, size=None) -> torch.Tensor:
        y = x
        #for _ in range(self.n_repeat):
        y = self.fourier2x(y)
        #for _ in range(self.n_repeat):
        y = self.fourier4x(y)
        # 如果最终与目标尺寸稍有出入，做一次最邻近 resize 兜底
        if size is not None and y.shape[-2:] != size:
            y = F.interpolate(y, size=size, mode="nearest")
        return y

class FourierUpAtt(nn.Module):
    """
    Fourier 上采样 (2^k 倍) + 频域简单自注意力
    """
    def __init__(self, channels: int, scale_factor: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        assert scale_factor in {2, 4, 8, 16}, "倍率必须是 2^k"
        self.n_repeat = int(np.log2(scale_factor))
        # 复用同一个带注意力的 freup_pad，节省显存
        self.fourier2x = FreAtt(channels, dropout=dropout, use_attention=True)
        self.fourier4x = FreAtt(channels, dropout=dropout, use_attention=False)

    def forward(self, x: torch.Tensor, size=None) -> torch.Tensor:
        y = x
        # for _ in range(self.n_repeat):
        y = self.fourier2x(y)          
        y = self.fourier4x(y)
        if size is not None and y.shape[-2:] != size:     # 尺寸兜底
            y = F.interpolate(y, size=size, mode="nearest")
        return y