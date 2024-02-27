import torch
from vit_pytorch import ViT, MAE
import torch.nn.functional as F

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange





class MAELOSS(nn.Module):
    def __init__(self, num_channels=4, channels=None, base_filter=None, args=None):
        super(MAELOSS, self).__init__()
        channels = base_filter
        v = ViT(
        image_size = 128,
        patch_size = 32,
        num_classes = 1,
        dim = 256,
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        channels = 4
        )
        self.mae = MAE(
            encoder = v,
            masking_ratio = 0.0,   # the paper recommended 75% masked patches
            decoder_dim = 128,      # paper showed good results with just 512
            decoder_depth = 6       # anywhere from 1 to 8
        ).cuda()
        self.mae.requires_grad_(False)
        self.load_state_dict(torch.load(""),strict=False)

    def forward(self, sr, hr):
        def _forward(x):
            x = self.mae(x)
            return x
        with torch.no_grad():
            mae_hr = _forward(hr.detach())
            mae_sr = _forward(sr)
        loss = F.l1_loss(mae_sr, mae_hr,reduction='mean')
        return loss
