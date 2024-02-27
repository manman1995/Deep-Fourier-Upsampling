import math
from weakref import ref
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss, rgb_to_grayscale, gradient, gaussian_kernel, rgb2lab
from PIL import Image
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
from scipy.stats import pearsonr
from torch.autograd import Variable
from math import exp

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class L1LossGaming(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, min_weight=0.5, reduction='mean'):
        super(L1LossGaming, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.min_weight = min_weight
        self.reduction = reduction

    def forward(self, pred, target, current_iter, total_iter, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        self.calibrated_weight = (1 - current_iter / total_iter) * self.loss_weight if current_iter < (total_iter // 2) else self.min_weight
        return self.calibrated_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)
    

@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, loss_weight=1.0, reduction='mean'):
        super(SSIMLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.window_size = window_size
        self.size_average = size_average
        self.loss_weight = loss_weight
        self.reduction = reduction

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average = True):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def ssim(self, img1, img2, window_size = 11, size_average = True):
        (_, channel, _, _) = img1.size()
        window = self.create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return self._ssim(img1, img2, window, window_size, channel, size_average)

    def forward(self, img1, img2):

        return self.loss_weight * (1-self.ssim(img1, img2, self.window_size, self.size_average))
    

# Define GAN loss: [vanilla | lsgan | wgan-gp]
@LOSS_REGISTRY.register()
class vanillaGANLoss(nn.Module):
    def __init__(self, gan_type, loss_weight, real_label_val=1.0, fake_label_val=0.0):
        super(vanillaGANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real, **kwargs):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return self.loss_weight * loss

################ loss functions for personalized Enhancement (ACM MM2022) ################


    ################# loss functions for Retinex decomposition #################

@LOSS_REGISTRY.register()
class DecompLoss(nn.Module):
    """Decomposition loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(DecompLoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred_lq, pred_gt, lq, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        self.reflect_lq = pred_lq[0]
        self.illumiantion_lq = pred_lq[1]
        self.illumiantion_lq3 = torch.cat((self.illumiantion_lq, self.illumiantion_lq, self.illumiantion_lq), dim=1)

        self.reflection_gt = pred_gt[0]
        self.illumiantion_gt = pred_gt[1]
        self.illumiantion_gt3 = torch.cat((self.illumiantion_gt, self.illumiantion_gt, self.illumiantion_gt), dim=1)

        recon_loss_low = torch.mean(torch.abs(self.reflect_lq * self.illumiantion_lq3 -  lq))
        recon_loss_high = torch.mean(torch.abs(self.reflection_gt * self.illumiantion_gt3 - target))

        equal_R_loss = torch.mean(torch.abs(self.reflect_lq - self.reflection_gt))

        i_mutual_loss = self.mutual_i_loss(self.illumiantion_lq, self.illumiantion_gt)

        i_input_mutual_loss_high = self.mutual_i_input_loss(self.illumiantion_gt, target)
        i_input_mutual_loss_low = self.mutual_i_input_loss(self.illumiantion_lq, lq)

        loss_Decom = 1 * recon_loss_high + 1 * recon_loss_low \
                    + 0.01 * equal_R_loss \
                    + 0.2 * i_mutual_loss \
                    + 0.15 * i_input_mutual_loss_high + 0.15 * i_input_mutual_loss_low

        # loss_Decom = 1 * recon_loss_low + 0.15 * i_input_mutual_loss_low

        return loss_Decom

    def mutual_i_loss(self, input_I_low, input_I_high):
        low_gradient_x = gradient(input_I_low, "x")
        high_gradient_x = gradient(input_I_high, "x")
        x_loss = (low_gradient_x + high_gradient_x) * torch.exp(-10 * (low_gradient_x + high_gradient_x))
        low_gradient_y = gradient(input_I_low, "y")
        high_gradient_y = gradient(input_I_high, "y")
        y_loss = (low_gradient_y + high_gradient_y) * torch.exp(-10 * (low_gradient_y + high_gradient_y))
        mutual_loss = torch.mean(x_loss + y_loss)
        return mutual_loss

    def mutual_i_input_loss(self, input_I_low, input_im):
        input_gray = rgb_to_grayscale(input_im)
        # print(input_gray.shape)
        # Image.fromarray((input_gray[0][0].data.cpu().numpy() * 255.0).astype(np.uint8)).save(
        #                     '/ghome/zhengns/code/BasicSR/eval_low.png')
        low_gradient_x = gradient(input_I_low, "x")
        input_gradient_x = gradient(input_gray, "x")
        x_loss = torch.abs(torch.div(low_gradient_x, torch.max(input_gradient_x, torch.tensor(0.01).cuda())))
        low_gradient_y = gradient(input_I_low, "y")
        input_gradient_y = gradient(input_gray, "y")
        y_loss = torch.abs(torch.div(low_gradient_y, torch.max(input_gradient_y, torch.tensor(0.01).cuda())))
        mut_loss = torch.mean(x_loss + y_loss)
        return mut_loss

    ################# loss functions for Retinex decomposition #################

    ################# loss functions for color #################

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out


def rgb2lab(rgb,ab_norm = 110.,l_cent = 50.,l_norm = 100.):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:,[0],:,:]-l_cent)/l_norm
    ab_rs = lab[:,1:,:,:]/ab_norm
    out = torch.cat((l_rs,ab_rs),dim=1)
    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2lab')
        # embed()
    return out


@LOSS_REGISTRY.register()
class LabLoss(nn.Module):
    """Lab color space loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(LabLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        pred_ab = rgb2lab(pred)
        target_ab = rgb2lab(target)

        return self.loss_weight * l1_loss(pred_ab, target_ab, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class YCbCrColorLoss(nn.Module):
    """Lab color space loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(YCbCrColorLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def RGB2YCbCr_torch(self, img): #B C H W, 255
        R = img[:,0:1,:,:]
        G = img[:,1:2,:,:]
        B = img[:,2:3,:,:]
        Y = 0.256789 * R + 0.504129 * G + 0.097906 * B + 16
        Cb = -0.148223 * R - 0.290992 * G + 0.439215 * B + 128
        Cr = 0.439215 * R - 0.367789 * G - 0.071426 * B + 128
        return Y,Cb,Cr


    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        Y_pred, Cb_pred, Cr_pred = self.RGB2YCbCr_torch(torch.clamp(pred, 0, 1) * 255.0)
        Y_gt, Cb_gt, Cr_gt = self.RGB2YCbCr_torch(torch.clamp(target, 0, 1) * 255.0)

        return self.loss_weight * (l1_loss(Cb_pred, Cb_gt, weight, reduction=self.reduction) + l1_loss(Cr_pred, Cr_gt, weight, reduction=self.reduction))
    
    ################# loss functions for color #################


@LOSS_REGISTRY.register()
class GramL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GramL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        b, c, h, w = pred.size()
        pred = pred.view(b, c, -1)
        target = target.view(b, c, -1)

        return (self.loss_weight * l1_loss(pred.bmm(pred.permute(0, 2, 1) / (c * h * w)),
                target.bmm(target.permute(0, 2, 1) / (c * h * w)), weight, reduction=self.reduction))


@LOSS_REGISTRY.register()
class L_histogram(nn.Module):
    def __init__(self, loss_weight, reduction='sum'):
        super(L_histogram, self).__init__()
        self.cri = nn.SmoothL1Loss()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class L_hsvHistogram(nn.Module):
    def __init__(self, loss_weight, reduction='sum'):
        super(L_hsvHistogram, self).__init__()
        self.cri = nn.SmoothL1Loss()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, predHueHisto, predSaturationHisto, targetHueHisto, targetSaturationHisto, hueSimilarity, saturationSimilarity, weight=None, **kwargs):

        return self.loss_weight * ((1 - hueSimilarity) * l1_loss(predHueHisto, targetHueHisto, weight, reduction=self.reduction) +
                                    (1 - saturationSimilarity) * l1_loss(predSaturationHisto, targetSaturationHisto, weight, reduction=self.reduction))


@LOSS_REGISTRY.register()
class L_identity(nn.Module):
    def __init__(self, loss_weight, reduction='sum'):
        super(L_identity, self).__init__()
        self.cri = nn.SmoothL1Loss()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, refl_refTextRefCont, refl_ref, refl_lowTextLowCont, refl_low, weight=None, **kwargs):
        return self.loss_weight * (l1_loss(refl_refTextRefCont, refl_ref, weight, reduction=self.reduction) +
                                    l1_loss(refl_lowTextLowCont, refl_low, weight, reduction=self.reduction))


@LOSS_REGISTRY.register()
class L_consistencyy(nn.Module):
    def __init__(self, loss_weight, reduction='sum'):
        super(L_consistencyy, self).__init__()
        self.cri = nn.SmoothL1Loss()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, refl_lowTextLowEnhCont, refl_low, refl_refTextRefEnhCont, refl_ref, weight=None, **kwargs):
        return self.loss_weight * (l1_loss(refl_lowTextLowEnhCont, refl_low, weight, reduction=self.reduction) +
                                    l1_loss(refl_refTextRefEnhCont, refl_ref, weight, reduction=self.reduction))


@LOSS_REGISTRY.register()
class L_KL(nn.Module):
    def __init__(self, loss_weight, reduction='sum'):
        super(L_KL, self).__init__()
        self.cri = nn.SmoothL1Loss()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, textureVectorLowEnhanced, textureVectorRef, weight=None, **kwargs):
        # textureVectorLowEnhanced_2 = torch.pow(textureVectorLowEnhanced, 2)
        # textureVectorRef_2 = torch.pow(textureVectorRef, 2)
        # encoding_loss = (textureVectorLowEnhanced_2 + textureVectorRef_2 - torch.log(textureVectorRef_2)).mean()
        # return encoding_loss

        loss = 0.5 * torch.sum(torch.pow(textureVectorLowEnhanced, 2) + torch.exp(textureVectorRef) - 1 - textureVectorRef, axis=-1)
        loss = torch.mean(loss)
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class L_color(nn.Module):

    def __init__(self, loss_weight):
        super(L_color, self).__init__()
        self.cri = nn.SmoothL1Loss()
        self.loss_weight = loss_weight

    def forward(self, x, y, **kwargs):
        b, c, h, w = x.shape
        x_norm = F.layer_norm(x, x.size()[1:], eps=1e-4)
        y_norm = F.layer_norm(y, y.size()[1:], eps=1e-4)
        mean_rgb_x = torch.mean(x_norm, [2, 3], keepdim=False)
        mean_rgb_y = torch.mean(y_norm, [2, 3], keepdim=False)

        mean_rgb_diff = mean_rgb_x - mean_rgb_y
        mr, mg, mb = torch.split(mean_rgb_diff, 1, dim=1)

        zero_tensor = torch.zeros_like(mr)

        Drg = self.cri(mr - mg, zero_tensor)
        Drb = self.cri(mr - mb, zero_tensor)
        Dgb = self.cri(mb - mg, zero_tensor)
        k = Drg + Drb + Dgb
        # Drg = torch.pow(mr - mg, 2)
        # Drb = torch.pow(mr - mb, 2)
        # Dgb = torch.pow(mb - mg, 2)
        # k = torch.pow(Drg + Drb + Dgb + 1e-10, 0.5)
        return self.loss_weight * k


@LOSS_REGISTRY.register()
class L_spa(nn.Module):
    def __init__(self, loss_weight, spa_kernel):
        super(L_spa, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        # self.pool = nn.AvgPool2d(spa_kernel,stride=1)
        self.gaussian = torch.FloatTensor(gaussian_kernel(spa_kernel, spa_kernel/4.0)).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        self.loss_weight = loss_weight

    def forward(self, enhance, org, **kwargs):
        b, c, h, w = org.shape

        # # ### gray version
        # # org_mean = torch.mean(org, 1, keepdim=True)
        # # enhance_mean = torch.mean(enhance, 1, keepdim=True)
        # ### color version
        org_pool = F.conv2d(org, self.gaussian, padding=0, groups=3)
        enhance_pool = F.conv2d(enhance, self.gaussian, padding=0, groups=3)

        # weight_diff = torch.max(
        #     torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
        #                                                       torch.FloatTensor([0]).cuda()),
        #     torch.FloatTensor([0.5]).cuda())
        # E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=0, groups=3)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=0, groups=3)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=0, groups=3)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=0, groups=3)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=0, groups=3)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=0, groups=3)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=0, groups=3)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=0, groups=3)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = D_left + D_right + D_up + D_down

        return self.loss_weight * E


@LOSS_REGISTRY.register()
class L_spaRefl(nn.Module):
    def __init__(self, loss_weight, spa_kernel):
        super(L_spaRefl, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        # self.pool = nn.AvgPool2d(spa_kernel,stride=1)
        self.gaussian = torch.FloatTensor(gaussian_kernel(spa_kernel, spa_kernel/4.0)).cuda().unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
        self.loss_weight = loss_weight

    def forward(self, enhance, org, **kwargs):
        b, c, h, w = org.shape

        # # ### gray version
        # # org_mean = torch.mean(org, 1, keepdim=True)
        # # enhance_mean = torch.mean(enhance, 1, keepdim=True)
        # ### color version
        org_pool = F.conv2d(org, self.gaussian, padding=0, groups=3)
        enhance_pool = F.conv2d(enhance, self.gaussian, padding=0, groups=3)

        # weight_diff = torch.max(
        #     torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
        #                                                       torch.FloatTensor([0]).cuda()),
        #     torch.FloatTensor([0.5]).cuda())
        # E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=0, groups=3)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=0, groups=3)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=0, groups=3)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=0, groups=3)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=0, groups=3)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=0, groups=3)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=0, groups=3)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=0, groups=3)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = D_left + D_right + D_up + D_down

        print(torch.mean(E))

        return self.loss_weight * E


def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])
    rgb = torch.abs(rgb)

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)


    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out

################ loss functions for personalized Enhancement (ACM MM2022) ################



################# loss functions for custmoized unfolding enhancer (ICCV 2023) #################
@LOSS_REGISTRY.register()
class illuMutualInputLoss(nn.Module):
    """Lab color space loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(illuMutualInputLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction


    def forward(self, illu, input_im, weight=None, **kwargs):
        rgb_weights = torch.Tensor([0.2989, 0.5870, 0.1140]).cuda()
        input_gray = torch.tensordot(input_im, rgb_weights, dims=([1], [-1])).unsqueeze(1)

        h_x = illu.size()[2]
        w_x = illu.size()[3]
        h_tv_illu = illu[:, :, 1:, :] - illu[:, :, :h_x-1, :]
        h_tv_illu = (h_tv_illu - h_tv_illu.min()) / (h_tv_illu.max() - h_tv_illu.min() + 1e-8)
        w_tv_illu = illu[:, :, :, 1:] - illu[:, :, :, :w_x-1]
        w_tv_illu = (w_tv_illu - w_tv_illu.min()) / (w_tv_illu.max() - w_tv_illu.min() + 1e-8)
        h_tv_gt = input_gray[:, :, 1:, :] - input_gray[:, :, :h_x-1, :]
        h_tv_gt = (h_tv_gt - h_tv_gt.min()) / (h_tv_gt.max() - h_tv_gt.min() + 1e-8)
        w_tv_gt = input_gray[:, :, :, 1:] - input_gray[:, :, :, :w_x-1]
        w_tv_gt = (w_tv_gt - w_tv_gt.min()) / (w_tv_gt.max() - w_tv_gt.min() + 1e-8)

        h_loss = torch.abs(torch.div(h_tv_illu, torch.max(h_tv_gt, torch.tensor(0.01).cuda())))
        w_loss = torch.abs(torch.div(w_tv_illu, torch.max(w_tv_gt, torch.tensor(0.01).cuda())))

        mut_loss = torch.mean(h_loss) + torch.mean(w_loss)

        return self.loss_weight * mut_loss


@LOSS_REGISTRY.register()
class illuMutualLoss(nn.Module):
    """Lab color space loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(illuMutualLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction


    def forward(self, illu, illu_gt, weight=None, **kwargs):

        h_x = illu.size()[2]
        w_x = illu.size()[3]
        h_tv_illu = illu[:, :, 1:, :] - illu[:, :, :h_x-1, :]
        h_tv_illu = (h_tv_illu - h_tv_illu.min()) / (h_tv_illu.max() - h_tv_illu.min() + 1e-8)
        w_tv_illu = illu[:, :, :, 1:] - illu[:, :, :, :w_x-1]
        w_tv_illu = (w_tv_illu - w_tv_illu.min()) / (w_tv_illu.max() - w_tv_illu.min() + 1e-8)
        h_tv_gt = illu_gt[:, :, 1:, :] - illu_gt[:, :, :h_x-1, :]
        h_tv_gt = (h_tv_gt - h_tv_gt.min()) / (h_tv_gt.max() - h_tv_gt.min() + 1e-8)
        w_tv_gt = illu_gt[:, :, :, 1:] - illu_gt[:, :, :, :w_x-1]
        w_tv_gt = (w_tv_gt - w_tv_gt.min()) / (w_tv_gt.max() - w_tv_gt.min() + 1e-8)

        h_loss = (h_tv_illu + h_tv_gt) * torch.exp(-10 * (h_tv_illu + h_tv_gt))
        w_loss = (w_tv_illu + w_tv_gt) * torch.exp(-10 * (w_tv_illu + w_tv_gt))

        mut_loss = torch.mean(h_loss) + torch.mean(w_loss)

        return self.loss_weight * mut_loss
    

@LOSS_REGISTRY.register()
class IlluTVLoss(nn.Module):
    def __init__(self, loss_weight, reduction):
        super(IlluTVLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, output, gt):
        h_x = output.size()[2]
        w_x = output.size()[3]
        h_tv_output = torch.pow((output[:, :, 1:, :] - output[:, :, :h_x-1, :]), 2)
        w_tv_output = torch.pow((output[:, :, :, 1:] - output[:, :, :, :w_x-1]), 2)

        h_tv_gt = torch.pow((gt[:, :, 1:, :] - gt[:, :, :h_x-1, :]), 2)
        w_tv_gt = torch.pow((gt[:, :, :, 1:] - gt[:, :, :, :w_x-1]), 2)

        tvloss = F.mse_loss(h_tv_output, h_tv_gt, reduction=self.reduction) + F.mse_loss(w_tv_output, w_tv_gt, reduction=self.reduction)

        return tvloss * self.loss_weight
    
################# loss functions for custmoized unfolding enhancer (ICCV 2023) #################



@LOSS_REGISTRY.register()
class NoiseLoss(nn.Module):
    """Lab color space loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(NoiseLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def gradient(self, img):
        height = img.size(2)
        width = img.size(3)
        gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
        gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
        gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
        gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
        gradient2_h = (img[:,:,4:,:]-img[:,:,:height-4,:]).abs()
        gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
        gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
        gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
        return gradient_h*gradient2_h, gradient_w*gradient2_w

    def normalize01(self, img):
        minv = img.min()
        maxv = img.max()
        return (img-minv)/(maxv-minv)

    def reflectance_smooth_loss(self, image, illumination, reflectance):
        gray_tensor = 0.299*image[0,0,:,:] + 0.587*image[0,1,:,:] + 0.114*image[0,2,:,:]
        gradient_gray_h, gradient_gray_w = self.gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
        gradient_reflect_h, gradient_reflect_w = self.gradient(reflectance)
        weight = 1/(illumination*gradient_gray_h * gradient_gray_w+0.0001)
        weight = self.normalize01(weight)
        weight.detach()
        loss_h = weight * gradient_reflect_h
        loss_w = weight * gradient_reflect_w
        refrence_reflect = image/illumination
        refrence_reflect.detach()
        return loss_h.mean() + loss_w.mean()

    def noise_loss(self, image, illumination, reflectance, noise):
        weight_illu = illumination
        weight_illu.detach()
        loss = weight_illu*noise
        return torch.norm(loss, 2)

    def forward(self, low_light, illumination, reflectance, noise, **kwargs):
        reflLoss = self.reflectance_smooth_loss(low_light, illumination, reflectance)
        noiseLoss = self.noise_loss(low_light, illumination, reflectance, noise)

        # print(reflLoss)
        # print(1e-5 * noiseLoss)

        return self.loss_weight * (reflLoss + 1e-5 * noiseLoss)


################# loss functions for MAE-based #################

@LOSS_REGISTRY.register()
class MaskFreloss(nn.Module):
    def __init__(self, win_size, loss_weight=1.0, reduction='mean'):
        super(MaskFreloss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.win_size = win_size
        self.loss_weight = loss_weight
        self.reduction = reduction

    def window_partition(self, x, win_size):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
        return windows

    def amplitude(self, img, img1):
        fre = torch.fft.rfft2(img, norm='backward')
        amp = torch.abs(fre)
        fre1 = torch.fft.rfft2(img1, norm='backward')
        amp1 = torch.abs(fre1)
        return self.loss_weight * l1_loss(amp, amp1, reduction='mean')

    def phase(self, img, img1):
        fre = torch.fft.rfft2(img, norm='backward')
        pha = torch.angle(fre)
        fre1 = torch.fft.rfft2(img1, norm='backward')
        pha1 = torch.angle(fre1)
        return self.loss_weight * l1_loss(pha, pha1, reduction='mean')

    def forward(self, pred, target, mask_disruption, weight=None, **kwargs):
        pred = self.window_partition(pred, self.win_size)
        target = self.window_partition(target, self.win_size)
        pred_masked = pred[mask_disruption, :, :, :]
        target_masked = target[mask_disruption, :, :, :]

        return self.amplitude(pred_masked, target_masked) + self.phase(pred_masked, target_masked)


@LOSS_REGISTRY.register()
class MaskL1loss(nn.Module):
    def __init__(self, win_size, loss_weight=1.0, reduction='mean'):
        super(MaskL1loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.win_size = win_size
        self.loss_weight = loss_weight
        self.reduction = reduction

    def window_partition(self, x, win_size):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
        return windows

    def forward(self, pred, target, mask_disruption, weight=None, **kwargs):
        pred = self.window_partition(pred, self.win_size)
        target = self.window_partition(target, self.win_size)
        pred_masked = pred[mask_disruption, :, :, :]
        target_masked = target[mask_disruption, :, :, :]

        return self.loss_weight * l1_loss(pred_masked, target_masked, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MaskTextureMixL1loss(nn.Module):
    def __init__(self, win_size, loss_weight=1.0, reduction='mean'):
        super(MaskTextureMixL1loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.win_size = win_size
        self.loss_weight = loss_weight
        self.reduction = reduction

    def window_partition(self, x, win_size):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
        return windows

    def forward(self, lowFre, pred, gt, mask_disruption, highFre=None, weight=None, **kwargs):
        lowFre = self.window_partition(lowFre, self.win_size)
        pred = self.window_partition(pred, self.win_size)
        gt = self.window_partition(gt, self.win_size)
        pred_mixed = lowFre + pred
        pred_mixed_masked = pred_mixed[mask_disruption, :, :, :]
        gt_masked = gt[mask_disruption, :, :, :]

        return self.loss_weight * l1_loss(pred_mixed_masked, gt_masked, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MaskTextureL1loss(nn.Module):
    def __init__(self, win_size, loss_weight=1.0, reduction='mean'):
        super(MaskTextureL1loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.win_size = win_size
        self.loss_weight = loss_weight
        self.reduction = reduction

    def window_partition(self, x, win_size):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
        return windows

    def forward(self, lowFre, pred, gt, mask_disruption, highFre=None, weight=None, **kwargs):
        pred = self.window_partition(pred, self.win_size)
        gt = self.window_partition(highFre, self.win_size)
        pred_masked = pred[mask_disruption, :, :, :]
        gt_masked = gt[mask_disruption, :, :, :]
        mask = torch.zeros_like(pred_masked)
        mask[gt_masked != 0] = 1


        return self.loss_weight * l1_loss(pred_masked * mask, gt_masked * mask, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MAELoss(nn.Module):
    """

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_pix_loss=False, loss_weight=1.0, reduction='mean'):
        super(MAELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.norm_pix_loss = norm_pix_loss
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.in_chans, self.embed_dim)
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


@LOSS_REGISTRY.register()
class MAETextureLoss(nn.Module):
    """

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_pix_loss=False, loss_weight=1.0, reduction='mean'):
        super(MAETextureLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.norm_pix_loss = norm_pix_loss
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.in_chans, self.embed_dim)
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, target, pred_highFre, lowFre, mask, highFre):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(target)
        lowFre = self.patchify(lowFre)
        highFre = self.patchify(highFre)
        mask_fre = torch.zeros_like(highFre)
        mask_fre[highFre != 0] = 1
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = ((pred_highFre + lowFre - target) * mask_fre) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


@LOSS_REGISTRY.register()
class MAEHOGLoss(nn.Module):
    """

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_pix_loss=False, loss_weight=1.0, reduction='mean'):
        super(MAEHOGLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.norm_pix_loss = norm_pix_loss
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.in_chans, self.embed_dim)
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x

    def forward(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


@LOSS_REGISTRY.register()
class MAEMS2MSLoss(nn.Module):
    """

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_pix_loss=False, loss_weight=1.0, reduction='mean'):
        super(MAEMS2MSLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.norm_pix_loss = norm_pix_loss
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.in_chans, self.embed_dim)
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 4, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 4))
        return x

    def forward(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


@LOSS_REGISTRY.register()
class MAEwoMaskLoss(nn.Module):
    """

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_pix_loss=False, loss_weight=1.0, reduction='mean'):
        super(MAEwoMaskLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.norm_pix_loss = norm_pix_loss
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.in_chans, self.embed_dim)
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        mask = self.patchify(mask)
        print(target.shape)
        print(pred.shape)
        print(mask.shape)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = ((loss * mask).sum() / mask.sum()).mean(-1)  # mean loss on removed patches
        return loss

################# loss functions for MAE-based #################


################# loss functions for CBDNet #################

@LOSS_REGISTRY.register()
class CBDNetLoss(nn.Module):
    def __init__(self):
        super(CBDNetLoss, self).__init__()

    def forward(self, out_image, gt_image, est_noise, gt_noise):
        l2_loss = F.mse_loss(out_image, gt_image)

        asym_loss = torch.mean(torch.abs(0.3 - torch.lt(gt_noise, est_noise).float()) * torch.pow(est_noise - gt_noise, 2))

        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, : ,1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = l2_loss +  50 * asym_loss + 5 * tvloss
        print(l2_loss)
        print(50 * asym_loss)
        print(5 * tvloss)

        return loss

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
################# loss functions for CBDNet #################


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class EdgeLoss(nn.Module):
    def __init__(self, loss_weight):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()
        self.weight = loss_weight

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.weight * self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super(WeightedTVLoss, self).forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super(WeightedTVLoss, self).forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


@LOSS_REGISTRY.register()
class MultiScaleGANLoss(GANLoss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(MultiScaleGANLoss, self).__init__(gan_type, real_label_val, fake_label_val, loss_weight)

    def forward(self, input, target_is_real, is_disc=False):
        """
        The input is a list of tensors, or a list of (a list of tensors)
        """
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    # Only compute GAN loss for the last layer
                    # in case of multiscale feature matching
                    pred_i = pred_i[-1]
                # Safe operaton: 0-dim tensor calling self.mean() does nothing
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


@LOSS_REGISTRY.register()
class GANFeatLoss(nn.Module):
    """Define feature matching loss for gans

    Args:
        criterion (str): Support 'l1', 'l2', 'charbonnier'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean'):
        super(GANFeatLoss, self).__init__()
        if criterion == 'l1':
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == 'l2':
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == 'charbonnier':
            self.loss_op = CharbonnierLoss(loss_weight, reduction)
        else:
            raise ValueError(f'Unsupported loss mode: {criterion}. Supported ones are: l1|l2|charbonnier')

        self.loss_weight = loss_weight

    def forward(self, pred_fake, pred_real):
        num_d = len(pred_fake)
        loss = 0
        for i in range(num_d):  # for each discriminator
            # last output is the final prediction, exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.loss_op(pred_fake[i][j], pred_real[i][j].detach())
                loss += unweighted_loss / num_d
        return loss * self.loss_weight
