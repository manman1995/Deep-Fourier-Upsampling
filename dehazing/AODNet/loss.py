import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1):
        super(VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()

    def forward(self, out, gt):
        loss = self.loss_fn(out, gt, feature_layers=[2])

        return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).cuda()
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)).cuda()
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)).cuda()
        self.resize = resize

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


def scharr(x):  # 输入前对RGB通道求均值在灰度图上算
    b, c, h, w = x.shape
    pad = nn.ReplicationPad2d(padding=(1, 1, 1, 1))
    x = pad(x)
    kx = F.unfold(x, kernel_size=3, stride=1, padding=0)  # b,n*k*k,n_H*n_W
    kx = kx.permute([0, 2, 1])  # b,n_H*n_W,n*k*k
    # kx=kx.view(1, b*h*w, 9) #1,b*n_H*n_W,n*k*k

    w1 = torch.tensor([-3, 0, 3, -10, 0, 10, -3, 0, 3]).float().cuda()
    w2 = torch.tensor([-3, -10, -3, 0, 0, 0, 3, 10, 3]).float().cuda()

    y1 = torch.matmul((kx * 255.0), w1)  # 1,b*n_H*n_W,1
    y2 = torch.matmul((kx * 255.0), w2)  # 1,b*n_H*n_W,1
    # y1=y1.view(b,h*w,1) #b,n_H*n_W,1
    y1 = y1.unsqueeze(-1).permute([0, 2, 1])  # b,1,n_H*n_W
    # y2=y2.view(b,h*w,1) #b,n_H*n_W,1
    y2 = y2.unsqueeze(-1).permute([0, 2, 1])  # b,1,n_H*n_W

    y1 = F.fold(y1, output_size=(h, w), kernel_size=1)  # b,m,n_H,n_W
    y2 = F.fold(y2, output_size=(h, w), kernel_size=1)  # b,m,n_H,n_W
    y1 = y1.clamp(-255, 255)
    y2 = y2.clamp(-255, 255)
    return (0.5 * torch.abs(y1) + 0.5 * torch.abs(y2)) / 255.0


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.reshape(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, input_fea, target_fea):
        target = gram_matrix(target_fea).detach()
        G = gram_matrix(input_fea)
        loss = F.mse_loss(G, target)
        return loss


def cos_loss(feat1, feat2):
    # maximize average cosine similarity
    return -F.cosine_similarity(feat1, feat2).mean()


def feat_scharr(x):
    x = torch.mean(x, dim=1, keepdim=True)
    x = (x - x.min()) / (x.max() - x.min())
    x = x * 255
    return scharr(x)


def feat_ssim(feat1, feat2, gt):
    mask = scharr(torch.mean(gt, dim=1, keepdim=True))
    # mask = torch.nn.MaxPool2d(5, 1, 2)(mask)
    mask = F.interpolate(mask, size=(feat1.shape[2], feat1.shape[3]), mode="bicubic")
    loss = torch.abs(feat1 - feat2) * mask
    return torch.mean(loss), mask


def similarity_loss(f_s, f_t):
    def at(f):
        return F.normalize(f.pow(2).mean(1).view(f.size(0), -1))

    return (at(f_s) - at(f_t)).pow(2).mean()


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2

        return torch.exp(
            -L2_distances[None, ...].cuda() / (self.get_bandwidth(
                L2_distances).cuda() * self.bandwidth_multipliers.cuda())[:, None,
                                              None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel.cuda()

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def charbonnier_loss(inputs, targets, epsilon=1e-6):
    diff = inputs - targets
    loss = torch.sqrt(diff.pow(2) + epsilon**2)
    mean_loss = torch.mean(loss)
    return mean_loss


class AmplitudeLoss(nn.Module):
    def __init__(self):
        super(AmplitudeLoss, self).__init__()

    def forward(self, img, img1):
        fre = torch.fft.rfft2(img, norm='backward')
        amp = torch.abs(fre)
        fre1 = torch.fft.rfft2(img1, norm='backward')
        amp1 = torch.abs(fre1)
        return torch.nn.functional.l1_loss(amp, amp1, reduction='mean')


class PhaseLoss(nn.Module):
    def __init__(self):
        super(PhaseLoss, self).__init__()

    def forward(self, img, img1):
        fre = torch.fft.rfft2(img, norm='backward')
        pha = torch.angle(fre)
        fre1 = torch.fft.rfft2(img1, norm='backward')
        pha1 = torch.angle(fre1)
        return torch.nn.functional.l1_loss(pha, pha1, reduction='mean')
