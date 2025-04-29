import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
# from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import os

from torchvision.utils import save_image
from cv2 import cv2
import time

# 可视化特征图
# print(AVW.shape) #[1, 12, 16384]
AV_show = AVW.detach().cpu()  # .transpose(1, 2).contiguous()
# print(AV_show.shape) #[1, 12, 16384]
show_AV = AV_show.view(b, c, h, -1)
# print(show_AV.shape) #[1, 12, 128, 128]
viz(show_AV)


# save_features(show_AV)
# save_features_pcolor(show_AV)
# print(AVW.shape)


def save_features(feature_map):
    for i in range(feature_map.size(1)):
        save_image(feature_map[0][i], os.path.join('./feature_maps', 'image_{}.jpg'.format(i)), nrow=1, padding=0)


def save_features_pcolor(feature_map):
    print(feature_map.shape)
    length = feature_map.shape[1]
    for i in range(length):
        feature = np.asanyarray(feature_map[0][i] * 255, dtype=np.uint8)
        features_pcolor = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join('./feature_maps', 'image_{}.jpg'.format(i)), features_pcolor)


def viz(input):
    x = input[0]
    print(x.shape)
    min_num = np.minimum(16, x.size()[0])
    for i in range(min_num):
        # plt.subplot(2, 8, i+1)
        plt.imshow(x[i])

        plt.axis('off')  # plt.show() 之前，plt.imshow() 之后
        # plt.xticks([])  #plt.show() 之前，plt.imshow() 之后
        # plt.yticks([])

        plt.savefig(os.path.join('./feature_maps', 'image_{}.jpg'.format(time.time())))
        # plt.show()
