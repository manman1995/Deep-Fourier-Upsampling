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
# from cv2 import cv2
import time

def feature_show(x,name, iter, stage):
    x_out = x.detach().cpu()
    x_out = x_out[0]
    x_out = np.average(x_out, axis=0)
    plt.imshow(x_out)
    plt.axis('off')  # plt.show() 之前，plt.imshow() 之后
    # plt.xticks([])  #plt.show() 之前，plt.imshow() 之后
    # plt.yticks([])
    plt.savefig(os.path.join('/home/zouzhen/zouz/derain_nips1/demo/demo', '{}_iter_{}_stage_{}.jpg'.format(name,iter, stage)))
