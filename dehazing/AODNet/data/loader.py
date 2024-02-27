import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as tf
from PIL import Image, ImageFile
import random
import math
from model import *
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True


class base_dataset(Dataset):
    def __init__(self, data_dir, img_size, transforms=False, crop=False):
        imgs = sorted(os.listdir(data_dir + "/hazy"))
        gt_imgs = [i.split("_")[0] for i in imgs]
        self.input_imgs = [os.path.join(data_dir + "/hazy", name) for name in imgs]

        self.gt_imgs = [os.path.join(data_dir + "/gt", name + ".png") for name in gt_imgs]
        self.transforms = transforms
        self.crop = crop
        self.img_size = img_size

    def __getitem__(self, index):
        inp_img_path = self.input_imgs[index]
        gt_img_path = self.gt_imgs[index]
        inp_img = Image.open(inp_img_path).convert("RGB")
        gt_img = Image.open(gt_img_path).convert("RGB")
        if self.transforms:
            inp_img = self.transforms(inp_img)
            gt_img = self.transforms(gt_img)

        if self.crop:
            inp_img, gt_img = self.crop_image(inp_img, gt_img)

        return inp_img, gt_img, inp_img_path

    def __len__(self):
        return len(self.gt_imgs)

    def crop_image(self, inp_img, gt_img):
        crop_h, crop_w = self.img_size
        i, j, h, w = tf.RandomCrop.get_params(
            inp_img, output_size=((crop_h, crop_w)))
        inp_img = TF.crop(inp_img, i, j, h, w)
        gt_img = TF.crop(gt_img, i, j, h, w)
        inp_img = TF.to_tensor(inp_img)
        gt_img = TF.to_tensor(gt_img)

        return inp_img, gt_img


class random_scale_dataset(Dataset):
    def __init__(self, data_dir, img_size, transforms=False, crop=False):
        imgs = sorted(os.listdir(data_dir + "/low"))
        self.input_imgs = [os.path.join(data_dir + "/low", name) for name in imgs]
        self.gt_imgs = [os.path.join(data_dir + "/high", name) for name in imgs]
        self.transforms = transforms
        self.crop = crop
        self.img_size = img_size

    def __getitem__(self, index):
        inp_img_path = self.input_imgs[index]
        gt_img_path = self.gt_imgs[index]
        inp_img = Image.open(inp_img_path).convert("RGB")
        gt_img = Image.open(gt_img_path).convert("RGB")

        random_scale_factor = random.randrange(self.img_size[0] * 0.25, self.img_size[0], 8)
        down_h = down_w = random_scale_factor

        if self.transforms:
            inp_img = self.transforms(inp_img)
            gt_img = self.transforms(gt_img)
            return inp_img, gt_img, down_h, down_w, inp_img_path

        if self.crop:
            inp_img, gt_img = self.crop_image(inp_img, gt_img)
            return inp_img, gt_img, down_h, down_w, inp_img_path

    def __len__(self):
        return len(self.gt_imgs)

    def crop_image(self, inp_img, gt_img):
        crop_h, crop_w = self.img_size
        i, j, h, w = tf.RandomCrop.get_params(
            inp_img, output_size=((crop_h, crop_w)))
        inp_img = TF.crop(inp_img, i, j, h, w)
        gt_img = TF.crop(gt_img, i, j, h, w)
        inp_img = TF.to_tensor(inp_img)
        gt_img = TF.to_tensor(gt_img)

        return inp_img, gt_img


def get_loader(data_dir, img_size, transforms, crop_flag, batch_size, num_workers, shuffle, random_flag=False):
    if random_flag:
        dataset = random_scale_dataset(data_dir, img_size, transforms, crop_flag)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    else:
        dataset = base_dataset(data_dir, img_size, transforms, crop_flag)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader
