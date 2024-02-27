import argparse
import yaml
import torchvision.transforms as transforms
from utils import read_args, save_checkpoint, AverageMeter, calculate_metrics, CosineAnnealingWarmRestarts
# import torchvision.transforms.InterpolationMode
import time
from tqdm import trange, tqdm
from torchvision.utils import save_image
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import time
import logging
import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F

from model import *
from data import *
from PIL import Image
from torchvision.transforms import Resize
import pyiqa
from thop import profile
from thop import clever_format

psnr_calculator = pyiqa.create_metric('psnr').cuda()
ssim_calculator = pyiqa.create_metric('ssimc', downsample=True).cuda()
lpips_calculator = pyiqa.create_metric('lpips').cuda()
niqe_calculator = pyiqa.create_metric('niqe').cuda()


def test(load_path, data_loader, args):
    if not os.path.exists(args.output_dir + '/image_test'):
        os.mkdir(args.output_dir + '/image_test')
        
    save_path = args.output_dir + '/image_test'

    model = guide_net(args.model["model_channel"])
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()

    psnrs = AverageMeter()
    ssims = AverageMeter()
    lpipss = AverageMeter()
    niqes = AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            input_img, gt_img, inp_img_path = batch

            name = inp_img_path[0].split("/")[-1]
            input_img = input_img.cuda()
            batch_size = input_img.size(0)
            start_time = time.time()
            output, _ = model(input_img)

            # metrics
            clamped_out = torch.clamp(output, 0, 1)
            psnr_val, ssim_val = psnr_calculator(clamped_out, gt_img), ssim_calculator(clamped_out, gt_img)
            psnrs.update(torch.mean(psnr_val).item(), batch_size)
            ssims.update(torch.mean(ssim_val).item(), batch_size)

            save_image(clamped_out[0], os.path.join(save_path, name))
            # lpips = lpips_calculator(clamped_out, gt_img)
            # lpipss.update(torch.mean(lpips).item(), batch_size)
            # niqe = niqe_calculator(clamped_out)
            # niqes.update(torch.mean(niqe).item(), batch_size)
            torch.cuda.empty_cache()

            if i % 20 == 0:
                logging.info(
                    "PSNR {:.4f}, SSIM {:.4f}, LPIPS {:.4F}, NIQE {:.4F}, Elapse time {:.2f}\n".format(psnrs.avg,
                                                                                                       ssims.avg,
                                                                                                       lpipss.avg,
                                                                                                       niqes.avg,
                                                                                                       time.time() - start_time))

        logging.info(
            "Finish test: avg PSNR: %.4f, avg SSIM: %.4F, avg LPIPS: %.4F, avg NIQE: %.4F, and takes %.2f seconds" % (
                psnrs.avg, ssims.avg, lpipss.avg, niqes.avg, time.time() - start_time))


def main(args, load_path):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    test_transforms = transforms.Compose([transforms.ToTensor()])

    log_format = "%(asctime)s %(levelname)-8s %(message)s"
    log_file = os.path.join(args.output_dir, "test_log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info("Building data loader")

    test_loader = get_loader(args.data["test_dir"],
                             eval(args.test_loader["img_size"]), test_transforms, False,
                             int(args.test_loader["batch_size"]), args.test_loader["num_workers"],
                             args.test_loader["shuffle"], random_flag=False)
    test(load_path, test_loader, args)


if __name__ == '__main__':
    parser = read_args("/home/yuwei/code/remote/FFT/config/config.yaml")
    args = parser.parse_args()
    main(args, "/home/yuwei/experiment/remote/thin_full_new/models/model.200.bin")
