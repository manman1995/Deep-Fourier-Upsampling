import argparse
import yaml
import torchvision.transforms as transforms
from utils import read_args, save_checkpoint, AverageMeter, calculate_metrics, CosineAnnealingWarmRestarts
import time
from tqdm import trange, tqdm
from torchvision.utils import save_image
# from tensorboardX import SummaryWriter
import os
import json
import time
import logging
import torch
from torch import nn, optim
import torchvision.utils as vutils
import torch.nn.functional as F

from data import *
from model import *
from loss import *


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

global_step = 0

amp_criterion = AmplitudeLoss()
pha_criterion = PhaseLoss()


def train(model, data_loader, criterion, optimizer, epoch, args):
    global global_step
    iter_bar = tqdm(data_loader, desc='Iter (loss=X.XXX)')
    nbatches = len(data_loader)

    total_losses = AverageMeter()
    pixel_losses = AverageMeter()
    amp_losses = AverageMeter()
    pha_losses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()

    optimizer.zero_grad()

    start_time = time.time()

    if not os.path.exists(args.output_dir + '/image_train'):
        os.mkdir(args.output_dir + '/image_train')

    if not os.path.exists(args.output_dir + "/models"):
        os.mkdir(args.output_dir + "/models")

    for i, batch in enumerate(iter_bar):
        optimizer.zero_grad()

        input_img, gt_img, image_path = batch
        input_img = input_img.cuda()
        gt_img = gt_img.cuda()
        batch_size = input_img.size(0)

        output = model(input_img)

        pixel_loss = criterion(output, gt_img)
        pixel_losses.update(pixel_loss.item(), batch_size)

        total_loss = pixel_loss
        total_losses.update(total_loss.item(), batch_size)

        total_loss.backward()
        optimizer.step()

        iter_bar.set_description('Iter (loss=%5.6f)' % total_losses.avg)

        if i % 2000 == 0:
            saved_image = torch.cat([input_img[0:2], output[0:2], gt_img[0:2]], dim=0)
            save_image(saved_image, args.output_dir + '/image_train/epoch_{}_iter_{}.jpg'.format(epoch, i))

        # metrics
        norm_out = torch.clamp(output, 0.0, 1.0)
        psnr_val, ssim_val = calculate_metrics(norm_out, gt_img)
        psnrs.update(psnr_val.item(), batch_size)
        ssims.update(ssim_val.item(), batch_size)

        if i % max(1, nbatches // 10) == 0:
            logging.info(
                "Epoch {}, learning rates {:}, Iter {}, total_loss {:.4f}, pixel_loss {:.4f}, amp_loss {:.4f}, pha_loss {:.4f}, PSNR {:.4f}, SSIM {:.4f}, Elapse time {:.2f}\n".format(
                    epoch, optimizer.param_groups[0]["lr"], i, total_losses.avg, pixel_losses.avg, amp_losses.avg, pha_losses.avg,
                    psnrs.avg, ssims.avg,
                    time.time() - start_time))

    if epoch % 1 == 0:
        logging.info("** ** * Saving model and optimizer ** ** * ")

        output_model_file = os.path.join(args.output_dir + "/models", "model.%d.bin" % (epoch))
        state = {"epoch": epoch, "state_dict": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "step": global_step}

        save_checkpoint(state, output_model_file)
        torch.save(model.state_dict(), os.path.join(args.output_dir + "/models", "model.%d.pth" % (epoch)))
        logging.info("Save model to %s", output_model_file)

    logging.info(
        "Finish training epoch %d, avg total_loss: %.4f, avg pixel_loss: %.4f, avg amp_loss: %.4f, avg pha_loss: %.4f, "
        "avg PSNR: %.2f, avg SSIM: %.2F, and takes %.2f seconds" % (
            epoch, total_losses.avg, pixel_losses.avg, amp_losses.avg, pha_losses.avg, psnrs.avg, ssims.avg,
            time.time() - start_time))

    logging.info("***** CUDA.empty_cache() *****\n")
    torch.cuda.empty_cache()


def evaluate(model, load_path, data_loader, epoch):

    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.cuda()
    model.eval()

    psnrs = AverageMeter()
    ssims = AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            input_img, gt_img, inp_img_path = batch
            input_img = input_img.cuda()
            batch_size = input_img.size(0)
            output = model(input_img)

            # metrics
            norm_out = torch.clamp(output, 0, 1)
            psnr_val, ssim_val = calculate_metrics(norm_out, gt_img)
            psnrs.update(psnr_val.item(), batch_size)
            ssims.update(ssim_val.item(), batch_size)
            torch.cuda.empty_cache()

            if i % 100 == 0:
                logging.info(
                    "PSNR {:.4f}, SSIM {:.4f}, Elapse time {:.2f}\n".format(psnrs.avg, ssims.avg,
                                                                            time.time() - start_time))

        logging.info(f"Finish test at epoch {epoch}: avg PSNR: %.4f, avg SSIM: %.4F, and takes %.2f seconds" % (
            psnrs.avg, ssims.avg, time.time() - start_time))


def main(args):
    global global_step

    start_epoch = 1
    global_step = 0

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, sort_keys=True, indent=2)

    log_format = "%(asctime)s %(levelname)-8s %(message)s"
    log_file = os.path.join(args.output_dir, "train_log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())

    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    logging.info(args.__dict__)

    if args.resume["flag"]:
        model = net(args)
        model.to(args.device)
        check_point = torch.load(args.resume["checkpoint"])
        model.load_state_dict(check_point["state_dict"])
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.optimizer["lr"],
                               betas=(0.9, 0.999))
        optimizer.load_state_dict(check_point["optimizer"])
        start_epoch = check_point["epoch"] + 1
        # start_epoch = check_point["epoch"]

    else:
        model = AODNet()
        model.to(args.device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.optimizer["lr"],
                               betas=(0.9, 0.999))

    logging.info("Building data loader")

    if args.train_loader["loader"] == "resize":
        train_transforms = transforms.Compose([transforms.Resize(eval(args.train_loader["img_size"])),
                                               transforms.ToTensor()])
        train_loader = get_loader(args.data["train_dir"],
                                  eval(args.train_loader["img_size"]), train_transforms, False,
                                  int(args.train_loader["batch_size"]), args.train_loader["num_workers"],
                                  args.train_loader["shuffle"])

    elif args.train_loader["loader"] == "crop":
        train_loader = get_loader(args.data["train_dir"],
                                  eval(args.train_loader["img_size"]), False, True,
                                  int(args.train_loader["batch_size"]), args.train_loader["num_workers"],
                                  args.train_loader["shuffle"])
    else:
        raise NotImplementedError

    if args.test_loader["loader"] == "default":

        test_transforms = transforms.Compose([transforms.ToTensor()])
        test_loader = get_loader(args.data["test_dir"],
                                  eval(args.test_loader["img_size"]), test_transforms, False,
                                  int(args.test_loader["batch_size"]), args.test_loader["num_workers"],
                                  args.test_loader["shuffle"])
    
    elif args.test_loader["loader"] == "resize":

        test_transforms = transforms.Compose([transforms.Resize(eval(args.test_loader["img_size"])),
                                               transforms.ToTensor()])
        test_loader = get_loader(args.data["test_dir"],
                                  eval(args.test_loader["img_size"]), test_transforms, False,
                                  int(args.test_loader["batch_size"]), args.test_loader["num_workers"],
                                  args.test_loader["shuffle"])

    criterion = nn.L1Loss()
    # vgg_loss = VGGLoss()

    if args.optimizer["type"] == "cos":
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.optimizer["T_0"],
                                                   T_mult=args.optimizer["T_MULT"],
                                                   eta_min=args.optimizer["ETA_MIN"],
                                                   last_epoch=-1)
    elif args.optimizer["type"] == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.optimizer["step"],
                                                       gamma=args.optimizer["gamma"])

    if args.resume["flag"]:
        for i in range(start_epoch):
            lr_scheduler.step()

    t_total = int(len(train_loader) * args.optimizer["total_epoch"])
    logging.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    logging.info("***** Running training *****")
    logging.info("  Batch size = %d", args.train_loader["batch_size"])
    logging.info("  Num steps = %d", t_total)
    logging.info("  Loader length = %d", len(train_loader))

    model.train()
    model.cuda()

    logging.info("Begin training from epoch = %d\n", start_epoch)
    for epoch in trange(start_epoch, args.optimizer["total_epoch"] + 1, desc="Epoch"):
        train(model, train_loader, criterion, optimizer, epoch, args)
        lr_scheduler.step()
        if epoch % args.evaluate_intervel == 0:
            logging.info("***** Running testing *****")
            load_path = os.path.join(args.output_dir + "/models", "model.%d.bin" % (epoch))
            evaluate(model, load_path, test_loader, epoch)
            logging.info("***** End testing *****")


if __name__ == '__main__':
    parser = read_args("config/config.yaml")
    args = parser.parse_args()
    main(args)
