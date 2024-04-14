# -*- coding: utf-8 -*-
"""
@Time ： 3/18/24 4:22 AM
@Auth ： woldier wong
@File ：train.py
@IDE ：PyCharm
@DESCRIPTION：train
"""
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
from utils.eval_utils import SSIM, PSNR, RMSE, denormalize, trunc
from utils.train_valid_utils import get_config, check_dir, config_backpack, init_model, load_dataset, init_optimizer
import datetime
from model import AbstractDenoiser
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, ToTensor
import pytz
from utils.img_utils import show_img
import random
from PIL import Image
from accelerate import Accelerator
import sys

sys.path.append('/')
# ================================================================================
# =========================load config=========================================
config_paths = [
    r'./config/dn_cnn/config.yml',  # _______________0 DnCNN
    r'./config/red_cnn/config.yml',  # ______________1 REDCNN
    r'./config/adnet/config.yml',  # ________________2 ADNet
    r'./config/ct_former/config.yml',  # ____________3 CTformer
    r'./config/nb_net/config.yml',  # _______________4 NBNet
    r'./config/uformer/config.yml',  # ______________5 Uformer
    r'./config/WiTUnet/config.yml',  # ______________6 WiTUnet-Tiny
]
config_path = config_paths[6]
# ================================================================================

tz = pytz.timezone('Asia/Shanghai')
date_str = datetime.datetime.now(tz).strftime("%Y_%m_%d_%H")


def train_loop(net: AbstractDenoiser, train_set, test_set, optimizer: torch.optim.Optimizer, config: dict):
    print("===========================woldier Deep Learning Distribution Framework====================================")
    data_loader_train = DataLoader(train_set, shuffle=True, batch_size=config["train"]["batch_size"])
    data_loader_test = DataLoader(test_set, shuffle=False, batch_size=config["train"]["batch_size"])
    # ==================================Accelerator Distribution Training===========================================
    accelerator = Accelerator()
    device = accelerator.device
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    net, optimizer, data_loader_train, data_loader_test, scheduler = accelerator.prepare(
        net, optimizer, data_loader_train, data_loader_test, scheduler
    )
    # =====================================================================================
    best_test_loss = 1000
    epochs = config["train"]["epochs"]
    start_e = config["train"].get("start-epochs", 0)
    base_path = date_str + "_" + config["logs"]["name"]  # YYYY_mm_DD_HH_XXX
    for epoch in range(start_e, start_e + epochs):
        start = time.time()
        # initialize  loss value for every epoch
        train_loss, test_loss = 0, 0
        # =============================train====================================
        train_loss = train_loop_one(
            data_loader_train,
            epoch,
            net,
            optimizer,
            train_loss,
            accelerator
        )
        # =============================test====================================

        ssim, psnr, rmse, test_loss = valid_loop_one(base_path,
                                                     data_loader_test,
                                                     epoch,
                                                     net,
                                                     test_loss,
                                                     accelerator,
                                                     (epoch + 1) % config["train"]["save_img_rate"] == 0)
        # =============================logging====================================
        best_test_loss = logging(base_path,
                                 best_test_loss,
                                 ssim,
                                 epoch,
                                 epochs,
                                 net,
                                 psnr,
                                 rmse,
                                 start,
                                 test_loss,
                                 train_loss,
                                 accelerator
                                 )
        scheduler.step()


def train_loop_one(data_loader_train, epoch, net: AbstractDenoiser, optimizer, train_loss, accelerator):
    with tqdm(total=len(data_loader_train), position=0, leave=True) as pbar:
        net.train()
        for step, batch in enumerate(data_loader_train):
            inputs = batch["LDCT"]
            labels = batch["FDCT"]
            # c = batch['c'].cuda()
            outputs, m_loss = net(inputs, labels)
            train_loss += m_loss.item()
            optimizer.zero_grad()
            # m_loss.backward()  # backward
            accelerator.backward(m_loss)  # backward
            optimizer.step()  # optimizer
            pbar.update()
            pbar.set_description("epoch%03d: Train Loss %.12f" % (epoch, train_loss / (step + 1)))  # 设置描述
        pbar.close()
    train_loss = train_loss / float(len(data_loader_train))
    return train_loss


def valid_loop_one(base_path, data_loader_test, epoch, net: AbstractDenoiser, test_loss, accelerator,
                   save_img: bool = False):
    index = np.random.randint(0, int(len(data_loader_test)), dtype="int")
    ssim, psnr, rmse = .0, .0, .0
    with tqdm(total=int(len(data_loader_test)), position=0, leave=True) as pbar, torch.no_grad():
        net.eval()
        for step, batch in enumerate(data_loader_test):
            inputs = batch["LDCT"]
            labels = batch["FDCT"]
            outputs, m_loss = net(inputs, labels)
            outputs, labels = accelerator.gather_for_metrics((outputs, labels))
            if index == step and save_img:
                show_img(labels,
                         inputs,
                         outputs,
                         name="./results/{}/img/{}.jpg".format(base_path, epoch),
                         )
                pass
            test_loss += m_loss.item()
            # =============回到原始尺度=================
            gt = trunc(denormalize(labels.cpu().detach().numpy()))
            pre = trunc(denormalize(outputs.cpu().detach().numpy()))
            ssim += SSIM(gt, pre)
            psnr += PSNR(gt, pre)
            rmse += RMSE(gt, pre)
            pbar.update()
            pbar.set_description("epoch%03d: Valid Loss %.12f" % (epoch, test_loss / (step + 1)))  # 设置描述
        pbar.close()
    test_loss = test_loss / float(len(data_loader_test))
    ssim = ssim / float(len(data_loader_test))
    psnr = psnr / float(len(data_loader_test))
    rmse = rmse / float(len(data_loader_test))
    return ssim, psnr, rmse, test_loss


def logging(base_path, best_test_loss, ssim, epoch, epochs, net, psnr, rmse, start, test_loss, train_loss, accelerator):
    if best_test_loss > test_loss:
        accelerator.wait_for_everyone()
        torch.save(accelerator.unwrap_model(net).state_dict(),
                   "./results/{}/weight/".format(base_path) + "best" + ".pth")
        best_test_loss = test_loss
    if epoch % 1 == 0:
        accelerator.wait_for_everyone()
        torch.save(accelerator.unwrap_model(net).state_dict(),
                   "./results/{}/weight/".format(base_path) + "EPOCH" + str(
                       epoch) + ".pth")
    log_str = '''Epoch #: {}/{}, Time taken: {} secs,\n\tLosses: train_MSE= {},test_MSE={}\n\tSSIM= {}, PSNR= {}, RMSE={}\n'''.format(
        epoch, epochs, time.time() - start, train_loss, test_loss, ssim, psnr, rmse)
    print(log_str)
    f = open("./results/{}/logs/log.txt".format(base_path), "a")
    f.writelines(log_str)
    f.close()  # close file
    return best_test_loss


def run():
    print(f"loading {config_path}")
    config = get_config(config_path)
    # check dir
    base_dir = "./results/{}/".format(date_str + "_" + config["logs"]["name"])
    check_dir(base_dir)
    # save config backpack
    config_backpack(config_path, base_dir)
    # init model
    model = init_model(config)
    # load dataset
    train_dataset, test_dataset = load_dataset(config)

    # set transform

    def transforms(examples):
        random_h = random.random()
        random_v = random.random()
        compose = Compose(
            [
                RandomHorizontalFlip(float(random_h > 0.5)),
                RandomVerticalFlip(float(random_v > 0.5)),
                ToTensor(),
            ]
        )
        examples["LDCT"] = [compose(Image.fromarray((np.array(image) * 255).astype(np.uint8))) for image in
                            examples["LDCT"]]
        examples["FDCT"] = [compose(Image.fromarray((np.array(image) * 255).astype(np.uint8))) for image in
                            examples["FDCT"]]

        return examples

    train_dataset.set_transform(transforms)
    test_dataset.set_transform(transforms)
    optim = init_optimizer(model, config)
    train_loop(model, train_dataset, test_dataset, optim, config)


if __name__ == '__main__':
    run()
