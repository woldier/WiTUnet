# -*- coding: utf-8 -*-
"""
@Time ： 3/19/24 3:06 AM
@Auth ： woldier wong
@File ：img_utils.py
@IDE ：PyCharm
@DESCRIPTION：TODO
"""
from typing import Optional
import torch
from torchvision.transforms import ToPILImage
import numpy as np
from PIL import Image

to_pil = ToPILImage()


def tensor_to_PILImage(img: torch.Tensor):
    if len(img.shape) >= 3:
        img = img[0].squeeze()
    pil_image = to_pil(img.cpu().detach())
    return pil_image


def tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    pil = tensor_to_PILImage(img)
    pil_numpy = np.array(pil)
    return pil_numpy


def show_img(x, y, pre, name: str = "./test.jpg"):
    name_new = name[:-4] + "{}" + name[-4:]
    tensor_to_PILImage(x).save(name_new.format("_FDCT"))
    tensor_to_PILImage(y).save(name_new.format("_LDCT"))
    if pre is not None:
        tensor_to_PILImage(pre).save(name_new.format("_PRE"))


if __name__ == '__main__':
    pass
