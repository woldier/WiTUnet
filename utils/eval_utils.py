# -*- coding: utf-8 -*-
"""
@Time ： 3/18/24 4:27 AM
@Auth ： woldier wong
@File ：eval_utils.py
@IDE ：PyCharm
@DESCRIPTION：评价指标
"""
import numpy as np
from skimage.metrics import structural_similarity as ssim


def denormalize(image, min=-1024.0, max=3072.0):
    image = image * (max - min) + min
    return image


def trunc(mat):
    mat[mat <= -160.0] = -160.0
    mat[mat >= 240.0] = 240.0
    return mat


def SSIM(x: np.ndarray, y: np.ndarray, data_range: int = 240 - (-160), is_mean: bool = True) -> np.ndarray:
    """
    Structural Similarity (SSIM) measures the similarity between images. 
    The mean was used to compare brightness, the standard deviation was used to compare contrast, 
    and the covariance was used to compare structural similarity. For samples x and y SSIM is defined as
    the following equation
    $SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\mu_{xy}+C_2)}{(\mu^2_x + \mu^2_y+C_1)(\mu^2_x + \mu^2_y+C_2)}$
    Where μx, μy, σ^2_x, σ^2_y denote means and standard devitions of x and y respectively, and σ_{xy} is the covariance
    between x and y. C_1 and C_2 are predefined constants. SSIM results range from 0 to 1, with values closer to 1 giving
    better results.
    x, y 是灰度图像, 值的范围为0-255
    Examples:


    :param x: 
    :param y:
    :param is_mea:
    :return: 
    """
    assert len(x.shape) == 4, "input shape mast be [B,1,H,W]"
    B, _, _, _ = x.shape
    res_list = []
    for i in range(B):
        xx = x[i].squeeze()
        yy = y[i].squeeze()
        res = ssim(xx, yy, data_range=data_range)
        res_list.append(res)
    # 返回平均SSIM值
    if is_mean:
        return np.mean(res_list)
    else:
        return res_list


def PSNR(x: np.ndarray, y: np.ndarray, data_range: int = 240 - (-160), is_mean: bool = True) -> np.ndarray:
    """
    Peak Signal-to-Noise Ratio (PSNR) is defined as the ratio between
    the maximum possible power in a signal and the noise power.
    PSNR is often used to measure the image quality after denoising.
    The higher the value, the higher the image quality after processing.
    For the clean image x and the noisy image y, PSNR is defined as the following equation.
    PSNR = 10 * log10(MAX^2/MSE)
    Where MAX denotes the maximum value of each pixel in the image. MSE is defined as the following equation.
    Example:
            >>> B = 4  # 批量大小
            >>> H = 256  # 图像高度
            >>> W = 256  # 图像宽度
            >>>  # 生成示例批量图像
            >>> x_batch = np.random.randn(B, 1, H, W)
            >>> y_batch = np.random.randn(B, 1, H, W)
            >>> print(PSNR(x_batch, y_batch))
    :param x:
    :param y:
    :param data_range:
    :param is_mean:
    :return:
    """
    mean_mask = [i for i in range(len(x.shape))]

    # 计算像素值的最大可能值
    max_pixel = data_range
    # mse = np.expand_dims(MSE(x, y) ,axis=mean_mask)
    mse = MSE(x, y)
    psnr = 10 * np.log10(max_pixel ** 2 / mse)
    if is_mean:
        return psnr.mean()
    else:
        return psnr


def MSE(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    >>> B = 4  # 批量大小
    >>> H = 256  # 图像高度
    >>> W = 256  # 图像宽度
    >>> # 生成示例批量图像
    >>> x_batch = np.random.randn(B, 1, H, W)
    >>> y_batch = np.random.randn(B, 1, H, W)
    >>> print(MSE(x_batch, y_batch))
    :param x:
    :param y:
    :return:
    """
    mean_mask = [i for i in range(len(x.shape))]
    mse = ((x - y) ** 2).mean(axis=tuple(mean_mask[1:]))
    return mse


def RMSE(x: np.ndarray, y: np.ndarray, is_mean: bool = True) -> np.ndarray:
    rmse = MSE(x, y) ** 0.5
    if is_mean:
        return rmse.mean()
    else:
        return rmse


if __name__ == '__main__':
    B = 4  # 批量大小
    H = 256  # 图像高度
    W = 256  # 图像宽度
    # 生成示例批量图像
    x_batch = np.random.randn(B, 1, H, W)
    y_batch = np.random.randn(B, 1, H, W)
    print(SSIM(x_batch, y_batch))
    pass
