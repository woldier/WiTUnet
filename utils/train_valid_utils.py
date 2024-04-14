# -*- coding: utf-8 -*-
"""
@Time ： 3/18/24 4:17 AM
@Auth ： woldier wong
@File ：train_valid_utils.py
@IDE ：PyCharm
@DESCRIPTION：tian valid 所用到的工具类
"""
import yaml
import importlib  # import model
import torch
from datasets import Dataset
import os


def get_config(path: str):
    """
    加载yaml 配置文件
    :param path:
    :return:
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def init_model(conf: dict) -> torch.nn.Module:
    m = _get_model(conf)
    w_path = conf["model"]["weight_path"]
    if w_path is not None and w_path != '':
        m.load_state_dict(torch.load(w_path))
        print("load model weight from {}".format(w_path))
    torch.cuda.empty_cache()
    return m


def _get_model(conf: dict) -> torch.nn.Module:
    """
    加载model
    :return:
    """
    model_path = conf["model"]["path"]
    model_name = conf["model"]["class_name"]
    m = importlib.import_module(model_path)
    clz = getattr(m, model_name)
    return clz(**conf["model"]["config"])  # 实例化对象


def load_dataset(conf: dict, fmt: str = "torch"):
    """

    :param conf: 配置文件
    :param fmt: 数据集中加载的数据的格式
        在datasets 中支持 [None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']
        此处仅仅支持 ['numpy', 'torch', 'pandas']
    :return:
    """
    _dataset_fmt_check(fmt)
    train_set = Dataset.load_from_disk(**conf["dataset"]["train"])
    train_set.set_format(fmt)  # set format to pt
    test_set = Dataset.load_from_disk(**conf["dataset"]["test"])
    test_set.set_format(fmt)  # set format to pt
    return train_set, test_set


def _dataset_fmt_check(fmt):
    """
    检查fmt的合法性
    :param fmt:
    :return:
    """
    assert fmt in ['numpy', 'torch', 'pandas'], \
        f'''not support data format! need ['numpy', 'torch', 'pandas'], but have {fmt}'''


def load_dataset_with_path(path: str, fmt: str = "torch"):
    """
    从路径加载数据集
    :param path: 数据集的路径
    :param fmt: 数据格式
    :return:
    """
    _dataset_fmt_check(fmt)
    return Dataset.load_from_disk(path)


# def load_dataset(train_path: str, test_path: str, fmt: str = "torch"):
#     """
#
#     :param train_path: train data path
#     :param test_path: valid data path
#     :param fmt: 数据集中加载的数据的格式
#         在datasets 中支持 [None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow', 'jax']
#         此处仅仅支持 ['numpy', 'torch', 'pandas']
#     :return:
#     """
#     assert fmt in ['numpy', 'torch', 'pandas'], \
#         f'''not support data format! need ['numpy', 'torch', 'pandas'], but have {fmt}'''
#     train_set = Dataset.load_from_disk(train_path)
#     train_set.set_format(fmt)  # set format to pt
#     test_set = Dataset.load_from_disk(test_path)
#     test_set.set_format(fmt)  # set format to pt
#     return train_set, test_set


def check_dir(base_path):
    """
    检查输出文件是否存在, 没有的话则创建
    :return:
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)  # 同时创建夫文件夹
        os.mkdir(os.path.join(base_path, 'img'))
        os.mkdir(os.path.join(base_path, 'logs'))
        os.mkdir(os.path.join(base_path, 'weight'))


def config_backpack(conf_path, save_dir):
    config_save_path = save_dir + "config.yml"
    os.system("cp {} {}".format(conf_path, config_save_path))


def init_optimizer(model, config: dict) -> torch.optim.Optimizer:
    """
    初始化优化器
    :param model:
    :param config:
    :return:
    """
    opti_dict = {
        "AdamW": torch.optim.AdamW,
        "Adam": torch.optim.Adam,
        "SDG": torch.optim.SGD,
        "RMSprop": torch.optim.RMSprop
    }
    # 配置优化器
    optim_conf = config["train"].get("optimizer")
    if optim_conf is None:  # 如果说optimizer没有配置, 则加载默认的
        optim = torch.optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"], betas=(0.9, 0.99), eps=1e-08)
        print(f'''using default optimizer AdamW(lr:{config["train"]["learning_rate"]},betas=(0.9, 0.99), eps=1e-08)''')
    else:
        keys = optim_conf.keys()
        assert len(keys) == 1, f"the optim_conf key must be have one, but found {[key for key in keys]}"
        key = list(keys)[0]
        assert key in opti_dict.keys(), f"un support optimizer! support {[item for item in opti_dict.keys()]}, but get{key}"
        # 如果没有lr参数, 则使用train.learning_rate, 或者是有这个key 但是没有值, 或者是值小于零
        if 'lr' not in optim_conf[key].keys() or optim_conf[key]['lr'] is None or optim_conf[key]['lr'] <= .0:
            optim_conf[key]['lr'] = config["train"]["learning_rate"]
        print(f'''using config optimizer {key}({optim_conf[key]})''')
        optim = opti_dict[key](model.parameters(), **optim_conf[key])
    return optim
