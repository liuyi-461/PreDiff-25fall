import warnings
from torch.utils.data import Dataset, DataLoader
import glob

from shutil import copyfile
import inspect
from collections import OrderedDict
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torchmetrics
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything, loggers as pl_loggers
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import (
    Callback, LearningRateMonitor, DeviceStatsMonitor,
    EarlyStopping, ModelCheckpoint, )
from lightning.pytorch.utilities import grad_norm
from omegaconf import OmegaConf
import os
import argparse
from einops import rearrange


class NPYDataset(Dataset):
    """
    自定义数据集：从指定目录读取单个或多个 .npy 文件。
    每个 .npy 文件形状为 (128, 128, 25) —— 即 (H, W, T)
    """

    def __init__(self, data_dir, limit=None, normalize=True):
        """
        Args:
            data_dir (str): 包含 .npy 文件的目录
            limit (int or None): 读取的文件数量，None 表示读取全部
            normalize (bool): 是否归一化到 [0, 1]
        """
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if limit is not None:
            self.files = self.files[:limit]

        if len(self.files) == 0:
            raise FileNotFoundError(f"未在 {data_dir} 中找到 .npy 文件")

        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])  # shape: (128, 128, 25)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.expand_dims(arr, axis=-1)  # -> (128, 128, 25, 1)

        if self.normalize:
            # arr = arr / np.max(arr) if np.max(arr) > 0 else arr
            arr = arr / 180.0  # 归一化到 [0, 1]

        # 转换为 torch.Tensor 并调整维度为 (T, H, W, C)
        arr = np.transpose(arr, (2, 0, 1, 3))  # (25, 128, 128, 1)
        arr = torch.tensor(arr, dtype=torch.float32)
        return arr
    
    def get_file_index(self, filename):
        """根据文件名查找索引"""
        basename_list = [os.path.basename(f) for f in self.files]
        if filename in basename_list:
            return basename_list.index(filename)
        return None
    
# 使用
data_dir = "/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/radar_npy_len13"
dataset = NPYDataset(data_dir=data_dir)
idx1 = dataset.get_file_index("2018-12-07_17-20-52.npy")
idx2 = dataset.get_file_index("2019-01-14_13-20-52.npy")
idx3 = dataset.get_file_index("2019-03-05_03-50-51.npy")
print(f"索引值1: {idx1}")
print(f"索引值2: {idx2}")
print(f"索引值3: {idx3}")
    