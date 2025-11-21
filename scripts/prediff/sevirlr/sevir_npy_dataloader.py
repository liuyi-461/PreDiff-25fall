"""
NPY-based SEVIR DataLoader for PreDiff
读取.npy格式的SEVIR数据，每个文件包含13帧(128, 128)数据
"""
import os
from typing import Union, Sequence, Tuple, List
import numpy as np
import datetime
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
from torchvision import transforms
from torch import nn
from einops import rearrange
from lightning import LightningDataModule, seed_everything


class TransformsFixRotation(nn.Module):
    """固定角度旋转增强"""
    def __init__(self, angles: List[int] = [0, 90, 180, 270]):
        super().__init__()
        self.angles = angles
    
    def forward(self, x):
        angle = np.random.choice(self.angles)
        k = angle // 90  # 旋转次数
        return torch.rot90(x, k=k, dims=[-2, -1])


class SEVIRNPYDataset(TorchDataset):
    """
    读取NPY格式的SEVIR数据集
    每个.npy文件包含一个样本: shape (13, 128, 128)
    """
    
    orig_layout = "THW"  # 原始npy文件的布局
    aug_layout = "THW"
    
    def __init__(self,
                 npy_data_dir: str,
                 file_list: List[str],
                 in_len: int = 7,
                 out_len: int = 6,
                 layout: str = "THWC",
                 output_type=np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 aug_mode: str = "0",
                 ret_contiguous: bool = True):
        """
        Parameters
        ----------
        npy_data_dir : str
            npy文件所在目录
        file_list : List[str]
            要加载的npy文件名列表
        in_len : int
            输入序列长度（上下文帧数）
        out_len : int
            输出序列长度（预测帧数）
        layout : str
            输出数据布局，如 "THWC" 或 "NTHWC"
        output_type : dtype
            输出数据类型
        preprocess : bool
            是否进行预处理（归一化）
        rescale_method : str
            归一化方法 "01" 或 "sevir"
        aug_mode : str
            数据增强模式
            "0": 无增强
            "1": 随机水平翻转 + 垂直翻转 + 随机旋转
            "2": 随机水平翻转 + 垂直翻转 + 固定角度旋转(0,90,180,270)
        ret_contiguous : bool
            是否返回连续内存的tensor
        """
        super(SEVIRNPYDataset, self).__init__()
        self.npy_data_dir = npy_data_dir
        self.file_list = file_list
        self.in_len = in_len
        self.out_len = out_len
        self.seq_len = in_len + out_len
        self.layout = layout.replace("C", "1")  # 移除batch维度
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.ret_contiguous = ret_contiguous
        
        # 数据增强
        self.aug_mode = aug_mode
        if aug_mode == "0":
            self.aug = lambda x: x
        elif aug_mode == "1":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=180),
            )
        elif aug_mode == "2":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                TransformsFixRotation(angles=[0, 90, 180, 270]),
            )
        else:
            raise NotImplementedError(f"aug_mode {aug_mode} not supported")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        """
        读取一个npy文件并返回处理后的数据
        
        Returns
        -------
        data : torch.Tensor
            shape根据layout确定，默认为 (seq_len, H, W, 1)
        """
        # 读取npy文件
        npy_path = os.path.join(self.npy_data_dir, self.file_list[index])
        data = np.load(npy_path)  # shape: (13, 128, 128)
        
        # 确保是seq_len帧
        assert data.shape[0] == self.seq_len, \
            f"Expected {self.seq_len} frames, got {data.shape[0]} in {self.file_list[index]}"
        
        # 转换为torch tensor
        data = torch.from_numpy(data).float()
        
        # 数据增强 (在THW格式下进行)
        if self.aug_mode != "0":
            data = self.aug(data)
        
        # 添加channel维度: THW -> THW1
        data = data.unsqueeze(-1)
        
        # 预处理（归一化）
        if self.preprocess:
            if self.rescale_method == "01":
                # 归一化到[0, 1]
                data = data / 255.0
            elif self.rescale_method == "sevir":
                # SEVIR原始预处理
                scale = 1 / 47.54
                offset = -33.44
                data = scale * (data + offset)
            else:
                raise ValueError(f"Unknown rescale_method: {self.rescale_method}")
        
        # 转换布局: THW1 -> target layout
        data = rearrange(data, f"T H W 1 -> {' '.join(self.layout)}")
        
        # 转换数据类型
        if self.output_type == np.float32:
            data = data.float()
        
        if self.ret_contiguous:
            return data.contiguous()
        else:
            return data


class SEVIRNPYLightningDataModule(LightningDataModule):
    """
    Lightning DataModule for NPY-based SEVIR dataset
    """
    
    def __init__(self,
                 npy_data_dir: str,
                 in_len: int = 7,
                 out_len: int = 6,
                 layout: str = "NTHWC",
                 output_type=np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 aug_mode: str = "0",
                 ret_contiguous: bool = True,
                 # datamodule specific
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 seed: int = 0):
        """
        Parameters
        ----------
        npy_data_dir : str
            npy文件所在目录
        in_len : int
            输入序列长度
        out_len : int
            输出序列长度
        layout : str
            数据布局，需要包含batch维度N，如 "NTHWC"
        train_ratio : float
            训练集比例
        val_ratio : float
            验证集比例
        test_ratio : float
            测试集比例
        batch_size : int
            批大小
        num_workers : int
            数据加载worker数量
        seed : int
            随机种子
        """
        super(SEVIRNPYLightningDataModule, self).__init__()
        
        self.npy_data_dir = npy_data_dir
        self.in_len = in_len
        self.out_len = out_len
        self.seq_len = in_len + out_len
        
        assert layout[0] == "N", "Layout must start with 'N' for batch dimension"
        self.layout = layout.replace("N", "")  # 移除batch维度用于Dataset
        
        self.output_type = output_type
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.aug_mode = aug_mode
        self.ret_contiguous = ret_contiguous
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "train_ratio + val_ratio + test_ratio must equal 1.0"
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # 数据集信息
        self.img_height = 128
        self.img_width = 128
        self.interval_real_time = 10  # SEVIR-LR的时间间隔是10分钟
    
    def prepare_data(self):
        """检查数据目录是否存在"""
        if not os.path.exists(self.npy_data_dir):
            raise FileNotFoundError(
                f"NPY data directory not found: {self.npy_data_dir}\n"
                f"Please prepare your .npy files in this directory."
            )
        
        # 检查是否有npy文件
        npy_files = [f for f in os.listdir(self.npy_data_dir) if f.endswith('.npy')]
        if len(npy_files) == 0:
            raise FileNotFoundError(
                f"No .npy files found in {self.npy_data_dir}"
            )
        
        print(f"Found {len(npy_files)} .npy files in {self.npy_data_dir}")
    
    def setup(self, stage=None):
        """划分训练/验证/测试集"""
        seed_everything(seed=self.seed)
        
        # 获取所有npy文件
        all_files = sorted([f for f in os.listdir(self.npy_data_dir) 
                           if f.endswith('.npy')])
        
        total_samples = len(all_files)
        print(f"Total samples: {total_samples}")
        
        # 划分数据集
        train_size = int(total_samples * self.train_ratio)
        val_size = int(total_samples * self.val_ratio)
        test_size = total_samples - train_size - val_size
        
        # 使用固定种子打乱文件列表
        rng = np.random.RandomState(self.seed)
        shuffled_files = rng.permutation(all_files).tolist()
        
        train_files = shuffled_files[:train_size]
        val_files = shuffled_files[train_size:train_size + val_size]
        test_files = shuffled_files[train_size + val_size:]
        
        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        if stage in (None, "fit"):
            self.sevir_train = SEVIRNPYDataset(
                npy_data_dir=self.npy_data_dir,
                file_list=train_files,
                in_len=self.in_len,
                out_len=self.out_len,
                layout=self.layout,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                aug_mode=self.aug_mode,  # 训练集使用数据增强
                ret_contiguous=self.ret_contiguous
            )
            
            self.sevir_val = SEVIRNPYDataset(
                npy_data_dir=self.npy_data_dir,
                file_list=val_files,
                in_len=self.in_len,
                out_len=self.out_len,
                layout=self.layout,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                aug_mode="0",  # 验证集不使用数据增强
                ret_contiguous=self.ret_contiguous
            )
        
        if stage in (None, "test"):
            self.sevir_test = SEVIRNPYDataset(
                npy_data_dir=self.npy_data_dir,
                file_list=test_files,
                in_len=self.in_len,
                out_len=self.out_len,
                layout=self.layout,
                output_type=self.output_type,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                aug_mode="0",  # 测试集不使用数据增强
                ret_contiguous=self.ret_contiguous
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.sevir_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.sevir_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.sevir_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    @property
    def num_train_samples(self):
        return len(self.sevir_train)
    
    @property
    def num_val_samples(self):
        return len(self.sevir_val)
    
    @property
    def num_test_samples(self):
        return len(self.sevir_test)


# 测试代码
if __name__ == "__main__":
    # 测试数据加载
    npy_dir = "/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/data_npy_2019_len13/"
    
    dm = SEVIRNPYLightningDataModule(
        npy_data_dir=npy_dir,
        in_len=7,
        out_len=6,
        layout="NTHWC",
        batch_size=4,
        num_workers=2,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    # 准备数据
    dm.prepare_data()
    dm.setup()
    
    # 测试训练集加载
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch.shape}")  # 应该是 (batch_size, seq_len, H, W, C)
    print(f"Batch dtype: {batch.dtype}")
    print(f"Batch min/max: {batch.min():.4f} / {batch.max():.4f}")
    
    print(f"\nDataset sizes:")
    print(f"Train: {dm.num_train_samples}")
    print(f"Val: {dm.num_val_samples}")
    print(f"Test: {dm.num_test_samples}")
