"""
自定义数据加载器 - 替代原来的SEVIRLightningDataModule
完全兼容原始接口，使用全局固定归一化参数
不使用数据增强
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from typing import Optional

# ========== 防止X11错误 ==========
import matplotlib
matplotlib.use('Agg')
# ===============================


class CustomDataset(Dataset):
    """
    自定义数据集类 - 完全兼容原始SEVIRDataLoader
    无数据增强版本
    """
    
    # 全局固定归一化参数
    VIL_MIN = 0.0
    VIL_MAX = 255.0
    
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 13,
        in_len: int = 7,
        out_len: int = 6,
        img_height: int = 128,
        img_width: int = 128,
        data_channels: int = 1,
        split: str = 'train',
        sample_mode: str = 'sequent',
        stride: int = 6,
        layout: str = 'NTHWC',
        preprocess: bool = True,
        rescale_method: str = '01',
        ret_contiguous: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.in_len = in_len
        self.out_len = out_len
        self.img_height = img_height
        self.img_width = img_width
        self.data_channels = data_channels
        self.split = split
        self.sample_mode = sample_mode
        self.stride = stride
        self.layout = layout
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.ret_contiguous = ret_contiguous
        
        # 加载数据文件列表
        self.data_list = self._load_data_list()
    
    def _load_data_list(self):
        """加载数据文件列表"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        pattern = os.path.join(self.data_dir, "*.npy")
        data_files = sorted(glob.glob(pattern))
        
        if len(data_files) == 0:
            raise FileNotFoundError(f"在 {self.data_dir} 中找不到.npy文件")
        
        return data_files
    
    def _load_single_sample(self, idx):
        """
        加载单个样本
        处理 (H, W, T) 格式的数据，转换为 (T, H, W, C)
        """
        data_path = self.data_list[idx]
        data = np.load(data_path)  # 形状: (128, 128, 25)
        
        # 转换形状: (H, W, T) -> (T, H, W)
        data = np.transpose(data, (2, 0, 1))  # (25, 128, 128)
        
        # 只取前 seq_len 张
        if data.shape[0] > self.seq_len:
            data = data[:self.seq_len]  # (13, 128, 128)
        elif data.shape[0] < self.seq_len:
            raise ValueError(f"数据时间步数 {data.shape[0]} < {self.seq_len}")
        
        # 添加通道维度: (T, H, W) -> (T, H, W, C)
        data = data[..., np.newaxis]  # (13, 128, 128, 1)
        
        return data
    
    def _preprocess(self, data):
        """使用全局固定参数归一化"""
        data = data.astype(np.float32)
        
        if self.rescale_method == '01':
            data = (data - self.VIL_MIN) / (self.VIL_MAX - self.VIL_MIN)
        elif self.rescale_method == 'neg1to1':
            data = 2 * (data - self.VIL_MIN) / (self.VIL_MAX - self.VIL_MIN) - 1
        elif self.rescale_method == 'std':
            data = (data - data.mean()) / (data.std() + 1e-8)
        
        return data
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """返回一个样本: (seq_len, H, W, C)"""
        # 加载数据
        data = self._load_single_sample(idx)
        
        # 预处理
        if self.preprocess:
            data = self._preprocess(data)
        
        # 转换为tensor
        data = torch.from_numpy(data).float()
        
        # 根据layout调整维度
        if self.layout == 'NTCHW':
            # 从 (T, H, W, C) 转换为 (T, C, H, W)
            data = data.permute(0, 3, 1, 2)
        elif self.layout == 'NTHWC':
            # 保持 (T, H, W, C)
            pass
        
        # 返回连续tensor
        if self.ret_contiguous:
            return data.contiguous()
        return data


class CustomLightningDataModule(LightningDataModule):
    """Lightning DataModule - 无数据增强版本"""
    
    def __init__(
        self,
        data_dir: str = None,
        seq_len: int = 13,
        in_len: int = 7,
        out_len: int = 6,
        img_height: int = 128,
        img_width: int = 128,
        data_channels: int = 1,
        batch_size: int = 2,
        num_workers: int = 8,
        sample_mode: str = 'sequent',
        stride: int = 6,
        layout: str = 'NTHWC',
        val_ratio: float = 0.1,
        preprocess: bool = True,
        rescale_method: str = '01',
        # 兼容原始接口的参数（但不使用）
        output_type = None,
        verbose: bool = False,
        aug_mode: str = "0",  # 默认不使用增强
        ret_contiguous: bool = True,
        dataset_name: str = None,
        start_date = None,
        train_test_split_date = None,
        end_date = None,
        seed: int = 0,
        **kwargs
    ):
        super().__init__()
        
        if data_dir is None:
            data_dir = '/data/public/nowcasting/sevirlr/data_npy_13/'
        
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.in_len = in_len
        self.out_len = out_len
        self.img_height = img_height
        self.img_width = img_width
        self.data_channels = data_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_mode = sample_mode
        self.stride = stride
        self.layout = layout
        self.val_ratio = val_ratio
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.output_type = output_type
        self.verbose = verbose
        self.ret_contiguous = ret_contiguous
        self.dataset_name = dataset_name
        self.seed = seed
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _get_all_files(self):
        """获取所有数据文件"""
        pattern = os.path.join(self.data_dir, "*.npy")
        return sorted(glob.glob(pattern))
    
    def prepare_data(self):
        """检查数据是否存在"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        all_files = self._get_all_files()
        if len(all_files) == 0:
            raise FileNotFoundError(f"在 {self.data_dir} 中找不到.npy文件")
        
        print(f"Found {len(all_files)} NPY files in {self.data_dir}")
    
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if stage == 'fit' or stage is None:
            all_files = self._get_all_files()
            num_files = len(all_files)
            num_val = int(num_files * self.val_ratio)
            num_train = num_files - num_val
            
            train_files = all_files[:num_train]
            val_files = all_files[num_train:]
            
            self.train_dataset = CustomDataset(
                data_dir=self.data_dir,
                seq_len=self.seq_len,
                in_len=self.in_len,
                out_len=self.out_len,
                img_height=self.img_height,
                img_width=self.img_width,
                data_channels=self.data_channels,
                split='train',
                sample_mode=self.sample_mode,
                stride=self.stride,
                layout=self.layout,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                ret_contiguous=self.ret_contiguous,
            )
            self.train_dataset.data_list = train_files
            
            self.val_dataset = CustomDataset(
                data_dir=self.data_dir,
                seq_len=self.seq_len,
                in_len=self.in_len,
                out_len=self.out_len,
                img_height=self.img_height,
                img_width=self.img_width,
                data_channels=self.data_channels,
                split='val',
                sample_mode=self.sample_mode,
                stride=self.stride,
                layout=self.layout,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                ret_contiguous=self.ret_contiguous,
            )
            self.val_dataset.data_list = val_files
            
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset size: {len(self.val_dataset)}")
        
        if stage == 'test' or stage is None:
            all_files = self._get_all_files()
            
            self.test_dataset = CustomDataset(
                data_dir=self.data_dir,
                seq_len=self.seq_len,
                in_len=self.in_len,
                out_len=self.out_len,
                img_height=self.img_height,
                img_width=self.img_width,
                data_channels=self.data_channels,
                split='test',
                sample_mode=self.sample_mode,
                stride=self.stride,
                layout=self.layout,
                preprocess=self.preprocess,
                rescale_method=self.rescale_method,
                ret_contiguous=self.ret_contiguous,
            )
            self.test_dataset.data_list = all_files
            
            print(f"Test dataset size: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    @property
    def num_train_samples(self):
        """返回训练样本数量"""
        return len(self.train_dataset) if self.train_dataset else 1000
    
    @property
    def num_val_samples(self):
        """返回验证样本数量"""
        return len(self.val_dataset) if self.val_dataset else 100
    
    @property
    def num_test_samples(self):
        """返回测试样本数量"""
        return len(self.test_dataset) if self.test_dataset else 100