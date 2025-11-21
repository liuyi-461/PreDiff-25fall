"""
自定义数据加载器 - 替代原来的SEVIRLightningDataModule
完全兼容原始接口，使用全局固定归一化参数
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from typing import Optional


class CustomDataset(Dataset):
    """
    自定义数据集类 - 完全兼容原始SEVIRDataLoader
    
    关键特性:
    1. 处理 (H, W, T) 格式的NPY文件
    2. 使用全局固定归一化参数（与SEVIR一致）
    3. 支持数据增强
    4. 返回连续tensor
    """
    
    # ========== 全局固定归一化参数（与原始SEVIR完全一致）==========
    # 这些参数是预先在整个SEVIR数据集上计算的，所有样本使用相同的值
    PREPROCESS_SCALE_SEVIR = {
        'vis': {'scale': 1.0 / 255.0, 'min_val': 0.0, 'max_val': 255.0},
        'ir069': {'scale': 1.0 / 1174.0, 'min_val': -109.0, 'max_val': 1174.0},
        'ir107': {'scale': 1.0 / 4295.0, 'min_val': -4295.0, 'max_val': 2895.0},
        'vil': {'scale': 1.0 / 255.0, 'min_val': 0.0, 'max_val': 255.0},
        'lght': {'scale': 1.0, 'min_val': 0.0, 'max_val': 1.0},
    }
    
    # 对于VIL数据（你的数据类型），使用固定参数
    VIL_MIN = 0.0
    VIL_MAX = 255.0
    # ==============================================================
    
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
        aug_mode: str = "1",
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
        self.aug_mode = aug_mode
        self.ret_contiguous = ret_contiguous
        
        # 设置数据增强（只在训练时使用）
        if split == 'train' and aug_mode != "0":
            if aug_mode == "1":
                from torchvision import transforms
                self.aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(degrees=180),
                ])
            elif aug_mode == "2":
                from torchvision import transforms
                # 固定角度旋转：0, 90, 180, 270度
                self.aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomChoice([
                        transforms.RandomRotation(degrees=(0, 0)),
                        transforms.RandomRotation(degrees=(90, 90)),
                        transforms.RandomRotation(degrees=(180, 180)),
                        transforms.RandomRotation(degrees=(270, 270)),
                    ])
                ])
            else:
                self.aug = lambda x: x
        else:
            self.aug = lambda x: x
        
        # 加载数据文件列表或索引
        self.data_list = self._load_data_list()
    
    def _load_data_list(self):
        """
        加载数据文件列表
        返回所有.npy文件的路径列表
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 获取所有.npy文件
        pattern = os.path.join(self.data_dir, "*.npy")
        data_files = sorted(glob.glob(pattern))
        
        if len(data_files) == 0:
            raise FileNotFoundError(f"在 {self.data_dir} 中找不到.npy文件")
        
        return data_files
    
    def _load_single_sample(self, idx):
        """
        加载单个样本
        
        处理 (H, W, T) 格式的数据，转换为 (T, H, W, C)
        只取前 seq_len 张图片
        """
        # 从单个文件加载
        data_path = self.data_list[idx]
        data = np.load(data_path)  # 形状: (128, 128, 25)
        
        # 转换形状: (H, W, T) -> (T, H, W)
        data = np.transpose(data, (2, 0, 1))  # 现在是 (25, 128, 128)
        
        # 只取前 seq_len 张（例如前13张）
        if data.shape[0] > self.seq_len:
            data = data[:self.seq_len]  # 现在是 (13, 128, 128)
        elif data.shape[0] < self.seq_len:
            raise ValueError(f"数据时间步数 {data.shape[0]} 小于所需的 {self.seq_len}")
        
        # 添加通道维度: (T, H, W) -> (T, H, W, C)
        data = data[..., np.newaxis]  # 现在是 (13, 128, 128, 1)
        
        return data
    
    def _preprocess(self, data):
        """
        数据预处理 - 使用全局固定参数归一化（与原始SEVIR完全一致）
        
        关键: 不使用每个样本自己的min/max，而是使用全局固定值
        这确保了:
        1. 所有样本使用相同的归一化参数
        2. 训练集和测试集归一化一致
        3. 与原始SEVIR DataLoader行为完全相同
        """
        # 确保数据类型
        data = data.astype(np.float32)
        
        # 归一化 - 使用全局固定参数
        if self.rescale_method == '01':
            # ✅ 与原始SEVIR一致: 使用全局固定的min/max
            # 归一化到 [0, 1]
            data = (data - self.VIL_MIN) / (self.VIL_MAX - self.VIL_MIN)
            
        elif self.rescale_method == 'neg1to1':
            # 归一化到 [-1, 1]
            data = 2 * (data - self.VIL_MIN) / (self.VIL_MAX - self.VIL_MIN) - 1
            
        elif self.rescale_method == 'std':
            # 标准化 (这个方法仍需要逐样本计算)
            data = (data - data.mean()) / (data.std() + 1e-8)
        
        return data
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        返回一个样本
        
        返回格式: (seq_len, H, W, C) 如果layout是'NTHWC'
        """
        # 加载数据（已经是正确形状: (13, 128, 128, 1)）
        data = self._load_single_sample(idx)
        
        # 预处理（使用全局固定参数归一化）
        if self.preprocess:
            data = self._preprocess(data)
        
        # 转换为tensor
        data = torch.from_numpy(data).float()
        
        # 数据增强（只在训练时，且aug_mode != "0"）
        if self.split == 'train' and self.aug_mode != "0":
            # 数据增强需要 (T, H, W) 格式
            # 当前是 (T, H, W, C=1)，先去掉通道维度
            data = data.squeeze(-1)  # (T, H, W)
            
            # 应用数据增强
            data = self.aug(data)  # (T, H, W)
            
            # 恢复通道维度
            data = data.unsqueeze(-1)  # (T, H, W, C=1)
        
        # 根据layout调整维度
        if self.layout == 'NTCHW':
            # 从 (T, H, W, C) 转换为 (T, C, H, W)
            data = data.permute(0, 3, 1, 2)
        elif self.layout == 'NTHWC':
            # 保持 (T, H, W, C)
            pass
        
        # 确保返回连续的tensor（与原始DataLoader一致）
        if self.ret_contiguous:
            return data.contiguous()
        else:
            return data


class CustomLightningDataModule(LightningDataModule):
    """
    Lightning DataModule - 完全兼容原始SEVIRLightningDataModule接口
    """
    
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
        # ========== 兼容原始接口的参数 ==========
        output_type = None,
        verbose: bool = False,
        aug_mode: str = "1",           # 数据增强模式
        ret_contiguous: bool = True,   # 返回连续tensor
        dataset_name: str = None,
        start_date = None,
        train_test_split_date = None,
        end_date = None,
        seed: int = 0,
        **kwargs  # 接受所有其他参数
        # ===================================================
    ):
        super().__init__()
        
        # 如果data_dir为None，使用默认路径
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
        
        # 原始接口参数
        self.output_type = output_type
        self.verbose = verbose
        self.aug_mode = aug_mode
        self.ret_contiguous = ret_contiguous
        self.dataset_name = dataset_name
        self.seed = seed
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _get_all_files(self):
        """获取所有数据文件"""
        pattern = os.path.join(self.data_dir, "*.npy")
        all_files = sorted(glob.glob(pattern))
        return all_files
    
    def prepare_data(self):
        """
        下载或准备数据（在单个进程中执行）
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"数据目录不存在: {self.data_dir}\n"
                f"请确保NPY数据文件在正确的位置"
            )
        
        # 检查是否有数据文件
        all_files = self._get_all_files()
        if len(all_files) == 0:
            raise FileNotFoundError(
                f"在 {self.data_dir} 中找不到.npy文件\n"
                f"请检查数据路径是否正确"
            )
        
        print(f"Found {len(all_files)} NPY files in {self.data_dir}")
    
    def setup(self, stage: Optional[str] = None):
        """
        设置数据集（在每个进程中执行）
        注意: 所有split使用相同的data_dir，不再使用子目录
        """
        if stage == 'fit' or stage is None:
            # 训练集和验证集从同一个目录读取
            all_files = self._get_all_files()
            
            # 按比例分割训练集和验证集
            num_files = len(all_files)
            num_val = int(num_files * self.val_ratio)
            num_train = num_files - num_val
            
            # 训练集使用前num_train个文件
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
                aug_mode=self.aug_mode,  # 训练时使用数据增强
                ret_contiguous=self.ret_contiguous,
            )
            # 覆盖数据列表
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
                aug_mode="0",  # 验证时不使用数据增强
                ret_contiguous=self.ret_contiguous,
            )
            # 覆盖数据列表
            self.val_dataset.data_list = val_files
            
            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Val dataset size: {len(self.val_dataset)}")
        
        if stage == 'test' or stage is None:
            # 测试集也从同一个目录读取（可以使用全部数据或最后部分）
            all_files = self._get_all_files()
            test_files = all_files  # 或者可以选择特定部分
            
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
                aug_mode="0",  # 测试时不使用数据增强
                ret_contiguous=self.ret_contiguous,
            )
            # 覆盖数据列表
            self.test_dataset.data_list = test_files
            
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
        """返回训练样本数量（用于计算total_num_steps）"""
        if self.train_dataset is not None:
            return len(self.train_dataset)
        else:
            # 如果还没setup，估算一个值
            return 1000
    
    @property
    def num_val_samples(self):
        """返回验证样本数量"""
        if self.val_dataset is not None:
            return len(self.val_dataset)
        else:
            return 100
    
    @property
    def num_test_samples(self):
        """返回测试样本数量"""
        if self.test_dataset is not None:
            return len(self.test_dataset)
        else:
            return 100