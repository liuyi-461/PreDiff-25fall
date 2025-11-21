# prediff/datasets/sevir/sevir_torch_wrap.py
"""
Code is adapted from https://github.com/amazon-science/earth-forecasting-transformer/blob/e60ff41c7ad806277edc2a14a7a9f45585997bd7/src/earthformer/datasets/sevir/sevir_torch_wrap.py
Add data augmentation.
Only return "VIL" data in `torch.Tensor` format instead of `Dict`

+ Add: SEVIRNPYDataset (for *.npy files)
+ Add: SEVIRLightningDataModule supports dataset_name="sevirlr_npy"
"""
import os
import glob
from typing import Union, Dict, Sequence, Tuple, List
import numpy as np
import datetime
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
from torchvision import transforms
from einops import rearrange
from lightning import LightningDataModule, seed_everything

from .sevir_dataloader import SEVIRDataLoader
from ...utils.path import default_dataset_sevir_dir, default_dataset_sevirlr_dir
from ..augmentation import TransformsFixRotation


# =========================================
# 原有：HDF5 路线（保留，不动）
# =========================================

def check_aws():
    if os.system("which aws") != 0:
        raise RuntimeError("AWS CLI is not installed! Please install it first. See https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")


def download_SEVIR(save_dir=None):
    check_aws()
    if save_dir is None:
        save_dir = default_dataset_sevir_dir
    else:
        save_dir = os.path.join(save_dir, "sevir")
    if os.path.exists(save_dir):
        raise FileExistsError(f"Path to save SEVIR dataset {save_dir} already exists!")
    else:
        os.makedirs(save_dir)
        os.system(f"aws s3 cp --no-sign-request s3://sevir/CATALOG.csv "
                  f"{os.path.join(save_dir, 'CATALOG.csv')}")
        os.system(f"aws s3 cp --no-sign-request --recursive s3://sevir/data/vil "
                  f"{os.path.join(save_dir, 'data', 'vil')}")


def download_SEVIRLR(save_dir=None):
    if save_dir is None:
        save_dir = default_dataset_sevirlr_dir
    else:
        save_dir = os.path.join(save_dir, "sevirlr")
    if os.path.exists(save_dir):
        raise FileExistsError(f"Path to save SEVIR-LR dataset {save_dir} already exists!")
    else:
        os.makedirs(save_dir)
        os.system(f"wget https://deep-earth.s3.amazonaws.com/datasets/sevir_lr.zip "
                  f"-P {os.path.abspath(save_dir)}")
        os.system(f"unzip {os.path.join(save_dir, 'sevir_lr.zip')} "
                  f"-d {save_dir}")
        os.system(f"mv {os.path.join(save_dir, 'sevir_lr', '*')} "
                  f"{save_dir}\n"
                  f"rm -rf {os.path.join(save_dir, 'sevir_lr')}")


class SEVIRTorchDataset(TorchDataset):
    """原 HDF5 数据集（保留）"""
    orig_dataloader_layout = "NHWT"
    orig_dataloader_squeeze_layout = orig_dataloader_layout.replace("N", "")
    aug_layout = "THW"

    def __init__(self,
                 seq_len: int = 25,
                 raw_seq_len: int = 49,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 layout: str = "THWC",
                 split_mode: str = "uneven",
                 sevir_catalog: Union[str, pd.DataFrame] = None,
                 sevir_data_dir: str = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 datetime_filter = None,
                 catalog_filter = "default",
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 output_type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 verbose: bool = False,
                 aug_mode: str = "0",
                 ret_contiguous: bool = True):
        super(SEVIRTorchDataset, self).__init__()
        self.layout = layout.replace("C", "1")
        self.ret_contiguous = ret_contiguous
        self.sevir_dataloader = SEVIRDataLoader(
            data_types=["vil", ],
            seq_len=seq_len,
            raw_seq_len=raw_seq_len,
            sample_mode=sample_mode,
            stride=stride,
            batch_size=1,
            layout=self.orig_dataloader_layout,
            num_shard=1,
            rank=0,
            split_mode=split_mode,
            sevir_catalog=sevir_catalog,
            sevir_data_dir=sevir_data_dir,
            start_date=start_date,
            end_date=end_date,
            datetime_filter=datetime_filter,
            catalog_filter=catalog_filter,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            output_type=output_type,
            preprocess=preprocess,
            rescale_method=rescale_method,
            downsample_dict=None,
            verbose=verbose)
        self.aug_mode = aug_mode
        if aug_mode == "0":
            self.aug = lambda x:x
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
            raise NotImplementedError

    def __getitem__(self, index):
        data_dict = self.sevir_dataloader._idx_sample(index=index)
        data = data_dict["vil"].squeeze(0)  # THW
        if self.aug_mode != "0":
            data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.aug_layout)}")
            data = self.aug(data)
            data = rearrange(data, f"{' '.join(self.aug_layout)} -> {' '.join(self.layout)}")
        else:
            data = rearrange(data, f"{' '.join(self.orig_dataloader_squeeze_layout)} -> {' '.join(self.layout)}")
        return data.contiguous() if self.ret_contiguous else data

    def __len__(self):
        return self.sevir_dataloader.__len__()


# =========================================
# 新增：NPY 路线
# =========================================

class SEVIRNPYDataset(TorchDataset):
    """
    直接读取 *.npy（形状 H*W*T），在时间维按滑动窗口切分：
      T_window = input_len + pred_len (= seq_len)
      start in range(0, T - seq_len + 1, stride)
    例如 T=25, seq_len=13, stride=6 => [0:13], [6:19], [12:25]
    返回张量布局：THWC（无 batch 维）
    """
    aug_layout = "THW"

    def __init__(self,
                 root_dir: str,
                 seq_len: int = 13,
                 input_len: int = 7,
                 pred_len: int = 6,
                 stride: int = 6,
                 layout: str = "THWC",
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 aug_mode: str = "0",
                 ret_contiguous: bool = True):
        super().__init__()
        assert os.path.isdir(root_dir), f"NPY root_dir not found: {root_dir}"
        self.root_dir = os.path.abspath(root_dir)
        self.seq_len = int(seq_len)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.layout = layout.replace("C", "1")
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.ret_contiguous = ret_contiguous

        # 列出所有 *.npy
        self.files: List[str] = sorted(glob.glob(os.path.join(self.root_dir, "*.npy")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy found under {self.root_dir}")

        # 为每个文件计算所有窗口起点，形成全局索引
        self.index: List[Tuple[int, int]] = []  # (file_idx, start_t)
        for fi, f in enumerate(self.files):
            arr = np.load(f, mmap_mode="r")  # H, W, T
            if arr.ndim != 3:
                raise ValueError(f"{f} must be shape (H, W, T), got {arr.shape}")
            T = arr.shape[2]
            if T < self.seq_len:
                continue
            for s in range(0, T - self.seq_len + 1, self.stride):
                self.index.append((fi, s))
        if len(self.index) == 0:
            raise RuntimeError(f"No valid windows in {self.root_dir}. Check seq_len/stride.")

        # 数据增强：与原版保持一致（在 THW 上做 flip/rot）
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
            raise NotImplementedError
        self.aug_mode = aug_mode

    def __len__(self):
        return len(self.index)

    def _preprocess_01(self, x: torch.Tensor):
        # x: THW (float)
        # 原版 '01' 是 VIL / 255
        return x / 255.0

    # def __getitem__(self, idx: int):
    #     fi, s = self.index[idx]
    #     path = self.files[fi]
    #     arr = np.load(path)  # H, W, T
    #     H, W, T = arr.shape
    #     t_slice = slice(s, s + self.seq_len)
    #     # 转为 THW（且加 Channel=1）
    #     thw = torch.from_numpy(arr[:, :, t_slice]).permute(2, 0, 1).float()  # THW

    #     # 预处理
    #     if self.preprocess:
    #         if self.rescale_method == "01":
    #             thw = self._preprocess_01(thw)
    #         else:
    #             raise NotImplementedError(f"rescale_method={self.rescale_method}")

    #     # 数据增强（在 THW 上）
    #     if self.aug_mode != "0":
    #         thw = self.aug(thw)

    #     # THW -> 目标布局（THWC 或 TCHW 等；这里只有 C=1）
    #     if "C" in self.layout:
    #         # 插入 C=1
    #         thwc = thw.unsqueeze(-1)  # THW1
    #         out = rearrange(thwc, "t h w c -> " + " ".join(self.layout))
    #     else:
    #         out = rearrange(thw, "t h w -> " + " ".join(self.layout))

    #     return out.contiguous() if self.ret_contiguous else out

    def __getitem__(self, i):
        fi, s = self.index[i]
        path = self.files[fi]
        vol = np.load(path).astype(np.float32)  # (H, W, T)
        clip = vol[..., s:s + self.seq_len]     # (H, W, seq_len)

        # Normalize
        if clip.max() > 1.0:
            clip = clip / 255.0

        # (T, H, W, 1)
        clip = clip.transpose(2, 0, 1)[..., None]
        x = torch.from_numpy(clip)  # [T, H, W, 1]

        # 数据增强
        if self.aug is not None:
            thw = rearrange(x, "t h w 1 -> t h w")
            thw = self.aug(thw)
            x = rearrange(thw, "t h w -> t h w 1")

        return x.contiguous()


# =========================================
# DataModule：新增 sevirlr_npy 分支
# =========================================

class SEVIRLightningDataModule(LightningDataModule):

    def __init__(self,
                 seq_len: int = 25,
                 sample_mode: str = "sequent",
                 stride: int = 12,
                 layout: str = "NTHWC",
                 output_type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 verbose: bool = False,
                 aug_mode: str = "0",
                 ret_contiguous: bool = True,
                 # datamodule_only
                 dataset_name: str = "sevir",
                 sevir_dir: str = None,
                 start_date: Tuple[int] = None,
                 train_test_split_date: Tuple[int] = (2019, 6, 1),
                 end_date: Tuple[int] = None,
                 val_ratio: float = 0.1,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 seed: int = 0,
                 # NPY only
                 npy_input_len: int = 7,
                 npy_pred_len: int = 6,
                 npy_stride: int = 6,
                 ):
        super(SEVIRLightningDataModule, self).__init__()
        self.seq_len = seq_len
        self.sample_mode = sample_mode
        self.stride = stride
        assert layout[0] == "N"
        self.layout = layout.replace("N", "")
        self.output_type = output_type  # for HDF 路线
        self.preprocess = preprocess
        self.rescale_method = rescale_method
        self.verbose = verbose
        self.aug_mode = aug_mode
        self.ret_contiguous = ret_contiguous
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # for NPY
        self.npy_input_len = npy_input_len
        self.npy_pred_len = npy_pred_len
        self.npy_stride = npy_stride

        if sevir_dir is not None:
            sevir_dir = os.path.abspath(sevir_dir)
        self.dataset_name = dataset_name
        self.sevir_dir = sevir_dir

        if dataset_name == "sevir":
            if sevir_dir is None:
                sevir_dir = default_dataset_sevir_dir
            catalog_path = os.path.join(sevir_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_dir, "data")
            raw_seq_len = 49
            interval_real_time = 5
            img_height = 384
            img_width = 384
        elif dataset_name == "sevirlr":
            if sevir_dir is None:
                sevir_dir = default_dataset_sevirlr_dir
            catalog_path = os.path.join(sevir_dir, "CATALOG.csv")
            raw_data_dir = os.path.join(sevir_dir, "data")
            raw_seq_len = 25
            interval_real_time = 10
            img_height = 128
            img_width = 128
        elif dataset_name == "sevirlr_npy":
            # NPY 路线：只用目录；无 catalog
            catalog_path = None
            raw_data_dir = sevir_dir  # 直接就是 *.npy 所在目录
            raw_seq_len = self.npy_input_len + self.npy_pred_len  # 通常=13
            interval_real_time = 10
            img_height = 128
            img_width = 128
        else:
            raise ValueError(f"Wrong dataset name {dataset_name}. Must be 'sevir', 'sevirlr' or 'sevirlr_npy'.")

        self.catalog_path = catalog_path
        self.raw_data_dir = raw_data_dir
        self.raw_seq_len = raw_seq_len
        self.interval_real_time = interval_real_time
        self.img_height = img_height
        self.img_width = img_width

        # for HDF 路线的按日期切分；NPY 路线不用
        self.start_date = datetime.datetime(*start_date) if start_date is not None else None
        self.train_test_split_date = datetime.datetime(*train_test_split_date) if train_test_split_date is not None else None
        self.end_date = datetime.datetime(*end_date) if end_date is not None else None
        self.val_ratio = val_ratio

    # ---------- 准备数据 ----------
    def prepare_data(self) -> None:
        if self.dataset_name == "sevirlr_npy":
            # 只检查目录和 .npy 是否存在
            assert self.sevir_dir is not None, "For 'sevirlr_npy', please set sevir_dir to your *.npy folder."
            assert os.path.isdir(self.sevir_dir), f"NPY dir not found: {self.sevir_dir}"
            npys = glob.glob(os.path.join(self.sevir_dir, "*.npy"))
            assert len(npys) > 0, f"No .npy files under {self.sevir_dir}"
        else:
            if os.path.exists(self.sevir_dir):
                assert os.path.exists(self.catalog_path), f"CATALOG.csv not found! Should be located at {self.catalog_path}"
                assert os.path.exists(self.raw_data_dir), f"SEVIR data not found! Should be located at {self.raw_data_dir}"
            else:
                if self.dataset_name == "sevir":
                    download_SEVIR(save_dir=os.path.dirname(self.sevir_dir))
                elif self.dataset_name == "sevirlr":
                    download_SEVIRLR(save_dir=os.path.dirname(self.sevir_dir))
                else:
                    raise NotImplementedError

    # ---------- 构建 Dataset ----------
    def setup(self, stage = None) -> None:
        seed_everything(seed=self.seed)
        if self.dataset_name == "sevirlr_npy":
            # NPY 路线
            if stage in (None, "fit"):
                ds = SEVIRNPYDataset(
                    root_dir=self.raw_data_dir,
                    seq_len=self.seq_len,
                    input_len=self.npy_input_len,
                    pred_len=self.npy_pred_len,
                    stride=self.npy_stride,
                    layout=self.layout,
                    preprocess=self.preprocess,
                    rescale_method=self.rescale_method,
                    aug_mode=self.aug_mode,
                    ret_contiguous=self.ret_contiguous,
                )
                n_total = len(ds)
                n_val = max(1, int(self.val_ratio * n_total)) if n_total > 1 else 0
                n_train = max(0, n_total - n_val)
                if n_train == 0:  # 只有一个样本也让 val 能跑
                    self.sevir_train = ds
                    self.sevir_val = ds
                else:
                    self.sevir_train, self.sevir_val = random_split(
                        ds, [n_train, n_val],
                        generator=torch.Generator().manual_seed(self.seed)
                    )
            if stage in (None, "test"):
                self.sevir_test = SEVIRNPYDataset(
                    root_dir=self.raw_data_dir,
                    seq_len=self.seq_len,
                    input_len=self.npy_input_len,
                    pred_len=self.npy_pred_len,
                    stride=self.npy_stride,
                    layout=self.layout,
                    preprocess=self.preprocess,
                    rescale_method=self.rescale_method,
                    aug_mode="0",  # 测试不做数据增强
                    ret_contiguous=self.ret_contiguous,
                )
        else:
            # 原 HDF5 路线
            if stage in (None, "fit"):
                sevir_train_val = SEVIRTorchDataset(
                    sevir_catalog=self.catalog_path,
                    sevir_data_dir=self.raw_data_dir,
                    raw_seq_len=self.raw_seq_len,
                    split_mode="uneven",
                    shuffle=True,
                    seq_len=self.seq_len,
                    stride=self.stride,
                    sample_mode=self.sample_mode,
                    layout=self.layout,
                    start_date=self.start_date,
                    end_date=self.train_test_split_date,
                    output_type=self.output_type,
                    preprocess=self.preprocess,
                    rescale_method=self.rescale_method,
                    verbose=self.verbose,
                    aug_mode=self.aug_mode,
                    ret_contiguous=self.ret_contiguous,)
                n_total = len(sevir_train_val)
                n_val = max(1, int(self.val_ratio * n_total)) if n_total > 1 else 0
                n_train = max(0, n_total - n_val)
                if n_train == 0:
                    self.sevir_train = sevir_train_val
                    self.sevir_val = sevir_train_val
                else:
                    self.sevir_train, self.sevir_val = random_split(
                        dataset=sevir_train_val,
                        lengths=[n_train, n_val],
                        generator=torch.Generator().manual_seed(self.seed))
            if stage in (None, "test"):
                self.sevir_test = SEVIRTorchDataset(
                    sevir_catalog=self.catalog_path,
                    sevir_data_dir=self.raw_data_dir,
                    raw_seq_len=self.raw_seq_len,
                    split_mode="uneven",
                    shuffle=False,
                    seq_len=self.seq_len,
                    stride=self.stride,
                    sample_mode=self.sample_mode,
                    layout=self.layout,
                    start_date=self.train_test_split_date,
                    end_date=self.end_date,
                    output_type=self.output_type,
                    preprocess=self.preprocess,
                    rescale_method=self.rescale_method,
                    verbose=self.verbose,
                    aug_mode="0",
                    ret_contiguous=self.ret_contiguous,)

    # ---------- DataLoader ----------
    def train_dataloader(self):
        return DataLoader(self.sevir_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.sevir_val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.sevir_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    @property
    def num_train_samples(self):
        return len(self.sevir_train)

    @property
    def num_val_samples(self):
        return len(self.sevir_val)

    @property
    def num_test_samples(self):
        return len(self.sevir_test)