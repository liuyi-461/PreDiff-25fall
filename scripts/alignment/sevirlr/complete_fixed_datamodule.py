import sys
sys.path.append('/data/25fall_nowcasting/lsy/PreDiff-25fall')

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from omegaconf import OmegaConf

class CompleteFixedSEVIRDataModule(LightningDataModule):
    """完全替换的修复数据模块"""
    
    def __init__(self, dataset_cfg, micro_batch_size=1, target_length=13):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.micro_batch_size = micro_batch_size
        self.target_length = target_length
        
        from prediff.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule
        
        # 修正数据集名称
        dataset_name = dataset_cfg["dataset_name"]
        #if dataset_name == "sevir_lr":
         #   dataset_name = "sevirlr"  # 修正为正确的名称
        
        # 创建原始数据模块
        self.original_dm = SEVIRLightningDataModule(
            seq_len=dataset_cfg["seq_len"],
            sample_mode=dataset_cfg["sample_mode"],
            stride=1,  # 使用安全的步长
            batch_size=micro_batch_size,
            layout=dataset_cfg["layout"],
            output_type=np.float32,
            preprocess=True,
            rescale_method="01",
            verbose=False,
            aug_mode=dataset_cfg["aug_mode"],
            ret_contiguous=False,
            dataset_name=dataset_name,  # 使用修正后的名称
            start_date=dataset_cfg["start_date"],
            train_test_split_date=dataset_cfg["train_test_split_date"],
            end_date=dataset_cfg["end_date"],
            val_ratio=dataset_cfg["val_ratio"],
            num_workers=8,
            npy_dir="/home/user01/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/radar_npy_len13"
        )
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        self.original_dm.prepare_data()
    
    def setup(self, stage=None):
        self.original_dm.setup(stage)
        
        # 创建过滤后的数据集
        self.train_dataset = self._create_filtered_dataset(self.original_dm.sevir_train)
        self.val_dataset = self._create_filtered_dataset(self.original_dm.sevir_val)
        self.test_dataset = self._create_filtered_dataset(self.original_dm.sevir_test)
        
        print(f"过滤后样本数 - 训练: {len(self.train_dataset)}, 验证: {len(self.val_dataset)}, 测试: {len(self.test_dataset)}")
    
    def _create_filtered_dataset(self, original_dataset):
        valid_indices = []
        for i in range(len(original_dataset)):
            try:
                sample = original_dataset[i]
                if sample.shape[0] == self.target_length:
                    valid_indices.append(i)
            except:
                continue
        
        from torch.utils.data import Subset
        return Subset(original_dataset, valid_indices)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.micro_batch_size, shuffle=False, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.micro_batch_size, shuffle=False, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.micro_batch_size, shuffle=False, num_workers=0)
    
    @property
    def num_train_samples(self):
        return len(self.train_dataset) if self.train_dataset else 0
    
    @property
    def num_val_samples(self):
        return len(self.val_dataset) if self.val_dataset else 0
    
    @property
    def num_test_samples(self):
        return len(self.test_dataset) if self.test_dataset else 0

# 测试函数
def test_complete_module():
    from train_sevirlr_avg_x import SEVIRAlignmentPLModule
    
    dataset_cfg = OmegaConf.to_object(SEVIRAlignmentPLModule.get_dataset_config())
    
    # 修正数据集名称
    #if dataset_cfg["dataset_name"] == "sevir_lr":
       # dataset_cfg["dataset_name"] = "sevirlr"
    
    dm = CompleteFixedSEVIRDataModule(
        dataset_cfg=dataset_cfg,
        micro_batch_size=2,
        target_length=13
    )
    
    dm.prepare_data()
    dm.setup()
    
    print(f"训练样本: {dm.num_train_samples}")
    print(f"验证样本: {dm.num_val_samples}")
    print(f"测试样本: {dm.num_test_samples}")
    
    # 测试数据加载
    print("测试训练数据加载...")
    train_loader = dm.train_dataloader()
    for i, batch in enumerate(train_loader):
        print(f"批次 {i}: {batch.shape}")
        if i >= 2:
            break
    
    print("测试验证数据加载...")
    val_loader = dm.val_dataloader()
    for i, batch in enumerate(val_loader):
        print(f"批次 {i}: {batch.shape}")
        if i >= 2:
            break

if __name__ == "__main__":
    test_complete_module()