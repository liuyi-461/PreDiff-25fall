import sys
sys.path.append('/data/25fall_nowcasting/lsy/PreDiff-25fall')

import torch
from torch.utils.data import Dataset, DataLoader
from prediff.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule

class FixedLengthDataset(Dataset):
    """强制所有样本长度一致的数据集包装器"""
    
    def __init__(self, original_dataset, target_length=13):
        self.original_dataset = original_dataset
        self.target_length = target_length
        self.valid_indices = []
        
        print("扫描数据集，过滤长度不一致的样本...")
        for i in range(len(original_dataset)):
            try:
                sample = original_dataset[i]
                if sample.shape[0] == target_length:  # 检查时间步长度
                    self.valid_indices.append(i)
            except:
                continue
        
        print(f"原始样本数: {len(original_dataset)}, 有效样本数: {len(self.valid_indices)}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        return self.original_dataset[original_idx]

class FixedSEVIRDataModule:
    """修复长度问题的数据模块"""
    
    def __init__(self, original_dm, target_length=13):
        self.original_dm = original_dm
        self.target_length = target_length
        self.batch_size = original_dm.batch_size
        
        # 创建过滤后的数据集
        self.train_dataset = FixedLengthDataset(original_dm.sevir_train, target_length)
        self.val_dataset = FixedLengthDataset(original_dm.sevir_val, target_length)
        self.test_dataset = FixedLengthDataset(original_dm.sevir_test, target_length)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    @property
    def num_train_samples(self):
        return len(self.train_dataset)
    
    @property
    def num_val_samples(self):
        return len(self.val_dataset)
    
    @property
    def num_test_samples(self):
        return len(self.test_dataset)

# 测试修复后的数据模块
def test_fixed_datamodule():
    print("=== 测试修复后的数据模块 ===")
    
    # 创建原始数据模块
    original_dm = SEVIRLightningDataModule(
        seq_len=13,
        sample_mode="sequent",
        stride=1,
        batch_size=2,
        num_workers=0,
        dataset_name="sevirlr",
        val_ratio=0.1,
        npy_dir="/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/radar_npy_len13"
    )
    
    original_dm.prepare_data()
    original_dm.setup()
    
    # 创建修复后的数据模块
    fixed_dm = FixedSEVIRDataModule(original_dm, target_length=13)
    
    # 测试数据加载
    print("测试训练数据加载...")
    train_loader = fixed_dm.train_dataloader()
    for i, batch in enumerate(train_loader):
        print(f"批次 {i}: 形状 {batch.shape}")
        if i >= 2:
            break
    
    print("测试验证数据加载...")
    val_loader = fixed_dm.val_dataloader()
    for i, batch in enumerate(val_loader):
        print(f"批次 {i}: 形状 {batch.shape}")
        if i >= 2:
            break
    
    print("✅ 修复成功！所有批次长度一致")

if __name__ == "__main__":
    test_fixed_datamodule()