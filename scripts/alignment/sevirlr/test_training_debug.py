import sys
sys.path.append('/data/25fall_nowcasting/lsy/PreDiff-25fall')

import torch
from prediff.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule
from train_sevirlr_avg_x import SEVIRAlignmentPLModule
import numpy as np
import os

def debug_training():
    print("=== 调试训练流程 ===")
    
    # 1. 创建数据模块
    dm = SEVIRLightningDataModule(
        seq_len=13,
        sample_mode="sequent",
        stride=6,
        layout="NTHWC",
        output_type=np.float32,
        preprocess=True,
        rescale_method="01",
        verbose=True,
        aug_mode="1",
        ret_contiguous=False,
        dataset_name="sevirlr",
        start_date=None,
        train_test_split_date=None,
        end_date=None,
        val_ratio=0.1,
        batch_size=1,
        num_workers=0,
        npy_dir="/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/radar_npy_len13"
    )
    
    dm.prepare_data()
    dm.setup()
    
    print(f"数据形状检查:")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"原始批次形状: {batch.shape}")
    
    # 2. 测试模型
    print("\n=== 测试模型数据处理 ===")
    
    total_num_steps = 100
    pl_module = SEVIRAlignmentPLModule(
        total_num_steps=total_num_steps,
        save_dir="/tmp/debug_test",
        oc_file=None
    )
    
    try:
        out_seq, cond_dict, verbose_dict = pl_module._get_input_sevirlr(batch, return_verbose=True)
        print(f"处理后输出序列形状: {out_seq.shape}")
        print("数据处理成功！")
        
    except Exception as e:
        print(f"模型处理错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training()
