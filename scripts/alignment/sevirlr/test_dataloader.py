import sys
sys.path.append('/data/25fall_nowcasting/lsy/PreDiff-25fall')

from prediff.datasets.sevir.sevir_torch_wrap import SEVIRLightningDataModule
import numpy as np

def test_dataloader():
    print("=== 测试数据加载 ===")
    
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
        batch_size=1,  # 先用最小的批次
        num_workers=0,  # 禁用多进程
        npy_dir="/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/radar_npy_len13"
    )
    
    dm.prepare_data()
    dm.setup()
    
    print("数据模块设置完成")
    print(f"训练样本数: {dm.num_train_samples}")
    print(f"验证样本数: {dm.num_val_samples}")
    print(f"测试样本数: {dm.num_test_samples}")
    
    # 测试加载一个批次
    try:
        train_loader = dm.train_dataloader()
        print("训练数据加载器创建成功")
        
        batch = next(iter(train_loader))
        print(f"批次加载成功，数据类型: {type(batch)}")
        if hasattr(batch, 'shape'):
            print(f"批次形状: {batch.shape}")
        else:
            print(f"批次内容: {batch}")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataloader()
