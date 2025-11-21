import sys
sys.path.append('/data/25fall_nowcasting/lsy/PreDiff-25fall')

# 直接测试 NPYSEVIRDataLoader
from prediff.datasets.sevir.npy_sevir_dataloader import NPYSEVIRDataLoader
import glob

print("=== 测试原始数据加载器 ===")

npy_dir = "/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/radar_npy_len13"
npy_files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))[:10]  # 只测试前10个文件

print(f"测试文件数量: {len(npy_files)}")

loader = NPYSEVIRDataLoader(
    file_list=npy_files,
    seq_len=13,
    raw_seq_len=128,  # 原始长度
    stride=6,
    batch_size=1,
    layout='NHWT',
    output_type=np.float32,
    preprocess=True,
    rescale_method="01"
)

print("数据加载器创建成功")

# 测试加载几个批次
for i in range(5):
    try:
        batch = loader[i]  # 直接索引获取
        print(f"批次 {i}: 形状 {batch.shape}")
    except Exception as e:
        print(f"批次 {i} 失败: {e}")
        break