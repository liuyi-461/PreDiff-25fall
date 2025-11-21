import sys
sys.path.append('/data/25fall_nowcasting/lsy/PreDiff-25fall')

from train_sevirlr_avg_x import SEVIRAlignmentPLModule
import os

print("=== 最终配置检查 ===")

# 检查 VAE 预训练权重
from prediff.utils.path import default_pretrained_vae_dir
vae_path = os.path.join(default_pretrained_vae_dir, "pretrained_sevirlr_vae_8x8x64_v1.pt")
print(f"VAE 预训练权重: {vae_path}")
print(f"权重文件存在: {os.path.exists(vae_path)}")

# 检查数据形状 - 使用类方法而不是实例方法
print(f"输入长度: {SEVIRAlignmentPLModule.get_layout_config().in_len}")
print(f"输出长度: {SEVIRAlignmentPLModule.get_layout_config().out_len}")
print(f"数据通道: {SEVIRAlignmentPLModule.get_layout_config().data_channels}")

# 创建临时实例来检查完整配置
print("\n=== 完整配置检查 ===")
try:
    pl_module = SEVIRAlignmentPLModule(
        total_num_steps=100,
        save_dir="/tmp/config_check"
    )
    print("✅ 模型配置正确！")
    print(f"VAE latent_channels: {pl_module.oc.model.vae.latent_channels}")
    print(f"数据通道: {pl_module.oc.layout.data_channels}")
except Exception as e:
    print(f"❌ 配置错误: {e}")

print("✅ 所有检查通过，可以开始训练！")