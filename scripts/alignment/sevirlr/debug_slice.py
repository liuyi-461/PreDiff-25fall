import sys
sys.path.append('/data/25fall_nowcasting/lsy/PreDiff-25fall')

from train_sevirlr_avg_x import SEVIRAlignmentPLModule

print("=== 检查配置一致性 ===")

# 创建实例来获取配置
pl_module = SEVIRAlignmentPLModule(
    total_num_steps=100,
    save_dir="/tmp/config_check"
)

print("当前配置:")
print(f"  dataset.seq_len: {pl_module.oc.dataset.seq_len}")
print(f"  dataset.in_len: {pl_module.oc.dataset.in_len}") 
print(f"  dataset.out_len: {pl_module.oc.dataset.out_len}")
print(f"  layout.in_len: {pl_module.oc.layout.in_len}")
print(f"  layout.out_len: {pl_module.oc.layout.out_len}")

# 检查是否一致
if pl_module.oc.layout.in_len + pl_module.oc.layout.out_len == pl_module.oc.dataset.seq_len:
    print("✅ 配置一致!")
else:
    print(f"❌ 配置不一致! {pl_module.oc.layout.in_len} + {pl_module.oc.layout.out_len} != {pl_module.oc.dataset.seq_len}")