#!/usr/bin/env python3
import torch
import os

def inspect_checkpoint(ckpt_path):
    print(f"\n=== 详细检查 {os.path.basename(ckpt_path)} ===")
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        print("Checkpoint中的所有键:")
        for key in checkpoint.keys():
            print(f"  - {key}: {type(checkpoint[key])}")
            
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"\nState_dict中的键数量: {len(state_dict)}")
            print("前15个键:")
            for i, key in enumerate(list(state_dict.keys())[:15]):
                print(f"  {i+1:2d}. {key} - {state_dict[key].shape}")
            
            # 特别关注alignment相关的键
            print("\nAlignment相关键:")
            align_keys = [key for key in state_dict.keys() if 'align' in key.lower()]
            for key in align_keys:
                print(f"  - {key}: {state_dict[key].shape}")
                
        # 检查其他重要信息
        for key in ['hyper_parameters', 'cfg', 'config']:
            if key in checkpoint:
                print(f"\n{key}:")
                if isinstance(checkpoint[key], dict):
                    for k, v in list(checkpoint[key].items())[:5]:
                        print(f"  - {k}: {v}")
                
    except Exception as e:
        print(f"读取失败: {e}")

# 检查checkpoint
checkpoint_dir = '/data/25fall_nowcasting/lsy/PreDiff-25fall/experiments-25fall/KA_training_1/checkpoints'
checkpoint_to_check = 'last.ckpt'

ckpt_path = os.path.join(checkpoint_dir, checkpoint_to_check)
if os.path.exists(ckpt_path):
    inspect_checkpoint(ckpt_path)
else:
    print(f"文件不存在: {ckpt_path}")
    # 检查其他可用的
    for f in os.listdir(checkpoint_dir):
        if f.endswith('.ckpt'):
            ckpt_path = os.path.join(checkpoint_dir, f)
            print(f"尝试检查: {f}")
            inspect_checkpoint(ckpt_path)
            break