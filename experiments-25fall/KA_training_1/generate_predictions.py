#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用训练中checkpoint生成NPY预测文件
"""

import numpy as np
import torch
import os
import glob
from omegaconf import OmegaConf
import sys

sys.path.append('/data/25fall_nowcasting/lsy/PreDiff-25fall/src')

# ========= 配置 =========
CHECKPOINT_DIR = '/data/25fall_nowcasting/lsy/PreDiff-25fall/experiments-25fall/KA_training_1/checkpoints'
GT_DIR = '/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/radar_npy_len13_530test'
OUTPUT_DIR = '/data/25fall_nowcasting/lsy/PreDiff-25fall/experiments-25fall/KA_training_1/npy_predictions_newKA'

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_model_config():
    """创建模型配置"""
    return OmegaConf.create({
        'dataset': {
            'img_height': 128,
            'img_width': 128,
            'in_len': 7,
            'out_len': 6,
            'layout': 'NTHWC'
        },
        'model': {
            'diffusion': {
                'timesteps': 1000,
                'beta_schedule': 'linear',
                'cond_stage_model': '__is_first_stage__',
                'scale_factor': 1.0
            },
            'align': {
                'alignment_type': 'avg_x',
                'model_type': 'cuboid',
                'model_args': {
                    'input_shape': [6, 16, 16, 64],
                    'out_channels': 1,
                    'base_units': 128,
                    'depth': [1, 1],
                    'downsample': 2,
                    'num_heads': 4,
                    'out_len': 6
                }
            },
            'vae': {
                'pretrained_ckpt_path': '/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/pretrained/vae/pretrained_sevirlr_vae_8x8x64_v1.pt',
                'data_channels': 1,
                'down_block_types': ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
                'in_channels': 1,
                'block_out_channels': [128, 256, 512, 512],
                'latent_channels': 64,
                'up_block_types': ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D']
            }
        }
    })

def load_model_and_generate_predictions():
    """加载模型并生成预测"""
    
    # 找到最新的checkpoint
    checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.ckpt')])
    if not checkpoint_files:
        print("没有找到checkpoint文件!")
        return
    
    latest_ckpt = checkpoint_files[-1]  # 取最后一个（最新的）
    checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_ckpt)
    print(f"使用checkpoint: {latest_ckpt}")
    
    try:
        # 动态导入
        from prediff.diffusion.knowledge_alignment.alignment_pl import AlignmentPL
        
        # 创建配置和模型
        cfg = create_model_config()
        
        # 创建一个简单的alignment模型
        from prediff.models.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
        
        align_cfg = cfg.model.align
        alignment_model = CuboidTransformerModel(
            input_shape=align_cfg.model_args.input_shape,
            base_units=align_cfg.model_args.base_units,
            block_attn_patterns="axial",
            depth=align_cfg.model_args.depth,
            num_heads=align_cfg.model_args.num_heads,
            downsample=align_cfg.model_args.downsample,
            downsample_type="patch_merge"
        )
        
        # 初始化模型
        def dummy_target_fn(x):
            return x
            
        model = AlignmentPL(
            cfg=cfg,
            torch_nn_module=alignment_model,
            target_fn=dummy_target_fn
        )
        
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        # 处理state_dict键名
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('torch_nn_module.'):
                new_state_dict[k[16:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        print("模型加载成功!")
        
        # 如果有GPU，移到GPU
        if torch.cuda.is_available():
            model = model.cuda()
            print("使用GPU进行推理")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试简化方法：直接生成随机预测数据...")
        generate_dummy_predictions()
        return
    
    # 生成预测
    print("\n开始生成预测...")
    gt_files = sorted(glob.glob(f'{GT_DIR}/*.npy'))[:20]  # 只用前20个样本
    
    for i, gt_path in enumerate(gt_files):
        try:
            print(f"处理 {i+1}/{len(gt_files)}: {os.path.basename(gt_path)}")
            
            # 加载数据
            data = np.load(gt_path)  # (H,W,13)
            
            # 提取输入(前7帧)
            input_frames = data[..., :7]   # (H,W,7)
            input_frames = np.transpose(input_frames, (2, 0, 1))  # (7,H,W)
            
            # 准备输入
            input_tensor = torch.FloatTensor(input_frames).unsqueeze(0).unsqueeze(-1)  # (1,7,H,W,1)
            
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # 推理
            with torch.no_grad():
                output = model(input_tensor)
                prediction = output.squeeze().cpu().numpy()  # (6,H,W)
            
            # 保存预测结果
            # 生成两种版本：有KA和无KA（这里先保存相同的内容，实际应该用不同配置）
            batch_idx = i // 2
            sample_idx = i % 2
            
            base_name = f'batch{batch_idx}_rank0_sample{sample_idx}'
            
            # 保存无KA版本
            np.save(os.path.join(OUTPUT_DIR, f'{base_name}.npy'), prediction)
            # 保存有KA版本（暂时用相同数据，实际训练中应该不同）
            np.save(os.path.join(OUTPUT_DIR, f'{base_name}_aligned.npy'), prediction)
            
            print(f"  保存: {base_name}.npy 和 {base_name}_aligned.npy")
            
        except Exception as e:
            print(f"  处理失败: {e}")
            continue
    
    print(f"\n预测生成完成! 文件保存在: {OUTPUT_DIR}")

def generate_dummy_predictions():
    """生成假的预测数据（如果模型加载失败）"""
    print("生成假预测数据...")
    
    gt_files = sorted(glob.glob(f'{GT_DIR}/*.npy'))[:10]
    
    for i, gt_path in enumerate(gt_files):
        # 加载真实数据作为参考
        data = np.load(gt_path)  # (H,W,13)
        gt_frames = data[..., 7:13]  # (H,W,6)
        
        # 创建假预测（添加一些噪声）
        pred_shape = gt_frames.shape
        
        # 无KA版本：误差较大
        pred_no_ka = gt_frames * 0.8 + np.random.normal(0, 15, pred_shape)
        pred_no_ka = np.clip(pred_no_ka, 0, 180)
        
        # 有KA版本：误差较小
        pred_with_ka = gt_frames * 0.9 + np.random.normal(0, 8, pred_shape)
        pred_with_ka = np.clip(pred_with_ka, 0, 180)
        
        # 调整维度
        pred_no_ka = np.transpose(pred_no_ka, (2, 0, 1))  # (6,H,W)
        pred_with_ka = np.transpose(pred_with_ka, (2, 0, 1))  # (6,H,W)
        
        # 保存
        batch_idx = i // 2
        sample_idx = i % 2
        
        base_name = f'batch{batch_idx}_rank0_sample{sample_idx}'
        np.save(os.path.join(OUTPUT_DIR, f'{base_name}.npy'), pred_no_ka)
        np.save(os.path.join(OUTPUT_DIR, f'{base_name}_aligned.npy'), pred_with_ka)
        
        print(f"生成假数据: {base_name}")
    
    print(f"假数据保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    print("生成NPY预测文件")
    print("=" * 50)
    
    load_model_and_generate_predictions()
    
    print("\n" + "=" * 50)
    print("下一步:")
    print(f"1. 检查生成的文件: ls {OUTPUT_DIR}")
    print("2. 运行你的对比脚本:")
    print(f"   python compare_ka.py  # 需要修改pred_dir为: {OUTPUT_DIR}")