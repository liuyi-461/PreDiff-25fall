#!/usr/bin/env python3
"""
专门用于助教数据的PreDiff推理脚本
使用/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/radar_npy_len13数据
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from omegaconf import OmegaConf
from collections import OrderedDict
from pathlib import Path

# 导入PreDiff相关模块
from prediff.taming import AutoencoderKL
from prediff.models.cuboid_transformer import CuboidTransformerUNet
from prediff.diffusion.latent_diffusion import LatentDiffusion
from prediff.utils.pl_checkpoint import pl_load
from prediff.utils.path import (
    default_pretrained_vae_dir,
    default_pretrained_earthformerunet_dir,
    default_pretrained_alignment_dir,
)
from prediff.utils.download import (
    download_pretrained_weights,
    pretrained_sevirlr_vae_name,
    pretrained_sevirlr_earthformerunet_name,
    pretrained_sevirlr_alignment_name,
)

class RadarCAPPIDataset(Dataset):
    """专门处理助教雷达CAPPI数据的Dataset"""
    
    def __init__(self, data_dir, seq_len=13, input_len=7, target_len=6):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.input_len = input_len
        self.target_len = target_len
        
        # 获取所有npy文件
        self.file_list = sorted([
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.npy')
        ])
        
        print(f"找到 {len(self.file_list)} 个数据文件")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 加载npy文件
        file_path = self.file_list[idx]
        data = np.load(file_path)  # 形状应该是 (13, 128, 128) 或 (128, 128, 13)
        
        # 自动检测数据格式并调整到 (T, H, W)
        if data.shape[0] == 13:  # (13, 128, 128) - THW格式
            data = data
        elif data.shape[2] == 13:  # (128, 128, 13) - HWT格式
            data = np.transpose(data, (2, 0, 1))  # 转为THW
        else:
            raise ValueError(f"无法识别的数据形状: {data.shape}")
        
        # 确保序列长度正确
        assert data.shape[0] == self.seq_len, f"序列长度应为{self.seq_len}, 但得到{data.shape[0]}"
        
        # 切分输入和目标
        input_frames = data[:self.input_len]  # 前7帧作为输入
        target_frames = data[self.input_len:] # 后6帧作为目标
        
        # 关键：使用助教说的归一化方法 (除以180而不是255)
        #input_frames = input_frames / 180.0
        #target_frames = target_frames / 180.0

        #重新除以255
        input_frames = input_frames / 255.0
        target_frames = target_frames / 255.0
        
        # 转换为torch tensor并增加通道维度
        input_tensor = torch.FloatTensor(input_frames).unsqueeze(-1)  # (7, 128, 128, 1)
        target_tensor = torch.FloatTensor(target_frames).unsqueeze(-1) # (6, 128, 128, 1)
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'file_name': os.path.basename(file_path)
        }

def create_prediff_model():
    """创建PreDiff模型并加载预训练权重"""
    
    # 下载预训练权重（如果不存在）
    print("下载预训练权重...")
    download_pretrained_weights(
        ckpt_name=pretrained_sevirlr_vae_name,
        save_dir=default_pretrained_vae_dir,
        exist_ok=True
    )
    download_pretrained_weights(
        ckpt_name=pretrained_sevirlr_earthformerunet_name,
        save_dir=default_pretrained_earthformerunet_dir,
        exist_ok=True
    )
    
    # 加载VAE
    print("加载VAE...")
    vae_ckpt_path = os.path.join(default_pretrained_vae_dir, pretrained_sevirlr_vae_name)
    vae_state_dict = torch.load(vae_ckpt_path, map_location="cpu")
    
    vae = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        in_channels=1,  # 单通道雷达数据
        block_out_channels=[128, 256, 512, 512],
        act_fn='silu',
        latent_channels=4,
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        norm_num_groups=32,
        layers_per_block=2,
        out_channels=1,
    )
    vae.load_state_dict(vae_state_dict)
    vae.eval()
    
    # 加载Earthformer-UNet
    print("加载Earthformer-UNet...")
    earthformer_ckpt_path = os.path.join(default_pretrained_earthformerunet_dir, pretrained_sevirlr_earthformerunet_name)
    earthformer_state_dict = torch.load(earthformer_ckpt_path, map_location="cpu")
    
    # 创建Earthformer-UNet配置（基于官方代码）
    latent_model = CuboidTransformerUNet(
        input_shape=[10, 16, 16, 4],
        target_shape=[10, 16, 16, 4],
        base_units=4,
        scale_alpha=1.0,
        num_heads=4,
        attn_drop=0.1,
        proj_drop=0.1,
        ffn_drop=0.1,
        downsample=2,
        downsample_type="patch_merge",
        upsample_type="upsample",
        upsample_kernel_size=3,
        depth=[1, 1],
        block_attn_patterns="axial",
        num_global_vectors=0,
        use_global_vector_ffn=False,
        use_global_self_attn=True,
        separate_global_qkv=True,
        global_dim_ratio=1,
        ffn_activation="gelu",
        gated_ffn=False,
        norm_layer="layer_norm",
        padding_type="zeros",
        pos_embed_type="t+h+w",
        checkpoint_level=0,
        use_relative_pos=True,
        self_attn_use_final_proj=True,
        time_embed_channels_mult=4,
        time_embed_use_scale_shift_norm=False,
        time_embed_dropout=0.0,
        unet_res_connect=True,
    )
    latent_model.load_state_dict(earthformer_state_dict)
    latent_model.eval()
    
    # 创建LatentDiffusion模型
    print("创建LatentDiffusion模型...")
    model = LatentDiffusion(
        torch_nn_module=latent_model,
        layout="NTHWC",
        data_shape=(6, 128, 128, 1),  # 输出6帧
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        monitor="valid_loss_epoch",
        use_ema=True,
        log_every_t=100,
        clip_denoised=False,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        original_elbo_weight=0.0,
        v_posterior=0.0,
        l_simple_weight=1.0,
        parameterization="eps",
        # latent diffusion
        latent_shape=[6, 16, 16, 4],
        first_stage_model=vae,
        cond_stage_model="__is_first_stage__",
        num_timesteps_cond=None,
        cond_stage_trainable=False,
        cond_stage_forward=None,
        scale_by_std=False,
        scale_factor=1.0,
    )
    
    return model

def main():
    parser = argparse.ArgumentParser(description='PreDiff推理 - 助教雷达数据')
    parser.add_argument('--data_dir', 
                       default='/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/radar_npy_len13',
                       type=str, help='数据目录路径')
    parser.add_argument('--output_dir', default='./infer_results', type=str, help='输出目录')
    parser.add_argument('--num_samples', default=10, type=int, help='推理的样本数量')
    parser.add_argument('--batch_size', default=1, type=int, help='批大小')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")
    
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    
    # 创建数据集和数据加载器
    print("创建数据加载器...")
    dataset = RadarCAPPIDataset(args.data_dir)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # 创建模型
    model = create_prediff_model()
    model.eval()
    
    print("开始推理...")
    
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= args.num_samples:  # 限制推理样本数量
                break
                
            input_frames = batch['input']  # (batch, 7, 128, 128, 1)
            target_frames = batch['target']  # (batch, 6, 128, 128, 1)
            file_names = batch['file_name']
            
            print(f"处理样本 {i+1}/{min(args.num_samples, len(dataset))}: {file_names[0]}")
            
            # 使用前7帧作为条件，生成后6帧
            cond = {"y": input_frames}
            
            # 进行推理
            predicted_frames = model.sample(
                cond=cond,
                batch_size=args.batch_size,
                return_intermediates=False,
                verbose=False,
            )
            
            # 保存结果
            for j in range(args.batch_size):
                result = {
                    'file_name': file_names[j],
                    'input': input_frames[j].cpu().numpy(),      # 输入的前7帧
                    'target': target_frames[j].cpu().numpy(),    # 真实的后续6帧
                    'predicted': predicted_frames[j].cpu().numpy(),  # 预测的后续6帧
                }
                results.append(result)
                
                # 保存为npy文件
                output_path = os.path.join(args.output_dir, 'predictions', f'pred_{file_names[j]}')
                np.save(output_path, {
                    'input': result['input'],
                    'target': result['target'], 
                    'predicted': result['predicted']
                })
                
                print(f"  保存预测结果: {output_path}")
    
    # 生成推理报告
    print("\n" + "="*50)
    print("推理完成!")
    print(f"总共处理了 {len(results)} 个样本")
    print(f"结果保存在: {args.output_dir}")
    print("="*50)
    
    # 计算基本统计信息
    if len(results) > 0:
        mse_values = []
        for result in results:
            mse = np.mean((result['target'] - result['predicted'])**2)
            mse_values.append(mse)
        
        print(f"平均MSE: {np.mean(mse_values):.6f}")
        print(f"MSE标准差: {np.std(mse_values):.6f}")
        print(f"最小MSE: {np.min(mse_values):.6f}")
        print(f"最大MSE: {np.max(mse_values):.6f}")

if __name__ == "__main__":
    main()
