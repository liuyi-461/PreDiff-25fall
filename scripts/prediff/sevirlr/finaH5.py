#!/usr/bin/env python3
"""
SEVIR VIL 雷达数据可视化工具
绘制每一帧雷达图像
"""

import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# VIL颜色映射和阈值 (根据参考代码)
VIL_LEVELS = np.array([0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 255])
VIL_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219],
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219],
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
]) / 255


def get_vil_cmap():
    """创建VIL的colormap"""
    # 使用标准的天气雷达配色方案
    colors = [
        '#000000',  # 0: 黑色 - 无回波
        '#04e9e7',  # 1: 青色 - 非常弱
        '#0300f4',  # 3: 蓝色 - 中等偏弱
        '#02fd02',  # 4: 绿色 - 中等
        '#008e00',  # 6: 暗绿 - 强
        '#fdf802',  # 7: 黄色 - 很强
        '#e5bc00',  # 8: 橙黄 - 非常强
        '#fd9500',  # 9: 橙色 - 极强
        '#fd0000',  # 10: 红色 - 强烈
        '#bc0000',  # 12: 暗红 - 极度强烈
        '#f800fd',  # 13: 品红 - 超强
    ]
    
    levels = [0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 255]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, len(colors))
    
    return cmap, norm


def plot_vil_frame(data, frame_idx, time_minutes, output_path, fs=12):
    """
    绘制单帧VIL数据
    
    Parameters:
    -----------
    data : np.ndarray
        单帧数据, shape=(384, 384)
    frame_idx : int
        帧序号
    time_minutes : int
        时间(分钟)
    output_path : str
        输出文件路径
    fs : int
        字体大小
    """
    fontproperties = FontProperties()
    fontproperties.set_family('serif')
    fontproperties.set_size(fs)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 获取colormap
    cmap, norm = get_vil_cmap()
    
    # 绘制数据
    im = ax.imshow(data, cmap=cmap, norm=norm, interpolation='nearest')
    
    # 设置标题
    ax.set_title(f'VIL Frame {frame_idx:02d} | Time: {time_minutes} min', 
                 fontproperties=fontproperties, pad=15)
    
    # 移除坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('VIL (kg/m²)', fontproperties=fontproperties)
    
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ 已保存: {output_path}")


def plot_vil_data(h5_file_path, output_dir, sample_idx=0, num_frames=49):
    """
    从H5文件读取并绘制VIL数据
    
    Parameters:
    -----------
    h5_file_path : str
        H5文件路径
    output_dir : str
        输出目录
    sample_idx : int
        要绘制的样本索引 (默认第一个样本)
    num_frames : int
        要绘制的帧数
    """
    print("="*80)
    print(f"开始处理VIL数据")
    print(f"文件: {h5_file_path}")
    print(f"样本索引: {sample_idx}")
    print(f"帧数: {num_frames}")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取H5文件
    with h5py.File(h5_file_path, 'r') as f:
        # 获取数据
        vil_data = f['vil'][sample_idx]  # shape: (384, 384, 49)
        sample_id = f['id'][sample_idx]
        
        print(f"\n样本ID: {sample_id}")
        print(f"数据形状: {vil_data.shape}")
        print(f"数据类型: {vil_data.dtype}")
        print(f"数值范围: [{np.min(vil_data)}, {np.max(vil_data)}]")
        print("\n开始绘制帧...\n")
        
        # 绘制每一帧
        for frame_idx in range(min(num_frames, vil_data.shape[2])):
            # 提取单帧数据
            frame_data = vil_data[:, :, frame_idx]
            
            # 计算时间 (每5分钟一帧)
            time_minutes = frame_idx * 5
            
            # 输出文件名
            output_filename = f"vil_sample{sample_idx:03d}_frame{frame_idx:02d}_t{time_minutes:03d}min.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 绘制并保存
            plot_vil_frame(frame_data, frame_idx, time_minutes, output_path)
        
        print("\n" + "="*80)
        print(f"✓ 完成! 共绘制 {min(num_frames, vil_data.shape[2])} 帧")
        print(f"✓ 输出目录: {output_dir}")
        print("="*80)


def plot_vil_grid(h5_file_path, output_path, sample_idx=0, 
                  max_frames=49, ncols=7, frame_stride=1):
    """
    绘制网格形式的多帧VIL数据 (类似参考代码的格式)
    
    Parameters:
    -----------
    h5_file_path : str
        H5文件路径
    output_path : str
        输出文件路径
    sample_idx : int
        样本索引
    max_frames : int
        最大帧数
    ncols : int
        列数
    frame_stride : int
        帧间隔
    """
    print(f"\n创建网格图...")
    
    with h5py.File(h5_file_path, 'r') as f:
        vil_data = f['vil'][sample_idx]  # shape: (384, 384, 49)
        sample_id = f['id'][sample_idx]
        
        # 选择要显示的帧
        frames = vil_data[:, :, ::frame_stride]
        num_frames = min(frames.shape[2], max_frames)
        frames = frames[:, :, :num_frames]
        
        # 计算网格布局
        nrows = int(np.ceil(num_frames / ncols))
        
        # 获取colormap
        cmap, norm = get_vil_cmap()
        
        # 创建图形
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                                 figsize=(3*ncols, 3*nrows))
        
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        fontproperties = FontProperties()
        fontproperties.set_family('serif')
        fontproperties.set_size(10)
        
        for idx in range(num_frames):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            
            frame_data = frames[:, :, idx]
            time_minutes = idx * 5 * frame_stride
            
            # 绘制
            ax.imshow(frame_data, cmap=cmap, norm=norm, interpolation='nearest')
            ax.set_title(f'{time_minutes} min', fontproperties=fontproperties)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 隐藏多余的子图
        for idx in range(num_frames, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')
        
        # 添加总标题
        fig.suptitle(f'VIL Data - Sample {sample_id.decode("utf-8")} - {num_frames} Frames', 
                     fontsize=14, y=0.995)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ 网格图已保存: {output_path}")


if __name__ == "__main__":
    # 配置参数
    h5_file = "/data/public/vil/2019/SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5"
    output_dir = "/data/25fall_nowcasting/dly/PreDiff-25fall/vil_frames"
    grid_output = "/data/25fall_nowcasting/dly/PreDiff-25fall/vil_grid.png"
    #"/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/data/vil/2019//SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5"
    # 检查文件是否存在
    if not os.path.exists(h5_file):
        print(f"错误: 文件不存在: {h5_file}")
        print("请修改脚本中的 h5_file 路径")
        exit(1)
    
    # 绘制单独的帧
    plot_vil_data(
        h5_file_path=h5_file,
        output_dir=output_dir,
        sample_idx=0,  # 第一个样本
        num_frames=25   # 49帧
    )
    
    # 绘制网格图
    plot_vil_grid(
        h5_file_path=h5_file,
        output_path=grid_output,
        sample_idx=0,
        max_frames=49,
        ncols=7,
        frame_stride=1
    )