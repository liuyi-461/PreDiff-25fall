#!/usr/bin/env python3
"""
SEVIR VIL 雷达数据可视化工具 (改进版 - 支持不同H5结构)
绘制每一帧雷达图像
"""

import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


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


def analyze_h5_structure(h5_file_path):
    """
    分析H5文件结构，返回关键信息
    
    Returns:
    --------
    dict: 包含文件结构信息
    """
    info = {
        'has_id': False,
        'vil_key': None,
        'vil_shape': None,
        'all_keys': []
    }
    
    with h5py.File(h5_file_path, 'r') as f:
        # 获取所有顶层键
        info['all_keys'] = list(f.keys())
        
        # 检查是否有'id'字段
        if 'id' in f.keys():
            info['has_id'] = True
        
        # 查找VIL数据的键（可能是'vil', 'VIL', 'data'等）
        for key in f.keys():
            if 'vil' in key.lower():
                info['vil_key'] = key
                info['vil_shape'] = f[key].shape
                break
        
        # 如果没找到vil相关的键，使用第一个数据集
        if info['vil_key'] is None:
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    info['vil_key'] = key
                    info['vil_shape'] = f[key].shape
                    break
    
    return info


def plot_vil_frame(data, frame_idx, time_minutes, output_path, sample_name="", fs=12):
    """
    绘制单帧VIL数据
    
    Parameters:
    -----------
    data : np.ndarray
        单帧数据, shape=(H, W)
    frame_idx : int
        帧序号
    time_minutes : int
        时间(分钟)
    output_path : str
        输出文件路径
    sample_name : str
        样本名称
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
    title = f'VIL Frame {frame_idx:02d} | Time: {time_minutes} min'
    if sample_name:
        title = f'{sample_name} - ' + title
    ax.set_title(title, fontproperties=fontproperties, pad=15)
    
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


def plot_vil_data(h5_file_path, output_dir, sample_idx=0, num_frames=49, time_interval=5):
    """
    从H5文件读取并绘制VIL数据 (支持不同结构的H5文件)
    
    Parameters:
    -----------
    h5_file_path : str
        H5文件路径
    output_dir : str
        输出目录
    sample_idx : int
        要绘制的样本索引
    num_frames : int
        要绘制的帧数
    time_interval : int
        时间间隔(分钟)，默认5分钟
    """
    print("="*80)
    print(f"开始处理VIL数据")
    print(f"文件: {h5_file_path}")
    print(f"样本索引: {sample_idx}")
    print(f"帧数: {num_frames}")
    print("="*80)
    
    # 先分析文件结构
    print("\n分析H5文件结构...")
    file_info = analyze_h5_structure(h5_file_path)
    print(f"  文件中的所有键: {file_info['all_keys']}")
    print(f"  VIL数据键: {file_info['vil_key']}")
    print(f"  VIL数据形状: {file_info['vil_shape']}")
    print(f"  是否包含ID字段: {file_info['has_id']}")
    
    if file_info['vil_key'] is None:
        print("\n❌ 错误: 未找到VIL数据!")
        print(f"   文件中的键: {file_info['all_keys']}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取H5文件
    with h5py.File(h5_file_path, 'r') as f:
        # 获取VIL数据
        vil_data_full = f[file_info['vil_key']]
        
        # 判断数据的形状并正确索引
        print(f"\n数据维度: {len(vil_data_full.shape)}")
        print(f"完整数据形状: {vil_data_full.shape}")
        
        # 根据数据形状决定如何提取
        if len(vil_data_full.shape) == 4:
            # 形状: (samples, H, W, T) 或 (samples, T, H, W)
            if sample_idx >= vil_data_full.shape[0]:
                print(f"\n❌ 错误: 样本索引 {sample_idx} 超出范围 (最大: {vil_data_full.shape[0]-1})")
                return
            
            # 检查哪个维度是时间维度
            if vil_data_full.shape[-1] < vil_data_full.shape[1]:
                # (samples, H, W, T)
                vil_data = vil_data_full[sample_idx]  # shape: (H, W, T)
                print(f"数据格式: (samples, H, W, T)")
            else:
                # (samples, T, H, W)
                vil_data = vil_data_full[sample_idx]  # shape: (T, H, W)
                # 转置为 (H, W, T)
                vil_data = np.transpose(vil_data, (1, 2, 0))
                print(f"数据格式: (samples, T, H, W) -> 已转置为 (H, W, T)")
                
        elif len(vil_data_full.shape) == 3:
            # 形状: (T, H, W) 或 (H, W, T)
            if vil_data_full.shape[0] > vil_data_full.shape[-1]:
                # (T, H, W)
                vil_data = np.transpose(vil_data_full[:], (1, 2, 0))  # -> (H, W, T)
                print(f"数据格式: (T, H, W) -> 已转置为 (H, W, T)")
            else:
                # (H, W, T)
                vil_data = vil_data_full[:]
                print(f"数据格式: (H, W, T)")
        else:
            print(f"\n❌ 错误: 不支持的数据形状 {vil_data_full.shape}")
            return
        
        # 获取样本ID（如果有）
        sample_name = ""
        if file_info['has_id']:
            try:
                sample_id = f['id'][sample_idx]
                if isinstance(sample_id, bytes):
                    sample_name = sample_id.decode('utf-8')
                else:
                    sample_name = str(sample_id)
            except:
                sample_name = f"sample_{sample_idx:03d}"
        else:
            sample_name = f"sample_{sample_idx:03d}"
        
        print(f"\n样本名称: {sample_name}")
        print(f"提取后的数据形状: {vil_data.shape}")
        print(f"数据类型: {vil_data.dtype}")
        print(f"数值范围: [{np.min(vil_data)}, {np.max(vil_data)}]")
        print("\n开始绘制帧...\n")
        
        # 确定实际可绘制的帧数
        actual_num_frames = min(num_frames, vil_data.shape[2])
        
        # 绘制每一帧
        for frame_idx in range(actual_num_frames):
            # 提取单帧数据
            frame_data = vil_data[:, :, frame_idx]
            
            # 计算时间
            time_minutes = frame_idx * time_interval
            
            # 输出文件名
            output_filename = f"vil_{sample_name}_frame{frame_idx:02d}_t{time_minutes:03d}min.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # 绘制并保存
            plot_vil_frame(frame_data, frame_idx, time_minutes, output_path, sample_name)
        
        print("\n" + "="*80)
        print(f"✓ 完成! 共绘制 {actual_num_frames} 帧")
        print(f"✓ 输出目录: {output_dir}")
        print("="*80)


def plot_vil_grid(h5_file_path, output_path, sample_idx=0, 
                  max_frames=49, ncols=7, frame_stride=1, time_interval=5):
    """
    绘制网格形式的多帧VIL数据 (支持不同结构的H5文件)
    
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
    time_interval : int
        时间间隔(分钟)
    """
    print(f"\n创建网格图...")
    
    # 分析文件结构
    file_info = analyze_h5_structure(h5_file_path)
    
    with h5py.File(h5_file_path, 'r') as f:
        vil_data_full = f[file_info['vil_key']]
        
        # 根据数据形状提取数据
        if len(vil_data_full.shape) == 4:
            if vil_data_full.shape[-1] < vil_data_full.shape[1]:
                vil_data = vil_data_full[sample_idx]  # (H, W, T)
            else:
                vil_data = vil_data_full[sample_idx]  # (T, H, W)
                vil_data = np.transpose(vil_data, (1, 2, 0))  # -> (H, W, T)
        elif len(vil_data_full.shape) == 3:
            if vil_data_full.shape[0] > vil_data_full.shape[-1]:
                vil_data = np.transpose(vil_data_full[:], (1, 2, 0))
            else:
                vil_data = vil_data_full[:]
        
        # 获取样本名称
        sample_name = ""
        if file_info['has_id']:
            try:
                sample_id = f['id'][sample_idx]
                if isinstance(sample_id, bytes):
                    sample_name = sample_id.decode('utf-8')
                else:
                    sample_name = str(sample_id)
            except:
                sample_name = f"sample_{sample_idx:03d}"
        else:
            sample_name = f"sample_{sample_idx:03d}"
        
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
            time_minutes = idx * time_interval * frame_stride
            
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
        fig.suptitle(f'VIL Data - {sample_name} - {num_frames} Frames', 
                     fontsize=14, y=0.995)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ 网格图已保存: {output_path}")


if __name__ == "__main__":
    # 配置参数 - 修改这里的路径
    h5_file = "/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/data/vil/2019/SEVIR_VIL_STORMEVENTS_2019_0701_1231.h5"
    output_dir = "/data/25fall_nowcasting/dly/PreDiff-25fall/vil_frames2"
    grid_output = "/data/25fall_nowcasting/dly/PreDiff-25fall/vil_grid.png"
    
    # 检查文件是否存在
    if not os.path.exists(h5_file):
        print(f"错误: 文件不存在: {h5_file}")
        print("请修改脚本中的 h5_file 路径")
        exit(1)
    
    # 绘制单独的帧
    plot_vil_data(
        h5_file_path=h5_file,
        output_dir=output_dir,
        sample_idx=0,       # 第一个样本
        num_frames=49,      # 49帧
        time_interval=10     # 5分钟间隔
    )
    
    # 绘制网格图
    plot_vil_grid(
        h5_file_path=h5_file,
        output_path=grid_output,
        sample_idx=0,
        max_frames=49,
        ncols=7,
        frame_stride=1,
        time_interval=5
    )