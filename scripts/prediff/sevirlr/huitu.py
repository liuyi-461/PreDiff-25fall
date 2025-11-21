"""
预测结果可视化对比程序
读取真值NPY、aligned预测NPY、unaligned预测NPY，生成对比图

使用方法:
python visualize_predictions.py --gt_dir /path/to/ground_truth --pred_dir /path/to/predictions --output_dir /path/to/output

或者直接修改下面的路径后运行
"""
import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 防止X11错误
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import math
from typing import Sequence, Union, Dict, Optional


# ========== SEVIR颜色映射 ==========
VIL_LEVELS = np.array([0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0])
VIL_COLORS = np.array([
    [0, 0, 0, 0],           # 0 transparent
    [0.3, 0.3, 0.3, 0.5],   # 1 gray
    [0.0, 1.0, 0.0, 0.75],  # 2 green  
    [0.0, 0.7, 0.0, 0.85],  # 3 dark green
    [1.0, 1.0, 0.0, 0.9],   # 4 yellow
    [0.9, 0.75, 0.0, 0.95], # 5 dark yellow
    [1.0, 0.5, 0.0, 1.0],   # 6 orange
    [1.0, 0.0, 0.0, 1.0],   # 7 red
    [0.7, 0.0, 0.0, 1.0],   # 8 dark red
    [1.0, 0.0, 1.0, 1.0],   # 9 magenta
    [0.6, 0.0, 0.6, 1.0]    # 10 dark magenta
])


def get_cmap(encoded=True):
    """获取SEVIR的VIL颜色映射"""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    cmap = ListedColormap(VIL_COLORS)
    norm = BoundaryNorm(VIL_LEVELS, cmap.N)
    vmin = VIL_LEVELS[0]
    vmax = VIL_LEVELS[-1]
    
    return cmap, norm, vmin, vmax


def vis_sevir_seq(
        save_path,
        seq: Union[np.ndarray, Sequence[np.ndarray]],
        label: Union[str, Sequence[str]] = "pred",
        norm: Optional[Dict[str, float]] = None,
        interval_real_time: float = 10.0,
        plot_stride: int = 2,
        label_rotation: int = 0,
        label_offset: tuple = (-0.06, 0.4),
        label_avg_int: bool = False,
        fs: int = 10,
        max_cols: int = 10):
    """
    可视化SEVIR序列
    
    Parameters
    ----------
    seq : Union[np.ndarray, Sequence[np.ndarray]]
        形状 = (T, H, W)，归一化后的值 0-1
    label : Union[str, Sequence[str]]
        每个序列的标签
    norm : Dict[str, float]
        归一化参数，seq_show = seq * norm['scale'] + norm['shift']
    interval_real_time : float
        每个时间步的实际分钟数
    plot_stride : int
        绘图步长
    max_cols : int
        最大列数
    """
    def cmap_dict():
        cmap, cnorm, vmin, vmax = get_cmap(encoded=True)
        # 只返回cmap和norm，不要同时传vmin/vmax
        return {'cmap': cmap, 'norm': cnorm}
    
    fontproperties = FontProperties()
    fontproperties.set_family('serif')
    fontproperties.set_size(fs)
    
    # 处理输入
    if isinstance(seq, Sequence):
        seq_list = [ele.astype(np.float32) for ele in seq]
        assert isinstance(label, Sequence) and len(label) == len(seq)
        label_list = list(label)
    elif isinstance(seq, np.ndarray):
        seq_list = [seq.astype(np.float32)]
        assert isinstance(label, str)
        label_list = [label]
    else:
        raise NotImplementedError
    
    # 添加平均强度标签
    if label_avg_int:
        label_list = [f"{ele1}\nAvgInt = {np.mean(ele2):.3f}"
                      for ele1, ele2 in zip(label_list, seq_list)]
    
    # 应用plot_stride
    seq_list = [ele[::plot_stride, ...] for ele in seq_list]
    seq_len_list = [len(ele) for ele in seq_list]
    
    max_len = min(max(seq_len_list), max_cols)
    
    # 处理长序列换行
    seq_list_wrap = []
    label_list_wrap = []
    seq_len_list_wrap = []
    
    for seq, label, seq_len in zip(seq_list, label_list, seq_len_list):
        num_row = math.ceil(seq_len / max_len)
        for j in range(num_row):
            slice_end = min(seq_len, (j + 1) * max_len)
            seq_list_wrap.append(seq[j * max_len: slice_end])
            if j == 0:
                label_list_wrap.append(label)
            else:
                label_list_wrap.append("")
            seq_len_list_wrap.append(min(seq_len - j * max_len, max_len))
    
    # 默认归一化参数
    if norm is None:
        norm = {'scale': 255, 'shift': 0}
    
    # 创建图形
    nrows = len(seq_list_wrap)
    fig, ax = plt.subplots(nrows=nrows, ncols=max_len,
                          figsize=(3 * max_len, 3 * nrows))
    
    # 如果只有一行，确保ax是二维的
    if nrows == 1:
        ax = np.array([ax])
    
    # 绘制每一行
    for i, (seq, label, seq_len) in enumerate(zip(seq_list_wrap, label_list_wrap, seq_len_list_wrap)):
        ax[i][0].set_ylabel(ylabel=label, fontproperties=fontproperties, rotation=label_rotation)
        ax[i][0].yaxis.set_label_coords(label_offset[0], label_offset[1])
        
        for j in range(max_len):
            if j < seq_len:
                x = seq[j] * norm['scale'] + norm['shift']
                ax[i][j].imshow(x, **cmap_dict())
                # 添加时间标签（只在最后一行）
                if i == len(seq_list_wrap) - 1:
                    ax[i][j].set_title(f"Min {int(interval_real_time * (j + 1) * plot_stride)}",
                                      y=-0.25, fontproperties=fontproperties)
            else:
                ax[i][j].axis('off')
    
    # 移除刻度
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].xaxis.set_ticks([])
            ax[i][j].yaxis.set_ticks([])
    
    # 添加图例
    num_thresh_legend = len(VIL_LEVELS) - 1
    legend_elements = [Patch(facecolor=VIL_COLORS[i],
                            label=f'{int(VIL_LEVELS[i-1])}-{int(VIL_LEVELS[i])}')
                      for i in range(1, num_thresh_legend + 1)]
    ax[0][0].legend(handles=legend_elements, loc='center left',
                   bbox_to_anchor=(-1.2, 0.), borderaxespad=0,
                   frameon=False, fontsize='10')
    
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {save_path}")


def load_npy_data(file_path):
    """
    加载NPY文件并转换格式
    
    输入格式: (H, W, T) 或 (T, H, W, C)
    输出格式: (T, H, W)
    """
    data = np.load(file_path)
    
    # 处理不同的输入格式
    if data.ndim == 3:
        # (H, W, T) -> (T, H, W)
        if data.shape[2] <= 25:  # 假设T <= 25
            data = np.transpose(data, (2, 0, 1))
        # 否则已经是 (T, H, W)
    elif data.ndim == 4:
        # (T, H, W, C) -> (T, H, W)
        data = data.squeeze(-1)
    
    return data


def find_matching_files(gt_dir, pred_dir):
    """
    找到匹配的真值和预测文件
    
    Returns
    -------
    matches : list of dict
        每个dict包含: {'date': str, 'gt': str, 'aligned': str, 'unaligned': str}
    """
    # 获取所有真值文件
    gt_pattern = os.path.join(gt_dir, "*.npy")
    gt_files = sorted(glob.glob(gt_pattern))
    
    matches = []
    
    for gt_file in gt_files:
        # 提取日期标识（文件名）
        gt_basename = os.path.basename(gt_file)
        date_str = os.path.splitext(gt_basename)[0]
        
        # 查找对应的预测文件
        aligned_file = os.path.join(pred_dir, f"{date_str}_aligned.npy")
        unaligned_file = os.path.join(pred_dir, f"{date_str}_unaligned.npy")
        
        # 检查文件是否存在
        if os.path.exists(aligned_file) and os.path.exists(unaligned_file):
            matches.append({
                'date': date_str,
                'gt': gt_file,
                'aligned': aligned_file,
                'unaligned': unaligned_file
            })
        else:
            if not os.path.exists(aligned_file):
                print(f"Warning: Missing aligned prediction for {date_str}")
            if not os.path.exists(unaligned_file):
                print(f"Warning: Missing unaligned prediction for {date_str}")
    
    return matches


def visualize_comparison(gt_file, aligned_file, unaligned_file, output_path,
                        in_len=7, out_len=6):
    """
    生成对比可视化
    
    Parameters
    ----------
    gt_file : str
        真值NPY文件路径
    aligned_file : str
        aligned预测NPY文件路径
    unaligned_file : str
        unaligned预测NPY文件路径
    output_path : str
        输出PNG文件路径
    in_len : int
        输入序列长度
    out_len : int
        输出序列长度
    """
    # 加载数据
    gt_data = load_npy_data(gt_file)  # (13, H, W)，范围 0-255
    aligned_pred = load_npy_data(aligned_file)  # (13, H, W)，范围 0-1（归一化后）
    unaligned_pred = load_npy_data(unaligned_file)  # (13, H, W)，范围 0-1（归一化后）
    
    # ========== 新增：反归一化预测数据到0-255范围 ==========
    # 预测数据是归一化到0-1的，需要乘以255恢复到真值相同的范围
    aligned_pred = aligned_pred * 255.0
    unaligned_pred = unaligned_pred * 255.0
    # ====================================================
    
    # 分离输入和输出
    gt_context = gt_data[:in_len]  # 前7帧
    gt_target = gt_data[in_len:in_len+out_len]  # 后6帧
    
    aligned_context = aligned_pred[:in_len]
    aligned_target = aligned_pred[in_len:in_len+out_len]
    
    unaligned_context = unaligned_pred[:in_len]
    unaligned_target = unaligned_pred[in_len:in_len+out_len]
    
    # 准备可视化序列
    seq_list = [
        gt_context,
        gt_target,
        aligned_target,
        unaligned_target
    ]
    
    label_list = [
        "Input (Context)",
        "Ground Truth (Target)",
        "Prediction (w/ KA - Aligned)",
        "Prediction (w/o KA - Unaligned)"
    ]
    
    # 生成可视化
    vis_sevir_seq(
        save_path=output_path,
        seq=seq_list,
        label=label_list,
        norm={'scale': 1.0, 'shift': 0.0},  # 数据现在都是0-255范围，不需要额外scale
        interval_real_time=10,
        plot_stride=1,
        label_rotation=0,
        label_offset=(-0.5, 0.5),
        label_avg_int=True,
        fs=20,
        max_cols=10
    )


def main():
    # ========== 路径配置（写死在这里，方便使用）==========
    GT_DIR = '/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/data_npy_2019_len13/'
    PRED_DIR = '/data/25fall_nowcasting/dly/PreDiff-25fall/tmp_sevirlr_prediff6/npy_outputs/'
    OUTPUT_DIR = '/data/25fall_nowcasting/dly/PreDiff-25fall/huitu/comparison_plotsm=1/'
    IN_LEN = 7
    OUT_LEN = 6
    MAX_SAMPLES = None  # None表示处理全部，可以改为数字如10来只处理前10个
    # ===================================================
    
    # 仍然保留命令行参数，可以覆盖默认值
    parser = argparse.ArgumentParser(description='可视化预测结果对比')
    parser.add_argument('--gt_dir', type=str, default=GT_DIR,
                       help='真值NPY文件目录')
    parser.add_argument('--pred_dir', type=str, default=PRED_DIR,
                       help='预测NPY文件目录')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='输出可视化PNG文件目录')
    parser.add_argument('--in_len', type=int, default=IN_LEN,
                       help='输入序列长度')
    parser.add_argument('--out_len', type=int, default=OUT_LEN,
                       help='输出序列长度')
    parser.add_argument('--max_samples', type=int, default=MAX_SAMPLES,
                       help='最大处理样本数（None表示处理全部）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找匹配的文件
    print(f"Searching for matching files...")
    print(f"  GT directory: {args.gt_dir}")
    print(f"  Prediction directory: {args.pred_dir}")
    
    matches = find_matching_files(args.gt_dir, args.pred_dir)
    
    if len(matches) == 0:
        print("Error: No matching files found!")
        print("Please check:")
        print("  1. GT directory exists and contains .npy files")
        print("  2. Prediction directory contains corresponding *_aligned.npy and *_unaligned.npy files")
        return
    
    print(f"Found {len(matches)} matching file sets")
    
    # 限制处理数量
    if args.max_samples is not None:
        matches = matches[:args.max_samples]
        print(f"Processing first {args.max_samples} samples")
    
    # 处理每个匹配
    for i, match in enumerate(matches):
        print(f"\n[{i+1}/{len(matches)}] Processing {match['date']}...")
        
        output_path = os.path.join(args.output_dir, f"{match['date']}_comparison.png")
        
        try:
            visualize_comparison(
                gt_file=match['gt'],
                aligned_file=match['aligned'],
                unaligned_file=match['unaligned'],
                output_path=output_path,
                in_len=args.in_len,
                out_len=args.out_len
            )
            print(f"  ✓ Saved to {output_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Completed! Generated {len(matches)} comparison plots")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()