#!/usr/bin/env python3
"""
NPY数据诊断脚本
使用方法: python check_npy.py
"""

import numpy as np
import os
import sys

# 根据你的错误日志，这是正确的路径
DATA_DIR = "/data/25fall_nowcasting/25fall_aiclass/dly/PreDiff/datasets/sevirlr/data_npy/"

def main():
    print("="*80)
    print("NPY数据诊断")
    print("="*80)
    print(f"\n数据目录: {DATA_DIR}")
    
    # 检查目录是否存在
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ 错误: 目录不存在: {DATA_DIR}")
        sys.exit(1)
    
    # 获取所有npy文件
    npy_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
    npy_files.sort()
    
    if len(npy_files) == 0:
        print(f"\n❌ 错误: 在 {DATA_DIR} 中没有找到NPY文件")
        sys.exit(1)
    
    print(f"\n✓ 找到 {len(npy_files)} 个NPY文件")
    print(f"\n文件示例: {npy_files[:5]}")
    
    # 读取第一个文件
    first_file = os.path.join(DATA_DIR, npy_files[0])
    print(f"\n" + "="*80)
    print(f"读取第一个文件: {npy_files[0]}")
    print("="*80)
    
    try:
        data = np.load(first_file)
        
        print(f"\n📊 基本信息:")
        print(f"  形状:      {data.shape}")
        print(f"  数据类型:  {data.dtype}")
        print(f"  数据范围:  [{data.min():.6f}, {data.max():.6f}]")
        print(f"  均值:      {data.mean():.6f}")
        print(f"  标准差:    {data.std():.6f}")
        print(f"  总元素数:  {data.size:,}")
        print(f"  内存大小:  {data.nbytes / 1024:.2f} KB")
        
        # 详细维度分析
        print(f"\n📐 维度分析:")
        if len(data.shape) == 4:
            dim0, dim1, dim2, dim3 = data.shape
            print(f"  维度0: {dim0}")
            print(f"  维度1: {dim1}")
            print(f"  维度2: {dim2}")
            print(f"  维度3: {dim3}")
            
            # 判断可能的布局
            print(f"\n🔍 布局判断:")
            if dim1 == 128 and dim2 == 128:
                print(f"  ✅ 布局: THWC")
                print(f"     时间步={dim0}, 高度={dim1}, 宽度={dim2}, 通道={dim3}")
                print(f"  ✅ 这是正确的格式!")
            elif dim1 == 1 and dim2 == 128 and dim3 == 128:
                print(f"  ⚠️  布局: TCHW")
                print(f"     时间步={dim0}, 通道={dim1}, 高度={dim2}, 宽度={dim3}")
                print(f"  ⚠️  需要转换为THWC格式: data.transpose(0,2,3,1)")
            elif dim0 == 13 or dim0 == 7 or dim0 == 6:
                print(f"  ❓ 可能的布局: T???")
                print(f"     时间步={dim0}")
                if dim1 < 20 and dim2 < 20:
                    print(f"  ❌ 警告: dim1={dim1}, dim2={dim2} 太小!")
                    print(f"     期望的空间分辨率是 128x128，但实际是 {dim1}x{dim2}")
                    print(f"  ❌ 你的数据可能有问题，需要重新生成!")
            else:
                print(f"  ❌ 未知布局")
                print(f"  期望形状: (13, 128, 128, 1) 或 (13, 1, 128, 128)")
                print(f"  实际形状: {data.shape}")
        elif len(data.shape) == 3:
            print(f"  ⚠️  3维数据: {data.shape}")
            print(f"  可能缺少通道维度，应该添加一个维度")
        else:
            print(f"  ❌ 不支持的维度数: {len(data.shape)}")
        
        # 显示每个时间步的信息
        print(f"\n⏱️  时间步信息 (前3个):")
        for t in range(min(3, data.shape[0])):
            if len(data.shape) == 4:
                frame = data[t]
                print(f"  时间步 {t}: 形状={frame.shape}, 范围=[{frame.min():.4f}, {frame.max():.4f}]")
            else:
                print(f"  时间步 {t}: (无法解析)")
        
        # 检查更多文件
        print(f"\n📋 检查所有文件的形状一致性:")
        shape_count = {}
        for fname in npy_files[:10]:  # 检查前10个
            fpath = os.path.join(DATA_DIR, fname)
            d = np.load(fpath)
            shape_str = str(d.shape)
            if shape_str not in shape_count:
                shape_count[shape_str] = []
            shape_count[shape_str].append(fname)
        
        for shape_str, files in shape_count.items():
            print(f"  形状 {shape_str}: {len(files)} 个文件")
            if len(files) <= 3:
                for f in files:
                    print(f"    - {f}")
        
        # 最终判断
        print(f"\n" + "="*80)
        print("🎯 结论:")
        print("="*80)
        expected_shape = (13, 128, 128, 1)
        if data.shape == expected_shape:
            print(f"✅ 数据格式完全正确!")
            print(f"   形状 {data.shape} 符合要求")
        elif data.shape == (13, 1, 128, 128):
            print(f"⚠️  数据需要转换!")
            print(f"   当前形状: {data.shape} (TCHW)")
            print(f"   需要转换为: (13, 128, 128, 1) (THWC)")
            print(f"   转换方法: data.transpose(0, 2, 3, 1)")
        else:
            print(f"❌ 数据格式不正确!")
            print(f"   期望形状: {expected_shape}")
            print(f"   实际形状: {data.shape}")
            if data.shape[1] < 20 or data.shape[2] < 20:
                print(f"\n❌❌❌ 严重问题: 空间分辨率太低!")
                print(f"   空间维度是 {data.shape[1]}x{data.shape[2]}")
                print(f"   但期望是 128x128")
                print(f"   你需要重新生成NPY文件!")
        
    except Exception as e:
        print(f"\n❌ 读取文件时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "="*80)

if __name__ == "__main__":
    main()