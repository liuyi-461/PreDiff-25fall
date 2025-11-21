#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键对比 Prediff 有无 KA 的预测结果
与 SEVIR-LR 真实值 (后 6 帧) 算 MSE/RMSE/Corr/TS
用法：python compare_ka.py
"""

import numpy as np
import glob, os
from skimage.transform import resize


# ========= 1. 路径 =========
pred_dir = '/data/25fall_nowcasting/cyr/PreDiff-25fall/experiments-25fall/1111_train/npy'
gt_dir   = '/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/radar_npy_len13_530test'


# ========= 2. 文件列表 =========
# 无KA：batch*_rank0_sample0.npy
# 有KA：batch*_rank0_sample0_aligned.npy
pred_no_files = sorted(glob.glob(f'{pred_dir}/batch*_rank0_sample0.npy'))
pred_ka_files = [f.replace('.npy', '_aligned.npy') for f in pred_no_files]
# 真实值：与预测文件数量一致，按字典序一一对应
gt_files = sorted(glob.glob(f'{gt_dir}/*.npy'))[:len(pred_no_files)]


# ========= 3. 工具函数 =========
def calc_csi(pred, gt, threshold=30.0, scale=180.0):
    """
    业务版 CSI
    pred, gt : 任意 shape 的 ndarray，值域 0-1（归一化）
    threshold:  dBZ 阈值，默认 30
    scale    :  归一化时用的最大值，默认 180（即 0-1 对应 0-180 dBZ）
    """
    pred_dBZ = pred * scale          # 回到物理单位
    gt_dBZ   = gt

    p = pred_dBZ >= threshold
    g = gt_dBZ   >= threshold

    hit  = (p & g).sum()
    miss = (~p & g).sum()
    fa   = (p & ~g).sum()

    return hit / (hit + miss + fa + 1e-8)

# ========= 4. 事件-样本级对齐（按第二段逻辑） =========
gt_files = sorted(os.listdir(gt_dir))      # 530 个 .npy，按事件名排序
n_event  = len(gt_files)
metrics = {'KA':   {'mse': [], 'rmse': [], 'corr': [], 'ts': [], 'csi30': []},
           'noKA': {'mse': [], 'rmse': [], 'corr': [], 'ts': [], 'csi30': []}}

for batch_idx in range(len(gt_files) // 2):
    for sample_idx in range(2):
        gt_index = batch_idx * 2 + sample_idx
        #if batch_idx  >= 40:
            #break
        if gt_index >= len(gt_files):
            continue
        # ---- 4.1 读 GT：固定 7:13 帧，转 (6,128,128) ----
        gt_path = os.path.join(gt_dir, gt_files[gt_index])
        gt = np.load(gt_path)                 # (H,W,13) 或 (128,128,13)
        if gt.ndim == 3 and gt.shape[-1] == 13:
            gt = gt[..., 7:13]                # (H,W,6)
        gt = np.transpose(gt, (2, 0, 1))      # (6,H,W)

        # ---- 4.2 读 pred ----
        base = f'batch{batch_idx}_rank0_sample{sample_idx}'
        pred_no_path = os.path.join(pred_dir, f'{base}.npy')
        pred_ka_path = os.path.join(pred_dir, f'{base}_aligned.npy')
        if not (os.path.exists(pred_no_path) and os.path.exists(pred_ka_path)):
            continue

        pred_no = np.load(pred_no_path)   # 无KA
        pred_ka = np.load(pred_ka_path)   # 有KA

        # ---- 4.3 去样本/通道维度 ----
        if pred_no.ndim == 5:                       # (N,T,H,W,C) 或 (N,T,H,W)
            pred_no = pred_no[0]
            pred_ka = pred_ka[0]
        if pred_no.shape[-1] == 1:                  # (T,H,W,1) → (T,H,W)
            pred_no = pred_no[..., 0]
            pred_ka = pred_ka[..., 0]

        # ---- 4.4 此时 pred 应该是 (6,128,128)，GT 也是 (6,128,128) ----
        assert pred_no.shape == gt.shape, f'shape mismatch {pred_no.shape} vs {gt.shape}'

        # ---- 4.5 拉平 & 算指标 ----
        pred_flat_no = pred_no.ravel()
        pred_flat_ka = pred_ka.ravel()
        gt_flat      = gt.ravel()

        for tag, pred_flat in [('noKA', pred_flat_no), ('KA', pred_flat_ka)]:
            mse  = np.mean((pred_flat - gt_flat/180.) ** 2)
            rmse = np.sqrt(mse)
            corr = np.corrcoef(pred_flat, gt_flat/180.)[0, 1]
            csi30 = calc_csi(pred_flat, gt_flat, threshold=30.0, scale=180.0)

            metrics[tag]['mse'].append(mse)
            metrics[tag]['rmse'].append(rmse)
            metrics[tag]['corr'].append(corr)
            metrics[tag]['csi30'].append(csi30)

# ========= 5. 输出对比 =========
for tag in ['noKA', 'KA']:
    print(f'----- {tag} -----')
    print(f'MSE  : {np.mean(metrics[tag]["mse"]):.6f}')
    print(f'RMSE : {np.mean(metrics[tag]["rmse"]):.6f}')
    print(f'Corr : {np.mean(metrics[tag]["corr"]):.6f}')
    print(f'CSI30: {np.mean(metrics[tag]["csi30"]):.6f}')