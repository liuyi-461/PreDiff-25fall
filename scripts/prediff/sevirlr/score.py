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
pred_dir = '/data/25fall_nowcasting/cyr/PreDiff-25fall/experiments-25fall/1027_train/npy'
gt_dir   = '/data/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/data_npy_test_13'


# ========= 2. 文件列表 =========
# 无KA：batch*_rank0_sample0.npy
# 有KA：batch*_rank0_sample0_aligned.npy
pred_no_files = sorted(glob.glob(f'{pred_dir}/batch*_rank0_sample0.npy'))
pred_ka_files = [f.replace('.npy', '_aligned.npy') for f in pred_no_files]
# 真实值：与预测文件数量一致，按字典序一一对应
gt_files = sorted(glob.glob(f'{gt_dir}/*.npy'))[:len(pred_no_files)]


# ========= 3. 工具函数 =========
def calc_ts(pred, gt, threshold=0.1):
    """Threat Score"""
    p, g = pred >= threshold, gt >= threshold
    hit, miss, fa = (p & g).sum(), (~p & g).sum(), (p & ~g).sum()
    return hit / (hit + miss + fa + 1e-8)


# ========= 4. 对齐 & 计算 =========
metrics = {'KA':   {'mse': [], 'rmse': [], 'corr': [], 'ts': []},
           'noKA': {'mse': [], 'rmse': [], 'corr': [], 'ts': []}}

for pf_no, pf_ka, gf in zip(pred_no_files, pred_ka_files, gt_files):
    if not (os.path.exists(pf_no) and os.path.exists(pf_ka) and os.path.exists(gf)):
        print('skip', os.path.basename(pf_no))
        continue

    # ---- 加载 ----
    pred_no = np.load(pf_no)   # 无KA
    pred_ka = np.load(pf_ka)   # 有KA
    gt      = np.load(gf)      # 真实值 (T,H,W) 或 (H,W,T)

    # ---- 4.1 去样本维度（若存在） ----
    if pred_no.ndim == 5:                       # (N,T,H,W,C) 或 (N,T,H,W)
        pred_no = pred_no[0]                    # 取第 1 个样本 → (T,H,W,C|none)
        pred_ka = pred_ka[0]

    # ---- 4.2 通道维度 ----
    if pred_no.shape[-1] == 1:                  # (T,H,W,1) → (T,H,W)
        pred_no = pred_no[..., 0]
        pred_ka = pred_ka[..., 0]

    # ---- 4.3 时间：gt 取后 6 帧与预测对齐 ----
    t_pred = pred_no.shape[0]                   # 6
    if gt.ndim == 3 and gt.shape[-1] == 13:     # (H,W,13) → 取后 6
        gt = gt[..., -t_pred:]                  # (H,W,6)
    elif gt.ndim == 3 and gt.shape[0] == 13:    # (13,H,W) → 取后 6
        gt = gt[-t_pred:]                       # (6,H,W)

    # ---- 4.4 维度顺序统一成 (T,H,W) ----
    if gt.ndim == 3 and gt.shape[-1] == t_pred:     # (H,W,T)
        gt = gt.transpose(2, 0, 1)                  # → (T,H,W)

    # ---- 4.5 空间：gt 缩小到 128×128（最近邻保雨区） ----
    if gt.shape[1:3] != (128, 128):
        gt_resize = np.empty((t_pred, 128, 128), dtype=gt.dtype)
        for t in range(t_pred):
            gt_resize[t] = resize(gt[t], (128, 128), order=0, preserve_range=True)
        gt = gt_resize

    # ---- 4.6 拉平 ----
    pred_flat_no = pred_no.ravel()
    pred_flat_ka = pred_ka.ravel()
    gt_flat      = gt.ravel()

    # ---- 4.7 指标 ----
    for tag, pred_flat in [('noKA', pred_flat_no), ('KA', pred_flat_ka)]:
        mse  = np.mean((pred_flat - gt_flat) ** 2)
        rmse = np.sqrt(mse)
        corr = np.corrcoef(pred_flat, gt_flat)[0, 1]
        ts   = calc_ts(pred_flat, gt_flat, threshold=0.1)
        metrics[tag]['mse'].append(mse)
        metrics[tag]['rmse'].append(rmse)
        metrics[tag]['corr'].append(corr)
        metrics[tag]['ts'].append(ts)


# ========= 5. 输出对比 =========
for tag in ['noKA', 'KA']:
    print(f'----- {tag} -----')
    print(f'MSE  : {np.mean(metrics[tag]["mse"]):.6f}')
    print(f'RMSE : {np.mean(metrics[tag]["rmse"]):.6f}')
    print(f'Corr : {np.mean(metrics[tag]["corr"]):.6f}')
    print(f'TS   : {np.mean(metrics[tag]["ts"]):.6f}')