import numpy as np
import glob, os
from skimage.metrics import structural_similarity as ssim

# ========= 1. 路径 =========
pred_dir = '/home/user01/personal_file/cyr/PreDiff-25fall/experiments-25fall/1127_train_1099ckpt/npy_sevir'
gt_dir   = '/home/user01/25fall_nowcasting/25fall_aiclass/lesson_resource/data/prediff/datasets/sevirlr/data_npy_2019_len19'

# ========= 2. 文件列表 =========
pred_no_files = sorted(glob.glob(f'{pred_dir}/batch*_rank0_sample0.npy'))
pred_ka_files = [f.replace('.npy', '_aligned.npy') for f in pred_no_files]
gt_files      = sorted(glob.glob(f'{gt_dir}/*.npy'))[:len(pred_no_files)]

# ========= 3. 工具函数 =========
def csi_sevir(pred, gt, thr):
    """pred,gt 都是亮度 0-255"""
    p = pred >= thr
    g = gt   >= thr
    hit  = (p & g).sum()
    miss = (~p & g).sum()
    fa   = (p & ~g).sum()
    return hit / (hit + miss + fa + 1e-8)

def ssim0(p, g):
    return ssim(p, g, data_range=1.0)

# ========= 4. SEVIR 官方亮度阈值 =========
sevir_thr = [16, 74, 133, 160, 181, 219]   # 对应 0.1/1/2.5/10/30/100 mm/h
lead_idx  = [0, 2, 5, 8, 11]               # 10 30 60 90 120 min
time_lab  = ['10min', '30min', '1h', '1h30min', '2h']

metrics   = {'noKA': {t: {'MAE': [], 'RMSE': [], 'SSIM': [],
                          **{f'CSI{thr}': [] for thr in sevir_thr}} for t in lead_idx},
             'KA':   {t: {'MAE': [], 'RMSE': [], 'SSIM': [],
                          **{f'CSI{thr}': [] for thr in sevir_thr}} for t in lead_idx}}

# ========= 5. 主循环 =========
for batch_idx in range(len(gt_files) // 2):
    for sample_idx in range(2):
        gt_index = batch_idx * 2 + sample_idx
        if gt_index >= len(gt_files):
            continue

        # ---- 5.1 读 GT：19帧 → 取后12帧（预报） ----------
        gt_path = os.path.join(gt_dir, gt_files[gt_index])
        gt = np.load(gt_path)                 # (H,W,19) or (19,H,W)
        if gt.ndim == 3 and gt.shape[-1] == 19:
            gt = gt[..., 7:19]                # (H,W,12)
        gt = np.transpose(gt, (2, 0, 1))      # (12,H,W)

        # ---- 5.2 读 pred ----------
        base = f'batch{batch_idx}_rank0_sample{sample_idx}'
        pred_no_path = os.path.join(pred_dir, f'{base}.npy')
        pred_ka_path = os.path.join(pred_dir, f'{base}_aligned.npy')
        if not (os.path.exists(pred_no_path) and os.path.exists(pred_ka_path)):
            continue

        pred_no = np.load(pred_no_path)
        pred_ka = np.load(pred_ka_path)
        if pred_no.ndim == 5:
            pred_no, pred_ka = pred_no[0], pred_ka[0]
        if pred_no.shape[-1] == 1:
            pred_no, pred_ka = pred_no[..., 0], pred_ka[..., 0]
        assert pred_no.shape == gt.shape, f'shape {pred_no.shape} vs {gt.shape}'

        # ---- 5.3 逐时刻算指标（只算圈内，mask=255 排除） ----------
        for t in lead_idx:
            g  = gt[t]                # 亮度 0-255
            pn = pred_no[t]
            pk = pred_ka[t]

            mask = g < 255            # 圈内有效像素
            if mask.sum() == 0:
                continue
            g_m   = g[mask]
            pn_m  = pn[mask]
            pk_m  = pk[mask]

            for tag, pred_m in [('noKA', pn_m), ('KA', pk_m)]:
                pred_m255 = pred_m * 255.          # 0-1 → 亮度
                mae  = float(np.mean(np.abs(pred_m - g_m/255.)))
                rmse = float(np.sqrt(np.mean((pred_m - g_m/255.)**2)))
                ssim_val = float(ssim0(pred_m, g_m/255.))

                # 官方多阈值 CSI（亮度空间）
                for thr in sevir_thr:
                    csi = float(csi_sevir(pred_m255, g_m, thr))
                    metrics[tag][t][f'CSI{thr}'].append(csi)

                metrics[tag][t]['MAE'].append(mae)
                metrics[tag][t]['RMSE'].append(rmse)
                metrics[tag][t]['SSIM'].append(ssim_val)

# ========= 6. 输出 ----------
for tag in ['noKA', 'KA']:
    print(f'----- {tag} -----')
    for i, t in enumerate(lead_idx):
        m = metrics[tag][t]
        csi_str = '  '.join([f'CSI{thr}={np.mean(m[f"CSI{thr}"]):.4f}' for thr in sevir_thr])
        print(f'{time_lab[i]:>6} | MAE={np.mean(m["MAE"]):.4f}  '
              f'RMSE={np.mean(m["RMSE"]):.4f}  SSIM={np.mean(m["SSIM"]):.4f}  {csi_str}')