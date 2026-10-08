# -*- coding: utf-8 -*-
"""
QC/step04_kdp.py — 第 4 部分: φdp 平滑 与 KDP 反演
=================================================================
输入必须是**已扣系统相位**的 φdp(QC/step03_system_phase.py 的输出), 否则起始残留
会被差分放大成虚假 KDP。

★★ 2026-10-07 用户决定: **不用 Bringi 的迭代/FIR 滤波, 改用 zch_phi 双尺度方案**
   (Park et al. 2009: 轻度 0.6 km / 重度 1.5 km 平滑 φdp -> 对**平滑后**的 φdp
    做最小二乘拟合, 再按 40 dBZ 阈值在轻/重之间切换)。实现全在 `QC/zch_phi.py`,
    本文件只做接线。
    ★ 2026-10-08: zch_phi 的拟合此前误用了**原始** φdp(平滑值只当 phidp_light/heavy
      存下来), 已修正为喂平滑后的 φdp —— 见 `zch_phi._smooth_and_fit` 第 ② 段。
      本次修正只动 kdp_* 三个字段, phidp_light/heavy 逐位不变(ZPHI 不受影响)。

   ★★★ 字段名也已**一并改掉**(同日, 用户要求): 不再有 `kdp_bringi` / `phidp_bringi`
      (那名字是 Bringi 时代的, 算法都换了留着只会误导)。现在写的是 zch_phi 原生名:
          kdp_dual     (°/km)  主产物, 轻/重按 Z 阈值切换          ← 原 kdp_bringi
          phidp_heavy  (°)     重度平滑 φdp, 供 ZPHI 衰减订正      ← 原 phidp_bringi
          kdp_light / kdp_heavy / phidp_light                   (诊断用)
      ⇒ 下游(FHC / KEEP_FIELDS / 精简缓存)已同步改成 kdp_dual。

★ 输入 φdp 由 zch_phi 自己做"负段修正"(fix_phi_negative_jump, |负|>180 且后接正 ->
  加 360); zch_phi 明确**不做 np.unwrap**(本批数据从未遇到 0/360 卷绕)。
  ⚠ 注意: 门控阶段(step02)的纹理计算仍会临时 unwrap, 那是门控内部的事, 与本步无关。
"""

import numpy as np

from . import config as C
from . import zch_phi


def compute_kdp_bringi(radar, phi_corr, ref_raw, verbose=True, window=None):
    """
    用 **zch_phi 双尺度方案**(轻 0.6 km / 重 1.5 km + 对平滑后 φdp 最小二乘 + 40 dBZ 切换)反演 KDP,
    并写入 radar 字段。

    ⚠ 函数名 `compute_kdp_bringi` 是**历史遗留**(改名会动 pipeline/__init__ 的导出表),
      内部已是 zch_phi 双尺度; 真要彻底改名请连同 pipeline.py / __init__.py 一起改。

    phi_corr: (nrays, ngates) float, QC 外/无效为 nan
    ref_raw:  (nrays, ngates) float, 无效为 nan(用于 40 dBZ 阈值切换)
    window:   ★ 已废弃(Bringi 的 FIR 窗参数)。传了会被忽略, 仅为兼容旧调用签名。
    返回 (kd_lin, dp_lin): kd_lin = kdp_dual, dp_lin = phidp_heavy(均为 masked array)
    """
    if window is not None and verbose:
        print('  [KDP] window 参数已废弃(zch_phi 用 light/heavy 双尺度), 已忽略')

    range_m = radar.range['data'].astype(float)

    # --- 负段修正 + 双尺度滤波 + 最小二乘拟合 + Z 阈值切换(全在 zch_phi 里) ---
    phi_fix = zch_phi.fix_phi_negative_jump(
        np.asarray(phi_corr, dtype=float))
    kd_lin, kl_lin, kh_lin, pl_lin, ph_lin, info = zch_phi.kdp_dual_filter(
        phi_fix, np.asarray(ref_raw, dtype=float), range_m, verbose=verbose)

    kd_lin = np.ma.masked_invalid(kd_lin)
    dp_lin = np.ma.masked_invalid(ph_lin)     # 重度平滑 φdp -> ZPHI 用
    if verbose:
        print(f'  KDP反演完成(zch_phi 双尺度), kdp_dual 有效值: {kd_lin.count()}')

    # --- 写字段: 全部用 zch_phi 原生名(无 bringi 遗留名) ---
    def _ma(a):
        return np.ma.masked_invalid(a)

    for _name, _data, _units, _std, _lname in (
            ('kdp_dual', kd_lin, 'degrees/km', 'specific_differential_phase',
             'KDP dual-scale (zch_phi, light/heavy switched at %.0f dBZ)'
             % info['z_switch_dbz']),
            ('kdp_light', kl_lin, 'degrees/km', 'specific_differential_phase',
             f'KDP light ({info["light_filter_km"]:.1f} km)'),
            ('kdp_heavy', kh_lin, 'degrees/km', 'specific_differential_phase',
             f'KDP heavy ({info["heavy_filter_km"]:.1f} km)'),
            ('phidp_light', pl_lin, 'degrees', 'differential_propagation_phase',
             f'PhiDP light ({info["light_filter_km"]:.1f} km)'),
            ('phidp_heavy', ph_lin, 'degrees', 'differential_propagation_phase',
             f'PhiDP heavy ({info["heavy_filter_km"]:.1f} km) -- use this for ZPHI')):
        radar.add_field(_name, {
            'data': _ma(_data), 'units': _units,
            'standard_name': _std, 'long_name': _lname,
        }, replace_existing=True)

    return kd_lin, dp_lin


def mask_reflectivity(radar, ref_raw):
    """
    把 QC 外的门用 -30 dBZ 填充并 mask(供 ZPHI / 画图统一使用).
    就地替换 radar.fields['reflectivity']['data']。
    """
    excl = np.isnan(ref_raw) | (ref_raw == C.ATTEN_BAD_FILL)
    ref_data = np.where(excl, -30.0, ref_raw)
    radar.fields['reflectivity']['data'] = np.ma.masked_where(excl, ref_data)
    return radar
