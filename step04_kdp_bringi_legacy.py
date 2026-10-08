# -*- coding: utf-8 -*-
"""
QC/step04_kdp_bringi_legacy.py — 第 4 部分(旧实现): CSU Bringi KDP
=================================================================
★ 这是 **2026-10-07 之前链上用的原实现**, 从 `QC.7z`(2026-10-06 打的包)里还原,
  代码逐行保持原样, 只是加了这段说明。它**不在当前处理链上** ——
  现在 `QC/step04_kdp.py` 走的是 zch_phi 双尺度方案; 本模块按需单独调用,
  用于和 zch_phi 对拍、或按用户要求补出 Bringi 口径的 KDP。

输入必须是**已扣系统相位**的 φdp(QC/step03_system_phase.py 的输出), 否则起始残留
会被差分放大成虚假 KDP。

调用 csu_radartools.csu_kdp.calc_kdp_bringi(逐射线 FIR 滤波 + 最小二乘斜率),
输出两个字段(注意字段名是 Bringi 时代的原名, 与 zch_phi 的
kdp_dual / phidp_heavy **并存不冲突**):
    kdp_bringi   (°/km)
    phidp_bringi (°, 滤波后, 已扣系统相位) -> 供 ZPHI 衰减订正使用

★ 输入 φdp 会先做**相位解缠**(修 0/360° 卷绕)再送进 Bringi —— csu_kdp 明确要求
  "phase has been unfolded already"。详见 compute_kdp_bringi 里的说明。
"""

import numpy as np

from csu_radartools import csu_kdp

from . import config as C


def compute_kdp_bringi(radar, phi_corr, ref_raw, verbose=True, window=None):
    """
    用 Bringi 方法从已扣系统相位的 φdp 反演 KDP, 并写入 radar 字段.
    phi_corr: (nrays, ngates) float, QC 外/无效为 nan
    ref_raw:  (nrays, ngates) float, 无效为 nan(仅用于给 Bringi 一个 DC 分量)
    window:   FIR 滤波窗口(km), 默认 None = 取 C.KDP_FIR_WINDOW(缺省 3.0)。
              见下方"FIR 滤波窗口"注释 —— 必须满足 window*1000/gs 为偶数。
    返回 (kd_lin, dp_lin): 均为 masked array
    """
    bad = C.ATTEN_BAD_FILL
    nrays, ngates = radar.nrays, radar.ngates

    # --- KDP(Bringi), 输入已是扣除系统相位的 φdp ---
    range_km = radar.range['data'] / 1000.0
    rng_2d = np.broadcast_to(range_km, (nrays, ngates))
    phi_in = np.where(np.isnan(phi_corr), bad, phi_corr).astype(np.float32)
    ref_in = np.where(np.isnan(ref_raw), bad, ref_raw).astype(np.float32)

    # --- ★ 先把 φdp 解缠, 再交给 Bringi ---
    # csu_kdp 的文档写明: "It assumes differential phase has been unfolded already."
    # φdp 是 **0~360° 的卷绕量**(雷达原始矩), 一旦真实相位累积跨过 0/360 边界就会
    # 突然从 ~350° 掉到 ~10°。Bringi 的 FIR 滤波 + 最小二乘拟合对**折点**没有免疫力:
    #   滤波窗 (window/gs = 1.5km/30m ≈ 50 门) 骑在折点上时, 窗内相位被人为掰出一个
    #   几百度的跳变, 拟合斜率随之被放大成假的高 KDP;
    #   同时 sd_lin(相位标准差) 在折点附近飙高 -> 超过 thsd -> 那些门直接被判"坏"丢掉,
    #   输出 dp_lin 上出现一条条空洞。
    # 做法与 step06 的 ZPHI 相位解缠**同一套逻辑**(相邻差取 wrap 到 (-180,180] 后累加)。
    # ⚠ 只在**有效门**上解缠(中间可能隔着 QC 空洞, 按位置 index 解缠会把空洞前后的
    #   错位也算进去), 解完再原样放回。坏值(bad)不参与 wrap 计算, 全程保持 bad。
    # 注意: 只有从 -180 起跳的那种相位需要这一步; 本链的 φdp 是 0~360 规格, 故必做。
    for _ray in range(nrays):
        row = phi_in[_ray]
        _ok = (row != bad) & np.isfinite(row)
        if _ok.sum() < 2:
            continue
        vals = np.unwrap(row[_ok], period=360.0)
        # 解缠后若整体低于 0(起点在 300° 那一类), 平移到以起点为参照的连续段上,
        # 避免负相位经 KDP 拟合时被当作"负增长"。
        row[_ok] = vals - (np.floor(vals[0] / 360.0) * 360.0)
        phi_in[_ray] = row

    # ⚠ std_gate 必须是**奇数**(csu_kdp 里 `if std_gate % 2 != 1:` 会直接把它换回库默认
    #   STD_GATE=11), 而 config 写的是 55.0 —— 偶数, 所以该参数一直是**静默失效**的,
    #   实际跑的是 11 库窗口。这里显式归一化到最近的奇数, 免得再有人以为 55 生效了。
    #   (2026-09-15 实测: std_gate=11/28/138 结果完全一致, 55 因被换回 11 也一致。)
    std_gate = int(round(C.KDP_STD_GATE))
    if std_gate % 2 == 0:
        std_gate += 1

    # ⚠ FIR 滤波窗口 window(km): csu_kdp.get_fir 里 fir['order'] = window*1000/gs 必须**偶数**,
    #   否则 get_fir 返回 None、calc_kdp_bringi 直接返回 (None,None,None) —— KDP 全崩(无报错)。
    #   这里: 默认取 C.KDP_FIR_WINDOW(缺省 3.0), 并**强制把库数归到最近的偶数**, 再反推 window,
    #   保证两站(库长 30/75 m)下 window 都不会踩偶数坑。verbose 时打印实际生效的 window/order。
    if window is None:
        window = float(getattr(C, 'KDP_FIR_WINDOW', 3.0))
    fir_order = int(round(window * 1000.0 / C.KDP_GATE_SPACING))
    if fir_order % 2 != 0:
        fir_order += 1
    window = fir_order * C.KDP_GATE_SPACING / 1000.0
    if verbose:
        print(f'  [KDP/FIR] window={window:.3f}km({fir_order}库) std_gate={std_gate}库 '
              f'gs={C.KDP_GATE_SPACING:.0f}m')
    kd_lin, dp_lin, st_lin = csu_kdp.calc_kdp_bringi(
        dp=phi_in, dz=ref_in, rng=rng_2d, thsd=C.KDP_THSD, nfilter=C.KDP_NFILTER,
        gs=C.KDP_GATE_SPACING, window=window, std_gate=std_gate)
    kd_lin = np.ma.masked_invalid(np.ma.masked_equal(kd_lin, bad))
    dp_lin = np.ma.masked_invalid(np.ma.masked_equal(dp_lin, bad))
    radar.add_field('kdp_bringi', {
        'data': kd_lin.copy(), 'units': 'degrees/km',
        'standard_name': 'specific_differential_phase',
        'long_name': 'Specific differential phase (CSU Bringi algorithm)',
        'valid_min': -10.0, 'valid_max': 20.0,
    }, replace_existing=True)
    if verbose:
        print(f'  KDP反演完成, kd_lin 有效值: {kd_lin.count()}')

    # φdp(滤波后, 已扣系统相位) -> ZPHI 输入
    radar.add_field('phidp_bringi', {
        'data': dp_lin.copy(), 'units': 'degrees',
        'standard_name': 'differential_propagation_phase',
        'long_name': 'Filtered differential phase (system phase removed BEFORE filtering)',
        'valid_min': 0.0, 'valid_max': 360.0,
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
