# -*- coding: utf-8 -*-
"""
QC/step07_classification.py — 第 7 部分: FHC 水凝物分类
=================================================================
用 CSU 夏季模糊逻辑(CSU_HIDRO / csu_fhc_summer)对每个库点分类:
    7 = 低密度霰, 8 = 高密度霰(两者都算霰), 9 = 冰雹(hail), 10 = 大滴;
    其余为雨/干雪/冰晶等。完整对照见下面的 FHC_CATEGORIES。
输入: cor_z(订正后反射率) / cor_zdr / ρhv / kdp_dual / temperature
      ★ 2026-10-07 起 KDP 字段名由 kdp_bringi 改为 kdp_dual(算法换成 zch_phi 双尺度)
输出: 字段 'FH'(Hydrometeor ID)

注意: 用的是**衰减订正后**的 cor_z/cor_zdr, 所以必须在衰减订正之后调用。
"""

import numpy as np

import csu_radartools.csu_fhc as csu_fhc

from . import config as C
from . import plotstyle as cs     # 绘图风格/色标(原 core/csu_function.py, 2026-09-12 搬进包内)

# csu_fhc_summer 返回的类别编号是 **1..10**(源码里 argmax+1), 不是 0..9:
#   1 Drizzle / 2 Rain / 3 Ice Crystals / 4 Aggregates / 5 Wet Snow
#   6 Vertical Ice / 7 Low-Density Graupel / 8 High-Density Graupel / 9 Hail / 10 Big Drops
FHC_CATEGORIES = {
    1: 'Drizzle', 2: 'Rain', 3: 'Ice Crystals', 4: 'Aggregates', 5: 'Wet Snow',
    6: 'Vertical Ice', 7: 'Low-Density Graupel', 8: 'High-Density Graupel',
    9: 'Hail', 10: 'Big Drops',
}
FHC_HAIL = (9,)          # 冰雹
FHC_GRAUPEL = (7, 8)     # 霰(低密度 + 高密度)


def fhc_counts(radar, field='FH'):
    """统计 FH 各类别的门数, 返回 {类别编号: 门数}"""
    fh = radar.fields[field]['data']
    vals = np.ma.filled(np.ma.masked_invalid(fh), -1).astype(np.int32).ravel()
    keys, cnt = np.unique(vals, return_counts=True)
    return {int(k): int(c) for k, c in zip(keys, cnt) if k > 0}


def add_fhc_field(radar, verbose=True):
    """
    就地添加 'FH' 字段并返回 radar。
    """
    dz = radar.fields['cor_z']['data']
    dr = radar.fields['cor_zdr']['data']
    rho = radar.fields['cross_correlation_ratio']['data']
    kdp = radar.fields['kdp_dual']['data']
    radar_T = radar.fields['temperature']['data']

    fh = csu_fhc.csu_fhc_summer(dz=dz, zdr=dr, rho=rho, kdp=kdp, use_temp=True, band='X', T=radar_T)

    # csu_function.add_field_to_radar_object 会读 cor_z 字段里的 '_FillValue'
    # (pyart 的 add_field 不会自动补这个键), 这里先补齐, 否则 KeyError: '_FillValue'
    if '_FillValue' not in radar.fields['cor_z']:
        radar.fields['cor_z']['_FillValue'] = C.ATTEN_BAD_FILL

    radar = cs.add_field_to_radar_object(fh, radar)
    if verbose:
        print(f'  FHC 完成, FH 已添加')
    return radar
