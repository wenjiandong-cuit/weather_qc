# -*- coding: utf-8 -*-
"""
QC/step05_temperature.py — 第 5 部分: 温度场(探空插值到雷达库点)
=================================================================
用探空的温度层结按**库点海拔高度**线性插值, 得到每个库点的环境温度,
落进字段 'temperature'。ZPHI 衰减订正(fzl/temp_ref)与 FHC 分类都用它,
所以必须在衰减订正之前生成。
超出探空高度范围的门置 nan(不参与分类)。
"""

import numpy as np


def add_temperature_field(radar, snd, verbose=True):
    """
    snd: QC/step01_sounding.read_sounding() 的返回 dict
    返回 fzl_for_corr(融化层高度, m)
    """
    gate_alt = np.ma.filled(radar.gate_altitude['data'], np.nan)
    heights_m = snd['heights_m']; temps_C = snd['temps_C']
    fzl_for_corr = snd['fzl']
    if verbose:
        print(f'  融化层高度: {fzl_for_corr:.0f} m')
    gaf = gate_alt.ravel()
    in_range = (gaf >= heights_m.min()) & (gaf <= heights_m.max())
    temp_flat = np.interp(gaf, heights_m, temps_C)
    temp_flat[~in_range] = np.nan
    radar.add_field('temperature', {
        'data': np.ma.masked_invalid(temp_flat.reshape(gate_alt.shape)),
        'units': 'degrees_C', 'standard_name': 'air_temperature',
        'long_name': 'Temperature from sounding interpolated to radar gates',
    }, replace_existing=True)
    return fzl_for_corr
