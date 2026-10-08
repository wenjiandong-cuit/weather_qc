# -*- coding: utf-8 -*-
"""
QC/step01_sounding.py — 第 1 部分: 探空资料读取
=================================================================
从探空文件读温度层结并求融化层(0°C)高度, 供温度场(QC/step05_temperature.py)
与 ZPHI 衰减订正(QC/step06_attenuation.py)使用。
支持 3 种格式: 中国气象局标准 CSV / 高表14 TXT / Wyoming TXT。
"""

import os

import numpy as np
import pandas as pd


def read_sounding(sounding_path):  # 成功
    """
    读取探空数据, 返回 dict(heights_m, temps_C, pressures, fzl, format)。
    支持 CSV / 高表14 TXT / Wyoming TXT 三种格式.
    文件不存在或读取失败时抛异常, 不再回退默认零度层.

    ★ 也可以直接传本函数自己返回的那个 dict —— 原样返回, 不重复读文件。
      这样上层能做到"探空只读一次、全链共享一份"(如 process_radar(file, snd)
      与指标提取用同一个 snd)。
    """
    if isinstance(sounding_path, dict):
        return sounding_path
    if not sounding_path:
        raise ValueError("未提供探空文件路径")
    if not os.path.exists(sounding_path):
        raise FileNotFoundError(f"探空文件不存在: {sounding_path}")

    ext = os.path.splitext(sounding_path)[1].lower()

    if ext == '.csv':
        # 中国气象局标准 CSV
        df = pd.read_csv(sounding_path, encoding='utf-8-sig', dtype=str)
        layer_num = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        df = df[layer_num.notna()].copy()
        p = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        t = pd.to_numeric(df.iloc[:, 2], errors='coerce').values
        h = pd.to_numeric(df.iloc[:, 5], errors='coerce').values
        heights_m = np.asarray(h, dtype=float)
        temps_C = np.asarray(t, dtype=float)
        pressures = np.asarray(p, dtype=float)
        fmt = 'CSV'
        print(f'  CSV格式, 读取 {len(heights_m)} 行')

    elif ext == '.txt':
        # 区分高表14 / Wyoming 格式
        with open(sounding_path, 'r', encoding='gbk', errors='ignore') as f:
            lines = f.readlines()
        is_wyoming = any('PRES' in line and 'HGHT' in line for line in lines[:5])

        heights, temps, pressures = [], [], []
        if is_wyoming:
            # Wyoming: PRES(hPa) HGHT(m) TEMP(C) ...
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        p_val = float(parts[0])
                        h_val = float(parts[1])
                        t_val = float(parts[2])
                        if p_val > 0 and h_val > 0:
                            pressures.append(p_val)
                            heights.append(h_val)
                            temps.append(t_val)
                    except (ValueError, IndexError):
                        pass
            fmt = 'Wyoming'
            print(f'  Wyoming格式, 读取 {len(heights)} 层')
        else:
            # 高表14: 规定等压面层 (>=13列, 第1/6/7列分别为压/高/温)
            for line in lines:
                parts = line.split()
                if len(parts) >= 13:
                    try:
                        p_val = float(parts[0])
                        h_val = float(parts[5])
                        t_val = float(parts[6])
                        if p_val > 0 and h_val > 0:
                            pressures.append(p_val)
                            heights.append(h_val)
                            temps.append(t_val)
                    except (ValueError, IndexError):
                        pass
            fmt = '高表14'
            print(f'  高表14格式, 读取 {len(heights)} 个规定等压面层')

        heights_m = np.array(heights, dtype=float)
        temps_C = np.array(temps, dtype=float)
        pressures = np.array(pressures, dtype=float)
    else:
        raise ValueError(f"不支持的探空文件格式: {ext}")

    # 清洗: 去 NaN, 按高度排序
    valid = ~np.isnan(heights_m) & ~np.isnan(temps_C)
    heights_m = heights_m[valid]
    temps_C = temps_C[valid]
    pressures = pressures[valid] if pressures.size == valid.size else None
    order = np.argsort(heights_m)
    heights_m = heights_m[order]
    temps_C = temps_C[order]
    if pressures is not None:
        pressures = pressures[order]

    if len(heights_m) == 0:
        raise ValueError(f"探空文件 {sounding_path} 中没有有效数据")

    # 计算融化层高度(0°C穿越点); 找不到则取最接近0°C的点
    cross = np.where(np.diff(np.sign(temps_C)))[0]
    if len(cross) > 0:
        i0 = cross[0]
        fzl = heights_m[i0] - temps_C[i0] * (heights_m[i0+1] - heights_m[i0]) / (temps_C[i0+1] - temps_C[i0])
    else:
        fzl = heights_m[np.argmin(np.abs(temps_C))]

    print(f'  探空[{fmt}]: {len(heights_m)} 点, {heights_m.min():.0f}~{heights_m.max():.0f} m, 融化层 {fzl:.0f} m')
    return {'heights_m': heights_m, 'temps_C': temps_C, 'pressures': pressures, 'fzl': float(fzl), 'format': fmt}
