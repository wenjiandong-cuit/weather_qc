# -*- coding: utf-8 -*-
"""
QC/wjd_cmaps.py — 除反射率以外的色标（ZDR / KDP / 相关系数）
=================================================================
颜色和分档间隔都取自张伟超师姐的程序：
    D:\\PAR_radar\\张伟超师姐程序\\雷达原始基数据即未质量控制数据直接读取画图\\plot_section_nowind.py
      · ZDR  -> plot_profile_uvw_ZDR  里的 colors / levels
      · KDP  -> plot_profile_uvw_kdp  里的 colors / levels
      · ROHV -> plot_profile_uvw_ROHV 里的 colors / levels

导入时自动把三个色标注册进 matplotlib，之后就能按名字取用
（和 cmweather 注册 wjd_ref / RefDiff / CM_rhohv 是一个路子）：

    import QC.wjd_cmaps
    cmap = matplotlib.colormaps['wjd_zdr']
    norm = QC.wjd_cmaps.get_norm('wjd_zdr')      # 按师姐的分档间隔

    WJD_COLORS['wjd_zdr']   颜色列表
    WJD_LEVELS['wjd_zdr']   分档边界
    get_norm(name)          BoundaryNorm（clip=True，与师姐一致）
"""

import matplotlib
import matplotlib.colors as mcolors

# ================= 色标定义（顺序即分档顺序，别改）=================

# ---- ZDR：13 色 / 13 档  -1 ~ 7.5 dB ----
WJD_ZDR_COLORS = ['#008c8f', '#02a8aa', '#00c8c6', '#00e6e7', '#dbff00', '#ffff01',
                  '#fedc00', '#ffb400', '#ff8201', '#ff5000', '#fe0000', '#c91401',
                  '#800000']
WJD_ZDR_LEVELS = [-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 7.5]

# ---- KDP：14 色 / 14 档  -2 ~ 5.5 °/km ----
WJD_KDP_COLORS = ['#6c6c6c', '#008c8c', '#00aaaa', '#00c8c8', '#00e6e6', '#dcff00',
                  '#ffff00', '#ffdc00', '#ffb400', '#ff8200', '#ff5000', '#ff0000',
                  '#c81400', '#800000']
WJD_KDP_LEVELS = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]

# ---- 相关系数：19 色 / 18 档  0.4 ~ 0.998 ----
WJD_RHOHV_COLORS = ['#0001fa', '#0551f7', '#00aaff', '#00d7fc', '#00fff9', '#00d885',
                    '#08a707', '#02d105', '#0eef0e', '#7eff01', '#f7ff03', '#ffd600',
                    '#ffa900', '#ff7c02', '#ff5700', '#fd2f00', '#f60408', '#fe007c',
                    '#fc03f2']
WJD_RHOHV_LEVELS = [0.400, 0.500, 0.600, 0.700, 0.750, 0.800, 0.850, 0.900,
                    0.920, 0.930, 0.950, 0.960, 0.970, 0.975, 0.980, 0.985,
                    0.990, 0.995, 0.998]

WJD_COLORS = {
    'wjd_zdr': WJD_ZDR_COLORS,
    'wjd_kdp': WJD_KDP_COLORS,
    'wjd_rhohv': WJD_RHOHV_COLORS,
}
WJD_LEVELS = {
    'wjd_zdr': WJD_ZDR_LEVELS,
    'wjd_kdp': WJD_KDP_LEVELS,
    'wjd_rhohv': WJD_RHOHV_LEVELS,
}
WJD_NAMES = list(WJD_COLORS)


# ================= 注册 / 取用 =================

def register(force=True, verbose=True):
    """把三个色标注册进 matplotlib，返回 {名字: ListedColormap}。

    force=True 时覆盖同名旧色标（重跑 notebook 不会报"已存在"）。
    """
    out = {}
    for name in WJD_NAMES:
        cmap = mcolors.ListedColormap(WJD_COLORS[name], name=name)
        if name in matplotlib.colormaps:
            if not force:
                out[name] = matplotlib.colormaps[name]
                continue
            matplotlib.colormaps.unregister(name)
        matplotlib.colormaps.register(cmap)
        out[name] = cmap
        if verbose:
            print('注册色标 %-10s %2d 色 / %2d 档'
                  % (name, cmap.N, len(WJD_LEVELS[name]) - 1))
    return out


def get_cmap(name):
    """按名字取色标（没注册就先注册）。"""
    if name not in matplotlib.colormaps:
        register()
    return matplotlib.colormaps[name]


def get_norm(name):
    """按师姐的分档间隔取 BoundaryNorm（clip=True，与师姐一致）。"""
    cmap = get_cmap(name)
    return mcolors.BoundaryNorm(WJD_LEVELS[name], ncolors=cmap.N, clip=True)


# 导入即注册，notebook 里 import 完就能用
CMAPS = register()
