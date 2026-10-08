# -*- coding: utf-8 -*-
"""
QC/step09_plotting.py — 第 9 部分: 出图(统一色标 + Science 风格)
=================================================================
所有图都走 core/csu_function.py 的统一样式:
  ref_cmap_params()  统一反射率色标(0~70 dBZ, 14 色)
  science_style()    Science 风格(白底/细框/加粗字体)
  add_latlon_grid()  加粗经纬度网格
  bold_colorbar()/bold_title()  加粗色标/标题

  plot_cr()         组合反射率**一行出图**(喂 radar 即可, 自动找 CR 字段; 可叠加降雹点)
  plot_cr_compare()  Nature 风格多 panel 对比: 未订正 CR / ZPHI 订正后 CR / ΔCR
  plot_ppi()        单层 PPI(任意字段)
  plot_composite()  组合反射率(把 comp_z 塞回 ref_sweep 画 PPI, 可叠加降雹点)

投稿级导出/审计: save_pub_fig()(PNG+PDF+SVG, 文字可编辑) / nature_panel_alignment()(多 panel 对齐门)

推荐: 先 QC.composite_radar(radar) 拿到 cr_radar, 再 QC.plot_cr(cr_radar)。
标题里不放文件名(按约定)。
"""

import os
import sys as _sys

import numpy as np

import pyart
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

from .step08_composite import (sweep_indices, composite_reflectivity,
                               build_cr_radar, fill_azimuth_gaps, composite_stats)
from . import plotstyle as cs     # 绘图风格/色标(原 core/csu_function.py, 2026-09-12 搬进包内)

REF_LIKE = ('reflectivity', 'cor_z', 'total_power')


def plot_ppi(radar, field='cor_z', sweep=0, out_path=None, title=None,
             figsize=(9, 7), dpi=130, vmin=None, vmax=None):
    """
    单层 PPI(统一色标 + 加粗经纬度)。
      field: reflectivity | cor_z | pia | kdp_dual | temperature | FH ...
      out_path: 给了就存图(不显示), 不给就 plt.show()
      title: 不给则自动用 "field  sweep=n  elev=x°"
    返回 (fig, ax)
    """
    s, e = sweep_indices(radar)[sweep]
    elev = float(np.mean(radar.elevation['data'][s:e]))
    if title is None:
        title = '%s  sweep=%d  elev=%.2f°' % (field, sweep, elev)

    cs.science_style()
    cmap, norm = cs.ref_cmap_params() if field in REF_LIKE else (None, None)
    if vmin is not None or vmax is not None:
        _vmin = 0.0 if vmin is None else float(vmin)
        _vmax = 70.0 if vmax is None else float(vmax)
        if cmap is None:
            cmap = plt.get_cmap('viridis')
        norm = mcolors.BoundaryNorm(np.arange(_vmin, _vmax + 1e-6, 5.0), cmap.N)

    disp = pyart.graph.RadarMapDisplay(radar)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    disp.plot_ppi_map(ax=ax, field=field, sweep=sweep, cmap=cmap, norm=norm, title='')
    cs.add_latlon_grid(ax, nx=5, ny=4, fontsize=9, bold=True)
    cs.bold_colorbar(disp, 0, fontsize=9, label=radar.fields[field].get('units', ''))
    cs.bold_title(ax, title, fontsize=10)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'  图已保存: {out_path}')
        plt.close(fig)
    else:
        plt.show()
    return fig, ax


def plot_composite(radar, composite_z, ref_sweep=0, out_path=None,
                   hail_location=None, vmin=0, vmax=70,
                   title='Composite Reflectivity', figsize=(10, 8), dpi=150):
    """
    画组合反射率 PPI 图.
    把 composite_z 塞回 ref_sweep 的位置, 用 pyart plot_ppi_map 画图(画完还原).
      hail_location: [[lon, lat], ...] 降雹点(黑色五角星)
      out_path:      给了就存图(不显示), 不给就 plt.show()
    返回 (fig, ax)
    """
    s0, e0 = sweep_indices(radar)[ref_sweep]
    original_z = radar.fields['cor_z']['data'][s0:e0].copy()
    radar.fields['cor_z']['data'][s0:e0] = np.ma.masked_invalid(composite_z)

    cmap, norm = cs.ref_cmap_params()
    cs.science_style()                     # Science 风格(白底/细框/加粗)

    display = pyart.graph.RadarMapDisplay(radar)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    display.plot_ppi_map(
        ax=ax, field='cor_z', sweep=ref_sweep,
        cmap=cmap, norm=norm,
        title=title,
    )
    # --- Science 风格: 经纬度网格(加粗) + 加粗色标/标题 ---
    cs.add_latlon_grid(ax, nx=5, ny=4, fontsize=9, bold=True)
    cs.bold_colorbar(display, 0, fontsize=9, label='dBZ')
    cs.bold_title(ax, title, fontsize=10)

    if hail_location:
        for loc in hail_location:
            ax.plot(loc[0], loc[1], color='black', marker='*',
                    markersize=11, markeredgewidth=0.9,
                    transform=ccrs.PlateCarree())

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'  图已保存: {out_path}')
        plt.close(fig)
    else:
        plt.show()          # notebook 里直接显示(不传 out_path)

    radar.fields['cor_z']['data'][s0:e0] = original_z
    return fig, ax


# ================= 组合反射率: 一行出图 =================
def _find_cr_field(radar, field=None):
    """在 radar 里找 CR 字段: 指定就返回; 否则优先 'CR', 再退到第一个 'CR*'。"""
    if field is not None:
        if field not in radar.fields:
            raise KeyError(f'radar 里没有字段 {field!r}; 现有: {sorted(radar.fields)}')
        return field
    if 'CR' in radar.fields:
        return 'CR'
    cands = sorted(k for k in radar.fields if k.startswith('CR'))
    if cands:
        return cands[0]
    raise KeyError(
        "radar 里没有 CR 字段 —— 先算一遍: "
        "cr_radar, comp_z, _ = QC.composite_radar(radar, field='cor_z')  "
        "然后 QC.plot_cr(cr_radar)")


def plot_cr(radar, field=None, sweep=0, out_path=None, hail_location=None,
            title='Composite Reflectivity', vmin=0, vmax=70,
            figsize=(10, 8), dpi=150, range_ring_km=None):
    """
    组合反射率**一行出图**(统一色标 + Science 风格 + 可选降雹星)。

    参数
      radar:         带 CR 字段的 radar
                     - QC.composite_radar() 返回的单层 slim radar -> sweep=0 直接画(最常用)
                     - 写回了 CR 字段的原多层 radar -> sweep 传 CR 所在层(即 ref_sweep)
      field:         None -> 自动找 'CR' / 第一个 'CR*' 字段; 也可显式指定 'CR_cor_z'
      sweep:         画哪一层(默认 0)
      out_path:      给了就存图(不显示), 不给就 plt.show()
      hail_location: [[lon, lat], ...] 降雹点(黑色五角星)
      vmin/vmax:     色标范围(默认 0~70 dBZ, 5 dBZ 一档)
      range_ring_km: 给了就用 pyart 自带 plot_range_rings 画一个该半径(km)的
                     最大范围圈(典型用法: 传 radar.range['data'][-1]/1000.0)
    返回 (fig, ax)

    用法
      cr_radar, comp_z, comp_alt = QC.composite_radar(radar, field='cor_z')
      QC.plot_cr(cr_radar)                                   # notebook 里直接显示
      QC.plot_cr(cr_radar, out_path=r'D:\\...\\cr.png',
                 hail_location=[[104.0153, 27.1483]])
    """
    field = _find_cr_field(radar, field)

    cmap, _norm = cs.ref_cmap_params()
    if vmin is not None or vmax is not None:
        _vmin = 0.0 if vmin is None else float(vmin)
        _vmax = 70.0 if vmax is None else float(vmax)
        norm = mcolors.BoundaryNorm(np.arange(_vmin, _vmax + 1e-6, 5.0), cmap.N)
    else:
        norm = _norm

    cs.science_style()
    display = pyart.graph.RadarMapDisplay(radar)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    display.plot_ppi_map(ax=ax, field=field, sweep=sweep,
                         cmap=cmap, norm=norm, title='')
    cs.add_latlon_grid(ax, nx=5, ny=4, fontsize=9, bold=True)
    cs.bold_colorbar(display, 0, fontsize=9,
                     label=radar.fields[field].get('units', 'dBZ'))
    cs.bold_title(ax, title, fontsize=10)

    if hail_location:
        for loc in hail_location:
            ax.plot(loc[0], loc[1], color='black', marker='*',
                    markersize=11, markeredgewidth=0.9,
                    transform=ccrs.PlateCarree())

    if range_ring_km:
        # 最大范围圈: 直接用 pyart 自带的 plot_range_rings
        display.plot_range_rings([float(range_ring_km)], ax=ax, col='k',
                                 ls='--', lw=0.8)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f'  图已保存: {out_path}')
        plt.close(fig)
    else:
        plt.show()          # notebook 里直接显示(不传 out_path)
    return fig, ax


# =====================================================================
# Nature 风格: 未订正 CR vs ZPHI 订正后 CR(多 panel 对比图)
# =====================================================================
# 先定"图的科学结论", 再写绘图代码:
#   未订正(reflectivity)的组合反射率, 在强降水/雹暴核心的**下游**被路径衰减系统性压低;
#   ZPHI 衰减订正(cor_z)把这段量值抬回来 —— ΔCR = 订正 - 未订正, 沿径向随累积衰减增大。
#
#   panel a  未订正 CR        —— 基准/背景(标出衰减造成的低值区)
#   panel b  ZPHI 订正后 CR   —— 处理结果
#   panel c  ΔCR = b - a      —— 主 panel: 订正量的空间分布(回答"订正改变了什么、改在哪")
#
#   a/b 共用同一 0~70 dBZ 色标(才可直接比较); c 用关于 0 对称的发散色标。
#   ΔCR 只在"两侧都有真实数据"的门上计算(不参与补方位角缺口), 所以 c 里的空白
#   = 无有效数据, 不会被补缺插值污染 —— 这是差值图必须守住的一点。
# ---------------------------------------------------------------------

_CR_UN = 'CR_uncorr'        # panel a 用的字段名
_CR_COR = 'CR_attcorr'      # panel b 用的字段名
_CR_DELTA = 'CR_delta'      # panel c 用的字段名

# 投稿要求: 矢量图里的文字必须可编辑(不能变成路径/位图), 背景必须白
_NATURE_RC = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans',
                        'Microsoft YaHei', 'SimHei', 'Noto Sans SC'],
    'pdf.fonttype': 42,        # TrueType 内嵌, PDF 里文字可编辑
    'ps.fonttype': 42,
    'svg.fonttype': 'none',    # SVG 里保留 <text>, 不转路径
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
}

_CR_COMPARE_LABELS = {
    'en': {
        'a': 'Composite reflectivity (uncorrected)',
        'b': 'Composite reflectivity (ZPHI-corrected)',
        'c': 'Difference (corrected − uncorrected)',
        'cbar': 'Composite reflectivity (dBZ)',
        'dcbar': 'Difference (dB)',
    },
    'zh': {
        'a': '组合反射率(未订正)',
        'b': '组合反射率(ZPHI 衰减订正)',
        'c': 'ΔCR(订正 − 未订正)',
        'cbar': '组合反射率 (dBZ)',
        'dcbar': 'ΔCR (dB)',
    },
}


def _cr_display_extent(radar, ref_sweep=0, margin=0.03, extent=None):
    """显示范围(经纬度)。三个 panel 共用同一 extent, 绘图区纵横比才完全一致。"""
    if extent is not None:
        return tuple(float(v) for v in extent)

    s, e = sweep_indices(radar)[ref_sweep]
    lon0 = lon1 = lat0 = lat1 = None
    glon = getattr(radar, 'gate_longitude', None)      # pyart 预计算的每门经纬度
    glat = getattr(radar, 'gate_latitude', None)
    if glon is not None and glat is not None:
        try:
            lo = np.ma.filled(np.asarray(glon['data'][s:e], dtype=float), np.nan)
            la = np.ma.filled(np.asarray(glat['data'][s:e], dtype=float), np.nan)
            if np.isfinite(lo).any() and np.isfinite(la).any():
                lon0, lon1 = float(np.nanmin(lo)), float(np.nanmax(lo))
                lat0, lat1 = float(np.nanmin(la)), float(np.nanmax(la))
        except Exception:
            lon0 = None
    if lon0 is None:                                   # 兜底: 中心点 + 距离/方位角推
        lat_c = float(radar.latitude['data'][0])
        lon_c = float(radar.longitude['data'][0])
        rng = np.asarray(radar.range['data'], dtype=float) / 1000.0          # km
        az = np.deg2rad(np.asarray(radar.azimuth['data'][s:e], dtype=float))
        dy = np.cos(az)[:, None] * rng[None, :]
        dx = np.sin(az)[:, None] * rng[None, :]
        lat0 = lat_c + float(dy.min()) / 111.195
        lat1 = lat_c + float(dy.max()) / 111.195
        k = 111.320 * max(abs(np.cos(np.deg2rad(lat_c))), 1e-6)
        lon0 = lon_c + float(dx.min()) / k
        lon1 = lon_c + float(dx.max()) / k

    pad = margin * max(lon1 - lon0, lat1 - lat0)
    return lon0 - pad, lon1 + pad, lat0 - pad, lat1 + pad


def _add_cr_field(radar, name, arr, long_name):
    """往单层 slim radar 里写一个 CR 字段(形状必须与那一层完全一致)。"""
    data = np.ma.masked_invalid(np.asarray(arr, dtype=np.float32))
    if data.shape != (radar.nrays, radar.ngates):
        raise ValueError('字段 %s 形状 %s 与 radar (%d, %d) 不匹配'
                         % (name, data.shape, radar.nrays, radar.ngates))
    radar.add_field(name, {
        'data': data, 'units': 'dBZ',
        'standard_name': 'equivalent_reflectivity_factor',
        'long_name': long_name,
    }, replace_existing=True)
    return name


def save_pub_fig(fig, out_path, formats=('png', 'pdf', 'svg'), dpi=600, verbose=True):
    """
    投稿级导出: 白底 + 可编辑文本(见 _NATURE_RC), 一次出多种格式。
      formats: 想出的后缀; 位图走 dpi, 矢量图不受 dpi 影响
    返回实际写出的文件路径列表。
    """
    base = os.path.splitext(str(out_path))[0]
    d = os.path.dirname(base)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    written = []
    for fmt in formats:
        p = base + '.' + str(fmt).lstrip('.')
        fig.savefig(p, dpi=dpi, bbox_inches='tight', facecolor='white')
        written.append(p)
        if verbose:
            try:
                print('  已导出: %s (%d KB)' % (p, os.path.getsize(p) // 1024))
            except OSError:            # 只影响这行提示, 不要因为读大小而中断
                print('  已导出: %s' % p)
    return written


def nature_scripts_dir():
    """nature-figure 技能的 scripts 目录; 可用环境变量 NATURE_FIGURE_SCRIPTS 覆盖。"""
    d = os.environ.get('NATURE_FIGURE_SCRIPTS')
    if d and os.path.isdir(d):
        return d
    cand = os.path.join(os.path.expanduser('~'), '.workbuddy', 'skills',
                        'nature-figure', 'scripts')
    return cand if os.path.isdir(cand) else None


def nature_panel_alignment(fig, json_out=None, overlay_svg=None,
                           tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                           row_groups=None, exclude_axes=(),
                           require_panel_labels=False,
                           strict=False, verbose=True):
    """
    跑 nature-figure 的多 panel **绘图区对齐门**: 量出每个 panel 最终渲染后的
    plot-area 矩形 -> 审计行/列是否等宽等高、间隔是否一致 -> 出 JSON/SVG 报告。

    找不到技能脚本或缺依赖时只提示、不打断出图(返回 None)。
      strict=True 时判定 FAIL 会抛异常 —— 投稿前用它卡住导出。
    """
    d = nature_scripts_dir()
    if d is None:
        if verbose:
            print('  [nature-qa] 未找到 nature-figure 脚本目录, 跳过对齐审计')
        return None
    if d not in _sys.path:
        _sys.path.insert(0, d)
    try:
        from audit_panel_alignment import (require_matplotlib_panel_alignment,
                                           exit_code, PanelAlignmentError)
    except Exception as exc:
        if verbose:
            print('  [nature-qa] 无法加载对齐审计脚本: %s' % exc)
        return None

    opts = {}
    if row_groups:
        opts['row_groups'] = row_groups
    if exclude_axes:
        opts['exclude_axes'] = list(exclude_axes)

    try:
        report = require_matplotlib_panel_alignment(
            fig, json_out=json_out, overlay_svg=overlay_svg,
            tolerance_pt=tolerance_pt, gutter_tolerance_pt=gutter_tolerance_pt,
            require_panel_labels=require_panel_labels, strict=strict, **opts)
    except PanelAlignmentError as exc:
        # 报告在抛异常之前已经落盘了, 这里把 verdict 一起说清楚
        verdict = ''
        if json_out and os.path.isfile(json_out):
            try:
                with open(json_out, encoding='utf-8') as fh:
                    import json as _json
                    verdict = _json.load(fh).get('verdict', '')
            except Exception:
                verdict = ''
        print('  [nature-qa] 对齐审计未通过%s%s:\n%s'
              % (' [%s]' % verdict if verdict else '',
                 '' if not json_out else ' | 报告: %s' % os.path.basename(json_out),
                 exc))
        if strict:
            raise
        return None

    if verbose:
        s = report.get('summary', {})
        print('  [nature-qa] 对齐审计 exit=%d | 可比组=%s fail=%s warn=%s%s'
              % (exit_code(report, strict=strict), s.get('comparisons'),
                 s.get('fail'), s.get('warn'),
                 '' if not json_out else ' | 报告: %s' % os.path.basename(json_out)))
    return report


def plot_cr_compare(radar, ref_sweep=0,
                    uncorrected='reflectivity', corrected='cor_z',
                    hail_location=None, out_path=None,
                    vmin=0.0, vmax=70.0, delta_vmax=None,
                    fill_gaps=True, show_delta=True,
                    extent=None, panel_labels=('a', 'b', 'c'),
                    lang='en', show_stats=True, title_panels=True,
                    fig_width_in=7.2, gap_in=0.09, base_fontsize=6.0,
                    embellish=True, emit=('png', 'pdf', 'svg'), dpi=600,
                    nature_qa=True, qa_strict=False,
                    close=False, verbose=True):
    """
    普通组合反射率 vs 衰减订正后组合反射率 —— Nature 风格多 panel 对比图。

    一图回答一个问题: **衰减订正把组合反射率改了多少、改在哪里**。

      panel a  未订正 CR('reflectivity' 组合)
      panel b  ZPHI 订正后 CR('cor_z' 组合)
      panel c  ΔCR = b - a(仅两侧都有真实数据的门; 白色=无有效数据)

    参数
      radar:          处理好的多层 radar(需同时含 uncorrected / corrected 两个字段)
      ref_sweep:      组合反射率所属层(默认 0), 也是三图的显示层
      uncorrected:    未订正字段名(默认 'reflectivity')
      corrected:      订正后字段名(默认 'cor_z')
      out_path:       给了就按 emit 导出(自动建目录); 不给则只在窗口/notebook 显示
      hail_location:  [[lon, lat], ...] 降雹点(黑色五角星)
      vmin/vmax:      a/b 共用色标范围(默认 0~70 dBZ, 5 dBZ 一档)
      delta_vmax:     Δ 色标半幅(dB); None=按数据 p99 自动, 取整
      fill_gaps:      a/b 是否补方位角缺口(默认 True, 与项目其它图一致)
      show_delta:     False 只出 a/b 两栏
      extent:         显式显示范围 (lon0, lon1, lat0, lat1); None=按雷达门位置自动+3% 边距
      panel_labels:   panel 字母
      lang:           'en' / 'zh' 面板标题与色标文字
      fig_width_in:   画布宽度(inch); 7.2 = Nature 双栏 183 mm
      emit:           导出格式
      nature_qa:      是否跑 nature-figure 的对齐门(找不到脚本会静默跳过)
      qa_strict:      True 时对齐 FAIL 直接抛异常

    返回 (fig, axs)。图对象保留着, notebook 里可直接继续改。

    提示
      - 图里用的是**未补缺**的两版 CR 算 Δ, 所以 c 与 a/b 的空白位置可能略有差别,
        这是刻意的: 差值只在两侧都有真实数据时才有意义。
      - a/b 共用一根色标(可比较); c 单独一根, 关于 0 对称(红=订正抬升)。
    """
    for f in (uncorrected, corrected):
        if f not in radar.fields:
            raise KeyError('radar 里没有字段 %r; 现有: %s' % (f, sorted(radar.fields)))
    lab = _CR_COMPARE_LABELS.get(lang) or _CR_COMPARE_LABELS['en']

    # ---- 1) 两版 CR: 都用"未补缺"的原始结果, Δ 才只落在两侧都有真实数据的门上 ----
    z_un, _alt_un = composite_reflectivity(radar, field=uncorrected, ref_sweep=ref_sweep,
                                           fill_gaps=False, write=False, verbose=False)
    z_co, _alt_co = composite_reflectivity(radar, field=corrected, ref_sweep=ref_sweep,
                                           fill_gaps=False, write=False, verbose=False)
    if np.shape(z_un) != np.shape(z_co):
        raise ValueError('两版组合反射率形状不一致: %s vs %s'
                         % (np.shape(z_un), np.shape(z_co)))
    z_un = np.ma.masked_invalid(np.asarray(z_un, dtype=float))
    z_co = np.ma.masked_invalid(np.asarray(z_co, dtype=float))
    dz = np.ma.masked_invalid(z_co - z_un)         # 任一侧无效 -> 自动变 masked

    if fill_gaps:                                  # 只影响 a/b 的显示; Δ 不补
        z_un = np.ma.masked_invalid(
            fill_azimuth_gaps(np.ma.filled(z_un, np.nan).copy())[0])
        z_co = np.ma.masked_invalid(
            fill_azimuth_gaps(np.ma.filled(z_co, np.nan).copy())[0])

    # ---- 2) 只含 ref_sweep 那层 + 三个字段的 slim radar(形状与字段完全吻合) ----
    slim = build_cr_radar(radar, z_co, ref_sweep=ref_sweep,
                          field_name=_CR_COR, verbose=False)
    _add_cr_field(slim, _CR_UN, z_un,
                  'Composite CR from %s (uncorrected)' % uncorrected)
    _add_cr_field(slim, _CR_DELTA, dz,
                  'Composite CR difference (%s - %s)' % (corrected, uncorrected))

    panels = [(lab['a'], _CR_UN), (lab['b'], _CR_COR)]
    if show_delta:
        panels.append((lab['c'], _CR_DELTA))
    ncol = len(panels)
    labels = [panel_labels[i] for i in range(ncol)]

    # ---- 3) 显示范围 + 版面: 由纵横比反推画布高度, 保证三栏绘图区等宽等高 ----
    lon0, lon1, lat0, lat1 = _cr_display_extent(slim, ref_sweep=0, extent=extent)
    aspect = (lon1 - lon0) / max(lat1 - lat0, 1e-9)

    base_fontsize = float(base_fontsize)
    left_in, right_in = 0.46, 0.07
    cbar_h_in, cbar_gap_in, cbar_lab_in, top_in = 0.10, 0.10, 0.26, 0.26
    panel_w = (float(fig_width_in) - left_in - right_in
               - (ncol - 1) * gap_in) / ncol
    panel_h = panel_w / aspect
    fig_h = cbar_lab_in + cbar_h_in + cbar_gap_in + panel_h + top_in

    cs.science_style(base_fontsize)
    plt.rcParams.update(_NATURE_RC)

    fig = plt.figure(figsize=(float(fig_width_in), fig_h),dpi=150)
    panel_y0 = (cbar_lab_in + cbar_h_in + cbar_gap_in) / fig_h
    axs = []
    for i in range(ncol):
        ax = fig.add_axes([(left_in + i * (panel_w + gap_in)) / fig_width_in,
                           panel_y0, panel_w / fig_width_in, panel_h / fig_h],
                          projection=ccrs.PlateCarree())
        ax.set_aspect('equal', adjustable='box')   # 等度纵横比: 不做地理拉伸
        ax.set_gid(labels[i])                      # 供对齐审计识别 panel id
        axs.append(ax)

    # ---- 4) 色标: a/b 共用 0~70 dBZ; Δ 关于 0 对称(红=订正抬升) ----
    cmap_ref, _ = cs.ref_cmap_params()
    norm_ref = mcolors.BoundaryNorm(np.arange(vmin, vmax + 1e-6, 5.0), cmap_ref.N)

    if delta_vmax is None:
        vals = np.asarray(dz.compressed(), dtype=float)
        vals = vals[np.isfinite(vals)]
        dmax = max(1.0, float(np.ceil(np.nanpercentile(np.abs(vals), 99)))) \
            if vals.size else 1.0
    else:
        dmax = abs(float(delta_vmax))
    cmap_d = plt.get_cmap('RdBu_r')
    try:
        norm_d = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax)
    except Exception:                              # 老版本 matplotlib 兜底
        norm_d = mcolors.Normalize(vmin=-dmax, vmax=dmax)

    # ---- 5) 画 panel ----
    disp = pyart.graph.RadarMapDisplay(slim)
    for i, (ptitle, fld) in enumerate(panels):
        ax = axs[i]
        is_delta = (fld == _CR_DELTA)
        disp.plot_ppi_map(ax=ax, field=fld, sweep=0,
                          cmap=cmap_d if is_delta else cmap_ref,
                          norm=norm_d if is_delta else norm_ref,
                          colorbar_flag=False, title_flag=False,
                          add_grid_lines=False, embellish=embellish)
        ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
        cs.add_latlon_grid(ax, nx=3, ny=3, fontsize=max(base_fontsize - 1, 5.0),
                           bold=True, grid_lw=0.4, grid_alpha=0.55)
        if title_panels:
            ax.set_title(ptitle, fontsize=base_fontsize, fontweight='bold', pad=3)
        if hail_location:
            for loc in hail_location:
                ax.plot(loc[0], loc[1], color='black', marker='*', markersize=5.5,
                        markeredgewidth=0.6, transform=ccrs.PlateCarree(), zorder=5)
        # panel 字母(左上角在地图圆外, 不会压住回波)
        ax.text(0.012, 0.988, labels[i], transform=ax.transAxes, ha='left', va='top',
                fontsize=base_fontsize + 2, fontweight='bold', color='black', zorder=6)

        if show_stats:
            if is_delta:
                v = np.asarray(dz.compressed(), dtype=float)
                txt = ('mean %+.1f   p95 %+.1f dB' % (v.mean(), np.percentile(v, 95))
                       if v.size else 'no valid data')
            else:
                v = np.asarray((z_un if fld == _CR_UN else z_co).compressed(), dtype=float)
                txt = ('max %.1f dBZ   n=%s' % (v.max(), format(v.size, ','))
                       if v.size else 'no valid data')
            ax.text(0.012, 0.935, txt, transform=ax.transAxes, ha='left', va='top',
                    fontsize=max(base_fontsize - 0.5, 5.0), color='black', zorder=6)

    # ---- 6) 横向色标: a/b 共用一根; Δ 单独一根(不同量纲, 不能共用) ----
    cbar_axes = []

    def _hbar(x0_in, w_in, cmap, norm, label, ticks=None):
        cax = fig.add_axes([x0_in / fig_width_in, cbar_lab_in / fig_h,
                            w_in / fig_width_in, cbar_h_in / fig_h])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                          cax=cax, orientation='horizontal')
        cb.set_label(label, fontsize=base_fontsize, fontweight='bold', labelpad=2)
        if ticks is not None:
            cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=max(base_fontsize - 1, 5.0),
                          length=1.8, width=0.5, pad=1.5)
        cb.outline.set_linewidth(0.5)
        for t in cb.ax.get_xticklabels():
            t.set_fontweight('bold')
        cbar_axes.append(cax)
        return cb

    # a/b 共用一根: 有 Δ 时只横跨前两栏, 否则横跨全部
    span_ref = (2 * panel_w + gap_in) if show_delta \
        else (ncol * panel_w + (ncol - 1) * gap_in)
    _hbar(left_in, span_ref, cmap_ref, norm_ref, lab['cbar'],
          ticks=np.arange(vmin, vmax + 1e-6, 10.0))
    if show_delta:
        _hbar(left_in + 2 * (panel_w + gap_in), panel_w,
              cmap_d, norm_d, lab['dcbar'], ticks=[-dmax, 0.0, dmax])

    # ---- 7) 数值摘要(图里的统计量也在终端复现一遍, 便于核对/写文) ----
    if verbose:
        print('  CR 对比 (基准 sweep=%d):' % ref_sweep)
        for tag, arr in (('未订正 %s' % uncorrected, z_un),
                         ('订正后 %s' % corrected, z_co)):
            st = composite_stats(arr)
            print('    %-26s max %.1f dBZ   mean %.1f   n=%s'
                  % (tag, st['max_dBZ'], st['mean_dBZ'], format(st['n_valid'], ',')))
        dv = np.asarray(dz.compressed(), dtype=float)
        if dv.size:
            print('    %-26s mean %+.2f   p50 %+.2f   p95 %+.2f   max %+.2f dB'
                  % ('ΔCR(订正-未订正)', dv.mean(), np.percentile(dv, 50),
                     np.percentile(dv, 95), dv.max()))
        print('    Δ 色标: 关于 0 对称 ±%.0f dB' % dmax)

    # ---- 8) Nature 对齐门(多 panel 必须过) + 导出 ----
    if nature_qa:
        base = os.path.splitext(str(out_path))[0] if out_path else None
        nature_panel_alignment(
            fig,
            json_out=(base + '.alignment.json') if base else None,
            overlay_svg=(base + '.alignment.svg') if base else None,
            row_groups=[labels] if ncol >= 2 else None,
            exclude_axes=cbar_axes, strict=qa_strict, verbose=verbose)

    if out_path:
        save_pub_fig(fig, out_path, formats=emit, dpi=dpi, verbose=verbose)
    elif not close:
        plt.show()
    if close:
        plt.close(fig)
    return fig, axs


# ======================================================================
# 多版本衰减订正并排对比(第一行 CR, 第二行 ΔCR)
# ======================================================================
def plot_cr_compare_multi(radar, ref_sweep=0,
                          uncorrected='reflectivity',
                          corrected_fields=(('ZPHI', 'cor_z'),),
                          hail_location=None, out_path=None,
                          vmin=0.0, vmax=70.0, delta_vmax=None,
                          fill_gaps=True, show_delta_row=True,
                          extent=None, lang='en', show_stats=True,
                          fig_width_in=7.2, gap_in=0.09, row_gap_in=0.30,
                          base_fontsize=6.0, embellish=True,
                          emit=('png', 'pdf', 'svg'), dpi=600,
                          nature_qa=False, qa_strict=False,
                          close=False, verbose=True):
    """
    多版本衰减订正并排对比 —— Nature 风格两行多列图。

      第一行  未订正 CR + 每个订正版的 CR      (共用 0~70 dBZ 色标)
      第二行  每个订正版的 ΔCR = 订正 − 未订正  (共用关于 0 对称的色标)

    典型用法(对比 pyart ZPHI 与论文版 park2005 的修正前后):

        from QC.step09_plotting import plot_cr_compare_multi
        plot_cr_compare_multi(radar, ref_sweep=0, lang='en',
                              corrected_fields=[('ZPHI (pyart)', 'cor_z'),
                                                ('Park2005 (fixed)', 'cor_z_park'),
                                                ('Park2005 (pre-fix)', 'cor_z_park_old')])

    参数
      radar:             处理好的多层 radar
      uncorrected:       未订正字段(默认 'reflectivity')
      corrected_fields:  [(面板标题, 字段名), ...] 多个订正版本, 按顺序排列
      show_delta_row:    False 只出第一行 CR
      delta_vmax:        ΔCR 色标半幅(dB); None = 按所有版本的 |Δ| p99 自动
      其余参数同 plot_cr_compare(色标/范围/字号/导出/nature_qa 等)

    返回 (fig, axs), axs 顺序 = 第一行各列 + 第二行(订正版对应的 Δ 列)。
    """
    # ---- 1) 每个版本算 CR(不补缺, Δ 才只落在两侧都有真实数据的门上) ----
    for _lab, f in corrected_fields:
        if f not in radar.fields:
            raise KeyError('radar 里没有字段 %r; 现有: %s' % (f, sorted(radar.fields)))
    if uncorrected not in radar.fields:
        raise KeyError('radar 里没有字段 %r' % (uncorrected,))

    z_un, _ = composite_reflectivity(radar, field=uncorrected, ref_sweep=ref_sweep,
                                     fill_gaps=False, write=False, verbose=False)
    z_un = np.ma.masked_invalid(np.asarray(z_un, dtype=float))

    vers = []                     # (label, field, z, dz)
    for lab_v, f in corrected_fields:
        z, _ = composite_reflectivity(radar, field=f, ref_sweep=ref_sweep,
                                      fill_gaps=False, write=False, verbose=False)
        z = np.ma.masked_invalid(np.asarray(z, dtype=float))
        if z.shape != z_un.shape:
            raise ValueError('组合反射率形状不一致: %r %s vs %r %s'
                             % (f, z.shape, uncorrected, z_un.shape))
        vers.append((lab_v, f, z, np.ma.masked_invalid(z - z_un)))

    # ---- 2) 显示用副本(补方位角缺口; 只影响图, 不影响 Δ) ----
    def _disp(a):
        if not fill_gaps:
            return a
        return np.ma.masked_invalid(fill_azimuth_gaps(np.ma.filled(a, np.nan).copy())[0])

    d_un = _disp(z_un)
    d_vers = [_disp(v[2]) for v in vers]

    # ---- 3) slim radar: 一层 + 所有 CR/Δ 字段(形状完全吻合) ----
    cor_names = ['CR_fixed_%d' % i for i in range(len(vers))]
    dlt_names = ['CR_delta_%d' % i for i in range(len(vers))]
    slim = build_cr_radar(radar, d_vers[0], ref_sweep=ref_sweep,
                          field_name=cor_names[0], verbose=False)
    _add_cr_field(slim, _CR_UN, d_un, 'Composite CR from %s (uncorrected)' % uncorrected)
    for i, (lab_v, f, z, dz) in enumerate(vers):
        _add_cr_field(slim, cor_names[i], d_vers[i],
                      'Composite CR from %s (%s)' % (f, lab_v))
        _add_cr_field(slim, dlt_names[i], dz,
                      'Composite CR difference (%s - %s)' % (f, uncorrected))

    # ---- 4) 面板标题 + 字母 ----
    lab_en = (lang != 'zh')
    titles_top = [('Uncorrected CR (%s)' % uncorrected) if lab_en
                  else ('组合反射率(未订正: %s)' % uncorrected)]
    titles_top += [(('%s corrected CR' % v[0]) if lab_en
                    else ('%s 订正后 CR' % v[0])) for v in vers]
    titles_bot = [(('ΔCR  %s' % v[0]) if lab_en else ('ΔCR  %s' % v[0])) for v in vers]
    ncol = 1 + len(vers)
    nrow = 2 if show_delta_row else 1
    n_pan = ncol + (len(vers) if show_delta_row else 0)
    letters = [chr(ord('a') + k) for k in range(n_pan)]
    labels = letters

    # ---- 5) 版面: 由纵横比反推画布高度, 两行绘图区等宽等高 ----
    lon0, lon1, lat0, lat1 = _cr_display_extent(slim, ref_sweep=0, extent=extent)
    aspect = (lon1 - lon0) / max(lat1 - lat0, 1e-9)

    base_fontsize = float(base_fontsize)
    left_in, right_in = 0.46, 0.07
    cbar_h_in, cbar_gap_in, cbar_lab_in, top_in = 0.10, 0.10, 0.26, 0.26
    panel_w = (float(fig_width_in) - left_in - right_in
               - (ncol - 1) * gap_in) / ncol
    panel_h = panel_w / aspect
    rows_h = nrow * panel_h + (nrow - 1) * row_gap_in
    fig_h = cbar_lab_in + cbar_h_in + cbar_gap_in + rows_h + top_in

    cs.science_style(base_fontsize)
    plt.rcParams.update(_NATURE_RC)
    fig = plt.figure(figsize=(float(fig_width_in), fig_h), dpi=150)

    def _x0(i):
        return (left_in + i * (panel_w + gap_in)) / fig_width_in

    def _y0(row):                      # row: 0=上(CR), 1=下(Δ)
        from_bottom = cbar_lab_in + cbar_h_in + cbar_gap_in + (nrow - 1 - row) * (panel_h + row_gap_in)
        return from_bottom / fig_h

    axs = []
    # 第一行: 未订正 + 各订正版
    for i in range(ncol):
        ax = fig.add_axes([_x0(i), _y0(0), panel_w / fig_width_in, panel_h / fig_h],
                          projection=ccrs.PlateCarree())
        ax.set_aspect('equal', adjustable='box')
        ax.set_gid(labels[i])
        axs.append(ax)
    # 第二行: 各订正版的 Δ(与上一行对应列对齐)
    axs_delta = []
    if show_delta_row:
        for j in range(len(vers)):
            ax = fig.add_axes([_x0(j + 1), _y0(1), panel_w / fig_width_in, panel_h / fig_h],
                              projection=ccrs.PlateCarree())
            ax.set_aspect('equal', adjustable='box')
            ax.set_gid(labels[ncol + j])
            axs_delta.append(ax)
            axs.append(ax)

    # ---- 6) 色标: CR 共用 0~vmax; Δ 关于 0 对称(红=订正抬升) ----
    cmap_ref, _ = cs.ref_cmap_params()
    norm_ref = mcolors.BoundaryNorm(np.arange(vmin, vmax + 1e-6, 5.0), cmap_ref.N)
    all_dz = [np.asarray(v[3].compressed(), dtype=float) for v in vers]
    all_dz = np.concatenate([a[np.isfinite(a)] for a in all_dz]) if all_dz else np.array([])
    if delta_vmax is None:
        dmax = max(1.0, float(np.ceil(np.nanpercentile(np.abs(all_dz), 99)))) \
            if all_dz.size else 1.0
    else:
        dmax = abs(float(delta_vmax))
    cmap_d = plt.get_cmap('RdBu_r')
    try:
        norm_d = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-dmax, vmax=dmax)
    except Exception:
        norm_d = mcolors.Normalize(vmin=-dmax, vmax=dmax)

    # ---- 7) 画 panel ----
    disp = pyart.graph.RadarMapDisplay(slim)
    grid_fs = max(base_fontsize - 1, 5.0)
    stat_fs = max(base_fontsize - 0.5, 5.0)

    def _draw(ax, fld, title, letter, is_delta, stat_txt):
        disp.plot_ppi_map(ax=ax, field=fld, sweep=0,
                          cmap=cmap_d if is_delta else cmap_ref,
                          norm=norm_d if is_delta else norm_ref,
                          colorbar_flag=False, title_flag=False,
                          add_grid_lines=False, embellish=embellish)
        ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
        cs.add_latlon_grid(ax, nx=3, ny=3, fontsize=grid_fs,
                           bold=True, grid_lw=0.4, grid_alpha=0.55)
        ax.set_title(title, fontsize=base_fontsize, fontweight='bold', pad=3)
        if hail_location:
            for loc in hail_location:
                ax.plot(loc[0], loc[1], color='black', marker='*', markersize=5.5,
                        markeredgewidth=0.6, transform=ccrs.PlateCarree(), zorder=5)
        ax.text(0.012, 0.988, letter, transform=ax.transAxes, ha='left', va='top',
                fontsize=base_fontsize + 2, fontweight='bold', color='black', zorder=6)
        if show_stats and stat_txt:
            ax.text(0.012, 0.935, stat_txt, transform=ax.transAxes, ha='left', va='top',
                    fontsize=stat_fs, color='black', zorder=6)

    def _txt_max(a):
        v = np.asarray(a.compressed(), dtype=float)
        return ('max %.1f dBZ   n=%s' % (v.max(), format(v.size, ','))) if v.size else 'no valid data'

    def _txt_delta(a):
        v = np.asarray(a.compressed(), dtype=float)
        return ('mean %+.1f   p95 %+.1f   max %+.1f dB'
                % (v.mean(), np.percentile(v, 95), v.max())) if v.size else 'no valid data'

    _draw(axs[0], _CR_UN, titles_top[0], letters[0], False, _txt_max(z_un))
    for j in range(len(vers)):
        _draw(axs[j + 1], cor_names[j], titles_top[j + 1], letters[j + 1], False,
              _txt_max(vers[j][2]))
    if show_delta_row:
        for j in range(len(vers)):
            _draw(axs_delta[j], dlt_names[j], titles_bot[j], letters[ncol + j], True,
                  _txt_delta(vers[j][3]))

    # ---- 8) 底部两根横向色标: 左=CR, 右=Δ ----
    cbar_axes = []
    total_w = float(fig_width_in) - left_in - right_in
    cb_w = 0.46 * total_w

    def _hbar(x0_in, w_in, cmap, norm, label, ticks=None):
        cax = fig.add_axes([x0_in / fig_width_in, cbar_lab_in / fig_h,
                            w_in / fig_width_in, cbar_h_in / fig_h])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                          cax=cax, orientation='horizontal')
        cb.set_label(label, fontsize=base_fontsize, fontweight='bold', labelpad=2)
        if ticks is not None:
            cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=grid_fs, length=1.8, width=0.5, pad=1.5)
        cb.outline.set_linewidth(0.5)
        for t in cb.ax.get_xticklabels():
            t.set_fontweight('bold')
        cbar_axes.append(cax)
        return cb

    _hbar(left_in, cb_w, cmap_ref, norm_ref,
          'Composite reflectivity (dBZ)' if lab_en else '组合反射率 (dBZ)',
          ticks=np.arange(vmin, vmax + 1e-6, 10.0))
    if show_delta_row:
        _hbar(left_in + total_w - cb_w, cb_w, cmap_d, norm_d,
              'ΔCR (corrected − uncorrected, dB)' if lab_en else 'ΔCR (订正 − 未订正, dB)',
              ticks=[-dmax, 0.0, dmax])

    # ---- 9) 数值摘要(终端复现, 便于写文/核对) ----
    if verbose:
        st_un = composite_stats(z_un)
        print('  CR 多版本对比 (基准 sweep=%d):' % ref_sweep)
        print('    %-28s max %6.1f dBZ   mean %5.1f   n=%s'
              % ('未订正 %s' % uncorrected, st_un['max_dBZ'], st_un['mean_dBZ'],
                 format(st_un['n_valid'], ',')))
        for lab_v, f, z, dz in vers:
            st = composite_stats(z)
            dv = np.asarray(dz.compressed(), dtype=float)
            dv = dv[np.isfinite(dv)]
            print('    %-28s max %6.1f dBZ   mean %5.1f   n=%-10s | ΔCR mean %+.2f '
                  'p50 %+.2f p95 %+.2f max %+.2f dB'
                  % ('%s (%s)' % (lab_v, f), st['max_dBZ'], st['mean_dBZ'],
                     format(st['n_valid'], ','),
                     dv.mean() if dv.size else np.nan,
                     np.percentile(dv, 50) if dv.size else np.nan,
                     np.percentile(dv, 95) if dv.size else np.nan,
                     dv.max() if dv.size else np.nan))
        print('    Δ 色标: 关于 0 对称 ±%.0f dB' % dmax)

    # ---- 10) 对齐审计(多行时按行分组) + 导出 ----
    if nature_qa:
        top_labels = labels[:ncol]
        bot_labels = labels[ncol:]
        row_groups = [top_labels] + ([bot_labels] if show_delta_row else [])
        base = os.path.splitext(str(out_path))[0] if out_path else None
        nature_panel_alignment(
            fig,
            json_out=(base + '.alignment.json') if base else None,
            overlay_svg=(base + '.alignment.svg') if base else None,
            row_groups=row_groups, exclude_axes=cbar_axes, strict=qa_strict,
            verbose=verbose)

    if out_path:
        save_pub_fig(fig, out_path, formats=emit, dpi=dpi, verbose=verbose)
    elif not close:
        plt.show()
    if close:
        plt.close(fig)
    return fig, axs
