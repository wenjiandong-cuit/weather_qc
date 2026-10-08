# -*- coding: utf-8 -*-
"""
QC/plotstyle.py — 绘图风格与色标工具
=================================================================
2026-09-12 从 core/csu_function.py 搬进 QC/(内容未改动), 让处理链与出图风格
都在一个文件夹里。原来的 core/csu_function.py 与 notebooks/csu_function.py
内容与本文件完全一致, 旧代码 from csu_function import ... 仍然可用。

主要函数:
    ref_cmap_params()            反射率统一色标
    science_style()              Science 期刊风格(白底/细框/加粗)
    add_latlon_grid(ax, ...)     经纬度网格
    bold_colorbar(display, ...)  加粗色标
    bold_title(ax, ...)          加粗标题
    add_field_to_radar_object()  把算好的数组加成 radar 字段
    two_panel_plot()             两联 PPI
    adjust_fhc_colorbar_for_pyart()  FHC 色标
    print_fields(radar)          列出字段
    calculate_angle(...)         辅助
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pyart


def add_field_to_radar_object(field, radar, field_name='FH', units='unitless', 
                              long_name='Hydrometeor ID', standard_name='Hydrometeor ID',
                              dz_field='cor_z'):
    """
    Adds a newly created field to the Py-ART radar object. If reflectivity is a masked array,
    make the new field masked the same as reflectivity.
    """
    fill_value = -32768
    masked_field = np.ma.asanyarray(field)
    masked_field.mask = masked_field == fill_value
    if hasattr(radar.fields[dz_field]['data'], 'mask'):
        setattr(masked_field, 'mask', 
                np.logical_or(masked_field.mask, radar.fields[dz_field]['data'].mask))
        fill_value = radar.fields[dz_field]['_FillValue']
    field_dict = {'data': masked_field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    radar.add_field(field_name, field_dict, replace_existing=True)
    return radar

def two_panel_plot(radar, sweep=0, var1='reflectivity', vmin1=0, vmax1=70,
                   cmap1=None, units1='dBZ', var2='differential_reflectivity',
                   vmin2=-5, vmax2=5, cmap2='RdYlBu_r', units2='dB', return_flag=False,
                   xlim=[-150,150], ylim=[-150,150]):
    if cmap1 is None:
        cmap1, _ = ref_cmap_params()      # 统一反射率色标
    display = pyart.graph.RadarDisplay(radar)
    fig = plt.figure(figsize=(13,5),dpi=150)
    ax1 = fig.add_subplot(121)
    display.plot_ppi(var1, sweep=sweep, vmin=vmin1, vmax=vmax1, cmap=cmap1, 
                     colorbar_label=units1, mask_outside=True)
    display.set_limits(xlim=xlim, ylim=ylim)
    ax2 = fig.add_subplot(122)
    display.plot_ppi(var2, sweep=sweep, vmin=vmin2, vmax=vmax2, cmap=cmap2, 
                     colorbar_label=units2, mask_outside=True)
    display.set_limits(xlim=xlim, ylim=ylim)
    if return_flag:

        return fig, ax1, ax2, display

def adjust_fhc_colorbar_for_pyart(cb):
    cb.set_ticks(np.arange(1.4, 10, 0.9))
    cb.ax.set_yticklabels(['Drizzle', 'Rain', 'Ice Crystals', 'Aggregates',
                           'Wet Snow', 'Vertical Ice', 'LD Graupel',
                           'HD Graupel', 'Hail', 'Big Drops'])
    cb.ax.set_ylabel('')
    cb.ax.tick_params(length=0)
    return cb

def print_fields(radar):
    print('Radar fields:')
    for key in radar.fields.keys():
        print(f'  {key}')

def calculate_angle(x,y):
    """
    Calculate the angle in degrees between the vector (x,y) and the positive x-axis.
    The angle is measured counterclockwise from the positive x-axis.
    """
    angle_rad = np.arctan2(y, x)  # Angle in radians
    angle_deg = np.degrees(angle_rad)  # Convert to degrees
    return -(angle_deg-90) if angle_deg<0 else angle_deg

def ref_cmap_params():
    Z_color = np.array([
    [210, 235, 255],
    [0, 255, 255],
    [2, 149, 246],
    [1, 2, 250],
    [0, 255, 3],
    [0, 150, 0],
    [248, 254, 0],
    [253, 201, 11],
    [253, 123, 1],
    [253, 1, 0],
    [212, 0, 8],
    [149, 0, 2],
    [191, 0, 255],
    [150, 0, 250]
    ]) / 255

    cmap = colors.ListedColormap(Z_color, name='my_cmap')

    boundaries = np.arange(0, 75, 5)   # [0,5,10,...,75]
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


# ==================== 出图风格(统一: 经纬度 + 加粗 + Science 风格) ====================
def science_style(base_fontsize=9):
    """Science 期刊风格: 白底、细框、加粗标题/轴标注, 无网格杂色

    字体: Arial/DejaVu Sans 打底(英文), 并挂中文字体兜底(Microsoft YaHei/SimHei/Noto Sans SC),
          避免标题里的中文变成方框乱码; unicode_minus=False 避免负号变方框。

    ★★ 2026-09-18 F 盘批量版修正(重要, 别再改回 'sans-serif') ★★
      原来只写 'font.family': 'sans-serif' —— matplotlib 会把字体**解析成
      font.sans-serif 列表里第一个能用的族(Arial)**, 然后只用它; 所谓"逐字回退"在
      走 'sans-serif' 这个别名时**并不生效**。结果: Arial 没有中文字形 ->
      标题里的中文全部渲染成方框(□), 实测缺字告警 8~9 条。
      改成**显式给一族列表** [Arial, Microsoft YaHei, ...] 后, matplotlib 才真的按
      字逐字回退: 西文用 Arial(保持原来的 Science 观感), 中文自动落到微软雅黑。
      实测同一句中英混排: 缺字告警 8 条 -> 0 条。
    """
    plt.rcParams.update({
        'font.family': ['Arial', 'Microsoft YaHei', 'SimHei', 'Noto Sans SC',
                        'DejaVu Sans'],
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Microsoft YaHei', 'SimHei',
                            'Noto Sans SC', 'Arial Unicode MS'],
        'axes.unicode_minus': False,        # 负号正常显示(不出现方框)
        'font.size': base_fontsize,
        'axes.titlesize': base_fontsize + 1,
        'axes.titleweight': 'bold',
        'axes.labelsize': base_fontsize,
        'axes.labelweight': 'bold',
        'xtick.labelsize': base_fontsize - 1,
        'ytick.labelsize': base_fontsize - 1,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.linewidth': 0.8,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'grid.linewidth': 0.5,
        'legend.frameon': False,
        'savefig.facecolor': 'white',
    })


# ---- 经纬度网格: 刻度取值 + 标注格式(2026-09-15 新增, 配合自绘标签) ----
_GRID_STEPS = (0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0)


def grid_nice_values(a, b, want, steps=_GRID_STEPS):
    """在 [a, b] 里挑一列"整"刻度: 取使刻度个数最接近 want 的那个步长。

    例: 26.51~27.59 / want=4 -> [26.75, 27.0, 27.25, 27.5]
    """
    a, b = float(a), float(b)
    if not np.isfinite([a, b]).all() or b <= a:
        return np.array([])
    best, best_gap = None, None
    for s in steps:
        vals = np.arange(np.ceil(a / s - 1e-9) * s, b + 1e-9, s)
        if len(vals) < 2:
            continue
        gap = abs(len(vals) - int(want))
        if best_gap is None or gap < best_gap:
            best, best_gap = vals, gap
    if best is None:
        best = np.array([a, b])
    return np.round(np.asarray(best, dtype=float), 6)


def fmt_deg(v, axis='E', dms=False):
    """经纬度文字: 104.25 + 'E' -> "104.25\u00b0E" (dms=True 时 "104\u00b015'E"); 负值自动换 W/S。"""
    hemi = axis if v >= 0 else ('W' if axis == 'E' else 'S')
    av = abs(float(v))
    if dms:
        d = int(av)
        m = int(round((av - d) * 60))
        if m == 60:
            d, m = d + 1, 0
        return "%d\u00b0%02d'%s" % (d, m, hemi)
    return '%g\u00b0%s' % (av, hemi)


def add_latlon_grid(ax, nx=5, ny=4, fontsize=9, bold=True,
                    grid_lw=0.5, grid_alpha=0.5, grid_color='0.35',
                    grid_ls='--', dms=False, native_labels=False):
    """给地图轴(cartopy GeoAxes)加经纬度网格线并标注; 标注默认加粗

    nx / ny : x/y 方向**大致**刻度个数(自动挑"整"步长: 0.05/0.1/0.2/0.25/0.5/1...)
    dms     : True 用度分格式(如 104\u00b015'E), False 用度(如 104.25\u00b0E)
    native_labels : True 切回 cartopy 原生标签(有下述 bug, 默认 False)

    *** 为什么标注要自己画(2026-09-15 定案, 别再改回去) ***
    cartopy 0.25 + matplotlib 3.11 下, 只要给 Gridliner 设 top_labels=False,
    GeoAxes 的 tight bbox 就会变成非有限值 -> savefig(bbox_inches='tight')
    会把这个"坏掉"的地图轴整个裁掉: 图里只剩右侧色标(实测 133x856 px 的假图),
    **地图框、经纬网、标题、降雹星全部消失**。同版本环境的对照实验:
        gridlines(draw_labels=True)                      -> 地图在(839 px)
        + right_labels=False                             -> 地图在
        + top_labels=False                               -> 地图丢(48 px)  <== 元凶
    本实现: 网格线还是交给 cartopy(裁剪随投影正确), 但**标签用 ax.text 自己画**在
    下边/左边, 完全不碰 Gridliner 的标签机制, 因此不受该 bug 影响。
    """
    import cartopy.crs as _ccrs
    pc = _ccrs.PlateCarree()
    try:
        x0, x1, y0, y1 = ax.get_extent(crs=pc)
    except Exception:
        return None

    if native_labels:                       # 旧写法(有 bug, 一般不用)
        return _native_latlon_grid(ax, nx, ny, fontsize, bold,
                                   grid_lw, grid_alpha, grid_color, grid_ls, dms)

    lons = grid_nice_values(x0, x1, nx)
    lats = grid_nice_values(y0, y1, ny)

    # 1) 网格线: cartopy gridliner 画(不带标签); 万一失败退回普通直线
    gl = None
    try:
        gl = ax.gridlines(xlocs=lons, ylocs=lats, draw_labels=False,
                          linewidth=grid_lw, color=grid_color,
                          alpha=grid_alpha, linestyle=grid_ls)
        try:
            gl.zorder = 4                   # 别被 pcolormesh 盖住
        except Exception:
            pass
    except Exception:
        gl = None
    if gl is None:
        for lo in lons:
            ax.plot([lo, lo], [y0, y1], transform=pc, color=grid_color, lw=grid_lw,
                    alpha=grid_alpha, ls=grid_ls, zorder=4)
        for la in lats:
            ax.plot([x0, x1], [la, la], transform=pc, color=grid_color, lw=grid_lw,
                    alpha=grid_alpha, ls=grid_ls, zorder=4)

    # 2) 标注: 下边经度 + 左边纬度(加粗, 画在框外)
    w = 'bold' if bold else 'normal'
    for lo in lons:
        ax.text(lo, y0, fmt_deg(lo, 'E', dms), transform=pc, ha='center', va='top',
                fontsize=fontsize, fontweight=w, color='black',
                zorder=6, clip_on=False)
    for la in lats:
        ax.text(x0, la, fmt_deg(la, 'N', dms), transform=pc, ha='right', va='center',
                fontsize=fontsize, fontweight=w, color='black',
                zorder=6, clip_on=False)

    try:
        ax.spines['geo'].set_linewidth(0.8)
    except Exception:
        pass
    return gl


def _native_latlon_grid(ax, nx=5, ny=4, fontsize=9, bold=True,
                        grid_lw=0.5, grid_alpha=0.5, grid_color='0.35',
                        grid_ls='--', dms=False):
    """旧的 cartopy 原生标签写法(保留但不默认使用): 见 add_latlon_grid 里的说明。
    ⚠ 它设了 top_labels=False -> 在 cartopy 0.25 + matplotlib 3.11 上,
       savefig(bbox_inches='tight') 会把整个地图轴裁掉, 只剩色标。
    """
    import cartopy.mpl.gridliner as _gridliner
    from matplotlib import ticker as _mticker

    gl = ax.gridlines(draw_labels=True, linewidth=grid_lw, color=grid_color,
                      alpha=grid_alpha, linestyle=grid_ls, dms=dms,
                      x_inline=False, y_inline=False, auto_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = _gridliner.LONGITUDE_FORMATTER
    gl.yformatter = _gridliner.LATITUDE_FORMATTER
    w = 'bold' if bold else 'normal'
    gl.xlabel_style = {'size': fontsize, 'weight': w, 'color': 'black'}
    gl.ylabel_style = {'size': fontsize, 'weight': w, 'color': 'black'}
    try:
        gl.xlocator = _mticker.MaxNLocator(nx)
        gl.ylocator = _mticker.MaxNLocator(ny)
    except Exception:
        pass
    for attr, val in (('xpadding', 6), ('ypadding', 6), ('rotate_labels', False)):
        try:
            setattr(gl, attr, val)
        except Exception:
            pass
    try:
        gl.zorder = 4
    except Exception:
        pass
    try:
        ax.spines['geo'].set_linewidth(0.8)
    except Exception:
        pass
    return gl


def bold_colorbar(display, index=0, fontsize=9, label=None):
    """把 pyart display 的色标标题/刻度加粗"""
    try:
        cb = display.cbs[index]
    except Exception:
        return None
    try:
        cb.set_label(label if label is not None else cb.get_label(),
                     fontweight='bold', fontsize=fontsize)
        for t in cb.ax.get_yticklabels():
            t.set_fontweight('bold')
            t.set_fontsize(fontsize - 1)
        cb.outline.set_linewidth(0.8)
    except Exception:
        pass
    return cb


def cjk_font_family():
    """
    带回退的字体族列表: 西文走 Arial, 中文自动落到微软雅黑(实测缺字 0 条)。

    ★ 2026-09-18 F 盘批量版新增 —— 用来**显式**指定给标题。

    为什么必须显式给, 光改 rcParams 不够(实测确认):
      pyart 的 plot_ppi_map 作图时会把坐标轴标题这个 Text 对象的 FontProperties
      **固化**成当时 rcParams 的值(即 science_style() 调用之前的 'sans-serif')。
      之后再调 science_style() 改 rcParams, rcParams 确实变了, 但这张图的标题
      仍然拿着旧的 ['sans-serif'] -> matplotlib 解析成 Arial -> 中文全成方框(□)。
      实测插桩: rcParams['font.family'] = ['Arial','Microsoft YaHei',...] 同时
                ax.title.get_fontproperties().get_family() = ['sans-serif']。
      => 只能在建标题时把 family 直接传进去, 绕开这个固化的对象。
    """
    return ['Arial', 'Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']


def bold_title(ax, text=None, fontsize=10, family=None):
    """
    加粗标题(不改文字就只改字重)。

    family: 显式字体族(默认用 cjk_font_family(), 保证中文不乱码);
            传 None 以外的自定义列表就按你给的来。
    """
    try:
        ax.set_title(text if text is not None else ax.get_title(),
                     fontweight='bold', fontsize=fontsize,
                     fontfamily=family or cjk_font_family())
    except Exception:
        pass
