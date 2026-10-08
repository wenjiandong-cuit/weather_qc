# -*- coding: utf-8 -*-
"""
metric 指标

QC/step10_metrics.py — 第 10 部分: 指标提取(框选 → 掩膜 → 各项指标)
=================================================================
原先在 core/renwu.py 里的"指标提取层", 2026-09-12 搬进 QC/ 包, 让处理链与指标
都待在一个文件夹里, 不用再跨目录导入。

指标清单(从图中识别):
  (1) 最大回波强度                      metric_max_z
  (2) 18dBZ / 45dBZ 回波顶高            metric_echo_top(threshold=18 / 45)
  (3) >=45dBZ 回波体积(库数 + 占比)     metric_echo_count
  (4) 垂直累积液态水含量(VIL)           metric_vil / compute_vil_field
      (网格化 + Greene&Clark1972 逐列积分, 2026-09-13 新增)
  (5) 冰雹粒子 + 霰粒子数量(需先做 FHC) metric_hail_graupel_count
  (6) 最大回波强度出现 30dBZ 时最强回波高度 metric_zmax_height
      (框内 >=30dBZ 的库点里最强回波所在高度, 2026-09-13 新增)
  (7) ZDR 柱                             metric_zdr_column / compute_zdr_column_field
      (Zhao et al. 2025 "3D mapping columns" 法: 阈值 + 柱状连通 + 向上非增;
       自己网格化一次, 与 VIL 分开(VIL 产物不带 ZDR); 2026-09-14 新增。
       进统计表只三项: 柱内最大 ZDR / 柱顶高度 / 柱顶距 0 °C 层高度;
       高度范围 [融化层-1km, 混合相区顶]: 0 °C 层优先用**探空 fzl**, 上界(-20 °C)直接查
       探空原始廓线 —— 传 snd=QC.read_sounding(...) 即可, 不受雷达库点高度限制)
  (8) (1)-(4) 变化速率                  all_metrics_series / metric_trends
      (多时次同框差分 -> 每分钟速率, 2026-09-13 新增)

用法(都从这个文件直接取, 不用再进 core/):
    from QC import (select_box_ginput, build_box_mask_3d,
                    metric_max_z, metric_echo_top, metric_echo_count,
                    metric_hail_graupel_count, metric_vil)

    radar = QC.process_radar(FILE, SOUNDING)       # 或 prepare_radar_products
    snd = QC.read_sounding(SOUNDING)               # 推荐: 探空给指标(7) 定 0 °C 层与 -20 °C 上界
    # ★ 弹窗里画的是**组合反射率 CR**(各仰角取 max) —— 用户要求, 不是最低层 PPI;
    #   想退回画某一层 PPI 就传 composite=False(那时 sweep 才是"画哪一层")。
    lon1, lat1, lon2, lat2 = select_box_ginput(radar, sweep=0, field='cor_z')
    box = build_box_mask_3d(radar, lon1, lat1, lon2, lat2)
    print(metric_max_z(radar, box))
    print(metric_echo_top(radar, box, threshold=45.0))
    vil_max, vil_mean = metric_vil(radar, box)     # VIL(kg/m²), 默认 55 dBZ 封顶
    print(metric_zdr_column(radar, box, snd=snd))  # ZDR 柱(需要 cor_zdr); 不传 snd 也能跑
    #  想换成老文献阈值/手给融化层: metric_zdr_column(radar, box, zdr_min_db=1.0,
    #                                                z0c_m=snd['fzl'])

    # 指标(8): 多时次变化速率 —— 主流程: 每个文件 ginput 框选 -> 框内指标 -> 差分
    series = QC.all_metrics_series_ginput(radar_list)   # 逐文件弹图框选(带 time/box)
    rates = QC.metric_trends(series)                    # 相邻时次差分, 每分钟速率
    QC.print_trends(rates)                              # 含 VIL 最陡增幅
    best_v, best_i = QC.max_trend(rates)                # 单取 VIL 最大增大速率
    QC.save_metrics_series(series, 'D:/cases/metrics_20250509.csv')           # 序列可落盘
    rates = QC.metric_trends(QC.load_metrics_series('D:/cases/metrics_20250509.csv'))
    # 变体: 想严格同框(框选一次复用到所有时次):
    #   固定经纬度: QC.all_metrics_series(radar_list, lon1, lat1, lon2, lat2)
    #   JSON 存框:  QC.select_and_save_box(...) + QC.all_metrics_series_from_file(...)

注: 框选靠鼠标点图, 只能在 notebook / 有图形界面的脚本里用。
    弹窗底图 = **组合反射率 CR**(各仰角 max, 2026-09-15 起), 不再是某一层的 PPI。
    批量跑(无界面)时自己给定 lon/lat 四个数, 直接调 build_box_mask_3d 即可。
"""

import contextlib
import os
import shutil
import tempfile

import numpy as np

from . import config as C


# =============================================================================
# 通用: NetCDF 路径兜底 —— Windows 上 netCDF4/HDF5 打不开非 ASCII 路径
# =============================================================================

@contextlib.contextmanager
def _nc_path(path, for_write=False):
    """
    把"可能含中文的路径"换成 netCDF4 能打开的 ASCII 路径, 用完再搬回去。

    实测(本机 netCDF4 1.7.4 / HDF5 2.1.0, 2026-09-15):
        F:/hail_growth_cases/.../result_out/x.nc  -> PermissionError: [Errno 13]
        F:/_nctest_ascii/x.nc                        -> 写成功
      同一中文目录下, 文件名用中文还是英文**都一样失败** -> 是**目录**的问题, 与文件名无关。
      而 CSV / Excel 能写进同一个目录(Python 的 open 走 Windows UTF-16 API, 不受影响),
      所以这个坑只在 netCDF4 / pyart.io.write_grid / read_grid 这条路上出现。

    用法(上下文管理器, with 里的变量一定是 ASCII 路径):
        with _nc_path(path, for_write=True) as p:
            pyart.io.write_grid(p, grid)     # 写: 先落临时 ASCII 目录, 收尾搬回目标
        with _nc_path(path) as p:
            grid = pyart.io.read_grid(p)     # 读: 先把目标拷到临时 ASCII 目录

    路径本来就是 ASCII 时原样返回, 不碰临时目录(零开销)。
    """
    p = os.fspath(path)
    if str(p).isascii():
        yield p
        return

    tmp_dir = tempfile.mkdtemp(prefix='qc_nc_')
    tmp = os.path.join(tmp_dir, 'swap.nc')
    try:
        if not for_write:
            shutil.copy2(p, tmp)              # 读: 先拷一份成 ASCII 路径
        yield tmp
    finally:
        try:
            if for_write and os.path.exists(tmp):
                d = os.path.dirname(os.path.abspath(p))
                if d:
                    os.makedirs(d, exist_ok=True)
                shutil.move(tmp, p)           # 写: 收尾搬回原(可能是中文的)路径
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def read_grid_nc(path):
    """
    读回 save_grid_nc 存的网格 .nc —— 内部走 _nc_path 兜底, 中文路径也能读。

    ★ 别直接用 pyart.io.read_grid 读中文路径的 .nc: 会 PermissionError(原因见 _nc_path)。
        grid = QC.read_grid_nc(r'F:/hail_growth_cases/.../vilgrid_xxx.nc')
    """
    import pyart

    with _nc_path(path, for_write=False) as p:
        return pyart.io.read_grid(p)


# =============================================================================
# Step 1: 框选
# =============================================================================

def _mark_box_and_save(fig, ax, lon1, lat1, lon2, lat2, out_path, dpi=150,
                       mark_box=True, cs=None):
    """
    【F 盘批量版新增 2026-09-18】在 ginput 那张 CR 图上画出框选矩形 + 标注经纬度范围, 存 PNG。

    存的**就是刚点完的那张图**(不是另画一张) -> 所见即所得, 事后能回头核查框选是否合理。

    画法: 红色矩形(先垫一条白色粗线) —— 在彩色 CR 底图上, 深色区和浅色区都看得清。
    用 PlateCarree 变换保证矩形与底图地理坐标一致; 失败则退回无变换直画(PlateCarree 的
    数据坐标本身就是经纬度, 差别只在投影畸变)。
    """
    if mark_box:
        xs = [lon1, lon2, lon2, lon1, lon1]
        ys = [lat1, lat1, lat2, lat2, lat1]
        # ★ 2026-09-18: light 渲染用的是**普通坐标轴**(不走 cartopy), 不能传 transform。
        #   普通轴的数据坐标本身就是经纬度, 直接画即可 —— 与地图版的差别只有投影畸变,
        #   对"看框落在哪"没有影响。这里先判断轴类型, 再决定要不要带投影变换。
        kw = {}
        try:
            from cartopy.mpl.geoaxes import GeoAxes as _GeoAxes
            if isinstance(ax, _GeoAxes):
                import cartopy.crs as _ccrs
                kw['transform'] = _ccrs.PlateCarree()
        except Exception:
            pass
        try:
            ax.plot(xs, ys, color='white', linewidth=3.4, alpha=0.8,
                    zorder=20, clip_on=False, **kw)   # 打底, 提对比
            ax.plot(xs, ys, color='red', linewidth=1.7,
                    zorder=21, clip_on=False, **kw)
        except Exception:
            ax.plot(xs, ys, color='red', linewidth=1.7, zorder=21, clip_on=False)

        # 标题: 保留 CR 那一行(丢掉"请点击…"的交互提示), 第二行写框的范围。
        #   两行刚好; 三行会顶到地图上边框。并排写会太长被右侧色标压住/裁掉。
        if cs is not None:
            first = (ax.get_title() or '').split('\n')[0]
            new = ((first + '\n') if first else '') + (
                f'框选范围  经度 [{lon1:.4f}, {lon2:.4f}]   纬度 [{lat2:.4f}, {lat1:.4f}]')
            cs.bold_title(ax, new, fontsize=10)

    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    print(f'  -> 框选图已保存: {out_path}')


def _plot_ppi_light(ax, fig, radar, field, sweep, cs, decimate=1):
    """
    【F 盘批量版新增 2026-09-18】轻量绘制 PPI: 直接用 pcolormesh 画在**普通坐标轴**上,
    **完全不碰 cartopy 投影 / 地理底图**。

    为什么要有这条路:
      原来走 pyart 的 plot_ppi_map —— 每次都要建 GeoAxes、初始化 PlateCarree 投影、
      往上面挂地理要素、再把极坐标数据重采样到投影网格。定框只需要"看回波长什么样",
      地理底图对判断框的位置没有额外帮助(经纬度网格已经够用), 但那套投影开销每次都要付。
      改成本函数后: 不 import cartopy、不建投影、直接 pcolormesh -> 弹窗明显更快。

    关键性质: **轴的数据坐标仍然是经度/纬度** —— 所以 plt.ginput() 的返回值、
    以及 _mark_box_and_save 画红框的坐标, 与地图版**完全一致**, 上下游无需改动。

    decimate: 抽稀倍数(1=不抽稀)。定框看形状不需要全分辨率, 抽 2 倍 -> 网格数降到 1/4,
              渲染更快; 抽稀后每个网格只是变大, 不会出现空洞。
    """
    import numpy as np

    # --- 取这一层的字段 + 每个库点的经纬度 ---
    sl = radar.get_slice(sweep)
    data = np.ma.asarray(radar.fields[field]['data'][sl], dtype=float)
    glon = np.asarray(radar.gate_longitude['data'][sl], dtype=float)
    glat = np.asarray(radar.gate_latitude['data'][sl], dtype=float)
    data = np.ma.masked_invalid(data)

    if decimate and int(decimate) > 1:
        d = int(decimate)
        glon = glon[::d, ::d]
        glat = glat[::d, ::d]
        data = data[::d, ::d]

    cmap, norm = cs.ref_cmap_params()
    # ★ edgecolors='face': 让每个网格的边跟自己的填充同色。
    #   极坐标数据在 pcolormesh 里是"曲面四边形", 相邻格之间会露出细细的白缝,
    #   弱回波区看上去像一圈圈辐条(实测很明显)。描边同色就把缝填上了。
    #   antialiased=False: 几十万个网格开抗锯齿会明显变慢, 定框图不需要。
    mesh = ax.pcolormesh(glon, glat, data, cmap=cmap, norm=norm,
                         shading='nearest', antialiased=False,
                         edgecolors='face', linewidth=0.2, rasterized=True)

    # --- 范围 / 纵横比 ---
    lon0, lon1 = float(np.nanmin(glon)), float(np.nanmax(glon))
    lat0, lat1 = float(np.nanmin(glat)), float(np.nanmax(glat))
    pad_lon = (lon1 - lon0) * 0.02
    pad_lat = (lat1 - lat0) * 0.02
    ax.set_xlim(lon0 - pad_lon, lon1 + pad_lon)
    ax.set_ylim(lat0 - pad_lat, lat1 + pad_lat)
    # 经纬度要按纬度做长宽比修正, 否则图会被拉扁(威宁 ~26.9°N, cos≈0.89)
    ax.set_aspect(1.0 / max(0.2, float(np.cos(np.deg2rad(0.5 * (lat0 + lat1))))))

    # --- 经纬度刻度: 与地图版同款"整"步长 + 度数标注 ---
    try:
        xt = cs.grid_nice_values(lon0, lon1, 5)
        yt = cs.grid_nice_values(lat0, lat1, 4)
        if len(xt):
            ax.set_xticks(xt)
            ax.set_xticklabels([cs.fmt_deg(v, 'E') for v in xt])
        if len(yt):
            ax.set_yticks(yt)
            ax.set_yticklabels([cs.fmt_deg(v, 'N') for v in yt])
    except Exception:
        pass
    ax.grid(True, color='0.35', linewidth=0.5, alpha=0.5, linestyle='--')
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_fontweight('bold')
        t.set_fontsize(8)

    # --- 色标 ---
    #   label 用字段名(+单位), 与 pyart 地图版一致('CR (dBZ)') —— 免得新旧存档图
    #   同一批个例里色标文字不一样, 回看时对不上。
    units = str(radar.fields[field].get('units') or '')
    lab = str(field)
    cb = fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.94)
    try:
        cb.set_label(f'{lab} ({units})' if units else lab,
                     fontweight='bold', fontsize=9)
        for t in cb.ax.get_yticklabels():
            t.set_fontweight('bold')
            t.set_fontsize(8)
        cb.outline.set_linewidth(0.8)
    except Exception:
        pass
    return mesh


def select_box_ginput(radar, sweep=0, field='cor_z', composite=True, verbose=True,
                      fig_save_path=None, fig_dpi=150, mark_box=True, title=None,
                      render='map', decimate=1):
    """
    Step 1: 画**组合反射率** + 鼠标点两点框选 -> 返回 (lon1, lat1, lon2, lat2)

    左上角 (lon1, lat1), 右下角 (lon2, lat2), 保证 lat1 >= lat2。

    ★ 2026-09-15 起(用户要求"框选窗口里要组合反射率, 不要第一层"):
      弹窗里画的是 **CR = 各仰角取 max 的二维组合反射率**(投影到单层),
      **不再是某一层的 PPI** —— 框选时看到的就是最大回波, 不会被某个低层切面误导。
      CR 走 QC.composite_radar 同一条路(方位角插值对齐 + 补方位角缺口),
      此时 `sweep` 只当"方位角基准层 / CR 所属层"(默认 0, 一般不用改);
      `field` 是**参与组合的源字段**(默认 'cor_z' 订正后反射率)。

    composite=False -> 退回老行为: 画 `field` 在 `sweep` 那一层的 PPI。

    依赖 matplotlib 的 ginput —— 需要交互式后端(notebook / GUI)。
    函数内部会**主动弹窗**(fig.show() + plt.pause), 所以脚本里 python xxx.py 也能点选;
    非 GUI 后端下弹不出窗, ginput 会一直卡住(调用方应先判后端, 见 debug_flow.check_gui_backend)。

    ★ F 盘批量版新增参数(2026-09-18):
      fig_save_path: 给了就在点完两点后, **把框画到这张图上并存成 PNG**(不另画一张),
                     路径自行给全(含文件名)。存图失败只告警, 不影响返回的框。
      fig_dpi:       存图 dpi(默认 150)。
      mark_box:      False 则只存原图, 不画红色矩形。
      title:         自定义标题(含 \n 可写多行); None = 用默认标题。
                     ★ 供"分阶段批量"用: 先批量处理并把 CR 存成单层 slim radar,
                       之后再弹图框选 —— 此时底图数据来自缓存, 但标题要与
                       原来一模一样(否则与历史框选图不一致)。
      render:        'map'   = 老行为: pyart plot_ppi_map + cartopy 地图底图(慢)
                     'light' = pcolormesh 直接画经纬度, 不碰 cartopy(快, 无地理底图)
                     ★ 两种方式的轴数据坐标都是经纬度, ginput 返回值完全一致;
                       红框也都能正确落在同一位置(_mark_box_and_save 已按轴类型自适应)。
      decimate:      render='light' 时的抽稀倍数(1=不抽稀)。定框只需看回波形状,
                     抽 2 倍网格数降到 1/4 -> 渲染更快, 不会出现空洞。
    """
    from matplotlib import pyplot as plt

    from . import plotstyle as cs

    # --- 画什么: 组合反射率(默认) 还是某一层 PPI(composite=False) ---
    if composite:
        from .step08_composite import composite_radar as _composite_radar
        # write_back=False: 只借它算 CR + 造一个"单层 + CR 字段"的 slim radar 来画,
        #                   不往原 radar 里塞 CR 字段(指标那边用不着, 免得越跑字段越多)
        disp_radar, comp_z, comp_alt = _composite_radar(
            radar, field=field, ref_sweep=sweep, write_back=False,
            field_name='CR', verbose=verbose)
        disp_field, disp_sweep = 'CR', 0
        ok = np.isfinite(np.ma.filled(comp_z, np.nan)).any()
        cr_max = float(np.nanmax(np.ma.filled(comp_z, np.nan))) if ok else float('nan')
        if title is None:
            title = (f'组合反射率 CR (各仰角 max, ref sweep={sweep})   max={cr_max:.1f} dBZ'
                     f'\n点击【左上角】+【右下角】 (顺序不影响)')
    else:
        disp_radar, disp_field, disp_sweep = radar, field, sweep
        if title is None:
            title = f'{field} sweep={sweep} - 点击左上角和右下角'

    cs.science_style()
    fig = plt.figure(figsize=(10, 8), dpi=100)

    if str(render).lower() == 'light':
        # ★ 2026-09-18 轻量路径: **不 import pyart / cartopy**, 直接 pcolormesh。
        ax = fig.add_subplot(111)
        _plot_ppi_light(ax, fig, disp_radar, disp_field, disp_sweep, cs,
                        decimate=decimate)
    else:
        import pyart
        import cartopy.crs as ccrs
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        display = pyart.graph.RadarMapDisplay(disp_radar)
        cmap, norm = cs.ref_cmap_params()
        display.plot_ppi_map(ax=ax, field=disp_field, sweep=disp_sweep,
                             cmap=cmap, norm=norm)
        cs.add_latlon_grid(ax, nx=5, ny=4, fontsize=9, bold=True)
        cs.bold_colorbar(display, 0, fontsize=9)

    cs.bold_title(ax, title, fontsize=10)

    print('请在图上点击两点: 第一点=框左上角, 第二点=框右下角'
          + ('  (窗口里显示的是组合反射率 CR)' if composite else ''))

    # ★ 2026-09-15: 脚本模式(非 notebook)下必须**显式把窗口弹出来**。
    #   matplotlib 在脚本里默认非交互: 不调 show() 就不会建/显示窗口; 而 plt.ginput()
    #   是在**画布**上收鼠标事件的 -> 结果是"根本没弹窗、光标一直卡住、点了也没反应"。
    #   notebook 由前端自动渲染, 所以那边不写这两行也能用; 脚本里不写必然卡死。
    #   非 GUI 后端(agg/svg/pdf...)下 show() 只发一条警告, 不会抛错, 故这里不拦。
    fig.canvas.draw()      # 先把 PPI 画上去(窗口弹出来时直接就是成品, 不是空白)
    fig.show()             # 弹窗(仅 GUI 后端有窗口; Agg 下无效但不报错)
    plt.pause(0.2)         # 跑一下 GUI 事件循环, 窗口才真正可见

    pts = plt.ginput(2, timeout=-1)

    if len(pts) != 2:
        plt.close(fig)
        raise ValueError('未点够两点, 框选失败')

    (lon1, lat1), (lon2, lat2) = pts
    if lat1 < lat2:
        lat1, lat2 = lat2, lat1
    if lon1 > lon2:
        lon1, lon2 = lon2, lon1

    print(f'框选范围: lon [{lon1:.4f}, {lon2:.4f}], lat [{lat2:.4f}, {lat1:.4f}]')

    # ★★ 2026-09-18 F 盘批量版: 点完就把框画上 -> 存图 ★★
    #   存的**就是刚点完的这张 CR 图**, 所见即所得(事后可回头核查框选是否合理)。
    #   整段 fail-soft: 画框/存图任何异常都只告警, 绝不丢掉已经点好的框
    #   —— 人工框选一个体扫要点两下, 一天上百个文件, 不能因为存图失败白点。
    if fig_save_path:
        try:
            _mark_box_and_save(fig, ax, lon1, lat1, lon2, lat2, fig_save_path,
                               dpi=fig_dpi, mark_box=mark_box, cs=cs)
        except Exception as e:
            print(f'  [!] 框选图保存失败({type(e).__name__}: {e}) —— 框已拿到, 继续往下跑')
            print(f'      目标路径: {fig_save_path}')

    plt.close(fig)
    return lon1, lat1, lon2, lat2


def select_box_drag(radar, sweep=0, field='cor_z', composite=True, verbose=True,
                    fig_save_path=None, fig_dpi=150, mark_box=True, title=None,
                    render='light', decimate=1, auto_test=None):
    """
    Step 1(拖拽版): 画**组合反射率** + 鼠标**拖拽拉出红框** -> 返回 (lon1, lat1, lon2, lat2)

    与 select_box_ginput 的差别只在"怎么选框":
      - ginput   : 点两下(左上角 + 右下角), 点完才看到框
      - 本函数   : 按住左键从左上角拖到右下角, 拖动过程中红框**实时跟随**, 松开即定框

    用 matplotlib.widgets.RectangleSelector 实现:
      - 交互感与 PS/截图工具一致(左上角按住拖到右下角);
      - 拖完还能拖动/调整边角(active='box'), 满意后按 Enter 或双击确认;
      - 也可以直接关闭窗口/按 Esc 取消(返回 ValueError, 与 ginput 未点够两点一致)。

    其余参数、画图逻辑、返回约定与 select_box_ginput **完全一致**(轴数据坐标都是经纬度,
    返回 lon1/lat1/lon2/lat2, 且保证 lat1>=lat2, lon1<=lon2), 下游无需改动。

    ★ auto_test: 只为"无人值守的自检"准备(默认 None = 正常人工交互)。
      给一个 dict {'lon1','lat1','lon2','lat2'} 时, 函数自己合成一串事件
      (左键按下 -> 拖动 -> 松开 -> Enter), 走的是**与真人完全相同的那套回调**,
      用来验证"拖框 + Enter"这条链在批处理里确实能跑通(不需要有人点鼠标)。
      自检窗口是 qtagg 也没关系: 事件是直接喂给画布的, 不依赖真鼠标。

    依赖交互式后端(需 GUI 窗口), 与 ginput 相同; 调用方应先判后端(见 debug_flow.check_gui_backend)。
    """
    from matplotlib import pyplot as plt
    from matplotlib.widgets import RectangleSelector

    from . import plotstyle as cs
    # --- 画什么: 与 select_box_ginput 共用同一段(组合反射率 CR / 单层 PPI) ---
    if composite:
        from .step08_composite import composite_radar as _composite_radar
        disp_radar, comp_z, comp_alt = _composite_radar(
            radar, field=field, ref_sweep=sweep, write_back=False,
            field_name='CR', verbose=verbose)
        disp_field, disp_sweep = 'CR', 0
        ok = np.isfinite(np.ma.filled(comp_z, np.nan)).any()
        cr_max = float(np.nanmax(np.ma.filled(comp_z, np.nan))) if ok else float('nan')
        if title is None:
            title = (f'组合反射率 CR (各仰角 max, ref sweep={sweep})   max={cr_max:.1f} dBZ'
                     f'\n按住左键从【左上角】拖到【右下角】, 松手后 Enter 确认')
    else:
        disp_radar, disp_field, disp_sweep = radar, field, sweep
        if title is None:
            title = f'{field} sweep={sweep} - 按住左键拖拽框选, Enter 确认'

    cs.science_style()
    fig = plt.figure(figsize=(10, 8), dpi=100)

    if str(render).lower() == 'light':
        ax = fig.add_subplot(111)
        _plot_ppi_light(ax, fig, disp_radar, disp_field, disp_sweep, cs,
                        decimate=decimate)
    else:
        import pyart
        import cartopy.crs as ccrs
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        display = pyart.graph.RadarMapDisplay(disp_radar)
        cmap, norm = cs.ref_cmap_params()
        display.plot_ppi_map(ax=ax, field=disp_field, sweep=disp_sweep,
                             cmap=cmap, norm=norm)
        cs.add_latlon_grid(ax, nx=5, ny=4, fontsize=9, bold=True)
        cs.bold_colorbar(display, 0, fontsize=9)

    cs.bold_title(ax, title, fontsize=10)
    print('按住左键从【左上角】拖到【右下角】拉红框, 松手后可拖动调整, '
          '按 Enter 确认(或双击); 关闭窗口/Esc 取消'
          + ('  (窗口里显示的是组合反射率 CR)' if composite else ''))

    fig.canvas.draw()
    fig.show()
    plt.pause(0.2)

    # --- 拖拽框选状态 ---
    result = {'ok': False, 'extents': None}
    _selector = {}

    def _on_select(eclick, erelease):
        # eclick / erelease 的 xdata/ydata 就是经纬度(轴数据坐标)
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return
        result['extents'] = (float(x0), float(y0), float(x1), float(y1))

    def _on_enter(evt):
        if evt.key in ('enter', ' '):
            if result['extents'] is not None:
                result['ok'] = True
            plt.close(fig)

    def _on_close(evt):
        plt.close(fig)

    # active='box' 允许拖完再整体拖动/调角; interactive=True 拖动中实时刷新
    try:
        rs = RectangleSelector(
            ax, _on_select, useblit=True,
            button=[1], minspanx=0.0, minspany=0.0,
            spancoords='data', interactive=True,
            props=dict(facecolor='red', edgecolor='red', alpha=0.2, fill=True))
    except TypeError:
        # 极老版本无 useblit/interactive 关键字时退回去
        rs = RectangleSelector(ax, _on_select, button=[1],
                               spancoords='data',
                               props=dict(facecolor='red', edgecolor='red',
                                          alpha=0.2, fill=True))
    _selector['rs'] = rs

    fig.canvas.mpl_connect('key_press_event', _on_enter)
    fig.canvas.mpl_connect('close_event', _on_close)

    # ★ 自检(auto_test): 合成"按下 -> 拖动 -> 松开 -> Enter"四个事件, 走真人同一套回调。
    #   调用方(批处理 --selftest-pick)拿它验证拖框链能不能跑通, 不需要有人守在电脑前。
    if auto_test:
        try:
            from matplotlib.backend_bases import KeyEvent, MouseEvent
            fig.canvas.draw()
            ax_bb = ax.get_window_extent()

            def _px(lo, la):
                """经纬度 -> 像素(相对画布)"""
                xd, yd = ax.transData.transform((float(lo), float(la)))
                xf, yf = fig.transFigure.inverted().transform((xd, yd))
                return xf * fig.bbox.width, yf * fig.bbox.height

            lo1, la1 = float(auto_test['lon1']), float(auto_test['lat1'])
            lo2, la2 = float(auto_test['lon2']), float(auto_test['lat2'])
            x1, y1 = _px(lo1, la1)
            x2, y2 = _px(lo2, la2)
            # matplotlib 的 MouseEvent 签名: (name, canvas, x, y, button, key, step,
            # dblclick, guiEvent); inaxes 由 callbacks.process 自己补, 不能手传。
            ev = dict(button=1, key=None, step=1, dblclick=False, guiEvent=None)
            fig.canvas.callbacks.process(
                'button_press_event',
                MouseEvent('button_press_event', fig.canvas, x1, y1, **ev))
            fig.canvas.callbacks.process(
                'motion_notify_event',
                MouseEvent('motion_notify_event', fig.canvas, x2, y2, **ev))
            fig.canvas.callbacks.process(
                'button_release_event',
                MouseEvent('button_release_event', fig.canvas, x2, y2, **ev))
            fig.canvas.callbacks.process(
                'key_press_event',
                KeyEvent('key_press_event', fig.canvas, 'enter'))
            ex = result['extents']
            print(f'  [自检] 合成拖框: 请求 lon[{lo1:.4f},{lo2:.4f}] '
                  f'lat[{la2:.4f},{la1:.4f}]')
            print(f'  [自检] 回调收到: extents={ex}  ok={result["ok"]}')
            if ex is not None:
                _dx = max(abs(ex[0] - lo1), abs(ex[1] - la1),
                          abs(ex[2] - lo2), abs(ex[3] - la2))
                print(f'  [自检] 与请求的最大偏差 {_dx:.6f} deg '
                      f'({"一致" if _dx < 1e-6 else "不一致!"})')
        except Exception as e:
            import traceback as _tb
            print(f'  [自检] 合成拖框失败: {type(e).__name__}: {e}')
            _tb.print_exc()

    # 阻塞直到窗口被 Enter 确认或关闭
    try:
        while plt.fignum_exists(fig.number):
            plt.pause(0.05)
    except Exception:
        pass
    finally:
        try:
            rs.disconnect_events()
        except Exception:
            pass

    if not result['ok'] or result['extents'] is None:
        # 窗口被直接关掉 / Esc 取消
        try:
            plt.close(fig)
        except Exception:
            pass
        raise ValueError('未拖出有效框(窗口被关闭或取消), 框选失败')

    x0, y0, x1, y1 = result['extents']
    lon1, lat1 = min(x0, x1), max(y0, y1)   # 左上角 = 小经度、大纬度
    lon2, lat2 = max(x0, x1), min(y0, y1)   # 右下角 = 大经度、小纬度
    # 与 ginput 版保持同一条归一化约定
    if lat1 < lat2:
        lat1, lat2 = lat2, lat1
    if lon1 > lon2:
        lon1, lon2 = lon2, lon1

    print(f'框选范围: lon [{lon1:.4f}, {lon2:.4f}], lat [{lat2:.4f}, {lat1:.4f}]')

    if fig_save_path:
        try:
            _mark_box_and_save(fig, ax, lon1, lat1, lon2, lat2, fig_save_path,
                               dpi=fig_dpi, mark_box=mark_box, cs=cs)
        except Exception as e:
            print(f'  [!] 框选图保存失败({type(e).__name__}: {e}) —— 框已拿到, 继续往下跑')
            print(f'      目标路径: {fig_save_path}')

    plt.close(fig)
    return lon1, lat1, lon2, lat2


# =============================================================================
# Step 2: 框 -> 3D 掩膜
# =============================================================================

def build_box_mask_3d(radar, lon1, lat1, lon2, lat2):
    """
    Step 2: lon/lat 框 -> 库点 mask(覆盖所有仰角层)。
    返回 shape=(nsweeps*nrays_per_sweep, ngates) 的 bool 数组
    (= radar 每个字段 data 的形状, 直接拿去索引即可)。

    说明: gate_latitude / gate_longitude 本身就是 (总射线数, 库点数), 每个门点都有
    自己的经纬度, 所以"所有仰角层的同一个水平位置"会自动一起被框进来, 不需要再扩展。
    """
    gate_lat = radar.gate_latitude['data']
    gate_lon = radar.gate_longitude['data']

    box_2d = (gate_lon >= lon1) & (gate_lon <= lon2) & (gate_lat >= lat2) & (gate_lat <= lat1)
    print(f'框内 2D 库点数: {box_2d.sum()}')
    return box_2d


def _as_bool_mask(arr, like):
    """把可能是 MaskedArray 的字段取成普通 bool 的"无效"掩膜(无效=True)。"""
    if isinstance(arr, np.ma.MaskedArray):
        return np.ma.getmaskarray(arr)
    return np.zeros(np.shape(like), dtype=bool)


def _compressed_valid(values):
    """取有效值(去掉 masked / nan)。"""
    if isinstance(values, np.ma.MaskedArray):
        v = values.compressed()
        return v[np.isfinite(v)] if v.size else v
    v = np.asarray(values, dtype=float).ravel()
    return v[np.isfinite(v)]


# =============================================================================
# Step 1.5: 框选结果 / 指标序列的存取(JSON + CSV)
#   工作流: 每个个例交互框选一次(select_and_save_box) -> 存进 JSON
#           -> 批量计算时读框(load_box / all_metrics_series_from_file)
#           -> 指标序列可另存 CSV(save_metrics_series), 事后算速率(load + metric_trends)
# =============================================================================

def save_box(path, case, lon1, lat1, lon2, lat2, note=''):
    """
    把一个框选结果(经纬度)存进 JSON 文件, 按 case 名组织, 一个文件可存多个个例。
    同名 case 覆盖, 其它 case 保留。自动归一化(lat1>=lat2, lon1<=lon2)。
    返回存入的 dict。
    """
    import json
    import os
    if lat1 < lat2:
        lat1, lat2 = lat2, lat1
    if lon1 > lon2:
        lon1, lon2 = lon2, lon1

    data = {'boxes': {}}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception:
                data = {'boxes': {}}
        if not isinstance(data.get('boxes'), dict):
            data['boxes'] = {}

    data['boxes'][case] = {'lon1': float(lon1), 'lat1': float(lat1),
                           'lon2': float(lon2), 'lat2': float(lat2),
                           'note': str(note)}
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'框 {case!r} 已存到 {path}: '
          f'lon[{lon1:.4f}, {lon2:.4f}], lat[{lat2:.4f}, {lat1:.4f}]')
    return data['boxes'][case]


def load_box(path, case):
    """读取某个 case 的框, 返回 (lon1, lat1, lon2, lat2)。"""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    try:
        b = data['boxes'][case]
    except KeyError:
        raise KeyError(f'{path} 里没有 case {case!r}; 现有: {list(data.get("boxes", {}))}')
    print(f'框 {case!r} 读取自 {path}: '
          f'lon[{b["lon1"]:.4f}, {b["lon2"]:.4f}], lat[{b["lat2"]:.4f}, {b["lat1"]:.4f}]')
    return b['lon1'], b['lat1'], b['lon2'], b['lat2']


def load_boxes(path):
    """读取 JSON 里全部框, 返回 {case: {'lon1':.., 'lat1':.., 'lon2':.., 'lat2':.., 'note':..}}。"""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return dict(data.get('boxes', {}))


def select_and_save_box(radar, path, case, sweep=0, field='cor_z', note='',
                        composite=True):
    """
    一步到位: 画**组合反射率** + ginput 框选 + 存进 JSON。
    返回 (lon1, lat1, lon2, lat2)。
    composite=False 时退回画 sweep 那一层的 PPI(见 select_box_ginput)。
    """
    lon1, lat1, lon2, lat2 = select_box_ginput(radar, sweep=sweep, field=field,
                                               composite=composite)
    save_box(path, case, lon1, lat1, lon2, lat2, note=note)
    return lon1, lat1, lon2, lat2


def all_metrics_series_from_file(radars, box_path, case, field='cor_z', fhc=False, snd=None):
    """
    指标(8) 一站式: 从 JSON 里读 case 的框 -> 对多时次 radar 列表跑指标序列。
    等价于 all_metrics_series(radars, *load_box(box_path, case), ...)。
    """
    lon1, lat1, lon2, lat2 = load_box(box_path, case)
    return all_metrics_series(radars, lon1, lat1, lon2, lat2, field=field, fhc=fhc, snd=snd)


def all_metrics_ginput(radar, sweep=0, field='cor_z', fhc=False, vil=True,
                       vil_result_out=None, zdr_col=True, zdrcol_result_out=None,
                       zdrcol_kwargs=None, snd=None, composite=True,
                       fig_save_path=None, fig_dpi=150, mark_box=True):
    """
    指标提取的**逐文件交互版**: 弹**组合反射率 CR** 图 -> ginput 点左上角/右下角
    -> 在该矩形范围内算全部指标(max_z / 回波顶高 / >=45dBZ 库数体积 / VIL /
       ZDR 柱 / FHC)。

    ★ 2026-09-15: 弹窗内容 = 组合反射率(各仰角 max), 不再是第一层 PPI;
      composite=False 可退回旧行为(画 sweep 那一层), 见 select_box_ginput。
      注意: 图换成了 CR, **指标本身仍照旧在 3D 库点上算**(框是一个水平矩形,
      覆盖所有仰角层), 两者互不影响 —— 换的只是"给人看的底图"。

    每个时次的文件都弹一次图、各自框选; 返回的 dict 额外带:
      'box' : 本次框选的 (lon1, lat1, lon2, lat2), 便于事后核查各时次框的一致性
      'time': 体扫开始时刻(解析不到就不带)
    返回值直接可喂 metric_trends()。

    注意: 逐文件各自框选时, 相邻时次的框往往不完全一致 —— 速率对比的是
    "各自框内"的指标。若想严格同框, 请尽量对准同一目标点相同位置;
    或改用固定框(all_metrics_series 传同一组经纬度 / all_metrics_series_from_file)。
    vil=False 时不算 VIL(批量时省时), 结果里没有 vil_* 键。
    vil_result_out: 传 dict 时把 VIL 2D 场/Grid 对象带出来(见 all_metrics), 供存 .nc。
    zdr_col / zdrcol_result_out / zdrcol_kwargs: 见 all_metrics。
    snd: QC.read_sounding() 的返回 dict —— 传给指标(7) 用(见 all_metrics)。
    fig_save_path / fig_dpi / mark_box: 【F 盘批量版】点完框就把框画上并存图, 见
                      select_box_ginput。存图失败只告警, 不影响指标提取。
    """
    lon1, lat1, lon2, lat2 = select_box_ginput(radar, sweep=sweep, field=field,
                                               composite=composite,
                                               fig_save_path=fig_save_path,
                                               fig_dpi=fig_dpi, mark_box=mark_box)
    box = build_box_mask_3d(radar, lon1, lat1, lon2, lat2)
    out = all_metrics(radar, box, field=field, fhc=fhc, vil=vil,
                      vil_result_out=vil_result_out, zdr_col=zdr_col,
                      zdrcol_result_out=zdrcol_result_out,
                      zdrcol_kwargs=zdrcol_kwargs, snd=snd)
    out['box'] = (float(lon1), float(lat1), float(lon2), float(lat2))
    t = _volume_datetime(radar)
    if t is not None:
        out['time'] = t
    return out


def all_metrics_series_ginput(radars, sweep=0, field='cor_z', fhc=False, vil=True,
                              zdr_col=True, zdrcol_kwargs=None, snd=None,
                              composite=True, fig_save_paths=None, fig_dpi=150,
                              mark_box=True):
    """
    指标(8) 交互版流水线: 对 radar 列表**逐个文件**弹**组合反射率 CR** 图框选 + 算指标,
    返回指标序列(按输入顺序, 每个元素带 'time' 和 'box')。
    之后 QC.metric_trends(series) 即得变化速率。
    vil=False 时不算 VIL(批量时省时)。
    composite=False 时退回画 sweep 那一层的 PPI(见 select_box_ginput)。

    【F 盘批量版】fig_save_paths: 与 radars 等长的路径列表(或单一路径字符串),
    逐个把"点了框的那张图"存下来, 见 select_box_ginput。

    用法(notebook 里逐个时次弹图):
        series = QC.all_metrics_series_ginput(radar_list)
        rates  = QC.metric_trends(series)
        QC.print_trends(rates)
    """
    if fig_save_paths is None or isinstance(fig_save_paths, str):
        _paths = [fig_save_paths] * len(radars)
    else:
        _paths = list(fig_save_paths)
    series = []
    for i, radar in enumerate(radars):
        print(f'\n=== 第 {i + 1}/{len(radars)} 个时次: 请在图上点左上角和右下角 ===')
        series.append(all_metrics_ginput(radar, sweep=sweep, field=field, fhc=fhc,
                                         vil=vil, zdr_col=zdr_col,
                                         zdrcol_kwargs=zdrcol_kwargs, snd=snd,
                                         composite=composite,
                                         fig_save_path=_paths[i], fig_dpi=fig_dpi,
                                         mark_box=mark_box))
    return series


def save_metrics_series(series, path):
    """
    把 all_metrics_series() 的结果存成 CSV(首列 time, 其余各指标一列)。
    事后用 load_metrics_series() 读回来直接喂 metric_trends()。
    """
    import csv
    import os
    if not series:
        raise ValueError('series 为空, 没什么可存')
    keys = sorted({k for m in series for k in m if k != 'time'})
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['time'] + keys)
        for m in series:
            t = m.get('time')
            t_str = t.isoformat(sep=' ') if t is not None else ''
            row = [t_str]
            for k in keys:
                v = m.get(k, np.nan)
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    row.append(str(v))          # box 元组 / 文件名等非数值原样存
                    continue
                row.append(f'{fv:.6g}' if np.isfinite(fv) else '')
            w.writerow(row)
    print(f'指标序列({len(series)} 个时次)已存到 {path}')
    return path


def load_metrics_series(path):
    """
    读回 save_metrics_series() 存的 CSV -> all_metrics_series 同结构的 dict 列表
    (time 解析成 datetime, 数值转 float, 空格 = 缺测 NaN)。可直接喂 metric_trends()。
    """
    import csv
    out = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = {}
            for k, v in row.items():
                if v is None or v == '':
                    m[k] = np.nan
                elif k == 'time':
                    m[k] = _to_datetime(v)
                else:
                    try:
                        m[k] = float(v)
                    except ValueError:
                        m[k] = v            # box / 文件名等非数值列原样读回
            out.append(m)
    return out


# Excel 表头: (指标键 -> 中文列名); 顺序即列顺序
# ★ 2026-09-24 用户定稿: 只留 7 个指标。
#   被删掉的表头(需要时按注释恢复即可):
#     ('count_ge45dbz', '≥45dBZ库数'),
#     ('ratio_ge45dbz', '≥45dBZ占比'),
#     ('count_ge45dbz_vol', '≥45dBZ库数(体积)'),
#     ('vil_mean_kgm2', '平均VIL(kg/m²)'),
#     ('zdrcol_max_zdr_db', 'ZDR柱内最大ZDR(dB)'),
#     ('zdrcol_depth_m', 'ZDR柱顶距0°C层高度(m)'),   # 2026-10-06 起 depth_m 才是论文口径的柱高
#     ('hail_graupel_cells', '冰雹+霰库数'),
#     ('hail_graupel_ratio', '冰雹+霰占比'),
#     ('hail_ratio_in_hail_graupel', '冰雹占(冰雹+霰)比'),
_EXCEL_HEADERS = (
    ('file', '文件名'),
    ('box', '框选范围(lon1,lat1,lon2,lat2)'),
    ('max_z', '最大回波强度(dBZ)'),
    ('echo_top_18dbz_m', '18dBZ回波顶高(m)'),
    ('echo_top_45dbz_m', '45dBZ回波顶高(m)'),
    ('volume_ge45dbz_km3', '≥45dBZ回波体积(km³)'),
    ('vil_max_kgm2', '最大VIL(kg/m²)'),
    ('hail_cells', '冰雹粒子数量'),
    ('graupel_cells', '霰粒子数量'),
    ('zmax30_height_m', '最强回波高度(≥30dBZ)(m)'),
    # ★ 2026-10-06 起这一列 = **层数 n × 层高 dz**(论文原始口径), 不再是柱顶海拔 AGL
    ('zdrcol_height_m', 'ZDR柱高(m, 层数×层高)'),
)


def save_metrics_excel(series, path, skip_keys=(),
                       sheet_name='指标提取'):
    """
    把指标序列存成 Excel (.xlsx): **首列 = 提取时间(体扫时刻)**,
    之后每个参数一列(中文表头, 见 _EXCEL_HEADERS; 顺序固定, 一行一个时次)。

    ★ 2026-09-24: skip_keys 默认改成**空** —— VIL 现在只产出 max 一个值,
      用户要它进表(以前默认排除 VIL 两列是因为 .nc 里已有整个 VIL 场)。
      要排除就显式传 skip_keys=('vil_max_kgm2',)。
      表里出现表头映射之外的数值键时, 以键名作列名追加在最后。

    参数
      series: all_metrics_ginput / all_metrics_series 等返回的 dict 列表
      path:   输出 .xlsx 路径(目录不存在自动创建)
    返回 path。
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter
    import os

    if not series:
        raise ValueError('series 为空, 没什么可存')

    # 列 = 固定映射顺序里的(未跳过的)键 + 序列里多出来的键
    known = [(k, h) for k, h in _EXCEL_HEADERS if k not in skip_keys]
    known_keys = {k for k, _ in known}
    extra = sorted({k for m in series for k in m
                    if k not in known_keys and k not in skip_keys and k != 'time'})
    columns = known + [(k, k) for k in extra]
    col_keys = [k for k, _ in columns]

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    head_font = Font(bold=True)
    center = Alignment(horizontal='center', vertical='center')
    ws.cell(row=1, column=1, value='提取时间').font = head_font
    ws.cell(row=1, column=1).alignment = center
    for j, (_, header) in enumerate(columns, start=2):
        c = ws.cell(row=1, column=j, value=header)
        c.font = head_font
        c.alignment = center

    for i, m in enumerate(series, start=2):
        t = m.get('time')
        ws.cell(row=i, column=1,
                value=(t.strftime('%Y-%m-%d %H:%M:%S') if hasattr(t, 'strftime')
                       else (str(t) if t is not None else '')))
        for j, k in enumerate(col_keys, start=2):
            v = m.get(k)
            if v is None:
                continue
            if isinstance(v, (tuple, list)):        # box -> 字符串
                v = ','.join(f'{float(x):.4f}' for x in v)
            elif isinstance(v, str):
                pass
            else:
                try:
                    v = float(v)
                    if not np.isfinite(v):
                        v = None                     # NaN -> 空
                except (TypeError, ValueError):
                    v = str(v)
            if v is not None:
                ws.cell(row=i, column=j, value=v)

    # 列宽: 按表头长度粗调, 时间列加宽
    ws.column_dimensions['A'].width = 21
    for j, (k, header) in enumerate(columns, start=2):
        ws.column_dimensions[get_column_letter(j)].width = max(len(header) * 2 + 2, 10)

    ws.freeze_panes = 'B2'                           # 冻结表头和时间列
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    wb.save(path)
    print(f'指标 Excel({len(series)} 行, 排除 {sorted(skip_keys) if skip_keys else "无"}) '
          f'已存到 {path}')
    return path


# =============================================================================
# Step 3: 各项指标
# =============================================================================

def metric_max_z(radar, box_mask, field='cor_z'):
    """
    指标 (1): 最大回波强度 (dBZ) —— 框内所有库点里 field(默认 cor_z) 的最大值。
    """
    z = radar.fields[field]['data']
    z_in_box = z[box_mask]
    z_valid = _compressed_valid(z_in_box)
    if z_valid.size == 0:
        return np.nan
    return float(z_valid.max())


def metric_zmax_height(radar, box_mask, threshold=30.0, field='cor_z'):
    """
    指标 (6): 最大回波强度出现 30dBZ 时最强回波的高度 (m)。

    定义: 框内所有 >= threshold(默认 30 dBZ) 的库点中, 反射率最强的库点
    所在高度(gate_altitude, 已含 4/3 地球半径订正)。
    - 框内最大回波 < threshold 时该指标无意义, 返回 NaN(存储时写空);
    - 若最强回波在多个高度并列(同一 dBZ 值), 取其中最高的那个。

    返回: 高度 (m); 无 >=threshold 回波时 NaN。
    """
    zf = np.ma.filled(radar.fields[field]['data'], np.nan).astype(float)
    altf = np.ma.filled(radar.gate_altitude['data'], np.nan).astype(float)

    sel = np.asarray(box_mask, dtype=bool) & np.isfinite(zf) & (zf >= float(threshold))
    if not sel.any():
        return np.nan

    zmax = np.nanmax(zf[sel])
    cand = sel & (zf == zmax) & np.isfinite(altf)
    if not cand.any():
        return np.nan
    return float(np.nanmax(altf[cand]))


def _get_sweep_indices(radar):
    """每个 sweep 的起止 ray 索引 [(start, end_exclusive), ...]"""
    si = radar.sweep_start_ray_index['data']
    ne = radar.sweep_end_ray_index['data']
    return [(int(si[i]), int(ne[i]) + 1) for i in range(radar.nsweeps)]


def _sweep_gate_altitude(radar, sweep_idx):
    """取某 sweep 的 gate_altitude (2D: nrays_sweep x ngates)"""
    s, e = _get_sweep_indices(radar)[sweep_idx]
    return radar.gate_altitude['data'][s:e]


def metric_echo_top(radar, box_mask, threshold=18.0, field='cor_z'):
    """
    指标 (2): 回波顶高 (m) —— 框内 field >= threshold 的库点里, 高度最大者。

    18 dBZ 和 45 dBZ 各调一次:
        metric_echo_top(radar, box, threshold=18.0)
        metric_echo_top(radar, box, threshold=45.0)
    """
    z = radar.fields[field]['data']
    gate_alt = radar.gate_altitude['data']

    valid = box_mask & (np.ma.filled(z, -999.0) >= threshold)
    if not valid.any():
        return 0.0
    alts = _compressed_valid(gate_alt[valid])
    if alts.size == 0:
        return 0.0
    return float(alts.max())


def metric_echo_count(radar, box_mask, threshold=45.0, field='cor_z'):
    """
    指标 (3): >=threshold dBZ 的库数 + 占有效库比。
    返回 (count, ratio): count = 框内 field>=threshold 的库点数,
                        ratio = count / 框内有效库点数。
    """
    z = radar.fields[field]['data']
    mask_arr = _as_bool_mask(z, z)
    zf = np.ma.filled(z, np.nan)

    valid_in_box = box_mask & ~mask_arr & np.isfinite(zf)
    count = int((valid_in_box & (zf >= threshold)).sum())
    total = int(valid_in_box.sum())
    ratio = count / total if total > 0 else 0.0
    return count, ratio


def metric_echo_volume(radar, box_mask, threshold=45.0, field='cor_z'):
    """
    指标 (3) 的"体积"版: >=threshold dBZ 的库点换算成立方公里。

    每个门点的体积用雷达库体积近似: 库长 x (两个方位角之间的弧长) x (两个仰角之间的厚度)。
    直接对"该库点所属的那一小块"算, 比"库数 x 常数"更准(近距/远距体积差很多)。

    返回 (n_gates, volume_km3)
    """
    z = radar.fields[field]['data']
    rng_km = radar.range['data'] / 1000.0                      # (ngates,)
    gate_alt = radar.gate_altitude['data']                     # (nrays_total, ngates)

    sel = box_mask & (np.ma.filled(z, -999.0) >= threshold)
    if not sel.any():
        return 0, 0.0

    dr = float(rng_km[1] - rng_km[0]) if rng_km.size > 1 else 0.03
    r2d = np.broadcast_to(rng_km, sel.shape)
    # 方位角间隔: 用本 sweep 的射线数(pyart 的属性名是 rays_per_sweep)
    nray_sweep = int(np.asarray(radar.rays_per_sweep['data']).ravel()[0])
    dphi = 2.0 * np.pi / max(nray_sweep, 1)
    # 库体积 ≈ dr * (r * dphi) * 垂直厚度; 垂直厚度用相邻门的高度差, 退化时用 dr
    dalt = np.abs(np.gradient(np.ma.filled(gate_alt, np.nan), axis=1))
    dalt = np.where(np.isfinite(dalt) & (dalt > 0), dalt, dr)

    cell = dr * (r2d * dphi) * dalt
    vol = float(np.nansum(cell[sel]))
    return int(sel.sum()), vol


# =============================================================================
# Step 3.5: VIL(垂直累积液态水含量) —— 笛卡尔网格化 + 逐列积分
# =============================================================================

def _cartesian_grid(radar, box_mask, fields, dz, dxy, z_top, margin, verbose=False):
    """
    [内部] 体扫 -> 笛卡尔网格(被 VIL 与 ZDR 柱共用, 免得两套网格化代码各写一遍)。

    网格水平范围 = 框内(或全部有效回波)库点的经纬度范围外扩 margin;
    网格原点 = 范围中心, 影响半径用 dist_beam(随距离/仰角增长, 适合体扫)。

    返回 pyart Grid 对象。
    """
    import pyart

    glon_g = radar.gate_longitude['data']
    glat_g = radar.gate_latitude['data']
    if box_mask is not None and np.asarray(box_mask).any():
        sel = box_mask
    else:
        base = radar.fields[fields[0]]['data']
        sel = ~_as_bool_mask(base, base)
    lon_min, lon_max = float(glon_g[sel].min()), float(glon_g[sel].max())
    lat_min, lat_max = float(glat_g[sel].min()), float(glat_g[sel].max())

    lat0 = 0.5 * (lat_min + lat_max)
    lon0 = 0.5 * (lon_min + lon_max)
    m_per_deg_lat = 111320.0                      # WGS84 平均 1 纬度 ≈ 111.32 km
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat0))
    y0 = (lat_min - lat0) * m_per_deg_lat - margin
    y1 = (lat_max - lat0) * m_per_deg_lat + margin
    x0 = (lon_min - lon0) * m_per_deg_lon - margin
    x1 = (lon_max - lon0) * m_per_deg_lon + margin

    nz = int(round(z_top / dz)) + 1
    ny = max(int(round((y1 - y0) / dxy)) + 1, 2)
    nx = max(int(round((x1 - x0) / dxy)) + 1, 2)

    if verbose:
        print(f'  网格: nz={nz}(dz={dz:.0f}m), ny={ny}, nx={nx}(dxy={dxy:.0f}m), '
              f'范围 lon[{lon_min:.3f},{lon_max:.3f}] lat[{lat_min:.3f},{lat_max:.3f}]')

    return pyart.map.grid_from_radars(
        (radar,), grid_shape=(nz, ny, nx),
        grid_limits=((0.0, z_top), (y0, y1), (x0, x1)),
        grid_origin=(lat0, lon0), fields=list(fields),
        weighting_function='Barnes2', roi_func='dist_beam',
        min_radius=dxy)


def _grid_box_mask(radar, grid, box_mask, tol_m=0.0):
    """
    [内部] 把 radar 上的人工框选(3D bool 掩膜)映射成**网格点的 2D 水平掩膜**。

    为什么需要: 网格化的水平范围 = 框选包络**外扩 margin**(默认 2 km) —— 那是为了让
    边缘列的插值不缺数据, 是"给网格化用的", 不是给人统计的。指标必须只统计人工框选
    的范围, 否则每边 2 km 的外扩圈里的回波会被算进来; 对 ZDR 柱尤其致命: 框外 2 km
    内的一根柱子会被当成"框内的柱子"报出去。

    参数
      box_mask: 与 radar 形状一致的 bool 掩膜(build_box_mask_3d 的输出)
      tol_m:    容差 (m), 默认 0。pyart 由 x/y(米) 反算经纬度所用球半径常数与
                _cartesian_grid 中 111320 m/deg 略有差别(20 km 尺度约几十米),
                调用方通常给半个格距的余量, 免得把框边上的格点误杀。
    返回
      (ny, nx) 的 bool 数组; 拿不到框 / 网格没有经纬度 -> None(=不做水平限制)。
    """
    if box_mask is None:
        return None
    bm = np.asarray(box_mask, dtype=bool)
    if not bm.any():
        return None
    glon = np.ma.filled(radar.gate_longitude['data'], np.nan).astype(float)
    glat = np.ma.filled(radar.gate_latitude['data'], np.nan).astype(float)
    ok = bm & np.isfinite(glon) & np.isfinite(glat)
    if not ok.any():
        return None
    # ★ 2026-09-24 修复(两处配套, 少改一处都不行):
    #   ① 赋值: 上一版写成 lat1 = max(纬) / lat2 = min(纬), 与下面的判据
    #      (lat2d >= lat2) & (lat2d <= lat1) 含义正好相反 -> 纬度判据全程为假;
    #   ② 判据: 上一版写成 (lat2d >= lat2) & (lat2d <= lat1)。
    #   结果 _grid_box_mask 只剩经度约束 -> 框上下各多出 >11 km 的横带。本例
    #   (1109 帧) 框内从 3816 个格点涨到 4070 个, 多出的 254 个点上恰好有更大的
    #   VIL -> VIL max 由 1.5116 虚高到 1.6007(+5.9%)。ZDR 柱(第 1666 行)共用本
    #   函数, 同样偏大。修复后 _grid_box_mask 与"框的经纬度矩形"完全一致(4070→3816)。
    lon1, lon2 = float(glon[ok].min()), float(glon[ok].max())
    lat1, lat2 = float(glat[ok].min()), float(glat[ok].max())

    try:
        lon2d = np.ma.filled(grid.point_longitude['data'], np.nan).astype(float)
        lat2d = np.ma.filled(grid.point_latitude['data'], np.nan).astype(float)
    except (KeyError, AttributeError):
        return None
    if lon2d.ndim == 3:
        lon2d, lat2d = lon2d[0], lat2d[0]
    if lon2d.ndim != 2 or lon2d.shape != lat2d.shape:
        return None
    if not (np.isfinite(lon2d).any() and np.isfinite(lat2d).any()):
        return None

    tol = float(tol_m or 0.0)
    lat_mid = 0.5 * (lat1 + lat2)
    dlat = tol / 111320.0
    dlon = tol / max(111320.0 * np.cos(np.deg2rad(lat_mid)), 1.0)
    return ((lon2d >= lon1 - dlon) & (lon2d <= lon2 + dlon) &
            (lat2d >= lat1 - dlat) & (lat2d <= lat2 + dlat))


def compute_vil_field(radar, box_mask=None, field='cor_z', extra_fields=(),
                      z_cap_dbz='config', z_min_dbz='config', dz_m='config',
                      dxy_m='config', z_top_m='config', margin_m='config',
                      verbose=False):
    """
    把体扫网格化到笛卡尔坐标后逐列积分, 得到 VIL 的 2D 水平分布 (kg/m²)。

    公式(Greene & Clark 1972):
        M = 3.44e-6 * Z^(4/7)      Z 为线单位 mm⁶·m⁻³
        VIL = Σ (M_k + M_k+1)/2 · Δh     (自 0 高度到 z_top, 直接得 kg/m² = mm)

    实现要点:
      - 网格范围自动取 box_mask(或全部有效回波)的经纬度范围, 外扩 margin;
      - **指标统计范围 = 人工框选本身**(第 5.5 步映射出的 box2d), 外扩的那圈不参与;
        与指标(7) ZDR 柱共用同一个 box_mask -> 两边"同一区域"(2026-09-15);
      - 网格中心 = 框中心, 用 pyart.map.grid_from_radars + dist_beam 影响半径;
      - 缺测/无回波的格点按 M=0 处理(缺层不插值, 保守);
      - dBZ 先封顶(z_cap_dbz, 雹電容)再转线单位;
      - 梯形积分用 np.trapz(兼容 numpy>=2 的 trapezoid)。

    参数
      box_mask: 框选 3D 掩膜 —— **既决定网格水平范围, 也是最终的统计范围**
                (2026-09-15 明确: 网格化时包络外扩 margin 只为边缘列插值, 统计时用
                 _grid_box_mask 把范围收回到框内 —— 与指标(7) ZDR 柱同一区域);
                None = 用全部有效回波范围, 此时不做水平限制
      field:    用哪个反射率字段, 默认 'cor_z'(衰减订正后, X 波段必须用它)
      extra_fields: 额外一起网格化的字段(如 ('cor_zdr',)), 一般不用。
                注意: 工作.py **不再**给 VIL 网格塞 ZDR 字段(2026-09-14 用户要求:
                VIL 产物不带 ZDR 的东西), 指标(7) ZDR 柱自己网格化(见 metric_zdr_column)
      其余参数: 传 'config'(默认)取 config.py 的 VIL_* 值;
                传具体数值覆盖; **z_cap_dbz=None 表示不封顶**

    返回 dict:
      'vil'  : 2D VIL 场 (ny, nx), kg/m² —— **整张网格**(含框外的 margin 圈), 供出图
      'lon'/'lat': 各网格点的经纬度 (ny, nx)
      'grid' : pyart Grid 对象(要画垂直剖面可直接用)
      'z_axis': 网格各层高度 (nz,) m
      'box2d': (ny, nx) bool —— **人工框选**映射到网格点的水平掩膜 = 指标统计范围
               (与指标(7) ZDR 柱同一区域); box_mask 传 None 时为 None(不限制)
    """
    z_cap = C.VIL_Z_CAP_DBZ if z_cap_dbz == 'config' else z_cap_dbz
    z_min = C.VIL_Z_MIN_DBZ if z_min_dbz == 'config' else z_min_dbz
    dz = C.VIL_GRID_DZ_M if dz_m == 'config' else float(dz_m)
    dxy = C.VIL_GRID_DXY_M if dxy_m == 'config' else float(dxy_m)
    z_top = C.VIL_Z_TOP_M if z_top_m == 'config' else float(z_top_m)
    margin = C.VIL_GRID_MARGIN_M if margin_m == 'config' else float(margin_m)

    if verbose:
        print(f'  VIL 网格参数: dz={dz:.0f}m, dxy={dxy:.0f}m, z_top={z_top:.0f}m')

    # --- 1~2. 网格水平范围 + 网格化(dist_beam: 影响半径随距离/仰角增长) ---
    fields = [field] + [f for f in extra_fields if f != field]
    grid = _cartesian_grid(radar, box_mask, fields, dz, dxy, z_top, margin,
                           verbose=verbose)

    # --- 3. 逐库点算 M: 封顶 -> 转 Z 线单位 -> 液态水含量 ---
    zd = np.ma.filled(grid.fields[field]['data'], np.nan).astype(float)   # (nz,ny,nx) dBZ
    if z_cap is not None:                                                 # 雹電容
        zd = np.minimum(zd, float(z_cap))
    dbz = np.where(np.isfinite(zd), zd, np.nan)
    zlin = np.where(np.isfinite(dbz) & (dbz >= float(z_min)), 10.0 ** (dbz / 10.0), 0.0)
    # 标准 Greene-Clark 形式: VIL = Σ 3.44e-6 · Z̄^(4/7) · Δh, Z 用 mm⁶/m³、Δh 用 m,
    # 结果直接是 kg/m²(等价于 M = 3.44e-3 · Z^(4/7) g/m³ 再除以 1000 —— 两种写法数值相同)。
    # 这里按标准形式写, m_water 单位 kg/m³, 后面积分不再除 1000。
    m_water = 3.44e-6 * zlin ** (4.0 / 7.0)                               # kg/m³, 缺测处 0

    # --- 4. 逐列梯形积分: Σ M̄·Δh -> kg/m² ---
    z_axis = np.asarray(grid.point_z['data'])[:, 0, 0].astype(float)      # (nz,) m
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz   # numpy>=2 改名 trapezoid
    vil = _trapz(m_water, z_axis, axis=0)                                 # (ny,nx) kg/m²

    lon2d = np.ma.filled(grid.point_longitude['data'], np.nan).astype(float)
    lat2d = np.ma.filled(grid.point_latitude['data'], np.nan).astype(float)

    # --- 5. 把 VIL 作为"垂直方向常量"字段挂到 grid 上, 存 grid .nc 时一起带走 ---
    #     (VIL 本身是 2D 柱积分量, 垂直方向没有结构; 重复到每层只为方便
    #      pyart.io.read_grid 读回后直接画水平分布/与其他字段同框)
    vil3d = np.ma.masked_invalid(np.repeat(vil[np.newaxis, :, :], grid.nz, axis=0))
    grid.add_field('VIL', {
        'data': vil3d,
        'units': 'kg m-2',
        'long_name': 'Vertically Integrated Liquid Water (constant with height)',
        '_FillValue': -9999.0,
    }, replace_existing=True)

    # --- 6. ★ 指标统计范围 = 人工框选(**和指标(7) ZDR 柱同一区域**, 2026-09-15) ---
    #   _cartesian_grid 把网格水平范围定成"框的经纬度包络 + margin(默认 2 km)"。那圈外扩
    #   只是为了让**边缘列的插值**不缺数据, 不是用户框选的区域, 不能进统计:
    #     · 面积上它不小 —— 20 km×20 km 的框外扩 2 km 后是 24 km×24 km, 面积 +44%;
    #     · 外圈里基本都是弱回波/无回波, 于是 mean VIL 被系统性拉低(不是随机误差);
    #     · 更重要的是口径: 指标(7) ZDR 柱一直只用框内(见 _grid_box_mask 的说明),
    #       两边若不一致, 同一个框算出来的 VIL 与 ZDR 柱就不是"同一个区域"的量。
    #   故这里也映射出一份框内掩膜, 交给 metric_vil 去统计(max/mean); **vil 数组本身
    #   仍是整张网格**(出图/存 .nc 不变), 只是不再拿整张网格去算指标。
    box2d = _grid_box_mask(radar, grid, box_mask, tol_m=0.5 * float(dxy))

    return {'vil': vil, 'lon': lon2d, 'lat': lat2d, 'grid': grid, 'z_axis': z_axis,
            'box2d': box2d}


def save_grid_nc(vil_result, path, source_file=None, note=None):
    """
    把 compute_vil_field 的 **pyart Grid 对象整体**存成 NetCDF (.nc)
    (pyart.io.write_grid_nc; 字段含网格化反射率 + 常量垂直的 VIL)。

    之后读回与画图(★ 用 QC.read_grid_nc, 它自带中文路径兜底):
        import QC
        grid = QC.read_grid_nc(r'.../vilgrid_<个例>_<时刻>.nc')
        pyart.graph.GridMapDisplay(grid).plot_cross_section('cor_z', ...)   # 垂直剖面
        pyart.graph.GridMapDisplay(grid).plot_latitudinal_level('VIL', ...) # VIL 水平分布

    参数
      vil_result: compute_vil_field 的返回 dict(需含 'grid')
      path:       输出 .nc 路径(目录不存在自动创建)。
                  ★ 路径里有中文也没关系 —— 内部走 _nc_path 兜底
                    (netCDF4/HDF5 直接写非 ASCII 路径会 PermissionError errno 13)
      source_file: 对应的雷达基数据文件名(写入全局属性)
      note:       附加备注
    """
    import os

    import pyart

    grid = vil_result['grid']
    md = dict(grid.metadata)
    if source_file is not None:
        md['source_file'] = os.path.basename(str(source_file))
    if note:
        md['note'] = str(note)
    grid.metadata = md

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    # pyart 新版本改名为 write_grid(旧名 write_grid_nc)
    writer = getattr(pyart.io, 'write_grid_nc', None) or pyart.io.write_grid
    # ★ 中文路径兜底: netCDF4/HDF5 打不开非 ASCII 路径 -> 先写临时 ASCII 再搬回来
    with _nc_path(path, for_write=True) as _p:
        writer(_p, grid)
    print(f'VIL grid 对象(字段: {sorted(grid.fields)}, 形状 {grid.nz}x{grid.ny}x{grid.nx}) '
          f'已存到 {path}')
    return path


def metric_vil(radar, box_mask, field='cor_z', return_field=False, **kwargs):
    """
    指标 (4): 垂直累积液态水含量 VIL (kg/m²)。

    先 compute_vil_field 得到每根柱的 VIL, 再**只在人工框选的区域里**统计:
        max_vil  = 框内最大 VIL(雹云判别主指标, 业务上 >30~40 kg/m² 提示冰雹风险)
        mean_vil = 框内所有格点(含无回波格点)的平均 VIL
    参数透传给 compute_vil_field(见其 docstring; 默认参数在 config.py 的 VIL_*)。

    ★ 统计范围(2026-09-15): **与指标(7) ZDR 柱完全同一区域** = 人工框选的那个矩形,
      由 build_box_mask_3d 得到、两处共用同一个 box_mask。网格化的水平包络比框大一圈
      (每边外扩 margin=2 km, 只为边缘列插值), 那一圈**不计入** max/mean —— 否则
      mean 会被框外的弱回波系统性拉低, 且与 ZDR 柱的口径不一致。详见
      compute_vil_field 第 5.5 步。

    return_field=False(默认): 返回 (max_vil, mean_vil);
    return_field=True       : 返回 (max_vil, mean_vil, res) —— res 是
                              compute_vil_field 的完整结果(含 2D 场, 供存 .nc 用),
                              避免为了存场把 VIL 重复算一遍。
    """
    res = compute_vil_field(radar, box_mask=box_mask, field=field, **kwargs)
    vil = res['vil']
    box2d = res.get('box2d')
    if box2d is not None:
        if box2d.any():
            vil = vil[box2d]                      # 只留框内格点(与 ZDR 柱同区域)
        else:
            print('  [警告] 框选范围映射到网格后一个格点都没有 -> VIL 统计退回整张网格')
    if vil.size == 0:
        out = (np.nan, np.nan)
        return out + (res,) if return_field else out
    vals = (float(vil.max()), float(vil.mean()))
    return vals + (res,) if return_field else vals


def save_vil_nc(vil_result, path, time=None, source_file=None, note=None):
    """
    把 compute_vil_field 的 VIL 2D 水平场存成 NetCDF (.nc)。

    变量:
      vil (y, x)  kg/m², 压缩存储, 缺测填 -9999
      lon (y, x)  degrees_east
      lat (y, x)  degrees_north
    全局属性:
      volume_time(体扫时刻), source_file(雷达文件名),
      z_cap_dbz / grid_dz_m / grid_dxy_m / grid_z_top_m(取 config.py 当前值)

    参数
      vil_result: compute_vil_field 的返回 dict(只需 'vil'/'lon'/'lat' 三个键)
      path:       输出 .nc 路径(目录不存在自动创建)
      time:       体扫时刻(datetime 或字符串, 存为全局属性)
      source_file: 对应的雷达基数据文件名
      note:       附加备注
    """
    from netCDF4 import Dataset
    import os

    vil = np.asarray(vil_result['vil'], dtype='f4')
    lon = np.asarray(vil_result['lon'], dtype='f8')
    lat = np.asarray(vil_result['lat'], dtype='f8')
    ny, nx = vil.shape

    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    # ★ _nc_path: 中文路径兜底(netCDF4/HDF5 打不开非 ASCII 路径)
    with _nc_path(path, for_write=True) as _p, \
            Dataset(_p, 'w', format='NETCDF4_CLASSIC') as nc:
        nc.createDimension('y', ny)
        nc.createDimension('x', nx)

        v = nc.createVariable('vil', 'f4', ('y', 'x'), zlib=True, complevel=4,
                              fill_value=np.float32(-9999.0))
        v[:] = np.where(np.isfinite(vil), vil, np.float32(-9999.0))
        v.units = 'kg m-2'
        v.long_name = 'vertically integrated liquid water (Greene & Clark 1972)'

        lo = nc.createVariable('lon', 'f8', ('y', 'x'))
        lo[:] = lon
        lo.units = 'degrees_east'
        lo.long_name = 'longitude'
        la = nc.createVariable('lat', 'f8', ('y', 'x'))
        la[:] = lat
        la.units = 'degrees_north'
        la.long_name = 'latitude'

        nc.Conventions = 'CF-1.8'
        nc.title = 'VIL field from X-band phased-array radar QC chain'
        nc.vil_formula = 'VIL = sum(3.44e-6 * Z^(4/7) * dh), Z capped at VIL_Z_CAP_DBZ'
        if time is not None:
            nc.volume_time = (time.isoformat(sep=' ') if hasattr(time, 'isoformat')
                              else str(time))
        if source_file is not None:
            nc.source_file = os.path.basename(str(source_file))
        nc.z_cap_dbz = float(C.VIL_Z_CAP_DBZ)
        nc.grid_dz_m = float(C.VIL_GRID_DZ_M)
        nc.grid_dxy_m = float(C.VIL_GRID_DXY_M)
        nc.grid_z_top_m = float(C.VIL_Z_TOP_M)
        if note:
            nc.note = str(note)

    print(f'VIL 场({ny}x{nx}, kg/m2) 已存到 {path}')
    return path


def make_vil_radar(radar, vil_result, ref_field='cor_z', path=None,
                   max_dist_deg=0.02):
    """
    造一个**只含 反射率 + VIL 两个字段**的 pyart Radar, 几何(时间/方位/仰角/
    库点经纬纬高/sweep 结构)与原 radar 完全一致, 专门方便用 pyart 画图:

        import pyart, QC
        vradar = QC.make_vil_radar(radar, vil_out)
        disp = pyart.graph.RadarMapDisplay(vradar)   # 或 RadarDisplay
        disp.plot_ppi('VIL', sweep=0)                # VIL 水平分布
        disp.plot_ppi('reflectivity', sweep=0)       # 衰减订正后的反射率

    字段说明
      'reflectivity': ref_field(默认 cor_z, 即衰减订正后反射率)的数据,
                      只是改名为 pyart 标准字段名, long_name 里注明了来源
      'VIL'         : compute_vil_field 的 2D 网格场**最近邻映射回雷达库点**
                      (同一根柱所有仰角取同一 VIL 值, 画 PPI 即 VIL 水平分布);
                      距最近网格点超过 max_dist_deg(默认 0.02°, 约 2 km,
                      即网格覆盖范围外)的库点为缺测

    参数
      radar:      处理好的原 radar(只借用几何与 ref_field 数据, **不会被修改**)
      vil_result: compute_vil_field / all_metrics(vil_result_out=...) 的结果,
                  需含 'vil'/'lon'/'lat' 三个键
      ref_field:  反射率取哪个字段, 默认 'cor_z'
      path:      传输出路径时, 顺便把新 radar 写成 CF-Radial NetCDF(.nc),
                 之后 pyart.io.read_cfradial(path) 读回来即可画图, 无需重算

    返回: 新 Radar 对象; 传了 path 时返回 (radar, path)。
    """
    import copy as _copy
    import os

    import pyart
    from scipy.spatial import cKDTree

    vil2 = np.asarray(vil_result['vil'], dtype=float)
    lon2 = np.asarray(vil_result['lon'], dtype=float)
    lat2 = np.asarray(vil_result['lat'], dtype=float)

    glat = np.ma.filled(radar.gate_latitude['data'], np.nan).astype(float)
    glon = np.ma.filled(radar.gate_longitude['data'], np.nan).astype(float)
    valid = np.isfinite(glat) & np.isfinite(glon)

    # --- VIL 网格 -> 雷达库点最近邻映射(经度乘 cos(lat0) 修正成等距) ---
    lat0 = float(np.nanmean(lat2))
    coslat = np.cos(np.deg2rad(lat0))
    tree = cKDTree(np.column_stack([lat2.ravel(), lon2.ravel() * coslat]))
    dist, idx = tree.query(np.column_stack([glat[valid], glon[valid] * coslat]))
    vil_flat = np.full(glat.size, np.nan)               # 1D 中转, 再散回 2D
    vil_flat[valid.ravel()] = vil2.ravel()[idx]
    vil_flat[dist > float(max_dist_deg)] = np.nan       # 网格没覆盖到的库点 → 缺测
    vil_gates = vil_flat.reshape(glat.shape)

    # --- 组新 radar: 浅拷贝共享几何, 只替换 fields 字典(原 radar 不受影响) ---
    vradar = _copy.copy(radar)
    ref = dict(radar.fields[ref_field])                 # 元数据浅拷贝, 数据数组共享
    ref['long_name'] = (f'{ref_field} (attenuation-corrected reflectivity, '
                        f'renamed to reflectivity for plotting)')
    vradar.fields = {
        'reflectivity': ref,
        'VIL': {
            'data': np.ma.masked_invalid(vil_gates),
            'units': 'kg m-2',
            'long_name': 'Vertically Integrated Liquid Water (Greene & Clark 1972)',
            '_FillValue': -9999.0,
            'coordinates': 'elevation azimuth range',
        },
    }
    vradar.metadata = dict(radar.metadata)
    vradar.metadata['source'] = ('QC chain: cor_z + gridded VIL '
                                 '(nearest-neighbor mapped back to gates)')

    if path is not None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with _nc_path(path, for_write=True) as _p:     # 中文路径兜底, 见 _nc_path
            pyart.io.write_cfradial(_p, vradar, format='NETCDF4')
        print(f'VIL radar(字段: reflectivity / VIL) 已存到 {path}')
        return vradar, path
    return vradar


# =============================================================================
# Step 3.6: ZDR 柱(指标(7)) —— "3D mapping columns" 法
#   依据: Zhao et al. (2025), Atmos. Chem. Phys., 25, 13453–13473
#         "Bridging the polarimetric structure and lightning activity"
#         (2.4 节 + Fig. 3), 由他们改进; 早期研究见 Woodard et al. 2012 /
#         Snyder et al. 2015 / Sharma et al. 2024 / Krause & Klaus 2024。
# =============================================================================

def isotherm_height(radar=None, temp_c=0.0, temp_field='temperature', bin_m=100.0,
                    min_bin_n=5, strict=False, snd=None):
    """
    反推**某条等温线的高度** (m, AGL)。两个数据源, **优先用探空原始廓线**:

      snd 给了 → 直接查 snd['heights_m'] / snd['temps_C'](步长 2 的层结, 线性插值)。
                 ★ 首选: 不受"雷达库点高度范围"限制 —— 探空一般到 30 km 以上,
                 -20 °C 这类高层等温线必定落在里面。
      snd 没给 → 从 radar 的 'temperature' 场反推。注意这个场是**探空按雷达库点
                 高度插值**出来的, 有效高度上限 = min(探空顶, 雷达库点最高):
                 浅体扫 / 近距离框选时可能只到几 km, 高层等温线就找不到了。
                 做法: 按高度分箱(bin_m)取各箱温度中位数 -> 找穿越点 -> 线性插值。
                 分箱取中位数是为了压掉插值噪声。

    参数
      radar:      pyart Radar(仅"从温度场反推"这条路需要), 含 'temperature' 字段
      temp_c:     等温线温度 (°C)。0.0 = 0 °C 层/融化层; -20.0 = 混合相区顶(常用约定)。
      temp_field: 温度字段名, 默认 'temperature'(step05_temperature 产出, °C)
      bin_m:      高度分箱厚度 (m); 只在"从雷达温度场反推"这条路上用
      min_bin_n:  一个箱至少多少个库点才参与(太少不要); 同上
      strict:     True = 等温线**不在廓线范围内**时返回 None(调用方自己决定怎么办);
                  False(默认) = 退化成"廓线最低/最高层高度"的近似值(老行为, 向后兼容)。
                  ★ 给指标(7) 找混合相区顶时必须用 strict=True —— 否则若数据没覆盖到
                  -20 °C(例如浅体扫), 会拿"廓线顶"当上界, 把柱子**错误截短**。
      snd:        QC.read_sounding() 的返回 dict。给了就直接查探空原始廓线 —— 推荐。

    返回 float(高度 m); 没有可用数据 / 有效点太少 / 找不到穿越 -> None(见 strict)。
    ★ 2026-09-14: 由 zero_degree_height 泛化而来, 目的是给指标(7) 判据① 找一个
      "混合相区顶"上界(论文 Fig.3 Step 2: from low level to the mixed-phase region)。
      zero_degree_height 保留为 temp_c=0.0 的薄包装, 老调用不受影响。
    ★ 同日追加 snd 通路: 原先只认雷达温度场, 而那个场的高度上限由**雷达库点**决定
      (探空再高也进不来), 浅体扫时会拿"廓线顶"冒充等温线。探空原始廓线一直在手上,
      直接查它既准确又不依赖体扫覆盖。
    """
    tc = float(temp_c)

    # --- 1. 取廓线: 优先探空原始廓线 ---
    if snd is not None:
        h = np.asarray(snd.get('heights_m', []), dtype=float).ravel()
        t = np.asarray(snd.get('temps_C', []), dtype=float).ravel()
        ok = np.isfinite(h) & np.isfinite(t)
        if ok.sum() < 2:
            return None
        h, t = h[ok], t[ok]
        order = np.argsort(h)
        prof_z, prof_t = h[order], t[order]
    else:
        if radar is None or temp_field not in radar.fields:
            return None
        t = np.ma.filled(radar.fields[temp_field]['data'], np.nan).astype(float).ravel()
        z = np.ma.filled(radar.gate_altitude['data'], np.nan).astype(float).ravel()
        ok = np.isfinite(t) & np.isfinite(z)
        if ok.sum() < max(min_bin_n, 10):
            return None
        t, z = t[ok], z[ok]

        lo = np.floor(z.min() / bin_m) * bin_m
        edges = np.arange(lo, z.max() + bin_m, bin_m)
        if edges.size < 3:
            return None
        idx = np.clip(np.digitize(z, edges) - 1, 0, edges.size - 2)

        pz, pt = [], []
        for i in range(edges.size - 1):
            m = idx == i
            if m.sum() >= min_bin_n:
                pz.append(0.5 * (edges[i] + edges[i + 1]))
                pt.append(float(np.median(t[m])))
        if len(pz) < 3:
            return None
        prof_z = np.asarray(pz, dtype=float)
        prof_t = np.asarray(pt, dtype=float)

    # --- 2. 找穿越点 ---
    warmer = np.where(prof_t >= tc)[0]         # 该等温线以下(更暖)的箱
    if warmer.size == 0:                       # 整层都比 tc 冷 -> tc 在廓线底之下
        return None if strict else float(prof_z[0])
    i0 = int(warmer[-1])                       # 最上面一个 >= tc 的箱
    if i0 >= prof_t.size - 1:                  # 整层都比 tc 暖 -> tc 在廓线顶之上
        return None if strict else float(prof_z[-1])
    t0, t1 = prof_t[i0], prof_t[i0 + 1]
    z0, z1 = prof_z[i0], prof_z[i0 + 1]
    if t1 == t0:
        return float(z0)
    return float(z0 + (tc - t0) * (z1 - z0) / (t1 - t0))


def zero_degree_height(radar, temp_field='temperature', bin_m=100.0, min_bin_n=5):
    """0 °C 层高度 (m, AGL) = isotherm_height(radar, 0.0) 的薄包装(向后兼容)。"""
    return isotherm_height(radar, 0.0, temp_field=temp_field,
                           bin_m=bin_m, min_bin_n=min_bin_n)


def compute_zdr_column_field(radar, box_mask=None, grid=None, zdr_field='cor_zdr',
                             ref_field='cor_z', z0c_m=None, z_upper_m='config',
                             snd=None, zdr_min_db='config',
                             below_fzl_m='config', dz_m='config', dxy_m='config',
                             z_top_m='config', margin_m='config',
                             use_connectivity='config', use_negative_gradient='config',
                             grad_tol_db='config', min_voxels='config',
                             ref_min_dbz='config', attach_to_grid=True, verbose=False):
    """
    指标(7) 核心实现: 用 "3D mapping columns" 法(Zhao et al. 2025) 在网格上识别 ZDR 柱。

    三条判据(全部满足才是柱子里的格点):
      ① 阈值:      ZDR >= zdr_min_db(默认 1.5 dB), 且位于
                   [融化层 - below_fzl_m, 混合相区顶] **且落在人工框选的经纬度范围内**;
                   上界默认取探空温度场的 -20 °C 等温线高度(见 z_upper_m /
                   ZDRCOL_MIXED_PHASE_TOP_C), 与论文 Fig.3 Step 2 的
                   "from low level to the mixed-phase region" 对齐;
      ② 柱状形态:  必须与"融化层以下 1 km ~ 融化层"这段下部高值区**同列垂直连续**
                   (论文的 successive columnar: 逐层 `mask[k] &= mask[k-1]`, 无横向自由度),
                   排除混合相区里悬空的高 ZDR 假信号(三体散射/退偏振条纹/扁冰晶);
      ③ 梯度:      融化层以上, ZDR **向上非增**(相邻两层之差 <= 0, 等于 0 也算通过)
                   —— 尺度分选指纹。跟论文 Fig.3 Step 3 的 "y − x <= 0" 逐字一致。

    ★★ 与论文原口径的对照(逐条核对过, 详见 config.py 的 ZDRCOL_* 注释; Fig.3 流程图为证):
      【2026-10-06 已按论文改到位的三项】
      · 判据② ✔ **逐列垂直连续**: 从种子板层(融化层下1 km ~ 融化层)往上, 逐层做
        `mask[k] &= mask[k-1]` —— 严格同列 (i, j) 不变。此前用的是 6-邻域三维连通
        (ndimage.label), 它**只比6 邻域**(面邻居, 不含对角),但那已经包含
        "**同一层内左右相邻**": 柱子水平错位时, 错位出去的那一列只要与原列同层相邻,
        就仍被判为连通而留下 ⇒ **柱体偏胖、柱顶偏高**。**本条已于 2026-10-06 对齐论文。**
        (实测 cell1 前6 帧: 柱体格点 199 -> 104, 柱高最多矮 500 m)
      · 柱"高" ✔ **depth_m = 层数 n × 层高**(论文 Sect. 3.4: "从融化层往上数到柱内
        最高层的层数 n × 0.5 km"; Fig. 2c: 17:24 白点占 2 层 -> 高 1 km)。
        这是**离散的层数 × dz**, 不是"柱顶海拔 − 0 °C 层高度"那个连续差。
        旧的 `height_m` 已**删除**; 柱顶海拔改用 `z_top_agl`(辅助量, 明确不是论文柱高),
        `base_m`(柱底海拔)同理保留, 另有 `n_layers`(层数 n)可直接查。
      · 判据③ ✔ `y − x <= 0`(`grad_tol_db=0` 时与论文原式逐字等同, 等于 0 也保留)。
      【仍与论文不同的一处 —— 用户 2026-10-06 决定保持现状】
      · 网格: 论文 = 0.25 km 水平 / 500 m 垂直 / 500 m~20 km 共 40 层; 本链默认 150 m /
        250 m / 0~15 km —— 与 VIL 同套, **比论文细** ⇒ 高度量化(250 m vs 500 m)与体积
        单元不同。要与论文数值直接对比就把 ZDRCOL_GRID_DZ_M 改 500.0、
        ZDRCOL_GRID_DXY_M 改 250.0、ZDRCOL_Z_TOP_M 改 20000.0。
      【论文未给数值、本链取通行约定的两处】
      · 判据① 高度范围 [融化层−1 km, **混合相区顶**](Fig.3 Step 2 明写 "from low level
        to the mixed-phase region"); 上界默认取 **-20 °C 等温线高度**(探空温度场反推)。
        论文未给该高度的具体数值, -20 °C 是最常见约定 —— 想改就改
        C.ZDRCOL_MIXED_PHASE_TOP_C 一个数。
      · 论文体积 = 格点数 × 单格体积(论文固定 0.03125 km³); 本函数按实际格点体积算。
      【本链额外加的约束(论文没有, 但按项目口径保留)】
      · 判据① 额外要求格点**落在人工框选的经纬度范围内**(项目口径; 论文是整场)。
      · 论文不用 ZH 阈值(Woodard 等的 ZH≥40 dBZ 法被论文明确弃用), 故 ref_min_dbz 默认 None。

    ★ 柱体 = 上述格点里 **位于融化层(0°C 层)以上** 的那部分(2026-09-14 明确):
      定义上 ZDR 柱必须"伸出融化层", 融化层以下那截只是判据②的连通锚, 不算柱体。
      所以若三条判据筛完后 **融化层以上一个格点都没有**, 就判为"无柱"(各值 NaN),
      而不是把一坨贴在融化层下面的高 ZDR 雨区当成柱子。

    ★ ZDR 字段的选择(2026-09-14 已从源头解决, 现在没有坑):
      默认 'cor_zdr'。**2026-09-14 起衰减订正只订反射率 ⇒ cor_zdr 就是原始
      differential_reflectivity 的直通**(只套 QC 掩膜, 数值完全相同), 全域(含融化层
      以上)都有值, 指标(7) 直接可用; 换成 'differential_reflectivity' 结果也一样。
      历史坑(留档): 在这之前 cor_zdr 是 Park2005 的雨区产物(温度>0 + 有回波 + φdp 有效),
      **融化层以上整片缺测** —— 而柱子恰长在融化层以上, 于是判据① 永远 0 格点、指标(7)
      恒为"无柱"。实测威宁 20250509 单个体扫: 6200 万门里订正版 cor_zdr 只有 331 门有效,
      原始 differential_reflectivity 有 6.8 万门(遍布所有仰角)。差分订正代码现已删除。

    参数
      box_mask:  框选 3D 掩膜 —— **既决定网格水平范围, 也是最终的统计范围**
                 (2026-09-15 明确: 网格化时包络外扩 margin 只为边缘列插值, 统计时
                 用 _grid_box_mask 把范围收回到框选内, 见判据①; 传 None = 全部有效
                 回波, 此时不做水平限制)
      grid:      已网格化的 pyart Grid(需含 zdr_field)。传了就直接用, **不再网格化**。
                 (2026-09-14 起工作.py **不再**复用 VIL 的网格 —— VIL 产物要不含 ZDR;
                 不传本参数时自己网格化一次, 只装 zdr_field 一个字段。)
                 ⚠ 传 grid 时, dz_m/dxy_m/z_top_m/margin_m 四个"网格分辨率"参数
                 会被忽略(网格已经定好了), 只有阈值/0°C 层/判据开关仍然生效。
      zdr_field: ZDR 字段名, 默认 'cor_zdr'(= 原始 ZDR 直通, 全域可用); 传 'config' 取
                 config.py 的 ZDRCOL_ZDR_FIELD; 也可直接传 'differential_reflectivity'
      ref_field: 反射率字段名(仅当 ref_min_dbz 不为 None 时用)
      z0c_m:     0°C 层高度 (m)。None(默认) 时按这个顺序找: ① 传了 snd 就用探空的
                 fzl(read_sounding 用探空原始廓线算的融化层) -> ② 从 radar 的
                 temperature 场反推(见 zero_degree_height)。也可直接给米数。
      z_upper_m: 判据① 的**上界高度** (m, AGL) = 论文 Fig.3 的"混合相区顶"。
                 'config'(默认): 用 config.py 的 ZDRCOL_MIXED_PHASE_TOP_C(默认 -20.0 °C)
                 算该等温线高度; 若那是 None 或算不出来, 则不设上界(一路判到网格顶)
                 并告警。
                 给具体数值: 直接用该高度(如 8000 = 8 km)。
                 给 None: 显式不设上界。
      snd:       QC.read_sounding() 的返回 dict ★ 强烈建议传。
                 · 0 °C 层优先用它的 fzl(与衰减订正用的那条融化层同源, 全链一致);
                 · -20 °C 层(判据① 上界)直接查它的原始廓线 —— 探空一般到 30 km 以上,
                   一定找得到, 不受"雷达库点最高只到几 km"的浅体扫限制。
                 不传也能跑, 但两条等温线只能从雷达 temperature 场反推, 那个场的
                 高度上限 = min(探空顶, 雷达库点最高), 覆盖不足时会退化成"不设上界"。
      zdr_min_db / below_fzl_m / dz_m / dxy_m / z_top_m / margin_m /
      use_connectivity / use_negative_gradient / grad_tol_db / min_voxels /
      ref_min_dbz: 'config' 取 config.py 的 ZDRCOL_* 值(见其说明)
      attach_to_grid: True 时把柱子掩膜作为字段 'ZDRCOL'(0/1) 挂到 grid 上,
                 这样存 grid .nc 时柱体位置一起带走, 读回可直接画
      verbose:   打印判定过程的中间量

    返回 dict:
      'mask3d'       : (nz,ny,nx) bool —— **柱体**元掩膜(融化层以上部分)
      'mask_conn3d'  : (nz,ny,nx) bool —— 判据②③筛完后、尚未切掉融化层以下的那份
                       (含连通的雨区锚; 只在调试/画图对比时用)
      'top2d'        : (ny,nx) 每列柱顶高度 (m, 无柱处 NaN)
      'depth_m'      : **柱"高"**(m) = 层数 n × 层高 dz —— 论文 Sect. 3.4 的原始口径,
                       即"从融化层往上数到柱内最高层有多少层"。离散量, 必然是 dz 的整数倍。
      'n_layers'     : 柱高对应的**层数 n**(论文原始量; 无柱 -> 0)
      'z_top_agl'    : 柱顶**海拔**(m) —— 旧 `height_m` 的替代, 仅供核查/画图,
                       **不是**论文的柱高(论文柱高 = depth_m)
      'base_m'       : 柱底海拔 (m) —— 辅助量(论文无此定义)
      'volume_km3'   : 柱体体积 (km³, 只算融化层以上部分)
      'n_voxels'     : 柱体元个数(融化层以上)
      'n_voxels_conn': 判据②③筛完后的格点个数(含融化层以下, 便于判断是被哪一步筛没的)
      'max_zdr_db'   : 柱内最大 ZDR (dB)
      'z0c_m'        : 用到的 0°C 层高度 (m)
      'z_lo_m'       : 判定下界 = z0c_m - below_fzl_m (m)
      'zdr_field'    : 实际用到的 ZDR 字段名
      --- 水平位置(2026-09-14 新增, 单位: 度; 无柱时全 NaN) ---
      'lon_c'/'lat_c'    : 柱体水平质心 (经度 °E / 纬度 °N)
      'lon_min'/'lon_max': 柱体水平外包框的经度范围 (°E)
      'lat_min'/'lat_max': 柱体水平外包框的纬度范围 (°N)
      'top_lon'/'top_lat': 柱顶最高处的位置 (同高多列取平均)
      'n_cols'/'area_km2': 柱体水平投影占的列数 / 投影面积 (km²)
      'z_axis'/'lon'/'lat'/'grid'/'zdr_min_db'   (lon/lat 为 2D 网格经纬度)
    """
    out = {'mask3d': None, 'mask_conn3d': None, 'base_m': np.nan,
           'depth_m': np.nan, 'n_layers': 0, 'z_top_agl': np.nan,
           'volume_km3': np.nan, 'n_voxels': 0, 'n_voxels_conn': 0,
           'max_zdr_db': np.nan, 'z0c_m': None, 'z_lo_m': np.nan, 'z_upper_m': None,
           'z_upper_src': None,
           'zdr_min_db': np.nan,
           'zdr_field': None, 'top2d': None, 'grid': grid, 'z_axis': None,
           'lon': None, 'lat': None, 'box2d': None,     # box2d: 水平框选掩膜(网格点)
           # --- 柱体水平位置(2026-09-14 新增, 供统计文件记录) ---
           'lon_c': np.nan, 'lat_c': np.nan,            # 柱体水平质心 (2D 投影)
           'lon_min': np.nan, 'lon_max': np.nan,        # 柱体水平外包框
           'lat_min': np.nan, 'lat_max': np.nan,
           'top_lon': np.nan, 'top_lat': np.nan,        # 柱顶(最高)所在位置
           'area_km2': np.nan, 'n_cols': 0}             # 水平投影面积 / 占的列数

    # --- 0. 参数 ---
    if zdr_field == 'config':
        zdr_field = C.ZDRCOL_ZDR_FIELD
    out['zdr_field'] = zdr_field
    zdr_min = C.ZDRCOL_ZDR_MIN_DB if zdr_min_db == 'config' else float(zdr_min_db)
    below = C.ZDRCOL_BELOW_FZL_M if below_fzl_m == 'config' else float(below_fzl_m)
    dz = C.ZDRCOL_GRID_DZ_M if dz_m == 'config' else float(dz_m)
    dxy = C.ZDRCOL_GRID_DXY_M if dxy_m == 'config' else float(dxy_m)
    z_top = C.ZDRCOL_Z_TOP_M if z_top_m == 'config' else float(z_top_m)
    margin = C.ZDRCOL_GRID_MARGIN_M if margin_m == 'config' else float(margin_m)
    conn = C.ZDRCOL_USE_CONNECTIVITY if use_connectivity == 'config' else bool(use_connectivity)
    grad = C.ZDRCOL_USE_NEG_GRADIENT if use_negative_gradient == 'config' else bool(use_negative_gradient)
    g_tol = C.ZDRCOL_GRAD_TOL_DB if grad_tol_db == 'config' else float(grad_tol_db)
    min_vox = int(C.ZDRCOL_MIN_VOXELS if min_voxels == 'config' else min_voxels)
    ref_min = C.ZDRCOL_REF_MIN_DBZ if ref_min_dbz == 'config' else ref_min_dbz
    out['zdr_min_db'] = zdr_min

    # --- 1. 0°C 层高度(优先探空 fzl —— 与衰减订正用的那条融化层同源) ---
    _snd_fzl = None
    if snd is not None:
        try:
            _v = snd.get('fzl')
            if _v is not None and np.isfinite(float(_v)):
                _snd_fzl = float(_v)
        except (TypeError, ValueError, AttributeError):
            _snd_fzl = None
    if z0c_m is None:
        if _snd_fzl is not None:
            z0c = _snd_fzl
            if verbose:
                print(f'  0°C 层: {z0c:.0f} m (探空 fzl)')
        else:
            z0c = zero_degree_height(radar)
            if verbose and z0c is not None:
                print(f'  0°C 层: {z0c:.0f} m (雷达 temperature 场反推)')
        if z0c is None:
            print('  [警告] 既没探空 fzl 也推不出 0°C 层高度 -> 跳过指标(7) ZDR 柱。'
                  '请传 snd=read_sounding(...) 或 z0c_m=(融化层高度), 或先跑温度场')
            return out
    else:
        z0c = float(z0c_m)
    out['z0c_m'] = z0c
    out['z_lo_m'] = z_lo = z0c - below

    # --- 1.5 判据① 上界 = 混合相区顶 (论文 Fig.3 Step 2: from low level to the
    #        mixed-phase region)。默认取 -20 °C 等温线高度。
    #        ★ 传了 snd 就直接查探空原始廓线(不受雷达库点高度限制) ---
    if z_upper_m == 'config':
        _tc = C.ZDRCOL_MIXED_PHASE_TOP_C
        if _tc is None:
            z_upper, _src = None, 'off'
            if verbose:
                print('  判据① 上界: 未设(ZDRCOL_MIXED_PHASE_TOP_C=None) -> 判到网格顶')
        else:
            z_upper = isotherm_height(radar, float(_tc), strict=True, snd=snd)
            _src = 'sounding' if snd is not None else 'radar'
            if z_upper is None:
                print(f'  [警告] 找不出 {float(_tc):g} °C 等温线高度 -> 判据① 不设上界, '
                      f'一路判到网格顶。要上界就传 z_upper_m=<米>, 或传 snd=探空'
                      f'(探空廓线到 30 km 以上, 一定覆盖得到)。')
                _src = 'off'
            elif verbose:
                print(f'  判据① 上界: {float(_tc):g} °C 层 = {z_upper:.0f} m (混合相区顶, '
                      f'{"探空廓线" if _src == "sounding" else "雷达温度场"})')
    elif z_upper_m is None:
        z_upper, _src = None, 'off'
        if verbose:
            print('  判据① 上界: 显式关闭 -> 判到网格顶')
    else:
        z_upper, _src = float(z_upper_m), 'manual'
        if verbose:
            print(f'  判据① 上界: 手动指定 {z_upper:.0f} m')
    if z_upper is not None and z_upper <= z_lo:
        print(f'  [警告] 判据① 上界({z_upper:.0f} m)不高于下界({z_lo:.0f} m) '
              f'-> 判定范围为空, 必然判为无柱')
    out['z_upper_m'] = z_upper
    out['z_upper_src'] = _src          # 'sounding' / 'radar' / 'manual' / 'off'

    # --- 2. 网格(没现成的就自己网格化 ZDR; 这份网格只服务指标(7), 与 VIL 无关) ---
    if grid is None:
        if zdr_field not in radar.fields:
            print(f'  [警告] 无 {zdr_field} 字段 -> 跳过指标(7) ZDR 柱')
            return out
        # 2026-09-14 起 ZDR 柱不再复用 VIL 的网格 -> 若启用了反射率下限(ref_min_dbz),
        # 得把 ref_field 一起网格化进来, 否则下面的 "ref_field in grid.fields" 判据会
        # 静默跳过(以前靠 VIL 网格里正好有 cor_z 兜住)。默认 ref_min_dbz=None, 不额外网格化。
        fields = [zdr_field]
        if ref_min is not None:
            if ref_field in radar.fields and ref_field != zdr_field:
                fields.append(ref_field)
            else:
                print(f'  [警告] 要用反射率下限但 radar 里没有 {ref_field} 字段'
                      f' -> 该判据不生效')
        grid = _cartesian_grid(radar, box_mask, fields, dz, dxy, z_top, margin,
                               verbose=verbose)
    if zdr_field not in grid.fields:
        print(f'  [警告] 网格里没有 {zdr_field} 字段 -> 跳过指标(7) ZDR 柱')
        return out
    out['grid'] = grid

    z_axis = np.asarray(grid.point_z['data'])[:, 0, 0].astype(float)          # (nz,) m
    zdr3 = np.ma.filled(grid.fields[zdr_field]['data'], np.nan).astype(float)  # (nz,ny,nx)
    out['z_axis'] = z_axis
    # lon/lat 给 2D 的(ny,nx): 画水平分布/俯视对照用; 3D 版本可由 grid 自己取
    out['lon'] = np.ma.filled(grid.point_longitude['data'][0], np.nan).astype(float)
    out['lat'] = np.ma.filled(grid.point_latitude['data'][0], np.nan).astype(float)

    # --- 3. 判据①: 阈值 + 高度范围 + **人工框选的水平范围**(+ 可选反射率下限) ---
    mask = np.isfinite(zdr3) & (zdr3 >= zdr_min)
    # ★ 2026-09-15: 水平上只统计人工框选的区域。前面网格化的包络外扩了 margin
    #   (默认 2 km, 只为让边缘列插值不缺数据), 那圈不参与统计 —— 否则框外 2 km 内的
    #   柱子会被当成"框内的柱子"。容差取半个格距, 抵消 x/y(米)->经纬度 的换算差。
    box2d = _grid_box_mask(radar, grid, box_mask, tol_m=0.5 * float(dxy))
    out['box2d'] = box2d
    if box2d is not None:
        if not box2d.any():
            print('  [警告] 框选范围映射到网格后一个格点都没有(网格与框选不匹配)'
                  ' -> 跳过指标(7) ZDR 柱')
            return out
        mask &= box2d[None, :, :]
    mask &= (z_axis >= z_lo)[:, None, None]
    if z_upper is not None:
        mask &= (z_axis <= z_upper)[:, None, None]         # 上界 = 混合相区顶
    if ref_min is not None and ref_field in grid.fields:
        ref3 = np.ma.filled(grid.fields[ref_field]['data'], np.nan).astype(float)
        mask &= np.isfinite(ref3) & (ref3 >= float(ref_min))
    n0 = int(mask.sum())
    if verbose:
        _up = f'{z_upper:.0f} m' if z_upper is not None else '网格顶'
        _hx = (f'水平: 只用框选范围内的 {int(box2d.sum())}/{box2d.size} 列'
               if box2d is not None else '水平: 未限制(没给框选掩膜)')
        print(f'  判据① 阈值+范围: {n0} 格点 (ZDR>={zdr_min} dB, '
              f'{z_lo:.0f} m <= z <= {_up}; {_hx})')
    if n0 == 0:
        # 提示: 融化层以上整片没有有效 ZDR —— 通常意味着这份 radar 是 2026-09-14 之前
        # 用旧流程处理的(那时 cor_zdr 是 Park2005 的雨区产物, 融化层以上缺测, 而柱子恰
        # 长在那里)。给一条明确提示, 免得无声返回 NaN 让人以为"今天没柱子"。
        n_valid_above = int((np.isfinite(zdr3) & (z_axis >= z0c)[:, None, None]).sum())
        if n_valid_above == 0:
            print(f'  [提示] 融化层({z0c:.0f} m)以上 "{zdr_field}" 一个有效格点都没有 -> '
                  f'指标(7) 恒为"无柱"。当前处理链的 cor_zdr 是原始 ZDR 直通(全域有值), '
                  f'若这份 radar 是旧流程产物, 请用当前链重跑。')
        return out

    # --- 4. 判据③(先做, 免得它对"柱顶"的定义被②改掉): 融化层以上 ZDR 向上**非增** ---
    #  论文 Fig.3 Step 3 原式是 "y − x <= 0"(y=上层对应格, x=下层), **等于 0 也保留**;
    #  2026-09-14 起本实现跟论文一致: <=, 即上下两层 ZDR 完全相同**不算违反**。
    #  g_tol 的含义 = "要求下降的幅度至少这么多"(0 = 只要求不上升)。
    if grad:
        above = np.where(z_axis >= z0c)[0]
        for k in above:
            if k == 0:
                continue
            cur, low = zdr3[k], zdr3[k - 1]
            pair_ok = ~(np.isfinite(cur) & np.isfinite(low))       # 缺一层 -> 不判定
            dec = (cur - low) <= -g_tol                           # 向上非增(等于 0 也留)
            mask[k] &= (pair_ok | dec)
        if verbose:
            print(f'  判据③ 向上非增(>={z0c:.0f} m): 剩 {int(mask.sum())} 格点')

    # --- 5. 判据②: 柱状形态 —— 论文 Fig.3 Step 2 的**逐列垂直连续** ---
    #  ★★ 2026-10-06 重写: 此前用的是 6-邻域三维连通(ndimage.label), 与论文**不一致** ——
    #  6 邻域 = 面邻居(不含对角), 但**包含"同一层内左右相邻"**: 柱子水平错位时, 错位
    #  出去的那一列只要与原列同层相邻, 就仍被判为连通而留下 ⇒ **多留格点** ⇒ 柱体偏胖、
    #  柱顶偏高。论文要的是 **successive columnar**: 高值格点必须与 0 °C 层以下那片
    #  种子区**同列垂直**连续(同一个 (i, j) 不变)。
    #  正解 = 从种子板层往上逐层做 `mask[k] &= mask[k-1]`(纯逐列, 无横向自由度)。
    if conn:
        slab = (z_axis >= z_lo) & (z_axis <= z0c)
        if not slab.any():
            print('  [提示] 判定范围内没有"融化层以下"的层, 柱状约束无从谈起 -> 判为无柱')
            return out
        k_seed = np.where(slab)[0]                  # 种子板层的层号(连续区间)
        k_lo, k_hi = int(k_seed.min()), int(k_seed.max())
        if verbose:
            print(f'  判据② 逐列连续(论文口径): 种子板层 k={k_lo}..{k_hi} '
                  f'({z_axis[k_lo]:.0f}~{z_axis[k_hi]:.0f} m), 向上逐层 mask[k] &= mask[k-1]')
        # 种子层以下(判定范围下界之下)不参与向上约束, 但也不该留高值 ——
        # 判据① 已经把 z < z_lo 的格点全剔了, 所以这里只需从 k_lo+1 往上迭代。
        for k in range(k_lo + 1, mask.shape[0]):
            mask[k] &= mask[k - 1]
        if verbose:
            print(f'  判据② 逐列连续: 剩 {int(mask.sum())} 格点')

    # --- 6. 小碎块过滤 ---
    if min_vox > 1 and mask.any():
        from scipy import ndimage
        lab, nlab = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
        sizes = ndimage.sum(mask, lab, index=np.arange(1, nlab + 1))
        keep = np.where(sizes >= min_vox)[0] + 1
        mask = np.isin(lab, keep)

    n_sel = int(mask.sum())
    out['mask_conn3d'] = mask.copy()          # 判据②③筛完、尚未切掉融化层以下的版本
    out['n_voxels_conn'] = n_sel
    if n_sel == 0:
        if verbose:
            print('  最终: 无 ZDR 柱(三条判据后已无格点)')
        return out

    # --- 6.5 切出"融化层以上"的柱体 ---
    #  定义: ZDR 柱必须伸出融化层; 融化层以下那截只当判据②的连通锚, 不算柱体。
    #  若此处为空 -> 说明高 ZDR 区全贴在融化层下面(那是雨区不是柱), 判为无柱。
    above_fzl = (z_axis >= z0c)[:, None, None]
    col = mask & above_fzl
    n_col = int(col.sum())
    if n_col == 0:
        if verbose:
            print(f'  最终: 无 ZDR 柱 —— 判据②③后 {n_sel} 格点全部位于 0°C 层'
                  f'({z0c:.0f} m)以下, 未伸出融化层')
        return out
    if verbose:
        print(f'  融化层以上柱体: {n_col} 格点 (连通结构共 {n_sel}, '
              f'其中 {n_sel - n_col} 在 0°C 层以下)')

    # --- 7. 定量: 柱高 / 体积 / 最大 ZDR (全部只统计融化层以上的柱体) ---
    #  ★★ 2026-10-06 论文口径: 柱"高"= **从融化层往上数到柱内最高层的层数 n × 层高**
    #  (Zhao et al. 2025 Sect. 3.4 明写; Fig. 2c: 17:24 白点占 2 层 -> 高 1 km ✔)。
    #  ⇒ 高度是**离散的层数 × dz**, 不是"柱顶海拔 − 0 °C 层高度"那个连续差。
    #  旧的 height_m(柱顶海拔 AGL)已按用户要求删除, 只留 depth_m。
    z_per_vox = np.broadcast_to(z_axis[:, None, None], col.shape)[col]
    dvox = float(dxy) * float(dxy) * float(np.median(np.diff(z_axis)))
    top2d = np.where(col.any(axis=0),
                     np.where(col, z_axis[:, None, None], -np.inf).max(axis=0), np.nan)

    #融化层所在层号(第一个 z >= z0c 的层)= 论文的 n = a; 柱顶层号 k_top
    #  ⚠ 2026-10-06 修 bug: 原写作 col.any(axis=0).any(axis=1) —— 那只压掉了 ny 维,
    #  返回的是 (nx,) 的"每列是否有柱", 再 .max() 拿到的是**列号**不是层号
    #  (自测里柱在 ny=2 行 -> 误取 k_top=2 -> 柱高算成负数)。
    #  正确: 对 (ny, nx) 两维一起取 any, 得到 (nz,) 的逐层标记, 再取最大层号。
    k_fzl = int(np.where(z_axis >= z0c)[0].min())
    k_top = int(np.where(col.any(axis=(1, 2)))[0].max())
    n_layers = k_top - k_fzl                      # 论文的 n(层数)
    dz_eff = float(np.median(np.diff(z_axis)))
    if n_layers <= 0:
        # 走到这里说明 col非空, 但**所有柱体格点都恰好落在 0 °C 层那一层里**
        # (k_top == k_fzl) => 柱子没有"伸出"融化层, 层数 n = 0 => 柱高 0。
        # 这不是"没有柱", 而是"柱高为零"(高 ZDR 全贴在融化层上, 没往上长) ——
        # 物理上更像雨区而不是雹柱, 按"无柱"处理并说明, 别让上层误以为是 bug。
        print(f'  [提示] 柱体全部贴在 0 °C 层({z0c:.0f} m)那一层内, 未向上伸出 '
              f'=> 层数 n = 0, 柱高 0 -> 判为无柱')
        return out

    out['mask3d'] = col
    out['n_voxels'] = n_col
    out['top2d'] = top2d
    out['n_layers'] = n_layers                    # 柱高 = n 层(论文原始量)
    out['depth_m'] = n_layers * dz_eff            # 柱"高" = n × dz (论文口径)
    out['z_top_agl'] = float(z_axis[k_top])       # 柱顶海拔(辅助量: 旧 height_m 的替代,
                                                  #   只供核查/画图, **不是**论文柱高)
    out['base_m'] = float(z_per_vox.min())        # 柱底海拔 AGL(辅助量, 论文无此定义)
    out['volume_km3'] = n_col * dvox / 1e9
    out['max_zdr_db'] = float(np.nanmax(np.where(col, zdr3, np.nan)))

    # --- 7.5 柱体水平位置(2026-09-14 新增): 质心 / 外包框 / 柱顶所在处 ---
    #   位置一律取**网格经纬度**(格点是等经纬间距的, 见 _cartesian_grid 的 origin),
    #   供统计文件(CSV/Excel)记录"这根柱子长在哪"。
    lon2 = np.asarray(out['lon'], dtype=float)          # (ny,nx)
    lat2 = np.asarray(out['lat'], dtype=float)
    hcol = col.any(axis=0)                              # 柱体在水平面上的投影 (ny,nx)
    out['n_cols'] = int(hcol.sum())
    if hcol.any():
        ok = hcol & np.isfinite(lon2) & np.isfinite(lat2)
        if ok.any():
            lons, lats = lon2[ok], lat2[ok]
            out['lon_c'] = float(lons.mean())           # 质心(按列等权)
            out['lat_c'] = float(lats.mean())
            out['lon_min'], out['lon_max'] = float(lons.min()), float(lons.max())
            out['lat_min'], out['lat_max'] = float(lats.min()), float(lats.max())
            out['area_km2'] = out['n_cols'] * float(dxy) * float(dxy) / 1e6
        # 柱顶最高处的位置: 同高可能有多列, 取它们的平均(容差 0.5 个层厚)
        if n_col > 0:
            tol_top = 0.5 * dz_eff
            ktop = (np.isfinite(top2d) & (top2d >= z_axis[k_top] - tol_top)
                    & np.isfinite(lon2) & np.isfinite(lat2))
            if ktop.any():
                out['top_lon'] = float(lon2[ktop].mean())
                out['top_lat'] = float(lat2[ktop].mean())

    if verbose:
        print(f'  最终: {n_col} 格点, 柱高 {out["depth_m"]:.0f} m '
              f'(= {out["n_layers"]} 层 x {dz_eff:.0f} m, 论文口径; '
              f'柱顶海拔 {z_axis[k_top]:.0f} m, 0 °C 层 {z0c:.0f} m), '
              f'柱底 {out["base_m"]:.0f} m, '
              f'体积 {out["volume_km3"]:.3f} km^3, 柱内最大 ZDR {out["max_zdr_db"]:.2f} dB')
        print(f'  水平位置: 质心 ({out["lon_c"]:.4f}°E, {out["lat_c"]:.4f}°N), '
              f'范围 lon {out["lon_min"]:.4f}~{out["lon_max"]:.4f}°E / '
              f'lat {out["lat_min"]:.4f}~{out["lat_max"]:.4f}°N, '
              f'柱顶最高处在 ({out["top_lon"]:.4f}°E, {out["top_lat"]:.4f}°N), '
              f'投影 {out["n_cols"]} 列 / {out["area_km2"]:.2f} km^2')

    # --- 8. 把柱体掩膜挂到 grid(0/1 场), 存 .nc 时一起带走 ---
    if attach_to_grid:
        grid.add_field('ZDRCOL', {
            'data': np.ma.masked_where(~col, col.astype('float32')),
            'units': '1',
            'long_name': ('ZDR column mask above the freezing level '
                          '(3D mapping columns, Zhao et al. 2025)'),
            '_FillValue': -9999.0,
        }, replace_existing=True)

    return out


def metric_zdr_column(radar, box_mask, return_field=False, **kwargs):
    """
    指标 (7): ZDR 柱高 + 体积(另附柱底/柱内最大 ZDR/0°C 层高度)。

    实现见 compute_zdr_column_field(Zhao et al. 2025 "3D mapping columns" 法)。
    柱体只统计**融化层以上**的部分(定义上柱子必须伸出 0°C 层; 融化层以下那截只当
    判据②的连通锚), 三条判据筛完后融化层以上为空 -> 判为无柱。

    参数透传给它(如 zdr_min_db=1.0 可换成老文献阈值; grid=<已有 Grid> 可复用网格,
    此时网格分辨率跟那个 Grid 走; z0c_m=<探空 fzl> 可省掉从温度场反推)。
    ★ 推荐传 snd=QC.read_sounding(...): 0 °C 层用探空 fzl、判据① 上界(-20 °C 混合相区顶)
    直接查探空原始廓线 —— 探空到 30 km 以上, 不受"雷达库点最高只到几 km"的浅体扫限制。
    ★ 工作.py 现在**不传 grid**: ZDR 柱自己网格化(只装 cor_zdr), 与 VIL 完全分开 ——
    这样 VIL 存出来的 vilgrid_*.nc 里只有 cor_z + VIL, 不带任何 ZDR 东西
    (2026-09-14 用户要求)。代价是网格化多做一遍。

    ★ ZDR 字段: 默认 'cor_zdr'。2026-09-14 起衰减订正**只订反射率**, cor_zdr 是原始
    differential_reflectivity 的直通(只套 QC 掩膜), 全域可用 -> 指标(7) 可直接跑。
    此前 cor_zdr 是 Park2005 的雨区产物(融化层以上缺测, 实测一个体扫 6200 万门里只剩
    331 门有效), 那时必须换 zdr_field='differential_reflectivity' 才能识别柱子;
    现在两者数值相同, 差分订正代码已删除。

    return_field=False(默认): 返回 dict(键与 all_metrics 一致, 直接 update 进去)。
    ★ 2026-10-06 起进统计文件的 ZDR 柱量**只留两项**(其余一律不写表):
        'zdrcol_max_zdr_db'  柱内最大 ZDR (dB)            —— ZDR 柱的最大值
        'zdrcol_height_m'    柱**高** (m) = 层数 n × dz   —— 论文 Sect. 3.4 原始口径
    ⚠⚠ **口径变更(2026-10-06)**: `zdrcol_height_m` 这一列的**含义变了**。
       旧值 = 柱顶海拔 AGL(5~8 km 量级); 新值 = 层数 × dz(论文"柱高", 0.25~2 km 量级)。
       ⇒ **已有的 cell1~cell5 统计表(2026-10-06 之前跑的)里这一列与新跑出来的不可比**,
          混在一起会看到"柱高突然掉到 1/5"。要么全量重跑, 要么新列另起名(如
          `zdrcol_depth_m`)再单独处理。列名保留 `zdrcol_height_m` 是因为它字面就对应
          论文的"柱高", 且下游 notebook(make_flat / plot_metrics_*)按该名字读。
    柱底高度 / 体积 / 格点数 / 水平位置等**仍然算**(在 res 里, 供画图与核查),
    只是不再进 CSV/Excel。

    没有温度场 / 没有 ZDR 字段 / 框内无柱时: 两个值均为 NaN, 不报错。
    return_field=True       : 返回 (dict, res), res = compute_zdr_column_field 的完整结果
                              (含 3D 掩膜、层数 n、柱底/体积/位置等, 供画图/存 .nc)。
    """
    res = compute_zdr_column_field(radar, box_mask=box_mask, **kwargs)
    out = {
        'zdrcol_max_zdr_db': res['max_zdr_db'],           # 柱内最大 ZDR (dB)
        'zdrcol_height_m': res['depth_m'],                # 柱"高" = 层数 n × dz (论文口径)
    }
    return (out, res) if return_field else out


def metric_hail_graupel_count(radar, box_mask):
    """
    指标 (5): 冰雹 + 霰粒子数量 + 占有效库比。**需要先做 FHC**(radar 里有 FH 字段)。

    CSU-FHC 类别编号 1..10(csu_fhc_summer 返回 argmax+1):
        7 = 低密度霰, 8 = 高密度霰(两者都算霰), 9 = 冰雹, 10 = 大滴
    返回 (count_hail, count_graupel, count_total, ratio, hail_ratio)
        ratio      = count_total / 框内有效库点数
        hail_ratio = count_hail / count_total(冰雹在"冰雹+霰"里占多少)
    """
    if 'FH' not in radar.fields:
        print('  [警告] 无 FH 字段, 跳过指标(5) —— 请先跑 FHC(add_fhc_field / process_radar)')
        return 0, 0, 0, 0.0, 0.0

    fh = radar.fields['FH']['data']
    mask_arr = _as_bool_mask(fh, fh)
    fhf = np.ma.filled(fh, -1.0)

    valid_in_box = box_mask & ~mask_arr & (fhf >= 0)
    count_hail = int((valid_in_box & (fhf == 9)).sum())
    count_graupel = int((valid_in_box & ((fhf == 7) | (fhf == 8))).sum())
    count_total = count_hail + count_graupel
    total_valid = int(valid_in_box.sum())

    ratio = count_total / total_valid if total_valid > 0 else 0.0
    hail_ratio = count_hail / count_total if count_total > 0 else 0.0
    return count_hail, count_graupel, count_total, ratio, hail_ratio


# =============================================================================
# Step 3.7: 指标(8) —— (1)-(4) 的变化速率(多时次序列差分)
# =============================================================================

# 参与速率计算的指标: (源键 in all_metrics 结果) -> (速率键, 单位)
# ★ 2026-09-24: count_ge45dbz 已不产出 -> 速率表里去掉它(缺键会全 NaN);
#   体积的速率 d_vol45_per_min 已经覆盖"≥45dBZ 那团回波在怎么长"这件事。
_TREND_KEYS = (
    ('max_z',              'd_max_z_per_min',    'dBZ/min'),
    ('echo_top_18dbz_m',   'd_et18_per_min',     'm/min'),
    ('echo_top_45dbz_m',   'd_et45_per_min',     'm/min'),
    ('volume_ge45dbz_km3', 'd_vol45_per_min',    'km3/min'),
    ('vil_max_kgm2',       'd_vil_max_per_min',  'kg/m2/min'),
)


def _volume_datetime(radar):
    """解析体扫开始时刻: radar.time['units'] 里的基准时间 + 第一个门的时间偏移。"""
    import datetime as _dt
    import re
    m = re.search(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}',
                  str(radar.time.get('units', '')))
    if m is None:
        return None
    base = _dt.datetime.strptime(m.group(0).replace('T', ' '), '%Y-%m-%d %H:%M:%S')
    return base + _dt.timedelta(seconds=float(np.asarray(radar.time['data']).ravel()[0]))


def _to_datetime(t):
    """把 datetime / np.datetime64 / 字符串统一成 python datetime; 解析失败返回 None。"""
    import datetime as _dt
    import re
    if isinstance(t, _dt.datetime):
        return t
    if isinstance(t, np.datetime64):
        return t.astype('datetime64[s]').tolist()
    m = re.search(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}', str(t))
    if m is None:
        return None
    return _dt.datetime.strptime(m.group(0).replace('T', ' '), '%Y-%m-%d %H:%M:%S')


def all_metrics_series(radars, lon1, lat1, lon2, lat2, field='cor_z', fhc=False,
                       vil=True, zdr_col=True, zdrcol_kwargs=None, snd=None):
    """
    指标(8) 第 1 步: 多个时次的体扫在**同一个经纬度框**内依次跑 all_metrics。

    参数
      radars: 处理好的 radar 列表(按时间顺序; 各时次须为同一雷达/同一扫描策略)
      lon1, lat1, lon2, lat2: 固定框(四个时次共用, 不要每个时次重新框选)
      fhc:  是否算指标(5); 批量算趋势时默认 False(省时, 指标(8)也用不到)
      vil:  是否算指标(4) VIL(最耗时); 批量不需要时传 False
      zdr_col: 是否算指标(7) ZDR 柱(需要 cor_zdr; 与 VIL 各自网格化, 见 all_metrics)
      snd: QC.read_sounding() 的返回 dict —— 传给指标(7) 用(见 all_metrics)
    返回
      all_metrics 结果的列表(按输入顺序), 每个元素额外带 'time'(体扫开始时刻,
      从 radar.time 解析; 解析不到就不带该键)。

    用法
      series = QC.all_metrics_series(radar_list, lon1, lat1, lon2, lat2)
      rates  = QC.metric_trends(series)          # 相邻时次的变化速率
      QC.print_trends(rates)
    """
    series = []
    for k, radar in enumerate(radars):
        box = build_box_mask_3d(radar, lon1, lat1, lon2, lat2)
        out = all_metrics(radar, box, field=field, fhc=fhc, vil=vil,
                          zdr_col=zdr_col, zdrcol_kwargs=zdrcol_kwargs, snd=snd)
        t = _volume_datetime(radar)
        if t is not None:
            out['time'] = t
        series.append(out)
    return series


def metric_trends(metrics_list, dt_min=None):
    """
    指标(8): (1)-(4) 指标的**变化速率**(相邻时次差分, 单位换算成 每分钟)。

    参数
      metrics_list: all_metrics_series() 的返回(按时间排序); 每个元素带 'time'
                    (datetime/np.datetime64/可解析字符串)最好 —— 时间间隔逐对真实计算
      dt_min:       metrics 里没有 'time' 时必须给, 按等间隔处理(如体扫间隔 6.0 分钟)
    返回
      长度 len-1 的列表, 每个元素是一个 dict:
        'dt_min'             : 该时间对的间隔(分钟)
        't_from_start_min'   : 距首时次的分钟数(有 time 时才有)
        'd_max_z_per_min'    : 最大回波强度变化  (dBZ/min)
        'd_et18_per_min'     : 18dBZ 回波顶高变化 (m/min)
        'd_et45_per_min'     : 45dBZ 回波顶高变化 (m/min)
        'd_count45_per_min'  : >=45dBZ 库数变化   (库/min)
        'd_vol45_per_min'    : >=45dBZ 体积变化   (km3/min)
        'd_vil_max_per_min'  : 最大 VIL 变化      (kg/m2/min)  ← 雹云增长的关键量
    另见 max_trend() 取"最陡增幅"。
    """
    n = len(metrics_list)
    if n < 2:
        raise ValueError('指标(8)至少需要两个时次')
    if not all(isinstance(m, dict) for m in metrics_list):
        raise TypeError('metrics_list 元素须是 all_metrics 系列输出的 dict')

    times = [_to_datetime(m.get('time')) for m in metrics_list]
    if all(t is not None for t in times):
        dts = [(times[i + 1] - times[i]).total_seconds() / 60.0 for i in range(n - 1)]
        t0 = times[0]
    else:
        if dt_min is None:
            raise ValueError("metrics_list 里没有完整的 'time', 请传 dt_min(体扫间隔, 分钟)")
        dts = [float(dt_min)] * (n - 1)
        t0 = None

    rates = []
    for i in range(n - 1):
        dt = dts[i]
        if dt <= 0:
            raise ValueError(f'第 {i}/{i+1} 时次的时间间隔 <= 0, 请检查时间序列')
        row = {'dt_min': dt}
        if t0 is not None:
            row['t_from_start_min'] = (times[i] - t0).total_seconds() / 60.0
        for src, dst, _unit in _TREND_KEYS:
            v0 = metrics_list[i].get(src, np.nan)
            v1 = metrics_list[i + 1].get(src, np.nan)
            ok = np.isfinite(v0) and np.isfinite(v1)
            row[dst] = float(v1 - v0) / dt if ok else np.nan
        rates.append(row)
    return rates


def max_trend(rates, key='d_vil_max_per_min'):
    """
    取某个速率键的**最大增幅**(最陡上升速率)及其时次对序号。
    返回 (rate, index); 全为 NaN 时返回 (nan, -1)。
    """
    vals = np.array([r.get(key, np.nan) for r in rates], dtype=float)
    if not np.isfinite(vals).any():
        return float('nan'), -1
    i = int(np.nanargmax(vals))
    return float(vals[i]), i


def print_trends(rates):
    """把 metric_trends() 的结果打印成人看的表格。"""
    print('(8) 指标变化速率(相邻时次差分):')
    for i, r in enumerate(rates):
        head = f'  时次对[{i}]'
        if 't_from_start_min' in r:
            head += f' t=+{r["t_from_start_min"]:.0f}~{r["t_from_start_min"]+r["dt_min"]:.0f} min'
        head += f' (dt={r["dt_min"]:.1f} min)'
        print(head)
        print(f'      ΔZ(max)  = {r["d_max_z_per_min"]:+6.2f} dBZ/min   '
              f'ΔET18 = {r["d_et18_per_min"]:+6.0f} m/min   '
              f'ΔET45 = {r["d_et45_per_min"]:+6.0f} m/min')
        print(f'      ΔVIL(max)= {r["d_vil_max_per_min"]:+6.2f} kg/m²/min   '
              f'Δ库数 = {r["d_count45_per_min"]:+6.1f} 库/min  '
              f'Δ体积 = {r["d_vol45_per_min"]:+6.3f} km³/min')
    best_v, best_i = max_trend(rates, 'd_vil_max_per_min')
    if best_i >= 0:
        print(f'  → VIL 最陡增幅: {best_v:+.2f} kg/m²/min (时次对 {best_i})')


# =============================================================================
# 一次性把常用指标打包(方便直接调用)
# =============================================================================

def all_metrics(radar, box_mask, field='cor_z', fhc=True, vil=True,
                vil_result_out=None, zdr_col=True, zdrcol_result_out=None,
                zdrcol_kwargs=None, snd=None):
    """
    把定稿的 7 个指标一次算出来, 返回一个 dict(键 = 指标名)。

    ★★ 2026-09-24 用户定稿: **只产出下面 7 个量**, 其余一概不落表 ★★

      | 用户口径                              | 返回的键               |
      |---------------------------------------|------------------------|
      | (1) 最大回波强度                      | `max_z`                |
      | (2) 18dBZ 回波顶高                    | `echo_top_18dbz_m`     |
      | (2) 45dBZ 回波顶高                    | `echo_top_45dbz_m`     |
      | (3) >=45dBZ 回波体积                  | `volume_ge45dbz_km3`   |
      | (4) 垂直累积液态水含量 VIL (只要 max) | `vil_max_kgm2`         |
      | (5) 冰雹粒子数量                      | `hail_cells`           |
      | (5) 霰粒子数量                        | `graupel_cells`        |
      | (6) 最强回波(>=30dBZ 判据)的高度      | `zmax30_height_m`      |
      | (7) ZDR 柱高 (= 层数 × 层高)           | `zdrcol_height_m`      |

    已废弃(不再产出, 需要时把下面那几行注释恢复即可, 计算函数都还在):
      count_ge45dbz / ratio_ge45dbz / count_ge45dbz_vol / vil_mean_kgm2 /
      hail_graupel_cells / hail_graupel_ratio / hail_ratio_in_hail_graupel /
      zdrcol_depth_m / zdrcol_max_zdr_db

    参数
      field: 用哪个字段算强度类指标, 默认 'cor_z'(衰减订正后)
      fhc:   是否算指标(5)(需要 FH 字段); 没做 FHC 就传 fhc=False
      vil:   是否算指标(4) VIL(含网格化积分, 是各项里最耗时的);
             批量/不需要 VIL 时传 vil=False, 结果里就没有 vil_* 键
      vil_result_out: 传一个 dict(如 {}) 时, 把 VIL 的 2D 场、网格经纬度和
             Grid 对象写进去, 供 save_grid_nc / save_vil_nc 存 .nc
             —— 复用同一次网格化积分, 不会把 VIL 算两遍
      zdr_col: 是否算指标(7) ZDR 柱(Zhao et al. 2025 法, 需要 cor_zdr 字段)。
             ★ 与 VIL **各用各的网格**(2026-09-14 用户要求: VIL 产物不带 ZDR 的东西):
               ZDR 柱自己网格化一次, VIL 的网格只装 field 一个字段 -> 存的 .nc 干净
      zdrcol_result_out: 传 dict 时把 ZDR 柱的完整结果(3D 掩膜/网格等)写进去
      zdrcol_kwargs: 传给 compute_zdr_column_field 的额外参数
             (如 {'zdr_min_db': 1.0} 换成老文献阈值, 或 {'z0c_m': fzl})
      snd:   QC.read_sounding() 的返回 dict ★ 建议传 —— 指标(7) 的 0 °C 层优先用
             探空 fzl、判据① 上界(-20 °C 混合相区顶)直接查探空原始廓线, 不受
             "雷达库点只到几 km"的浅体扫限制。已显式写进 zdrcol_kwargs 的 snd
             优先级更高, 不会被本参数覆盖。
    """
    out = {}
    # ★ 2026-09-24 用户定稿: 只保留 7 个指标(见本函数 docstring 的对照表),
    #   count_ge45dbz / ratio_ge45dbz / count_ge45dbz_vol / hail_graupel_cells /
    #   hail_graupel_ratio / hail_ratio_in_hail_graupel / vil_mean_kgm2 /
    #   zdrcol_depth_m / zdrcol_max_zdr_db **一概不再产出** —— CSV 直接就是干净的 7 列。
    out['max_z'] = metric_max_z(radar, box_mask, field=field)
    out['zmax30_height_m'] = metric_zmax_height(radar, box_mask, threshold=30.0,
                                               field=field)

    et18 = metric_echo_top(radar, box_mask, threshold=18.0, field=field)
    et45 = metric_echo_top(radar, box_mask, threshold=45.0, field=field)
    out['echo_top_18dbz_m'] = et18
    out['echo_top_45dbz_m'] = et45

    # 指标(3) 只要体积; 库数/占比不产出 -> 用 _ 接住丢弃
    _, vol45 = metric_echo_volume(radar, box_mask, threshold=45.0, field=field)
    out['volume_ge45dbz_km3'] = vol45

    # ★ VIL 与 ZDR 柱**各用各的网格**(2026-09-14 用户要求: "VIL 中不要带 ZDR 的东西")。
    #   VIL 网格只装 field(默认 cor_z), 不再多带 cor_zdr -> 存出来的
    #   vilgrid_<CASE>_<时刻>.nc 里只有 cor_z + VIL, 干净, 不含任何 ZDR 字段/掩膜;
    #   ZDR 柱要 ZDR 场, 自己网格化一次(代价: 网格化多做一遍, 那是本链最耗时的一步)。
    kw = dict(zdrcol_kwargs or {})
    if snd is not None:
        kw.setdefault('snd', snd)          # 探空: 0 °C 层 + 判据① 上界都用它
    # 默认跟 config.py 的 ZDRCOL_ZDR_FIELD 走(可被 zdrcol_kwargs 覆盖)。
    # 注意 ZDR 柱长在融化层以上: 当前链里 cor_zdr 就是原始 ZDR 直通(全域有值), 默认可用。
    zdr_field = kw.get('zdr_field', C.ZDRCOL_ZDR_FIELD)
    zdr_ok = bool(zdr_col) and (zdr_field in radar.fields)
    if zdr_col and not zdr_ok:
        print(f'  [提示] radar 里没有 {zdr_field} 字段 -> 指标(7) ZDR 柱给 NaN'
              f'(先跑完整处理链, 或用 zdrcol_kwargs 指定 zdr_field)')
    if vil:
        if vil_result_out is not None:              # 要存 .nc -> 需要网格对象
            vil_max, _vil_mean, vres = metric_vil(radar, box_mask, field=field,
                                                  return_field=True)
            vil_result_out['vil'] = vres['vil']
            vil_result_out['lon'] = vres['lon']
            vil_result_out['lat'] = vres['lat']
            vil_result_out['grid'] = vres['grid']      # save_grid_nc 要用
            vil_result_out['z_axis'] = vres['z_axis']
        else:
            vil_max, _vil_mean = metric_vil(radar, box_mask, field=field)
        # ★ 只要 max(用户 2026-09-24 定稿); mean 照算但不落表
        out['vil_max_kgm2'] = vil_max

    if zdr_col:
        if zdr_ok:
            # 这里**故意不传 grid** —— 让 compute_zdr_column_field 自己网格化 ZDR,
            # 免得把 ZDR 字段 / ZDRCOL 掩膜挂到 VIL 的网格上(那样 .nc 就不干净了)。
            if field != 'cor_z':
                kw.setdefault('ref_field', field)
            m, zres = metric_zdr_column(radar, box_mask, return_field=True, **kw)
            # ★ 只要柱"高"(zdrcol_height_m = 层数 n × 层高 dz, 论文口径);
            # 柱内最大 ZDR / 层数 / 柱底 / 体积等不再落表(仍在 zres 里, 供核查与画图)
            out['zdrcol_height_m'] = m.get('zdrcol_height_m', np.nan)
            if zdrcol_result_out is not None:
                zdrcol_result_out.update(zres)
        else:
            # 没 ZDR 字段: 不调用(省一次无用告警), 直接给 NaN, 保证 CSV/Excel 列还在
            out['zdrcol_height_m'] = np.nan

    if fhc:
        # ★ 只要冰雹库数 + 霰库数两个数; 合计与两个占比不产出
        ch, cg = metric_hail_graupel_count(radar, box_mask)[:2]
        out['hail_cells'] = ch
        out['graupel_cells'] = cg

    return out


def print_metrics(m, prefix='  '):
    """把 all_metrics() 的结果打印成人看的格式(7 个指标, 2026-09-24 定稿)。"""
    print(f'{prefix}(1) 最大回波强度        : {m["max_z"]:.1f} dBZ')
    if 'zmax30_height_m' in m:
        v = m['zmax30_height_m']
        print(f'{prefix}(6) 最强回波高度(>=30dBZ): '
              + (f'{v:.0f} m ({v/1000:.2f} km)' if np.isfinite(v) else '框内 <30dBZ, 无'))
    print(f'{prefix}(2) 18dBZ 回波顶高       : {m["echo_top_18dbz_m"]:.0f} m '
          f'({m["echo_top_18dbz_m"]/1000:.2f} km)')
    print(f'{prefix}    45dBZ 回波顶高       : {m["echo_top_45dbz_m"]:.0f} m '
          f'({m["echo_top_45dbz_m"]/1000:.2f} km)')
    print(f'{prefix}(3) >=45dBZ 回波体积     : {m["volume_ge45dbz_km3"]:.2f} km^3')
    if 'vil_max_kgm2' in m:
        print(f'{prefix}(4) 最大 VIL            : {m["vil_max_kgm2"]:.1f} kg/m^2')
    if 'zdrcol_height_m' in m:
        v = m['zdrcol_height_m']
        if np.isfinite(v):
            print(f'{prefix}(7) ZDR 柱高(层数x层高) : {v:.0f} m ({v/1000:.2f} km)')
        else:
            print(f'{prefix}(7) ZDR 柱高            : 无柱(或无 ZDR 字段 / 无温度场)')
    if 'hail_cells' in m:
        print(f'{prefix}(5) 冰雹粒子数          : {m["hail_cells"]}, '
              f'霰粒子数 {m["graupel_cells"]}')
