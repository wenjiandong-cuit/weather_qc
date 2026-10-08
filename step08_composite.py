# -*- coding: utf-8 -*-
"""
QC/step08_composite.py — 第 8 部分: 组合反射率(Composite Reflectivity, CR)
=================================================================
把整卷扫描的各仰角层投影到同一套方位角上, 逐距离库取最大值, 得到组合反射率
(精确版还给出最大值所在高度)。两种实现, 各有用途:

  composite_reflectivity()       精确版: 各层按方位角线性插值对齐到 ref_sweep 的方位角,
                                 再取 max + 对应高度, 最后补方位角缺口。
                                 -> (comp_z, comp_alt)   # nan 表示无回波
  composite_reflectivity_fast()  快速版: 各层方位角一致时用 np.fmax.reduce 一次成型
                                 (整卷 68 层只需一次归约); 不一致时最近邻对齐再 fmax。
                                 -> (comp_masked, az)    # 批量出图用这个

但平时**推荐用 composite_radar()**: 一步算完, 既写回原 radar, 又直接给你一个
可 pyart 绘图的单层 radar 对象 -> (cr_radar, comp_z, comp_alt)。

配套工具: sweep_indices() / align_azimuth_nearest() / fill_azimuth_gaps() / composite_stats()
        build_cr_radar()(把二维 CR 做成单层 radar) / composite_field_name()

注意: 精确版用 scipy.interpolate.interp1d(安全); 不要用 scipy.signal / numpy.linalg(本环境会崩)。
"""

import numpy as np
from scipy.interpolate import interp1d


# ================= 小工具 =================
def sweep_indices(radar):
    """每个 sweep 的起止 ray 索引 [(start, end_exclusive), ...]"""
    si = radar.sweep_start_ray_index['data']
    ei = radar.sweep_end_ray_index['data']
    return [(int(si[i]), int(ei[i]) + 1) for i in range(radar.nsweeps)]


def align_azimuth_nearest(az_src, data_src, az_dst):
    """按方位角最近邻把 data_src 对齐到 az_dst(方位角循环, 0°/360° 相接)"""
    order = np.argsort(az_src)
    a = np.asarray(az_src, float)[order]
    d = data_src[order]
    a_ext = np.concatenate([a - 360.0, a, a + 360.0])
    d_ext = np.concatenate([d, d, d], axis=0)
    pos = np.clip(np.searchsorted(a_ext, az_dst), 1, len(a_ext) - 1)
    left, right = a_ext[pos - 1], a_ext[pos]
    idx = np.where(np.abs(az_dst - left) <= np.abs(right - az_dst), pos - 1, pos)
    return d_ext[idx]


def fill_azimuth_gaps(z, alt=None):
    """
    补组合反射率图的缺口(方位角循环):
      - 方位角是循环的: 0° 的邻居是 359°
      - 某个位置是 NaN -> 用方位角方向最近的非 NaN 组合反射率值复制
      - 整列全 NaN -> 不补
    就地修改 z / alt 并返回 (z, alt)。
    """
    n_az, n_gates = z.shape

    for g in range(n_gates):
        col = z[:, g]
        valid = np.isfinite(col)

        if not valid.any() or valid.all():
            continue

        # 循环扩展: 首尾各加一份
        col_ext = np.concatenate([col, col, col])
        valid_ext = np.concatenate([valid, valid, valid])
        if alt is not None:
            alt_ext_col = np.concatenate([alt[:, g], alt[:, g], alt[:, g]])

        offset = n_az
        padded = np.concatenate(([False], valid_ext, [False]))
        diff = np.diff(padded.astype(int))
        gap_starts = np.where(diff == -1)[0]
        gap_ends = np.where(diff == 1)[0]

        for gs, ge in zip(gap_starts, gap_ends):
            if ge <= offset or gs >= 2 * n_az:
                continue
            gs_orig = max(gs, offset)
            ge_orig = min(ge, 2 * n_az)
            if gs_orig >= ge_orig:
                continue

            has_before = valid_ext[gs_orig - 1] if gs_orig > 0 else False
            has_after = valid_ext[ge_orig] if ge_orig < len(valid_ext) else False

            if has_before and has_after:
                mid = (gs_orig + ge_orig) // 2
                z[gs_orig - offset : mid - offset, g] = col_ext[gs_orig - 1]
                z[mid - offset : ge_orig - offset, g] = col_ext[ge_orig]
                if alt is not None:
                    alt[gs_orig - offset : mid - offset, g] = alt_ext_col[gs_orig - 1]
                    alt[mid - offset : ge_orig - offset, g] = alt_ext_col[ge_orig]
            elif has_before:
                z[gs_orig - offset : ge_orig - offset, g] = col_ext[gs_orig - 1]
                if alt is not None:
                    alt[gs_orig - offset : ge_orig - offset, g] = alt_ext_col[gs_orig - 1]
            elif has_after:
                z[gs_orig - offset : ge_orig - offset, g] = col_ext[ge_orig]
                if alt is not None:
                    alt[gs_orig - offset : ge_orig - offset, g] = alt_ext_col[ge_orig]

    return z, alt


def composite_stats(comp, alt=None):
    """组合反射率(可选高度)的常用统计, 直接打印/存表用"""
    z = np.ma.filled(np.asarray(comp), np.nan).astype(float)
    st = {'n_valid': int(np.isfinite(z).sum()), 'size': int(z.size),
          'max_dBZ': float(np.nanmax(z)) if np.isfinite(z).any() else float('nan'),
          'mean_dBZ': float(np.nanmean(z)) if np.isfinite(z).any() else float('nan')}
    if alt is not None:
        a = np.ma.filled(np.asarray(alt), np.nan).astype(float)
        m = np.isfinite(z) & np.isfinite(a)
        st['alt_at_max_m'] = float(a[m][np.argmax(z[m])]) if m.any() else float('nan')
        st['alt_min_m'] = float(np.nanmin(a)) if np.isfinite(a).any() else float('nan')
        st['alt_max_m'] = float(np.nanmax(a)) if np.isfinite(a).any() else float('nan')
    return st


def composite_field_name(source_field):
    """组合反射率字段的默认名字: reflectivity -> 'CR', 其它 -> 'CR_<字段>'"""
    return 'CR' if source_field in ('reflectivity', 'ref') else 'CR_' + str(source_field)


def store_composite_field(radar, comp, source_field='reflectivity', name=None,
                          ref_sweep=0, verbose=True):
    """
    把组合反射率结果写进 radar 对象(字段名默认见 composite_field_name).

    组合反射率是"投影到某一层的二维产品", 尺寸是 (n_az, ngates):
      * 如果和 radar 的 (nrays, ngates) 一致(单层雷达 / 各层射线数相同且已抽层) -> 整块写入
      * 否则 -> 写进 ref_sweep 对应的射线位置, 其余位置为 mask
        (这样 plot_ppi(radar, field='CR', sweep=ref_sweep) 就能直接画)
    返回写入的字段名。
    """
    nname = name or composite_field_name(source_field)
    data = np.ma.masked_invalid(np.asarray(comp, dtype=np.float32))

    if data.ndim != 2 or data.shape[1] != radar.ngates:
        raise ValueError(f'组合反射率形状 {data.shape} 与雷达 (nrays={radar.nrays}, '
                         f'ngates={radar.ngates}) 不匹配')
    if data.shape[0] == radar.nrays:
        stored, where = data, '尺寸一致, 整块写入'
    else:
        s, e = sweep_indices(radar)[ref_sweep]
        if data.shape[0] != (e - s):
            raise ValueError(f'组合反射率有 {data.shape[0]} 条射线, 与 sweep {ref_sweep} 的 {e-s} 条不符')
        stored = np.ma.masked_all((radar.nrays, radar.ngates), dtype=np.float32)
        stored[s:e] = data
        where = f'写入 sweep {ref_sweep} ({e-s} 条射线), 其余为 mask'

    radar.add_field(nname, {
        'data': stored, 'units': 'dBZ',
        'standard_name': 'equivalent_reflectivity_factor',
        'long_name': f'Composite reflectivity from {source_field} ({where})',
    }, replace_existing=True)
    if verbose:
        print(f'  [CR] 已写入 radar.fields["{nname}"]: max={float(np.ma.max(data)):.1f} dBZ ({where})')
    return nname


# ================= 精确版 =================
def composite_reflectivity(radar, field='cor_z', ref_sweep=0, fill_gaps=True, verbose=True,
                           write=True, field_name=None):
    """
    组合反射率(精确): 以 ref_sweep 的方位角为基准, 其他层插值对齐后取 max.
    解决不同仰角层 rays 数不一致的问题.

    参数
      field:     用哪个字段组合(默认 'cor_z' 订正后反射率; 也可 'reflectivity')
      ref_sweep: 方位角基准层(默认 0)
      fill_gaps: 是否补方位角缺口(默认 True)
      write:     是否把结果写回 radar 成为字段(默认 True)
      field_name:写入的字段名; 默认 'CR'(源字段 reflectivity) 或 'CR_<field>' 其它
    返回
      composite_z:   (n_az_ref, ngates) 最大反射率
      composite_alt: (n_az_ref, ngates) 最大反射率对应的高度(m)
    """
    sweep_ranges = sweep_indices(radar)
    z_field = radar.fields[field]['data']
    gate_alt = radar.gate_altitude['data']
    ngates = radar.ngates

    # 基准方位角 = ref_sweep 的方位角
    s0, e0 = sweep_ranges[ref_sweep]
    base_az = np.array(radar.azimuth['data'][s0:e0], dtype=float)
    n_az = len(base_az)
    if verbose:
        print(f'  基准 sweep={ref_sweep}, n_az={n_az}, ngates={ngates}')

    z_aligned = []
    alt_aligned = []

    for i, (s, e) in enumerate(sweep_ranges):
        az_sweep = np.array(radar.azimuth['data'][s:e], dtype=float)
        z_sweep = np.ma.filled(z_field[s:e], np.nan)
        alt_sweep = np.ma.filled(gate_alt[s:e], np.nan)

        # 判断是否需要对齐: rays 数一致 + 方位角一致(容差 0.01°)
        need_interp = (len(az_sweep) != n_az)
        if not need_interp:
            diff = np.abs(az_sweep - base_az)
            diff = np.minimum(diff, 360.0 - diff)
            if np.max(diff) > 0.01:
                need_interp = True

        if not need_interp:
            z_aligned.append(z_sweep)
            alt_aligned.append(alt_sweep)
            if verbose:
                print(f'  sweep {i}: {len(az_sweep)} rays, 已对齐')
        else:
            # 排序 + 去重
            order = np.argsort(az_sweep)
            az_sorted = az_sweep[order]
            z_sorted = z_sweep[order]
            alt_sorted = alt_sweep[order]

            uniq_az, uniq_idx = np.unique(az_sorted, return_index=True)
            if len(uniq_az) < len(az_sorted):
                az_sorted = az_sorted[uniq_idx]
                z_sorted = z_sorted[uniq_idx]
                alt_sorted = alt_sorted[uniq_idx]

            # 方位角循环扩展: (az-360, az, az+360)
            az_ext = np.concatenate([az_sorted - 360.0, az_sorted, az_sorted + 360.0])
            z_ext = np.vstack([z_sorted, z_sorted, z_sorted])
            alt_ext = np.vstack([alt_sorted, alt_sorted, alt_sorted])

            f_z = interp1d(az_ext, z_ext, axis=0, kind='linear',
                           bounds_error=False, fill_value=np.nan)
            f_alt = interp1d(az_ext, alt_ext, axis=0, kind='linear',
                             bounds_error=False, fill_value=np.nan)
            z_aligned.append(f_z(base_az))
            alt_aligned.append(f_alt(base_az))
            if verbose:
                print(f'  sweep {i}: {len(az_sorted)} rays -> 插值到 {n_az} rays')

    # 堆叠: (nsweeps, n_az, ngates)
    z_stack = np.stack(z_aligned, axis=0)
    alt_stack = np.stack(alt_aligned, axis=0)

    # 取 max 和对应高度
    all_nan = np.all(np.isnan(z_stack), axis=0)
    z_safe = np.where(np.isnan(z_stack), -999, z_stack)

    max_idx = np.nanargmax(z_safe, axis=0)
    composite_z = np.where(all_nan, np.nan, np.nanmax(z_safe, axis=0))
    composite_alt = np.where(all_nan, np.nan,
                            np.take_along_axis(alt_stack, max_idx[None, ...], axis=0)[0])

    if verbose:
        print(f'  组合反射率完成: max Z={np.nanmax(composite_z):.1f} dBZ, '
              f'对应高度范围 {np.nanmin(composite_alt):.0f}~{np.nanmax(composite_alt):.0f} m')

    # 补缺
    if fill_gaps:
        composite_z, composite_alt = fill_azimuth_gaps(composite_z, composite_alt)
        if verbose:
            print(f'  补缺后: max Z={np.nanmax(composite_z):.1f} dBZ, '
                  f'有效库数 {np.isfinite(composite_z).sum()}/{composite_z.size}')

    # 写回 radar(字段名默认 'CR' / 'CR_<field>')
    if write:
        store_composite_field(radar, composite_z, source_field=field,
                              name=field_name, ref_sweep=ref_sweep, verbose=verbose)

    return composite_z, composite_alt


# ================= 快速版(批量出图用) =================
def composite_reflectivity_fast(radar, gatefilter=None, field='reflectivity', verbose=False,
                                write=True, field_name=None, ref_sweep=0):
    """
    组合反射率(快速): 各层取 max.
      各层方位角一致(rays 数与方位角都对得上) -> 直接 np.fmax.reduce, 一次归约整卷
      否则 -> 逐层最近邻对齐到第 0 层方位角再 fmax
    参数
      gatefilter: pyart GateFilter, 给了就把 QC 外的门当 NaN
      field:      默认 'reflectivity'(未订正), 也可 'cor_z'
      write:      是否把结果写回 radar 成为字段(默认 True; 批量出图自己建字段时可设 False 省内存)
    返回
      composite: masked array (n_az0, ngates), mask = 无回波
      az0:       基准方位角(第 0 层)
    """
    z = radar.fields[field]['data']
    zf = np.ma.filled(z, np.nan).astype(np.float32)
    if gatefilter is not None:
        zf[~gatefilter.gate_included] = np.nan
    si = np.asarray(radar.sweep_start_ray_index['data'], int)
    ei = np.asarray(radar.sweep_end_ray_index['data'], int) + 1
    az = np.asarray(radar.azimuth['data'], float)
    nsw, ng = radar.nsweeps, radar.ngates
    counts = ei - si
    az0 = az[si[0]:ei[0]]

    same = bool(np.all(counts == counts[0]))
    if same:
        for k in range(1, nsw):
            a = az[si[k]:ei[k]]
            d = np.abs(a - az0)
            if np.max(np.minimum(d, 360.0 - d)) > 0.05:
                same = False
                break

    if verbose:
        print(f'  CR(快速): {nsw} 层 x {ng} 库, 方位角{"一致" if same else "不一致(最近邻对齐)"}')

    if same:
        nr = int(counts[0])
        stack = np.empty((nsw, nr, ng), np.float32)
        for k in range(nsw):
            stack[k] = zf[si[k]:ei[k]]
        comp = np.fmax.reduce(stack, axis=0)     # fmax 忽略 NaN
        del stack
    else:
        comp = np.full((az0.size, ng), np.nan, np.float32)
        for k in range(nsw):
            a = az[si[k]:ei[k]]
            d = np.abs(a - az0)
            if a.size != az0.size or np.max(np.minimum(d, 360 - d)) > 0.05:
                aligned = align_azimuth_nearest(a, zf[si[k]:ei[k]], az0)
            else:
                aligned = zf[si[k]:ei[k]]
            np.fmax(comp, aligned, out=comp)
    comp = np.ma.masked_invalid(comp)
    if write:
        store_composite_field(radar, comp, source_field=field, name=field_name,
                              ref_sweep=ref_sweep, verbose=verbose)
    return comp, az0


# ================= 组合反射率 -> radar 对象(随时可调、可直接 pyart 绘图) =================
def build_cr_radar(radar, comp, ref_sweep=0, field_name='CR', verbose=True):
    """
    把二维组合反射率做成"只含 ref_sweep 那一层 + CR 字段"的 slim pyart Radar。

    为什么要单独做一个: 组合反射率是**投影到某一层的二维产品**, 直接塞回多层 radar 时
    只能放在 ref_sweep 的射线位置(其余为 mask), pyart 画起来别扭。这里抽出一层, 让
    形状 (nrays_of_sweep, ngates) 与字段完全吻合, 拿到就能画:

        import pyart
        pyart.graph.RadarMapDisplay(cr_radar).plot_ppi_map(field='CR', sweep=0)

    参数
      radar:      原多层 radar(pyart Radar)
      comp:       组合反射率数组 (nrays_of_ref_sweep, ngates)
      ref_sweep:  组合反射率所属层(默认 0)
      field_name: 写进 slim radar 的字段名(默认 'CR')
    返回
      slim radar(可直接 pyart 绘图 / QC.plot_cr())
    """
    slim = radar.extract_sweeps([ref_sweep])
    data = np.ma.masked_invalid(np.asarray(comp, dtype=np.float32))
    if data.ndim != 2 or data.shape != (slim.nrays, slim.ngates):
        raise ValueError(
            f'组合反射率形状 {data.shape} 与 sweep {ref_sweep} 的 '
            f'(nrays={slim.nrays}, ngates={slim.ngates}) 不匹配')
    slim.add_field(field_name, {
        'data': data, 'units': 'dBZ',
        'standard_name': 'equivalent_reflectivity_factor',
        'long_name': f'Composite reflectivity ({field_name}, sweep {ref_sweep})',
    }, replace_existing=True)
    if verbose:
        elev = float(np.mean(slim.elevation['data']))
        print(f'  [CR] 绘图用 radar: 1 层 elev={elev:.2f}°, '
              f'{slim.nrays} rays x {slim.ngates} gates, field="{field_name}"')
    return slim


def composite_radar(radar, field='cor_z', ref_sweep=0, fill_gaps=True,
                    write_back=True, field_name=None, verbose=True):
    """
    组合反射率"一次算完、落成 radar 对象" —— 平时就用这个(随时可调)。

    做三件事:
      1. 计算组合反射率(精确版: 方位角插值对齐 + 补缺)
      2. write_back=True -> 结果同时写回**原 radar** 的字段('CR' / 'CR_<field>')
      3. 另返回一个**只含 ref_sweep 那层 + CR 字段**的 slim radar, 可直接 pyart 绘图

    参数
      radar:      处理好的多层 radar(至少要有 field 指定的字段)
      field:      用哪个字段组合, 默认 'cor_z'(衰减订正后); 也可 'reflectivity'(未订正)
      ref_sweep:  方位角基准层 / CR 所在层(默认 0)
      fill_gaps:  是否补方位角缺口(默认 True)
      write_back: 是否把结果写回原 radar(默认 True)
      field_name: 字段名, 默认 'CR'(源字段 reflectivity) 或 'CR_<field>'
    返回
      (cr_radar, comp_z, comp_alt)
        cr_radar : 单层 slim radar, 直接 QC.plot_cr(cr_radar) 或 pyart 绘图
        comp_z   : (n_az, ngates) 组合反射率
        comp_alt : (n_az, ngates) 组合反射率对应高度(m)

    用法
      import QC
      cr_radar, comp_z, comp_alt = QC.composite_radar(radar, field='cor_z')
      QC.plot_cr(cr_radar)                                   # 显示
      QC.plot_cr(cr_radar, out_path='cr.png', hail_location=[[104.02, 27.15]])
    """
    comp_z, comp_alt = composite_reflectivity(
        radar, field=field, ref_sweep=ref_sweep, fill_gaps=fill_gaps,
        verbose=verbose, write=write_back, field_name=field_name)
    name = field_name or composite_field_name(field)
    cr_radar = build_cr_radar(radar, comp_z, ref_sweep=ref_sweep,
                              field_name=name, verbose=verbose)
    return cr_radar, comp_z, comp_alt
