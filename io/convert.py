# -*- coding: utf-8 -*-
"""
QC/io/convert.py —— fast_pa 的 dict -> pyart Radar（**全项目唯一实现**）
================================================================================

【为什么有这个文件】
    以前 `read_radar.py` 和 `fast_read.py` **各有一份 `to_pyart()`**，逻辑几乎重复。
    2026-09-21 目录重组时合并到这里，两份改一份，避免以后改了一处忘另一处。

    两份原本的差异（合并前的实况）：
      · `read_radar.to_pyart(pa, range_mode, verbose)`
          —— 元数据用它**自带 dump** 的 `_md()`；不支持字段筛选。
      · `fast_read.to_pyart(pa, fields, range_mode, verbose)`
          —— 元数据用 `pycwr...get_metadata()`；支持 `fields=` 筛选、
             `pa['fields_moment']` 回退取值。
    ★ 已实测验证：`read_radar._md()` 与 `pycwr.get_metadata()` 对 dump 表里的
      全部 31 个名字输出**逐字段完全一致**（`get_fillvalue()` 也同为 -9999.0），
      所以合并成"统一走 pycwr 的 get_metadata"不会改变任何一个字节的输出。

【为什么能保证和 pycwr 一模一样】
    桥接不是"凭理解重写"，而是**逐条照抄 pycwr 自己的导出逻辑**
    （pycwr/core/interop.py :: build_radar_from_prd）：

      · 字段名：用 pycwr 自己的 CINRAD_field_mapping 把矩名换成 pyart 名
        （'Wc'->'spectrum_width_corrected'、'SNRV'->'vertical_signal_noise_ratio' …），
        与 interop._canonical_field_name 同一张表 —— 输出键名逐字相同
      · 字段结构：每个字段包成 **dict**（含 data / _FillValue / units / standard_name /
        long_name / valid_min / valid_max），元数据取 pycwr 自己的 get_metadata
        （**不是** pyart 的那套，两者是两份独立副本，必须用 pycwr 的）
      · 字段顺序：照 pycwr 的 _PREFERRED_EXPORT_FIELDS 排序，保证 dict 顺序也一致
      · 各矩库数**一致**（威宁 12 个矩同库数）：每个字段本来就是 (nrays, ngates)，
        直接出数组，不需要铺到公共列宽
      · 时间轴：用 pycwr 自己的 julian2date_SEC / make_time_unit_str / date2num
      · fixed_angle / nyquist / unambiguous_range：都取 CutConfigurationBlock 的同一个字段
      · gate_x/y/z、gate_altitude、gate_latitude/longitude：**不给**，
        让 pyart 自己懒算 —— pycwr 也没给，所以两边算出来的必然一致。

    ★ 关键认知：gate_altitude / gate_latitude / gate_longitude 在 pyart 里是
      **由 range + azimuth + elevation + altitude 懒算出来的**（pyart/core/radar.py
      init_gate_x_y_z / init_gate_altitude / init_gate_longitude_latitude）。
      所以只要那四个输入一致，温度场、组合反射率、VIL、ZDR 柱用到的门坐标就自动一致。

【支持的雷达】
    ZWN01 / ZWN02 / ZWN03（威宁 X 波段双偏振相控阵，2025–）—— 12 个矩**库数一致**
    （实测 cell2 全 1472 库、cell3 全 2000 库）。
    ★ 本版只适配威宁：若某帧各矩库数不一致，**直接报错**，不做左对齐补齐
      —— 补齐会静默改变场的位置含义。

【对外提供的函数】
    to_pyart(pa, fields=None, range_mode='aligned', verbose=False) -> pyart Radar
    moment_to_pyart_name(name)  /  pyart_to_moment_name(name)

【注意】
    · 本模块会 import pyart / pycwr，必须走 conda 环境（wradlib）。
      调用方（read_radar / fast_read）负责在 import 之前先补好
      conda 环境的 DLL 目录进 PATH（见 fast_pa._ensure_env_dll_path）。
    · 只读，不写任何东西。
"""

from __future__ import annotations

import time

import numpy as np


# ---------------------------------------------------------------------------
# 矩名 <-> pycwr 导出的 pyart 字段名
# ---------------------------------------------------------------------------
def _moment_mapping():
    """
    取 pycwr 自己的矩名->pyart名映射表（CINRAD_field_mapping）。

    ★ 必须是 pycwr 那张表本身，不能自己抄一份 —— 输出键名才可能与 pycwr 逐字一致。
      拿不到时（版本变动/导入失败）退化成空表，此时函数退化为恒等映射。
    """
    try:
        from pycwr.configure.default_config import CINRAD_field_mapping as _M
        return _M or {}
    except Exception:
        return {}


#: 模块加载时快照一份（正是 pycwr 导出的那张表）
_MOM2PYART = _moment_mapping()
#: 反向表：pyart 字段名 -> 矩名
_PYART2MOM = {v: k for k, v in _MOM2PYART.items()}


def moment_to_pyart_name(name):
    """
    矩名 -> pycwr 导出的 pyart 字段名（与 interop._canonical_field_name 同表）。

    ★ 关键：映射表只对 **矩名**（'dBZ' / 'Wc' / 'SNRV' …）有定义。
      传进来的如果本来就是 pyart 名（'reflectivity' …），表里查不到，
      此时**原样返回** —— 所以这个函数是**幂等**的，可以放心反复调用。
    """
    if name in _MOM2PYART:
        return _MOM2PYART[name]
    return name


def pyart_to_moment_name(name):
    """pyart 字段名 -> 矩名（反向表；查不到原样返回）。"""
    if name in _PYART2MOM:
        return _PYART2MOM[name]
    return name


def _preferred_export_fields():
    """
    pycwr 写 netCDF 时的字段排列顺序。

    ★ 直接借 pycwr 自己的表（pycwr.core.interop._PREFERRED_EXPORT_FIELDS），
      不是我们自己抄 —— 输出 dict 的键序才能与 pycwr 一致。
      拿不到时用 1.0.8 的内容兜底（已核对一致）。
    """
    try:
        from pycwr.core.interop import _PREFERRED_EXPORT_FIELDS as pref
        if pref:
            return tuple(pref)
    except Exception:
        pass
    return ('dBZ', 'dBT', 'V', 'W', 'ZDR', 'CC', 'PhiDP', 'KDP', 'SQI',
            'CPA', 'LDR', 'HCL', 'CF', 'SNRH', 'SNRV')


# ---------------------------------------------------------------------------
# 核心：pa dict -> pyart Radar
# ---------------------------------------------------------------------------
#    逐条对齐 pycwr/core/interop.py :: build_radar_from_prd
# ---------------------------------------------------------------------------
def to_pyart(pa, fields=None, range_mode='aligned', verbose=False, align_uneven=False):
    """
    把 fast_pa.read_pa() 的返回值造成 pyart Radar。

    pa        : fast_pa.read_pa() 的 dict
    fields    : None = 把 pa 里已有的字段全部放进去（默认；pa 里没有的就不会有）
    range_mode: 写进 metadata 的 range_mode，pycwr 默认导出为 'aligned'
    verbose   : 打印对齐/造对象信息
    align_uneven: False(默认) = 各矩库数不一致时**直接报错**（威宁 12 矩同库数，必须一致）；
                  True = 短字段（如晴隆 V/W 1000 库 vs 强度 1203 库）**左对齐补齐**到整卷
                  最大库数，右侧补 NaN(mask)。物理语义：速度只测到 75 km，75~90 km 无速度。

    ★ 输出的字段名一律是 **pycwr 导出的 pyart 名**（CINRAD_field_mapping 的值），
      所以 'Wc' 出来是 'spectrum_width_corrected'、'SNRV' 出来是
      'vertical_signal_noise_ratio' —— 与 pycwr 逐字一致。

    ★ 故意**不传** gate_x / gate_y / gate_z / gate_altitude / gate_latitude /
      gate_longitude —— pycwr 也没传。pyart 会按 range + 方位角 + 仰角 + 台站高度
      **懒算**，只要这四个输入一致，算出来的门坐标必然一致（还省掉 8.3 s 的投影计算）。
    """
    import pyart
    from pycwr.configure.pyart_config import get_fillvalue, get_metadata
    from pycwr.io.util import date2num, julian2date_SEC, make_time_unit_str

    pref = _preferred_export_fields()
    t0 = time.time()

    nrays = int(pa['nrays'])
    nsweeps = int(pa['nsweeps'])
    ngates = int(pa['ngates'])

    # ---- 1. 字段：键名用 pycwr 导出的 pyart 名，顺序照 pycwr _PREFERRED_EXPORT_FIELDS ----
    #   ★ pa['fields'] 的键是 **pyart 名**（fast_pa 里已经过了一遍 MOMENT2PYART）；
    #     pa['fields_moment'] 的键才是矩名。两者都留着，下面按需要取。
    #     pycwr 导出时会把矩名过一遍 _canonical_field_name（= CINRAD_field_mapping），
    #     所以这里先把偏好表里的矩名过同一张表，再和 pa['fields'] 的键比对。
    mom = pa.get('fields_moment') or {}
    order = []
    for logical in pref:
        pyart_name = moment_to_pyart_name(logical)
        if pyart_name in pa['fields'] and pyart_name not in order:
            order.append(pyart_name)
    for name in pa['fields']:              # 剩下那些不在偏好表里的
        if name not in order:
            order.append(name)
    #   统一成 pycwr 导出的键名（幂等：已是 pyart 名的原样保留）
    order = [moment_to_pyart_name(n) for n in order]
    if fields is not None:
        want = {moment_to_pyart_name(n) for n in fields}
        order = [n for n in order if n in want]

    def _arr_of(pyart_name):
        """按 pycwr 字段名取数组：优先 pa['fields']；查不到再退回矩名字典。"""
        a = pa['fields'].get(pyart_name)
        if a is None:
            a = mom.get(pyart_to_moment_name(pyart_name))
        if a is None:
            raise KeyError('pa 里找不到字段 %s（fields=%s）'
                           % (pyart_name, sorted(pa['fields'])))
        return a

    fill = get_fillvalue()
    out_fields = {}
    #   ★ 每个字段包成 **dict**（不是裸数组）—— pycwr/build_radar_from_prd 最后一步就是
    #       field_metadata = get_metadata(target_name)
    #       field_metadata["data"] = masked_array
    #       field_metadata["_FillValue"] = fill
    #       fields[target_name] = field_metadata
    #     元数据表就是 pycwr 自己的 pycwr.configure.pyart_config.get_metadata，
    #     所以 units / standard_name / long_name / valid_min / valid_max 会自动同源。
    #   ★ 威宁版口径：各矩库数必须一致（12 个矩同库数），所以每个字段直接就是
    #     (nrays, ngates)，不需要往公共列宽铺缓冲。不一致就报错，见下。
    nb_own = {n: int(_arr_of(n).shape[1]) for n in order}
    uneven = {n: v for n, v in nb_own.items() if v != ngates}
    if uneven and not align_uneven:
        raise ValueError(
            '本版 IO 只适配"各矩库数一致"的 PA 文件（威宁 12 个矩同库数）：'
            '本帧最大库数 %d（共 %d 个字段），但有 %d 个字段库数不同 -> %s。'
            '本模块不做左对齐补齐（要补请传 align_uneven=True）。'
            % (ngates, len(order), len(uneven), uneven))
    if uneven and align_uneven and verbose:
        print('  [convert] 库数不一致，左对齐补齐到 %d 库: %s' % (ngates, uneven))

    for src in order:
        arr = _arr_of(src)
        if align_uneven and arr.shape[1] < ngates:
            buf = np.full((nrays, ngates), np.nan, dtype=arr.dtype)
            buf[:, :arr.shape[1]] = arr
            arr = buf
        bufd = np.ma.masked_array(arr, mask=np.isnan(arr), fill_value=fill)
        md = get_metadata(src)          # pycwr 自己的元数据表（同 build_radar_from_prd）
        md['data'] = bufd
        md['_FillValue'] = fill
        out_fields[src] = md
        del bufd

    #   按 pycwr 的偏好顺序重新排一遍, 保证 dict 的键序与 pycwr 一致
    out_fields = {n: out_fields[n] for n in order if n in out_fields}

    # ---- 2. 时间轴：julian2date_SEC(Seconds, MicroSeconds)，与 PAFile.get_scan_time 同一实现 ----
    dts = np.empty(nrays, dtype=object)
    sec = pa['seconds']
    usec = pa['microseconds']
    for i in range(nrays):
        dts[i] = julian2date_SEC(int(sec[i]), int(usec[i]))
    units = make_time_unit_str(min(dts))
    t_md = get_metadata('time')
    t_md['units'] = units
    t_md['data'] = date2num(dts, units).astype('float32')

    # ---- 3. 距离轴 ----
    r_md = get_metadata('range')
    r_md['data'] = pa['range'].astype('float32')
    if ngates:
        r_md['meters_to_center_of_first_gate'] = float(pa['range'][0])
    if ngates > 1:
        r_md['meters_between_gates'] = float(pa['range'][1] - pa['range'][0])

    # ---- 4. 方位/仰角 ----
    az_md = get_metadata('azimuth')
    az_md['data'] = pa['azimuth'].astype('float32')
    el_md = get_metadata('elevation')
    el_md['data'] = pa['elevation'].astype('float32')

    # ---- 5. 扫描层索引 ----
    sri_md = get_metadata('sweep_start_ray_index')
    sri_md['data'] = np.asarray(pa['sweep_start'][:nsweeps], dtype='int32')
    eri_md = get_metadata('sweep_end_ray_index')
    eri_md['data'] = np.asarray(pa['sweep_end'][:nsweeps], dtype='int32')

    # ---- 6. 站点 / 扫描类型 / metadata ----
    scan_type = pa['scan_type']
    md_meta = get_metadata('metadata')
    md_meta['site_name'] = pa['sitename']
    md_meta['instrument_name'] = pa['sitename']
    md_meta['source'] = 'pycwr.PRD'
    md_meta['range_mode'] = range_mode

    sweep_mode = get_metadata('sweep_mode')
    if scan_type == 'ppi':
        sweep_mode['data'] = np.array(nsweeps * ['azimuth_surveillance'], dtype='S')
    elif scan_type == 'rhi':
        sweep_mode['data'] = np.array(nsweeps * ['rhi'], dtype='S')
    else:
        sweep_mode['data'] = np.array(nsweeps * ['sector'], dtype='S')

    sweep_number = get_metadata('sweep_number')
    sweep_number['data'] = np.arange(nsweeps, dtype='int32')

    fixed_angle = get_metadata('fixed_angle')
    fixed_angle['data'] = np.asarray(
        pa['header']['cut_cfg']['Elevation'][:nsweeps], dtype='float32')

    lat_md = get_metadata('latitude')
    lat_md['data'] = np.array([float(pa['latitude'])], dtype='float64')
    lon_md = get_metadata('longitude')
    lon_md['data'] = np.array([float(pa['longitude'])], dtype='float64')
    alt_md = get_metadata('altitude')
    alt_md['data'] = np.array([float(pa['altitude'])], dtype='float64')

    # ---- 7. 仪器参数：逐层值按该层射线数展开 ----
    rays_per_sweep = (np.asarray(pa['sweep_end'][:nsweeps], dtype=np.int32)
                      - np.asarray(pa['sweep_start'][:nsweeps], dtype=np.int32) + 1)
    nyq = get_metadata('nyquist_velocity')
    nyq['data'] = np.repeat(np.asarray(pa['nyquist'][:nsweeps], dtype=np.float32),
                            rays_per_sweep)
    unamb = get_metadata('unambiguous_range')
    unamb['data'] = np.repeat(np.asarray(pa['unambiguous_range'][:nsweeps], dtype=np.float32),
                              rays_per_sweep)
    freq = get_metadata('frequency')
    freq['data'] = np.array([float(pa['frequency_ghz'])], dtype=np.float32)
    instrument_parameters = {'nyquist_velocity': nyq,
                             'unambiguous_range': unamb,
                             'frequency': freq}

    # ---- 8. 造 Radar ----
    #   ★ 故意不传 gate_x / gate_y / gate_z / gate_altitude / gate_latitude / gate_longitude
    #     —— pycwr 也没传，pyart 会按同一套公式懒算，两边必然一致。
    radar = pyart.core.Radar(
        time=t_md, _range=r_md, fields=out_fields, metadata=md_meta,
        scan_type=scan_type, latitude=lat_md, longitude=lon_md, altitude=alt_md,
        sweep_number=sweep_number, sweep_mode=sweep_mode, fixed_angle=fixed_angle,
        sweep_start_ray_index=sri_md, sweep_end_ray_index=eri_md,
        azimuth=az_md, elevation=el_md,
        instrument_parameters=instrument_parameters)

    if verbose:
        print('  [%s] %d 层 / %d rays / %d gates, 字段 %d 个, 造对象 %.2fs'
              % (_TAG[0], radar.nsweeps, radar.nrays, radar.ngates, len(out_fields),
                 time.time() - t0))
    return radar


# ---------------------------------------------------------------------------
# 日志前缀：让共用的 to_pyart 仍按"是哪个调用方在用"打前缀
# ---------------------------------------------------------------------------
#    以前 read_radar 打 "[read_radar]"、fast_read 打 "[fast_read]"，日志一眼能看出
#    是哪条路径在读文件。合并成一份实现后，用这个 mutable 单元素列表承载当前调用方：
#    调用方在自己的 read_*() 入口里 `set_log_tag('read_radar')` 一下即可。
#    （不用 contextvars，因为这里是单线程顺序读；也不用参数传递，避免改 to_pyart 签名。）
_TAG = ['convert']


def set_log_tag(tag):
    """设置 to_pyart verbose 输出里的方括号前缀（read_radar / fast_read / ...）。"""
    _TAG[0] = str(tag)


def get_log_tag():
    """当前的前缀。"""
    return _TAG[0]
