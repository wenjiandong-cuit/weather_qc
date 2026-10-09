# -*- coding: utf-8 -*-
r"""
fast_read.py —— 把 fast_pa 的读取结果"桥接"成 pyart Radar
================================================================

【它解决什么】
    pycwr.io.read_auto(fp).ToPyartRadar() 单个体扫 **38.5 s**（实测 2025-05-20 11:09 ZWN01）。
    fast_pa.read_pa() 把同样一份数据读出来只要 1.6~17 s（取决于解几个矩）。
    但下游（QC 处理链 / 组合反射率 / 指标 / 出图）要的是 **pyart Radar 对象**，
    不是 dict。所以这里补上"dict -> pyart Radar"这一步。

【为什么能保证和 pycwr 一模一样】
    桥接不是"凭理解重写"，而是**逐条照抄 pycwr 自己的导出逻辑**
    （pycwr/core/interop.py :: build_radar_from_prd）：

      · 字段名：用 pycwr 自己的 CINRAD_field_mapping 把矩名换成 pyart 名
        （'Wc'->'spectrum_width_corrected'、'SNRV'->'vertical_signal_noise_ratio' …），
        与 interop._canonical_field_name 同一张表 —— 输出键名逐字相同
      · 字段结构：每个字段包成 **dict**（含 data / _FillValue / units / standard_name /
        long_name / valid_min / valid_max），元数据同样取 pycwr 自己的 get_metadata
        （**不是** pyart 的那套，两者是两份独立副本，必须用 pycwr 的）
      · 字段顺序：照 pycwr 的 _PREFERRED_EXPORT_FIELDS 排序，保证 dict 顺序也一致
      · 各矩库数**一致**（威宁 12 个矩同库数）：字段直接出 (nrays, ngates)。
        若某帧各矩库数不同 -> 直接报错（威宁版不做左对齐补齐，详见 to_pyart 注释）
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
    · 实测（2025-05-20 11:09 ZWN01）：--verify 全绿（逐字段逐位一致 + geometry 全 OK）
    ★ 本版只适配威宁：各矩库数不一致的文件会在 to_pyart 处**直接报错**，
      不会静默产出错位的场。

【本模块提供】
    read_pa_radar(path, fields=..., **kw)  -> pyart Radar   ← 正常使用就用这个
    to_pyart(pa_dict, ...)                -> pyart Radar   ← 已经有 dict 时用
    compare_with_pycwr(path)              -> (ok, report)  ← 和 pycwr 逐项比对
    moment_to_pyart_name(n) / pyart_to_moment_name(n)     ← 矩名 <-> pyart 字段名
    FAST_FIELDS_CR / FAST_FIELDS_RUN / FAST_FIELDS_ALL   ← 预设的"要解哪几个矩"

    CLI:
        python fast_read.py <file> --verify      与 pycwr 逐项比对（最权威的自检）
        python fast_read.py <file> --bench       只计时，不比对
        python fast_read.py <file> --info        看文件头（转给 fast_pa）

【注意】
    · 本模块会 import pyart，所以必须走 conda 环境 —— 推荐
      `conda run -n wradlib --no-capture-output python fast_read.py <file> --verify`，
      直接敲 `<env>\python.exe` 只有在 PATH 已补好 DLL 目录时才稳（脚本内已
      调 _ensure_env_dll_path() 自救，但沙箱/父进程场景仍建议显式设 PATH）。
    · 只读文件，不写任何东西。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

# 先把 conda 环境的 DLL 目录补进 PATH —— 必须早于 import pyart 以及任何 pycwr 子模块
from .fast_pa import _ensure_env_dll_path  # noqa: E402

# dict -> pyart Radar 的**唯一实现**（与 read_radar 共用，见 convert.py 里的说明）
#   矩名 <-> pyart 名 的两张映射表也一并从那边取（同源，不会两边跑偏）
from .convert import (                              # noqa: E402
    to_pyart as _to_pyart,
    set_log_tag,
    moment_to_pyart_name,
    pyart_to_moment_name,
)

_ensure_env_dll_path()


# ---------------------------------------------------------------------------
# 预设：不同阶段要解哪几个矩
# ---------------------------------------------------------------------------
# cr 阶段：只算"未订正组合反射率"，只依赖原始 reflectivity 一个矩
FAST_FIELDS_CR = ('dBZ',)

# run 阶段：完整处理链真正用到的矩（逐个在 QC/ 里核实过）
#   dBZ  -> reflectivity                step02 门控 / step04 掩膜 / pipeline
#   CC   -> cross_correlation_ratio     step02 门控 / pipeline / FHC
#   V    -> velocity                    step02 静止地物
#   W    -> spectrum_width              step02 静止地物
#   dBT  -> total_power                 step02 整层损坏 / 近距异常
#   PhiDP-> differential_phase          pipeline 系统相位 / step04 KDP / step06 衰减
#   ZDR  -> differential_reflectivity   step06 -> cor_zdr
# 不用的矩：KDP(自己用 Bringi 重算)、SQI/CPA/LDR/CP/HCL/CF/SNRH/SNRV
FAST_FIELDS_RUN = ('dBZ', 'dBT', 'V', 'W', 'ZDR', 'CC', 'PhiDP')

# None = 全解（12 个矩），只在需要和 pycwr 全字段比对时用
FAST_FIELDS_ALL = None


def _hr(ch='-'):
    print(ch * 78)


def _as_masked(arr, fill):
    """与 pycwr 同款的字段打包: NaN -> masked, 保留 float32, 不复制数据。"""
    return np.ma.masked_array(arr, mask=np.isnan(arr), fill_value=fill)


def _field_data(md):
    """从容忍各种形态的"字段容器"里取出数据数组（dict / ndarray / MaskedArray）。"""
    if isinstance(md, dict):
        return md.get('data')
    if isinstance(md, np.ndarray):
        return md
    return None


def _field_meta(md):
    """取出字段的非数据元信息（dict 才有；数组形态返回空）。"""
    if isinstance(md, dict):
        return {k: v for k, v in md.items() if k != 'data'}
    return {}


# ---------------------------------------------------------------------------
# 核心：pa dict -> pyart Radar
# ---------------------------------------------------------------------------
#    ★ 2026-09-21 目录重组时，本函数与 read_radar.to_pyart() **合并成了一份公共
#      实现**，放在 `QC/io/convert.py`（唯一实现）。这里只做转发。
#
#      合并前的差异与验证：
#        · 旧 fast_read 版（就是这一段）信息量更大 —— 支持 `fields=` 筛选、
#          支持 `pa['fields_moment']` 回退取值；旧 read_radar 版两样都没有。
#        · 所以合并后**保留了本版的语义**，read_radar 那边改成继承这份能力。
#        · 元数据：本版本来就用 `pycwr...get_metadata()`；旧 read_radar 版用自带
#          dump 的 `_md()`。已实测两者对 dump 表里的全部 31 个名字输出**逐字段
#          完全一致**，`get_fillvalue()` 同为 -9999.0 ⇒ 输出无任何变化。
# ---------------------------------------------------------------------------
def to_pyart(pa, fields=None, range_mode='aligned', verbose=False, align_uneven=False):
    """
    把 fast_pa.read_pa() 的返回值造成 pyart Radar。

    ★ 真正的实现已合并到 `QC/io/convert.py :: to_pyart`（全项目唯一一份）。
      本文件只做转发，参数语义与那边完全一致 —— 详见 convert.to_pyart 的文档。

    pa        : fast_pa.read_pa() 的 dict
    fields    : None = 把 pa 里已有的字段全部放进去（默认；pa 里没有的就不会有）
    range_mode: 写进 metadata 的 range_mode，pycwr 默认导出为 'aligned'
    align_uneven: 各矩库数不一致时是否左对齐补齐（晴隆 V/W 1000 库 vs 强度 1203 库要传 True）

    ★ 输出的字段名一律是 **pycwr 导出的 pyart 名**（CINRAD_field_mapping 的值），
      所以 'Wc' 出来是 'spectrum_width_corrected'、'SNRV' 出来是
      'vertical_signal_noise_ratio' —— 与 pycwr 逐字一致。
    """
    set_log_tag('fast_read')
    return _to_pyart(pa, fields=fields, range_mode=range_mode, verbose=verbose,
                     align_uneven=align_uneven)


def read_pa_radar(path, fields=FAST_FIELDS_RUN, verbose=False, align_uneven=False, **kw):
    """
    快速读取一个 PA 体扫并返回 pyart Radar。

    fields: 要解哪几个矩（可用矩名 dBZ/ZDR/... 或 pyart 名 reflectivity/...；
            None = 全解）。预设见 FAST_FIELDS_CR / FAST_FIELDS_RUN / FAST_FIELDS_ALL。
            ★ 这里传进去的名字会被 fast_pa.read_pa 直接用（它只认矩名），
              所以传 pyart 名时会先反查回矩名。
    align_uneven: 各矩库数不一致时左对齐补齐（晴隆 V/W 1000 库 vs 强度 1203 库要传 True）。
    kw    : 透传给 fast_pa.read_pa（如 ray_limit / exact / check_layout）
    """
    from .fast_pa import read_pa
    t0 = time.time()
    if fields is not None:
        fields = [pyart_to_moment_name(n) for n in fields]
    pa = read_pa(path, fields=fields, verbose=verbose, **kw)
    t_read = time.time() - t0
    #   ★ 不必再传 fields —— fast_pa.read_pa 已经按 fields 只解了那几个矩，
    #     pa['fields'] 里本来就只有它们。
    radar = to_pyart(pa, verbose=verbose, align_uneven=align_uneven)
    if verbose:
        print('  [fast_read] 读取 %.2fs + 造对象 = 合计 %.2fs（pycwr 锚点 38.5s）'
              % (t_read, time.time() - t0))
    return radar


# ★ 2026-10-08 用户定的对外名字: 读**基数据**(.bin.zip, 全路径) -> pyart Radar。
#   read_base_data 是主推名字; read_pa_radar 保留为别名, 不破坏已有 30+ 处调用。
read_base_data = read_pa_radar


# ---------------------------------------------------------------------------
# 和 pycwr 逐项比对（--verify）
# ---------------------------------------------------------------------------
def _arr_eq(a, b):
    """返回 (是否完全相同, 最大绝对差, 说明)"""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False, float('nan'), '形状 %s vs %s' % (a.shape, b.shape)
    if a.dtype.kind in 'fc' and b.dtype.kind in 'fc':
        na, nb = np.isnan(a), np.isnan(b)
        if not np.array_equal(na, nb):
            return False, float('nan'), 'NaN 掩码不一致 (差 %d 个)' % int((na != nb).sum())
        both = ~na
        if not both.any():
            return True, 0.0, '全 NaN'
        d = float(np.max(np.abs(a[both].astype(np.float64) - b[both].astype(np.float64))))
        return (d == 0.0), d, ''
    if a.dtype.kind in 'US' or b.dtype.kind in 'US' or a.dtype == object:
        try:
            same = np.array_equal(a, b)
        except Exception:
            same = all(x == y for x, y in zip(a, b))
        return bool(same), 0.0 if same else float('nan'), ''
    same = np.array_equal(a, b)
    return bool(same), 0.0, ''


def compare_with_pycwr(path, verbose=True):
    """
    把"快速读取 + 桥接"与"pycwr 原路读取"逐项比对。
    返回 (ok, report_dict)。ok=True 表示可以放心替换。
    """
    import gc
    import pycwr

    rep = {'path': path, 'rows': [], 'ok': True}

    def _note(name, ok, detail='', extra=''):
        rep['rows'].append((name, ok, detail, extra))
        if not ok:
            rep['ok'] = False

    if verbose:
        _hr('=')
        print('快速读取 vs pycwr 逐项比对')
        print(os.path.basename(path))
        _hr('=')

    # --- 参考：pycwr 原路 ---
    if verbose:
        print('--- [参考] pycwr.io.read_auto().ToPyartRadar() ---')
    t0 = time.time()
    ref = pycwr.io.read_auto(path).ToPyartRadar()
    t_ref = time.time() - t0
    if verbose:
        print('  %.2f s   %d 层 / %d rays / %d gates, 字段 %s'
              % (t_ref, ref.nsweeps, ref.nrays, ref.ngates, sorted(ref.fields)))

    # --- 被测：fast_pa + 桥接（全字段解，才能逐字段比） ---
    if verbose:
        print('--- [被测] fast_pa.read_pa() + fast_read.to_pyart()  (全字段) ---')
    t0 = time.time()
    fast = read_pa_radar(path, fields=None, verbose=verbose)
    t_fast = time.time() - t0
    if verbose:
        print('  %.2f s   %d 层 / %d rays / %d gates, 字段 %s'
              % (t_fast, fast.nsweeps, fast.nrays, fast.ngates, sorted(fast.fields)))

    rep['t_pycwr'] = t_ref
    rep['t_fast'] = t_fast

    if verbose:
        _hr()
        print('%-30s %-8s %s' % ('比对项', '结论', '说明'))
        _hr()

    # ---- 基本形状 ----
    for name, va, vb in (('nsweeps', ref.nsweeps, fast.nsweeps),
                         ('nrays', ref.nrays, fast.nrays),
                         ('ngates', ref.ngates, fast.ngates)):
        _note(name, va == vb, '' if va == vb else '%s vs %s' % (va, vb))

    # ---- 几何 ----
    for name, va, vb in (('range', ref.range['data'], fast.range['data']),
                         ('azimuth', ref.azimuth['data'], fast.azimuth['data']),
                         ('elevation', ref.elevation['data'], fast.elevation['data']),
                         ('sweep_start_ray_index', ref.sweep_start_ray_index['data'],
                          fast.sweep_start_ray_index['data']),
                         ('sweep_end_ray_index', ref.sweep_end_ray_index['data'],
                          fast.sweep_end_ray_index['data']),
                         ('fixed_angle', ref.fixed_angle['data'], fast.fixed_angle['data']),
                         ('latitude', ref.latitude['data'], fast.latitude['data']),
                         ('longitude', ref.longitude['data'], fast.longitude['data']),
                         ('altitude', ref.altitude['data'], fast.altitude['data']),
                         ('time', ref.time['data'], fast.time['data']),
                         ('nyquist_velocity', ref.instrument_parameters['nyquist_velocity']['data'],
                          fast.instrument_parameters['nyquist_velocity']['data']),
                         ('unambiguous_range', ref.instrument_parameters['unambiguous_range']['data'],
                          fast.instrument_parameters['unambiguous_range']['data']),
                         ('frequency', ref.instrument_parameters['frequency']['data'],
                          fast.instrument_parameters['frequency']['data'])):
        ok, d, msg = _arr_eq(va, vb)
        _note(name, ok, msg or ('maxdiff=%.3g' % d if d else ''))

    _note('time.units', ref.time['units'] == fast.time['units'],
          '' if ref.time['units'] == fast.time['units']
          else '%r vs %r' % (ref.time['units'], fast.time['units']))

    # ---- 门坐标（懒算的，温度场/组合/VIL/ZDR柱真正用的就是这三个）----
    if verbose:
        print('  ... 正在算门坐标（这一步会把 pyart 的懒算逼出来，最费内存）')
    for name, va, vb in (('gate_altitude', ref.gate_altitude['data'], fast.gate_altitude['data']),
                         ('gate_latitude', ref.gate_latitude['data'], fast.gate_latitude['data']),
                         ('gate_longitude', ref.gate_longitude['data'], fast.gate_longitude['data'])):
        ok, d, msg = _arr_eq(va, vb)
        _note(name, ok, msg or ('maxdiff=%.4g' % d if d else ''))
    del va, vb
    gc.collect()

    # ---- 字段 ----
    #   ★ 2026-09-15: pycwr 导出的字段名 = CINRAD_field_mapping 的值
    #     （'Wc' -> 'spectrum_width_corrected'、'SNRV' -> 'vertical_signal_noise_ratio'）。
    #     我们 to_pyart() 走的是同一张表, 所以两边名字应当逐字相同;
    #     下面再兜一层"按矩名归一"——万一某版 pycwr 换了键名, 也能配对成功而不是
    #     变成"字段名集合不一致 + 所有字段 maxdiff=nan"这种没信息量的失败。
    _norm = lambda d: {pyart_to_moment_name(k): k for k in d}
    map_ref, map_fast = _norm(ref.fields), _norm(fast.fields)
    only_ref = sorted(set(map_ref) - set(map_fast))
    only_fast = sorted(set(map_fast) - set(map_ref))
    _note('字段名集合', not only_ref and not only_fast,
          '' if not (only_ref or only_fast)
          else 'pycwr独有%s / 快速独有%s' % (only_ref, only_fast))
    for key in sorted(set(map_ref) & set(map_fast)):
        name_r, name_f = map_ref[key], map_fast[key]
        da = _field_data(ref.fields[name_r])
        db = _field_data(fast.fields[name_f])
        if da is None or db is None:
            _note('field:%s' % key, False, '取不到 data（类型异常）')
            continue
        ma = np.ma.getmaskarray(da)
        mb = np.ma.getmaskarray(db)
        ok_mask = np.array_equal(ma, mb)
        va = np.ma.filled(da, np.nan).astype(np.float32)
        vb = np.ma.filled(db, np.nan).astype(np.float32)
        alias = '' if name_r == name_f else '  [pycwr:%s / 快速:%s]' % (name_r, name_f)
        #   ★ NaN 位置也要单独核一遍: mask 相同不代表"未 mask 处没有 NaN"
        #     （NaN 若没被 mask 住, 后面对全数组求 max 会直接得 nan, 掩盖真实差异）。
        na_, nb_ = np.isnan(va), np.isnan(vb)
        ok_nan = np.array_equal(na_, nb_)
        if not ok_nan:
            _note('field:%s' % key, False,
                  '未 mask 的 NaN 位置不一致 (差 %d 个)%s' % (int((na_ ^ nb_).sum()), alias))
            del va, vb, da, db
            continue
        both = ~na_ & ~nb_
        if ok_mask and both.any():
            #   只在两侧都非 NaN 的位置上比数值 —— 否则 max 会被 NaN 污染成 nan
            d = float(np.max(np.abs(va[both].astype(np.float64)
                                    - vb[both].astype(np.float64))))
            _note('field:%s' % key, d == 0.0,
                  'maxdiff=%.3g (有效 %d/%d 点)%s' % (d, int(both.sum()), va.size, alias))
        elif ok_mask:
            _note('field:%s' % key, True, '全 NaN（两侧一致）%s' % alias)
        else:
            _note('field:%s' % key, False,
                  'mask 差 %d 个%s'
                  % (int(np.abs(ma.astype(np.int8) - mb.astype(np.int8)).sum()), alias))
        # 元数据（只在真正不同时才列出来，列为提示不算失败）
        ka = _field_meta(ref.fields[name_r])
        kb = _field_meta(fast.fields[name_f])
        diff = {k: (ka.get(k), kb.get(k)) for k in set(ka) | set(kb) if ka.get(k) != kb.get(k)}
        if diff:
            _note('field:%s 元数据' % key, True, '有差异(不影响数值): %s' % diff)
        del va, vb, da, db
    gc.collect()

    # ---- 顶层元数据（只提示）----
    mdiff = {k: (ref.metadata.get(k), fast.metadata.get(k))
             for k in set(ref.metadata) | set(fast.metadata)
             if ref.metadata.get(k) != fast.metadata.get(k)}
    if mdiff:
        _note('metadata 差异', True, str(mdiff))

    if verbose:
        for name, ok, detail, _ in rep['rows']:
            print('%-30s %-8s %s' % (name, 'OK' if ok else '★不一致', detail))
        _hr()
        print('pycwr  耗时 %.2f s' % t_ref)
        print('快速   耗时 %.2f s   （加速 %.1f 倍）' % (t_fast, t_ref / max(t_fast, 1e-9)))
        _hr('=')
        print('结论: %s' % ('全部逐项一致 ✔ 可以放心替换成快速读取'
                            if rep['ok'] else '★存在不一致, 不要替换!'))
        _hr('=')
    return rep['ok'], rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    if not os.path.exists(path):
        print('文件不存在: %s' % path)
        return 1

    if '--info' in argv:
        from .fast_pa import _cmd_info
        _cmd_info(path)
        return 0

    if '--verify' in argv:
        ok, _rep = compare_with_pycwr(path)
        return 0 if ok else 3

    if '--fields' in argv:
        i = argv.index('--fields')
        fl = [x.strip() for x in argv[i + 1].split(',') if x.strip()]
    else:
        fl = FAST_FIELDS_RUN

    if '--bench' in argv:
        for tag, f in (('cr(只解 dBZ)', FAST_FIELDS_CR),
                       ('run(解 7 个矩)', FAST_FIELDS_RUN),
                       ('全解 12 个矩', None)):
            t0 = time.time()
            r = read_pa_radar(path, fields=f)
            dt = time.time() - t0
            print('%-16s %6.2f s   %d 层/%d rays/%d gates  字段 %s'
                  % (tag, dt, r.nsweeps, r.nrays, r.ngates, sorted(r.fields)))
            del r
            import gc
            gc.collect()
        print('pycwr 锚点: 38.5 s')
        return 0

    r = read_pa_radar(path, fields=fl, verbose=True)
    print('OK: %d 层 / %d rays / %d gates, 字段 %s' % (r.nsweeps, r.nrays, r.ngates,
                                                      sorted(r.fields)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
