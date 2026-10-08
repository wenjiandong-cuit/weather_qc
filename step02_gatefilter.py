# -*- coding: utf-8 -*-
"""
QC/step02_gatefilter.py — 第 2 部分: QC 门控(质量控制)
=================================================================
顺序(不可随意调换, 后面的规则依赖前面已剔除的结果):
  1. exclude_transition()                                 ---- 边界过渡门      ★默认关
  2. ρhv 下限 + 反射率下限                                                  ★恒定执行
  2b. **ρhv 上限**(config.QC_RHOHV_MAX)                                      ★**现已关闭(2026-10-08 用户要求移除)**
      config 里是 None ⇒ 整段跳过。置成数值(如 1.15)即可重新启用, 用于剔掉
      ρhv>1 的门(这批数据 20.3% 的门恰好 =1.200, 物理上不可能是相关系数)。
      设 config.QC_RHOHV_MAX = None 可关掉这条。
  3. 冰雹核豁免(Z>=45 且 0.5<=ρhv<0.75 保留)                                ★默认关
  4. 低层近距静止地物(低仰角+近距+近零速度+小谱宽)        ---- 地物杂波      ★默认关
  5. 整层损坏剔除(z - total_power 大面积异常)              ---- 环形/弧状假回波 ★默认关
  6. 雷达附近"同色大值区"整层剔除(数值几乎相同的连通大块)  ---- 平台/饱和坏层 ★默认关
  7. 近距 reflectivity 明显高于 total_power(去远区基线)    ---- 近场/地物伪回波 ★默认开
  8. pyart moment + texture 门控(沿射线纹理)               ---- 非气象回波/杂斑 ★恒定执行
     ★ 2026-09-22 起这里的纹理走"跳过空窗"快速版(见 fast_texture.py), 快约 13 倍、
       结果与 pyart 原版逐位一致; 环境变量 QC_FAST_TEXTURE=0 可关回原版。
     ★ 2026-10-03 起在算纹理**之前**先把 φdp 按射线解缠(0/360 卷绕 -> 连续)——
       否则 pyart 的 φdp 纹理规则会把卷绕当成杂波, 在晴隆 ZR903 上剔掉近一半真实回波。
  9. 椒盐斑点抑制(φdp / 反射率)                                            ★默认开
返回 pyart.filters.GateFilter; 不修改 radar。

★★ 2026-10-07 用户决定: **默认删掉第 1/3/4/5/6 条**, 现用口径 = 2 + 2b + 7 + 8 + 9。
★★ 2026-10-08 用户要求: **加上 ρhv 上限**(第 2b 条, 默认 1.15), 剔掉 ρhv>1.15 的门
   —— 它们物理上不可能是相关系数(主要是那个 1.200 的尖峰)。阈值在 config.QC_RHOHV_MAX。
   代码路径一条没删, 要恢复哪条就给 build_qc_gatefilter 传对应开关 True:
   transition / hail_keep / clutter / whole_layer。
"""

import numpy as np

import pyart

from . import config as C

# --- 沿射线纹理换成快速版(跳过空窗; 与 pyart 原版逐位等价, 见 fast_texture.py) ---
# 必须在本模块任何 pyart 门控调用之前装上: pyart 的 gatefilter.py 里写的是
# `from ..util import texture_along_ray` —— 函数名已经绑进它自己的模块全局了,
# 所以只改 sigmath 里的定义没用, 要改 pyart.filters.gatefilter 里的那个名字。
# 关掉: 环境变量 QC_FAST_TEXTURE=0 (排查/对拍时用)。
from .fast_texture import install_fast_texture as _install_fast_texture  # noqa: E402

_install_fast_texture()
#
# 效果(实测, 16678 rays x 2000 gates x 4 个场):
#   pyart 原版 41.7 s -> 快速版约 3.2 s (约 13 倍), 结果逐位一致。
# 原因: 这卷数据 97.6% 的门是 NaN, 窗口里夹一个 NaN 结果必然还是 NaN,
#       即 97.8% 的滑窗是白算的 —— 快速版先用前缀和把"全有值"的窗口挑出来
#       (只占 2.2%), 只对这些真算。


# =============================================================================
# pyart "moment + texture" 门控(沿射线纹理)
# =============================================================================
# 直接调 pyart.filters.moment_and_texture_based_gate_filter:
#     过渡带 -> rhoHV 下限 -> phiDP / rhoHV / ZDR / 反射率 的沿射线纹理上限
#     (每条纹理规则同时剔掉该纹理场里 masked / invalid 的门)
# 参数不传就用 pyart 默认值(wind_size=7, min_rhv=0.6,
# max_textphi=20.0, max_textrhv=0.30, max_textzdr=2.85, max_textrefl=8.0);
# 要改直接传 pyart 的同名参数, 例如 moment_texture_gatefilter(radar, wind_size=11)。

_MTF_TEX = ('differential_phase_texture', 'cross_correlation_ratio_texture',
            'differential_reflectivity_texture', 'reflectivity_texture')


def _unwrap_phi_by_ray(phi):
    """
    按射线(**每一行**)在有效门上把 φdp 解缠, 消掉 0/360 卷绕。
    输入 2D(或 1D) masked array / ndarray; 返回同形状 masked array, **掩膜保持不变**。

    ★ 2026-10-03 为什么必须做:
      晴隆 ZR903 原始 φdp 值域是 0~360°, 约 52% 的数据挤在 336-360 与 0-12 两端,
      相邻门跳变 |Δ|>180° 的占 **12.88%**(围着 0/360 来回翻)。
      pyart 的 φdp 沿射线纹理是 7 门标准差, 阈值 max_textphi=20.0° —— 这种"翻转"
      被当成非气象杂波, 实测把 **51.3% 的门判为超阈**, 单条规则剔掉近一半真实回波,
      最终整条链只剩 32% 的有回波门(威宁同一条规则超阈 0.0%、逐门无影响)。
      解缠后纹理中位从 22.5° 回到正常量级, φdp 规则剔除率 56.9% -> 29.2%。
      step04/step06 里本来就有 np.unwrap(period=360), 但它们跑在 step02 **之后**,
      救不了门控 —— 所以这里把解缠提到纹理之前。

    注意: np.unwrap 是累积器, 判卷绕看的是**相邻门跳变是否接近 360 的整数倍**,
          不是端点差的符号。掩膜外的 NaN 不参与解缠、也不会被改动。
    """
    a = np.ma.asarray(phi)
    data = np.ma.filled(a, np.nan).astype(np.float64)
    out = data.copy()
    rows = np.atleast_2d(out)
    for i in range(rows.shape[0]):
        row = rows[i]
        good = np.isfinite(row)
        if good.sum() < 2:
            continue
        idx = np.nonzero(good)[0]
        row[idx] = np.unwrap(row[idx], period=360.0)
    if np.ma.isMaskedArray(a):
        return np.ma.masked_array(out, mask=np.ma.getmaskarray(a))
    return np.ma.masked_array(out, mask=~np.isfinite(out))


def moment_texture_gatefilter(radar, unwrap_phi=True, **kwargs):
    """
    pyart 的 moment + texture 门控(沿射线纹理) —— 就是 pyart 那个函数的直接调用。
    参数照抄 pyart.filters.moment_and_texture_based_gate_filter, 不传用默认值。
    (4 个源场 reflectivity / differential_phase / cross_correlation_ratio /
     differential_reflectivity 少一个, pyart 自己会抛 UnboundLocalError。)
    返回 GateFilter。

    unwrap_phi: 默认 True —— 算纹理前在**有效门上按射线解缠 φdp**(见 _unwrap_phi_by_ray),
                调完 pyart 立刻还原, radar 本身一个字节都不改。设 False 退回旧行为(排查对拍用)。
    """
    # pyart 2.2.5 的 add_field 不带 replace_existing: radar 上已有同名纹理场时会 ValueError,
    # 先摘下来再放回, 这样同一个 radar 可以反复跑(不影响门控结果)。
    # 另外 phi_field=None 时 pyart 会去查配置里没有的 "uncorrected_differential_phase",
    # 直接 KeyError -> 这里补上标准名。
    kwargs.setdefault('phi_field', 'differential_phase')
    keep = {n: radar.fields.pop(n) for n in _MTF_TEX if n in radar.fields}

    # ★ 解缠 φdp(只在喂给 pyart 的那一份上做, 出来就还原)
    _phi_field = kwargs['phi_field']
    _phi_backup = None
    if unwrap_phi and _phi_field in radar.fields:
        _phi_backup = radar.fields[_phi_field]
        _phi_new = dict(_phi_backup)
        _phi_new['data'] = _unwrap_phi_by_ray(_phi_backup['data'])
        radar.fields[_phi_field] = _phi_new

    try:
        res = pyart.filters.moment_and_texture_based_gate_filter(radar, **kwargs)
        # 兼容两种 pyart 返回: 新版只回 GateFilter, 老版回 (GateFilter, texture) 元组
        return res[0] if isinstance(res, tuple) else res
    finally:
        radar.fields.update(keep)
        if _phi_backup is not None:
            radar.fields[_phi_field] = _phi_backup


def build_qc_gatefilter(radar, verbose=True, whole_layer=False, near_range=True,
                        despeckle=True, clutter=False, transition=False, hail_keep=False):
    """
    QC 门控: ρhv + 弱回波 + (可选)静止地物 + 反射率/总功率异常 + pyart moment/texture + 椒盐斑点.
    返回 GateFilter; 不修改 radar。

    ★ 2026-10-07 用户决定: **默认删掉第 1/3/4/5/6 条规则**(只留 2/7/8), 要用时传开关打开。
      改动的只是默认值, 代码路径一条没删。

    开关(默认 = 现用口径 2/7/8):
      transition:      规则 1 边界过渡带 (默认 False)
      hail_keep:       规则 3 冰雹核豁免 (默认 False)
      clutter:         规则 4 低层近距静止地物 (默认 False)
      whole_layer:     规则 5/6 两条整层剔除(z-tp 坏层 / 同色大值区) (默认 False)
      near_range:      规则 7 近距 reflectivity 明显高于 total_power (默认 True)
      despeckle:       规则 8b 椒盐斑点抑制(φdp/反射率) (默认 True)
      (规则 2 ρhv/Z 下限、规则 8a pyart moment+texture 无开关, 恒定执行)

    传 False 的那几条**整段跳过**, 连中间量(如 z/cc)都不算, 省时间。
    """
    gf = pyart.filters.GateFilter(radar)

    # --- 规则 1: 边界过渡带(默认关) ---
    if transition:
        gf.exclude_transition()

    # --- 规则 2: ρhv / 反射率下限(恒定执行) ---
    gf.exclude_below('cross_correlation_ratio', C.QC_RHOHV_MIN)
    gf.exclude_below('reflectivity', C.QC_Z_MIN)

    # --- 规则 2b: ρhv 上限(2026-10-08 用户要求, 恒定执行) ---
    #   ρhv 是相关系数, 物理上不可能 >1; 但这批数据里有大量门恰好是 1.200 —— 那是**记号**
    #   (全卷 20.3% 的有效门 = 1.200; 逐层 34%~100% 都有)。这种门不能当气象用。
    #   上限值在 config.QC_RHOHV_MAX(默认 1.15); 设 None 就关掉。
    _rhi = getattr(C, 'QC_RHOHV_MAX', None)
    if _rhi is not None:
        n_before = int(gf.gate_included.sum())
        gf.exclude_above('cross_correlation_ratio', float(_rhi))
        if verbose:
            print('  [QC] ρhv 上限 %.2f: 剔除 %d 门 (ρhv 是相关系数, >1 必然是记号)'
                  % (float(_rhi), n_before - int(gf.gate_included.sum())))

    # --- 规则 3: 冰雹核 ρhv 偏低(0.5~0.75)但 Z 很高 -> 保留(默认关) ---
    z = radar.fields['reflectivity']['data']
    cc = radar.fields['cross_correlation_ratio']['data']
    if hail_keep:
        hail_keep_mask = (z >= C.QC_HAIL_KEEP_Z) & (cc >= C.QC_HAIL_RHOHV_MIN) \
            & (cc < C.QC_RHOHV_MIN)
        gf.include_gates(np.ma.filled(hail_keep_mask, False))

    # --- 规则 4: 静止地物(低层+近距+低速度+低谱宽) -> 剔除(默认关) ---
    if clutter and 'velocity' in radar.fields and 'spectrum_width' in radar.fields:
        vel = radar.fields['velocity']['data']
        sw = radar.fields['spectrum_width']['data']
        rng_km = radar.range['data'] / 1000.0
        r2d = np.broadcast_to(rng_km, (radar.nrays, radar.ngates))
        elev = np.ma.filled(radar.elevation['data'][:, None], 0.0)
        clutter = (elev <= C.CLUTTER_ELEV_MAX) & (r2d <= C.CLUTTER_RANGE_KM) \
            & (np.abs(vel) <= C.CLUTTER_VEL_MAX) & (sw <= C.CLUTTER_SW_MAX) \
            & (z >= C.CLUTTER_Z_MIN) & (z <= C.CLUTTER_Z_MAX) & (cc >= 0.9)
        gf.exclude_gates(np.ma.filled(clutter, False))
        if verbose:
            print(f'  [QC] 低层近距静止地物剔除 {np.ma.filled(clutter, False).sum()} 门')

    if whole_layer:  # --- 两条整层剔除规则(坏层 + 同色大值区), 可关 ---
        # 整层损坏防护: 个别仰角层出现反射率整体远高于总功率的坏数据(如某次扫描校准错误),
        # 会形成环形/弧形假回波(实测整层 z-tp≈+45dB、z 顶在 59/91dBZ 的残环),
        # 这类“层”整层剔除(其它层高度覆盖近似, 对组合反射率无损失)
        if 'total_power' in radar.fields:
            si = radar.sweep_start_ray_index['data']
            ei = radar.sweep_end_ray_index['data']
            rng_km = radar.range['data'] / 1000.0
            r2d = np.broadcast_to(rng_km, (radar.nrays, radar.ngates))
            zf = np.ma.filled(z, np.nan); tf = np.ma.filled(radar.fields['total_power']['data'], np.nan)
            for swi in range(radar.nsweeps):
                s0, e1 = int(si[swi]), int(ei[swi]) + 1
                sub = (r2d[s0:e1] >= C.BAD_TILT_MIN_RANGE_KM) & np.isfinite(zf[s0:e1]) \
                    & np.isfinite(tf[s0:e1]) & (tf[s0:e1] > -100) & (zf[s0:e1] > -30)
                if sub.sum() < 1000:
                    continue
                gap = (zf[s0:e1] - tf[s0:e1])[sub]
                frac_bad = float((gap > C.BAD_TILT_GAP_DB).mean())
                if frac_bad > C.BAD_TILT_FRAC:
                    rows = np.zeros(radar.nrays, dtype=bool)
                    rows[s0:e1] = True
                    gf.exclude_gates(np.broadcast_to(rows[:, None], (radar.nrays, radar.ngates)))
                    if verbose:
                        print(f'  [QC] !!! 整层损坏剔除: sweep {swi} elev='
                              f'{np.mean(radar.elevation["data"][s0:e1]):.2f}° '
                              f'(z-tp>{C.BAD_TILT_GAP_DB:.0f}dB 比例 {frac_bad*100:.0f}%, '
                              f'z 中位 {np.nanmedian(zf[s0:e1][sub]):.0f} dBZ / tp 中位 '
                              f'{np.nanmedian(tf[s0:e1][sub]):.0f} dBZ)')

        # 雷达附近"同色大值区"整层剔除:
        #   某些仰角层在雷达附近存在 Z>=40dBZ、数值几乎完全相同的连通大块(同一色标颜色),
        #   是数据损坏/饱和平台(实测 59.0dBZ 占 100%、覆盖全部射线), 在组合反射率里表现为
        #   一圈/一片同色假回波 -> 整层剔除(相邻仰角高度覆盖近似, 对组合反射率无损失)
        if C.PLATEAU_CHECK:
            si = radar.sweep_start_ray_index['data']
            ei = radar.sweep_end_ray_index['data']
            rng_km = radar.range['data'] / 1000.0
            r2d = np.broadcast_to(rng_km, (radar.nrays, radar.ngates))
            zf = np.ma.filled(z, np.nan)
            for swi in range(radar.nsweeps):
                s0, e1 = int(si[swi]), int(ei[swi]) + 1
                hi = (r2d[s0:e1] <= C.PLATEAU_RANGE_KM) & np.isfinite(zf[s0:e1]) & (zf[s0:e1] >= C.PLATEAU_Z_MIN)
                n = int(hi.sum())
                if n < C.PLATEAU_MIN_BLOB:
                    continue
                vals = zf[s0:e1][hi]
                q = np.round(vals / C.PLATEAU_BIN_DB).astype(np.int32)
                uq, cnt = np.unique(q, return_counts=True)
                bi = int(np.argmax(cnt))
                peak_frac = cnt[bi] / n
                if peak_frac < C.PLATEAU_PEAK_FRAC:
                    continue
                blob = hi & (np.round(zf[s0:e1] / C.PLATEAU_BIN_DB).astype(np.int32) == uq[bi])
                try:
                    from scipy.ndimage import label as _ndlabel
                    lab, nlab = _ndlabel(blob)
                except Exception:
                    lab, nlab = None, 0
                if nlab:
                    sizes = np.bincount(lab.ravel())[1:]
                    big = int(sizes.max())
                    big_id = int(np.argmax(sizes)) + 1
                    blob_rays = int(np.unique(np.nonzero(lab == big_id)[0]).size)
                else:
                    big, blob_rays = n, int(blob.any(axis=1).sum())
                if big >= C.PLATEAU_MIN_BLOB and blob_rays >= C.PLATEAU_MIN_RAYS:
                    rows = np.zeros(radar.nrays, dtype=bool)
                    rows[s0:e1] = True
                    gf.exclude_gates(np.broadcast_to(rows[:, None], (radar.nrays, radar.ngates)))
                    if verbose:
                        print(f'  [QC] !!! 雷达附近同色大值区, 整层剔除: sweep {swi} elev='
                              f'{np.mean(radar.elevation["data"][s0:e1]):.2f}° '
                              f'(Z>={C.PLATEAU_Z_MIN:.0f}dBZ 门 {n} 个, 众数 {uq[bi]*C.PLATEAU_BIN_DB:.1f}dBZ '
                              f'占 {peak_frac*100:.0f}%, 最大连通块 {big} 门 / {blob_rays} 条射线)')

    # 近距离内 reflectivity 明显高于 total_power(去远区基线后, 近场/地物伪回波, 常见山区雷达 3-8km 环)
    # 自校准: 用 20-60km 的 z-tp 中值当基线(各雷达 total_power 与 reflectivity 系统偏差不同,
    #         中远距 z>tp 主要是衰减订正产物), 只在 8km 内且超出基线+阈值才剔, 避免误删真实回波
    if near_range and 'total_power' in radar.fields:
        tp = radar.fields['total_power']['data']
        rng_km = radar.range['data'] / 1000.0
        r2d = np.broadcast_to(rng_km, (radar.nrays, radar.ngates))
        zf = np.ma.filled(z, np.nan); tf = np.ma.filled(tp, np.nan)
        far = (r2d >= 20.0) & (r2d <= 60.0) & (zf >= 5) & (zf <= 45) & np.isfinite(tf) & (tf > -100)
        diff_far = (zf - tf)[far]
        if diff_far.size > 5000 and np.isfinite(diff_far).all():
            bias = float(np.nanmedian(diff_far))
            near_gap = (zf - tf) - bias
            bad = (r2d <= C.REFL_POWER_DIFF_RANGE_KM) & (zf >= 0) & np.isfinite(tf) \
                & (tf > -100) & (near_gap >= C.REFL_POWER_DIFF_DB)
            gf.exclude_gates(np.ma.filled(bad, False))
            if verbose:
                print(f'  [QC] 近距反射率/总功率异常剔除 {np.ma.filled(bad, False).sum()} 门 (远区基线 {bias:.1f} dB)')

    # pyart "moment + texture" 门控(沿射线纹理; 放在椒盐抑制之前)
    mtf_gf = moment_texture_gatefilter(radar)
    n0 = int(gf.gate_included.sum())
    gf.exclude_gates(mtf_gf.gate_excluded)
    if verbose:
        print('  [QC] moment/texture: %d -> %d 门' % (n0, int(gf.gate_included.sum())))

    # 椒盐斑点抑制
    if despeckle:
        gf = pyart.correct.despeckle_field(radar, 'differential_phase',
                                           gatefilter=gf, size=C.DESPECKLE_PHI_SIZE)
        gf = pyart.correct.despeckle_field(radar, 'reflectivity',
                                           gatefilter=gf, size=C.DESPECKLE_Z_SIZE)
    if verbose:
        print(f'  [QC] 保留 {gf.gate_included.sum()}/{radar.nrays*radar.ngates} 门')
    return gf


def build_qc_gatefilter_basic(radar, verbose=False, use_clutter=False, whole_layer=False):
    """
    基本 QC(快速版, 批量出组合反射率图用) = 完整 QC 关掉"近距 refl-tp"和"椒盐抑制":
      ρhv 下限 + 反射率下限 (+ 可选过渡带 / 冰雹核保留 / 静止地物)
    参数
      use_clutter: 静止地物规则(需要 velocity / spectrum_width 字段)。★默认 False ——
                   2026-10-07 起完整 QC 也默认不用它(用户决定删规则 4)。
      whole_layer: 是否加上"整层损坏 / 雷达附近同色大值区"两条整层剔除规则
                   —— 只用到 reflectivity(+total_power), 很便宜, **打开能去掉环形假回波**。
                   ★默认 False —— 2026-10-07 起用户决定删规则 5/6。
    返回 GateFilter; 不修改 radar。

    注: 完整处理链的 step02(build_qc_gatefilter) 默认**带**纹理门控(见 C.MTF_ENABLE)。
    """
    return build_qc_gatefilter(radar, verbose=verbose, whole_layer=whole_layer,
                               near_range=False, despeckle=False, clutter=use_clutter)
