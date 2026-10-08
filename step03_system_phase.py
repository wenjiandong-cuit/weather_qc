# -*- coding: utf-8 -*-
"""
QC/step03_system_phase.py — 系统相位估计(按射线, 取 ρhv 连续高值段的 φdp)
=================================================================
★ 本项目**唯一**的系统相位实现。旧法(前 40 门中位数)已按用户要求删除 ——
  它不看 ρhv, 弱信号/杂波门的相位噪声会一起进来。
  注: 本文件原名为 `QC/相位估计.py`, 2026-09-12 改为现名以统一英文命名
  (与同目录 stepNN_*.py 一致, 序号 3 正是本步骤在链上的位置)。

方法:

    每条射线从近到远扫描, 取**前 N_GATES(默认 30)个连续的
    ρhv > RHO_MIN(默认 0.97)的门**的 φdp 作为系统相位
    —— 只在信噪比好、纯气象的门上取, 相位更干净

找到的段取 30 个门的**平均**(STAT='mean'; 可改成 'median' 中位数或 'first' 首门),
找不到的射线给 nan, 再由全局兜底分位数填充(默认读 config.SYSPHASE_FALLBACK_PCT)。

入口(自底向上三层, 想用哪层用哪层):
    find_first_rho_run()        底层: 找每条射线第一个"连续 N 个 ρhv 高值"段的起点
    per_ray_system_phase()      数组级: (rho, phi) -> 逐射线系统相位
    estimate_system_phase()     对象级: radar -> 逐射线系统相位(自动取字段)
    remove_system_phase_by_rho() 管线入口: 逐射线估计 + 整场扣除 -> phi_corr
"""

import numpy as np

from . import config as C

# ================= 本方法的默认判据(要改就改这里, 或调用时传参) =================
RHO_MIN = 0.97        # ρhv 高值门阈值
N_GATES = 50          # 连续取几个门
STAT = 'mean'         # 这 N 个门的 φdp 怎么合成系统相位: 'mean' | 'median' | 'first'
RHO_FIELD = 'cross_correlation_ratio'    # radar 里的 ρhv 字段名
PHI_FIELD = 'differential_phase'         # radar 里的 φdp 字段名(未滤波的原始相位)

# ★ 这批数据的 φdp 有两个**填充值(记号)**: 0.0 和 30.0
#   证据(威宁 20250520110900 整卷 793,648 个数值型 φdp 门):
#       0.000 占 47.7%、30.000 占 16.1%(第三名只有 0.1%), 且"整条射线全 0" 1225 条、
#       "整条射线全 30" 2085 条、两者从不混在同一条里 —— 是记号不是相位。
#   ⚠ 它们**不是 NaN**, 所以 `np.isfinite(phi)` 拦不住; 要显式排除。
#   有回波(Z≥10 且 ρhv>0.9)的门里, φdp 是这种记号的占 60.7%。
PHI_SENTINELS = (0.0, 30.0)


def _rewrap_phi(seg):
    """回正：φdp 是 0~360 卷绕量，>180° 的门其实是负相位(晴隆 352° = -8°)。
    减 360 映射到 (-180,180]，让负相位显形，之后普通算术平均/中位数就是对的。"""
    return np.where(seg > 180.0, seg - 360.0, seg)


def valid_gate_mask(rho, phi=None, rho_min=None, refl=None, refl_min=None,
                    gate_range=None, max_range_km=None, sentinel=None):
    """
    逐门判"这一格能不能用来估系统相位", 返回 bool 掩膜 (nrays, ngates)。
    **不要求连续** —— 中间断开也没关系(见 per_ray_system_phase 的 sparse 模式)。

    条件: ρhv > rho_min ; φdp 有限(且不是 sentinel 指定的填充值) ;
          可选 refl >= refl_min ; 可选 gate_range <= max_range_km。
    """
    if rho_min is None:
        rho_min = RHO_MIN
    rho = np.asarray(rho)
    ok = np.isfinite(rho) & (rho > rho_min)
    if phi is not None:
        _p = np.asarray(phi, dtype=float)
        ok = ok & np.isfinite(_p)
        if sentinel is not None:
            _bad = np.zeros(_p.shape, dtype=bool)
            for _v in np.atleast_1d(sentinel):
                _bad |= np.isclose(_p, float(_v), atol=1e-6, rtol=0.0)
            ok = ok & ~_bad
    if refl is not None and refl_min is not None:
        ok = ok & np.isfinite(np.asarray(refl)) & (np.asarray(refl) >= refl_min)
    if max_range_km is not None and gate_range is not None:
        ok = ok & (np.asarray(gate_range, dtype=float) <= max_range_km * 1000.0)[None, :]
    return ok


def max_run_length(ok):
    """每条射线**最长连续 True** 的长度, 返回 (nrays,)。用于"必须有一段连续合格门"这类判据。"""
    ok = np.asarray(ok, dtype=bool)
    c = np.zeros(ok.shape, dtype=np.int32)
    if ok.shape[1]:
        c[:, 0] = ok[:, 0]
    for j in range(1, ok.shape[1]):
        c[:, j] = np.where(ok[:, j], c[:, j - 1] + 1, 0)
    return c.max(axis=1)


def combine_phase(seg, stat):
    """把一个门段(或一组合格门)的 φdp 合成一个数。stat 支持:
       'mean' | 'median' | 'first'(最靠近雷达的那个) |
       'min' | 'max' | 'p5' / 'p10' / 'p25' / 'p75' / 'p90'(分位数)。
       ★ 当"系统相位(基线)"用时, 应该取低分位: 'min' / 'p5' / 'p10' / 'median';
         取 'max' 会把整条廓线压到 0, 只适合"这条射线是个常数块"的病态数据。
    """
    seg = np.asarray(seg, dtype=float)
    seg = seg[np.isfinite(seg)]
    if seg.size == 0:
        return np.nan
    s = str(stat).lower()
    if s == 'mean':
        return float(np.mean(seg))
    if s == 'median':
        return float(np.median(seg))
    if s == 'first':
        return float(seg[0])
    if s == 'min':
        return float(seg.min())
    if s == 'max':
        return float(seg.max())
    if s.startswith('p') and s[1:].replace('.', '', 1).isdigit():
        return float(np.percentile(seg, float(s[1:])))
    raise ValueError("stat 只支持 mean/median/first/min/max/pN, 收到: %r" % (stat,))


def find_first_rho_run(rho, n_gates=None, rho_min=None, phi=None, require_phi=True,
                       gate_range=None, max_range_km=None, refl=None, refl_min=None,
                       sentinel=None):
    """
    每条射线找**第一个连续 n_gates 个 ρhv > rho_min** 的段, 返回该段的起始列号。
    找不到的射线返回 -1。
    (如果**不要求连续**, 用 valid_gate_mask + combine_phase, 或 per_ray_system_phase(sparse=True))。

    rho: (nrays, ngates) bool/float 的 ρhv
    phi: 传入且 require_phi=True 时, 该段还要求 φdp 有效(非 nan) —— 避免取到空相位
         ★ 注意: "非 nan"**挡不住填充值 0.0/30.0**(它们是合法浮点数)。要挡就用 sentinel。
    sentinel: 传入则这些值一律视为无效, 例如 PHI_SENTINELS=(0.0, 30.0)。
              实测(威宁 20250520110900 sweep 13, n_gates=50): 不传 -> 命中 65 条
              (其中 22 条估出 0.0、22 条估出 30.0 = 咬到填充值); 传 (0.0,30.0) -> 命中 22 条,
              全部落在 127~131°。⇒ 想按射线估系统相位, 必须传这个。
    refl: 传入且 refl_min 给定时, 该段还要求 reflectivity >= refl_min(dBZ) ——
          只取有真实降水回波的门, 滤掉晴空弱回波里的 ρhv 噪声(晴隆 ρhv 偏低时尤其有用)。
    gate_range / max_range_km: 给定则只在 max_range_km 以内找。**建议给**: 实测本个例
                 有 19.5% 的射线要等到 10 km 之外才凑够连续 30 个高 ρhv 门, 那里的 φdp
                 已经含了真实的传播相位, 拿来当"系统相位"会**高估**(把订正量扣多)。
                  传 None(默认) = 不限距离, 完全按原始判据。
    返回: (nrays,) int
    """
    if n_gates is None:
        n_gates = N_GATES
    ok = valid_gate_mask(rho, phi=(phi if require_phi else None), rho_min=rho_min,
                         refl=refl, refl_min=refl_min, gate_range=gate_range,
                         max_range_km=max_range_km, sentinel=sentinel)
    if not require_phi and phi is not None and sentinel is not None:
        # 不要求 φdp 有效, 但仍要排除填充值
        _p = np.asarray(phi, dtype=float)
        _bad = np.zeros(_p.shape, dtype=bool)
        for _v in np.atleast_1d(sentinel):
            _bad |= np.isclose(_p, float(_v), atol=1e-6, rtol=0.0)
        ok = ok & ~_bad

    # 连续 True 的累计长度: c[i, j] = 以 j 结尾的连续 True 个数
    c = np.zeros(ok.shape, dtype=np.int32)
    if ok.shape[1]:
        c[:, 0] = ok[:, 0]
    for j in range(1, ok.shape[1]):
        c[:, j] = np.where(ok[:, j], c[:, j - 1] + 1, 0)

    hit = c >= n_gates                      # 该门结尾已经凑满 n_gates 个
    any_hit = hit.any(axis=1)
    first_end = np.argmax(hit, axis=1)      # 没命中的行返回 0, 后面用 any_hit 盖掉
    start = np.where(any_hit, first_end - n_gates + 1, -1)
    return start.astype(np.int64)


def per_ray_system_phase(rho, phi, n_gates=None, rho_min=None, stat=None,
                         gate_range=None, max_range_km=None, refl=None, refl_min=None,
                         sentinel=None, sparse=False, require_run=None):
    """
    逐射线系统相位(基线) = 该射线上若干合格门的 φdp 合成值。找不到 -> nan(调用方兜底)。

    三种口径:
      ① sparse=False (默认)              取**前 n_gates 个连续** ρhv>rho_min 的门, 只用这一段。
      ② sparse=True, require_run=None    **不要求连续**: 该射线所有合格门一起用,
                                         门数 >= n_gates 即可。
      ③ sparse=True, require_run=K       **仍然要求连续**: 该射线必须存在一段
                                         连续 K 个合格门(质量锚), 但基线用**全部**合格门
                                         (含断开的部分)来算。★ 2026-10-08 用户要求新增。

    实测(威宁 20250520110900, ρhv 上限 1.15, stat='p5'):
        口径① 连续50门            : 整卷 483/16678
        口径② sparse 10门         : 整卷 1151/16678
        口径③ sparse 10门 + 锚50   : 见运行结果(介于两者之间, 既有质量锚又不被切碎)

    rho, phi: (nrays, ngates) float
    n_gates:  口径①② = 需要的门数(②里是"至少几个"); 口径③ = 统计时至少几个门
    require_run: 口径③的连续段长度 K; None = 不要求连续(口径②)
    stat:     合成方式, 见 combine_phase():
              'mean' | 'median' | 'first' | 'min' | 'max' | 'p5'/'p10'/'p25'/'p75'/'p90'
              ★ 当"基线"用时应取低分位('min'/'p5'/'p10'/'median')。
    sentinel: 这些 φdp 值视为无效(填充值记号), 例如 PHI_SENTINELS=(0.0, 30.0)。
              ★ 只加"非 nan"是不够的: 0.0/30.0 不是 nan。强烈建议传。
    gate_range / max_range_km: 见 find_first_rho_run(建议给 max_range_km, 默认 None 不限)
    refl / refl_min: 额外要求 reflectivity >= refl_min 的门才参与(有回波才取相位)
    返回: (nrays,) float
    """
    if n_gates is None:
        n_gates = N_GATES
    if rho_min is None:
        rho_min = RHO_MIN
    if stat is None:
        stat = STAT

    rho = np.asarray(rho)
    phi = np.asarray(phi)
    nrays, ngates = rho.shape

    if sparse:
        ok = valid_gate_mask(rho, phi=phi, rho_min=rho_min, refl=refl,
                             refl_min=refl_min, gate_range=gate_range,
                             max_range_km=max_range_km, sentinel=sentinel)
        if require_run is not None:
            # ★ 质量锚: 该射线必须有一整段连续 require_run 个合格门
            mr = max_run_length(ok)
            ok = ok & (mr >= int(require_run))[:, None]
        out = np.full(nrays, np.nan)
        for i in range(nrays):
            jj = np.flatnonzero(ok[i])
            if jj.size < int(n_gates):
                continue
            seg = phi[i, jj]
            seg = seg[np.isfinite(seg)]
            if seg.size < int(n_gates):
                continue
            # 'first' 在 sparse 下 = 最靠近雷达的那一个合格门
            s = str(stat).lower()
            out[i] = float(seg[0]) if s == 'first' else combine_phase(_rewrap_phi(seg), s)
        return out

    # --- 连续模式(原行为, 未改动) ---
    start = find_first_rho_run(rho, n_gates, rho_min, phi=phi, require_phi=True,
                               gate_range=gate_range, max_range_km=max_range_km,
                               refl=refl, refl_min=refl_min, sentinel=sentinel)
    out = np.full(nrays, np.nan)
    for i in np.where(start >= 0)[0]:
        j0 = int(start[i])
        seg = phi[i, j0:j0 + min(n_gates, ngates - j0)]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue
        if str(stat).lower() == 'first':
            out[i] = float(seg[0])
        else:
            out[i] = combine_phase(_rewrap_phi(seg), stat)
    return out


def estimate_system_phase(radar, rho_field=None, phi_field=None, gatefilter=None,
                          n_gates=None, rho_min=None, stat=None, max_range_km=None,
                          fallback_pct=None, fillna=True, verbose=True):
    """
    对 pyart Radar 直接估计逐射线系统相位(不修改 radar)。

    参数
      rho_field / phi_field: 字段名, 默认 cross_correlation_ratio / differential_phase
      gatefilter:            传入则只在它保留的门里挑(排除已被剔除的杂波门)
      n_gates / rho_min:     判据(默认 30 / 0.97)
      stat:                  'mean'(默认) | 'median' | 'first'
      max_range_km:          只在雷达这个距离以内找(建议给, 见 find_first_rho_run);
                             None = 不限
      fallback_pct:          找不到连续段的射线, 用已找到射线的该分位数兜底
                             (默认 config.SYSPHASE_FALLBACK_PCT=10); None 则不兜底
      fillna:                False 则不兜底, 找不到的射线保持 nan
    返回 (start_phase, info)
      start_phase: (nrays,) float
      info:        dict(n_found, n_fallback, p10, median, p90)
    """
    rho_field = rho_field or RHO_FIELD
    phi_field = phi_field or PHI_FIELD
    if n_gates is None:
        n_gates = N_GATES
    if rho_min is None:
        rho_min = RHO_MIN
    if stat is None:
        stat = STAT
    if fallback_pct is None:
        fallback_pct = getattr(C, 'SYSPHASE_FALLBACK_PCT', 10.0)

    rho = np.ma.filled(radar.fields[rho_field]['data'], np.nan).astype(float)
    phi = np.ma.filled(radar.fields[phi_field]['data'], np.nan).astype(float)
    if gatefilter is not None:
        keep = np.asarray(gatefilter.gate_included)
        rho = np.where(keep, rho, np.nan)
        phi = np.where(keep, phi, np.nan)

    start_phase = per_ray_system_phase(rho, phi, n_gates=n_gates,
                                       rho_min=rho_min, stat=stat,
                                       gate_range=radar.range['data'],
                                       max_range_km=max_range_km)
    n_found = int(np.isfinite(start_phase).sum())
    n_fallback = 0
    if fillna and n_found < start_phase.size:
        if n_found:
            fb = float(np.nanpercentile(start_phase, fallback_pct))
        else:
            vals = phi[np.isfinite(phi)]
            fb = float(np.percentile(vals, fallback_pct)) if vals.size else 0.0
        n_fallback = int(start_phase.size - n_found)
        start_phase = np.where(np.isfinite(start_phase), start_phase, fb)

    info = dict(n_found=n_found, n_fallback=n_fallback)
    if n_found:
        info.update(p10=float(np.nanpercentile(start_phase, 10)),
                    median=float(np.nanmedian(start_phase)),
                    p90=float(np.nanpercentile(start_phase, 90)))
    if verbose:
        msg = (f'  [系统相位/ρhv判据] 连续{n_gates}个ρhv>{rho_min} 的射线 '
               f'{n_found}/{start_phase.size}')
        if n_found:
            msg += (f', 相位 p10={info["p10"]:.1f} 中位={info["median"]:.1f} '
                    f'p90={info["p90"]:.1f}°')
        if n_fallback:
            msg += f', 兜底 {n_fallback} 条'
        print(msg)
    return start_phase, info


def remove_system_phase_by_rho(phi_raw, mask_array, rho_raw, n_gates=50,
                               rho_min=0.90, stat='mean', gate_range=None,
                               max_range_km=None, fallback_pct=None, refl=None,
                               refl_min=None, mode='per_ray', wrap=None,
                               drop_sentinel=True, sentinel=PHI_SENTINELS,
                               sparse=False, require_run=None, verbose=True):
    """
    **只做"去系统相位"这一步**: 估系统相位 -> 从 φdp 里扣掉 -> 返回扣好的 φdp。
    (不想跑整条 QC/ZPHI 链、只要一个干净 φdp 时用这个。)

    ★ 2026-10-08 用户定稿的默认口径(不传参数就是这个): 
        n_gates=50 / rho_min=0.90 / stat='mean' / mode='per_ray' / wrap=False(clip)
        / drop_sentinel=True(0.0/30.0 当空值) / sparse=False(要求连续 50 门)
      其中 drop_sentinel 这一条是为了修掉"还有几条射线没扣掉"—— 不剔空值的话,
      有些射线会把填充值 0.0/30.0 当成基线(扣 0), 或回退到中位数 30.0(只扣 30),
      于是残留 95~130°。实测 sweep 13: 不剔 -> 8 条残留, 剔了 -> 0 条。

    判据(默认): 逐射线取前 n_gates 个连续 ρhv>rho_min 门的 φdp 合成(可加 refl>=refl_min 条件)。
    判据(sparse=True): **不要求连续**(中间断开无所谓), 该射线所有合格门一起用,
                       门数 >= n_gates 才算数; 加 require_run=K 则仍要求存在一段连续 K 门。

    mode='per_ray' (默认)
        **每条射线扣各自估计值**(没估到的射线回退全局中位数)。
    mode='global'
        逐射线估完后取**全体命中射线的中位数**, 所有射线扣同一个常数(2026-09-27 起的旧口径)。

    参数
      phi_raw, rho_raw: (nrays, ngates) float, QC 外已置 nan
      mask_array:       (nrays, ngates) bool, QC 通过的门(掩膜外一律返回 nan)
      gate_range / max_range_km: 见 find_first_rho_run(建议给 radar.range['data'] 与距离上限)
      mode:             'global' | 'per_ray'
      wrap:             None(默认) = mode='global' 时 True(取模 360, 兼容旧行为)、
                        mode='per_ray' 时 False(clip 到 0)。
                        威宁/六枝 φdp 值域 0~150° ⇒ 用 False(clip);
                        晴隆那种 φ_sys≈349°、φdp 是 0~360 卷绕量的站才用 True。
      drop_sentinel:    True = 估计前把恰好 0.0 / 30.0 的门判为无效(这两个是填充值)
      sparse:           True = 不要求连续(见上); 此时 n_gates 是"至少要几个门"
      stat:             合成方式, 见 combine_phase()。当基线用建议 'p5'/'min'/'median';
                        sparse 模式下配合 'p5' 效果最好(实测 sweep 13: 命中 29/269)。
    返回 (phi_corr, sp_used)
      phi_corr: (nrays, ngates) 扣完系统相位的 φdp, 掩膜外 = nan
      sp_used : (nrays,) **每条射线实际扣掉的值**(global 时就是那个常数铺开)
    """
    if n_gates is None:
        n_gates = 50
    if rho_min is None:
        rho_min = 0.90
    if stat is None:
        stat = 'mean'
    if fallback_pct is None:
        fallback_pct = getattr(C, 'SYSPHASE_FALLBACK_PCT', 10.0)
    mode = str(mode).lower()
    if mode not in ('global', 'per_ray'):
        raise ValueError("mode 只能是 'global' / 'per_ray', 收到 %r" % (mode,))

    good = np.asarray(mask_array)
    phi_in = np.asarray(phi_raw, dtype=float)
    rho_use = np.where(good, np.asarray(rho_raw), np.nan)
    # ★ "φdp 不是 nan" 挡不住填充值 0.0/30.0(它们是合法浮点数) ⇒ 把记号交给判据显式排除
    _bad = np.zeros(phi_in.shape, dtype=bool)
    if drop_sentinel:
        for _v in np.atleast_1d(sentinel):
            _bad |= np.isclose(phi_in, float(_v), atol=1e-6, rtol=0.0)
    phi_for_crit = np.where(_bad, np.nan, phi_in)
    start_phase = per_ray_system_phase(rho_use, phi_for_crit,
                                       n_gates=n_gates, rho_min=rho_min, stat=stat,
                                       gate_range=gate_range, max_range_km=max_range_km,
                                       refl=refl, refl_min=refl_min,
                                       sentinel=(sentinel if drop_sentinel else None),
                                       sparse=sparse, require_run=require_run)
    n_found = int(np.isfinite(start_phase).sum())
    if n_found:
        global_phase = float(np.nanmedian(start_phase))  # start_phase 已回正成连续值，普通中位数即可
    else:
        vals = phi_for_crit[good & np.isfinite(phi_for_crit)]
        global_phase = float(np.percentile(vals, fallback_pct)) if vals.size else 0.0

    if mode == 'per_ray':
        sp_used = np.where(np.isfinite(start_phase), start_phase, global_phase)
        n_fallback = int(start_phase.size - n_found)
    else:
        sp_used = np.full(phi_in.shape[0], global_phase)
        n_fallback = 0

    do_wrap = (mode == 'global') if wrap is None else bool(wrap)
    if do_wrap:
        # wrap 回 0~360(晴隆: φdp 是 0~360 卷绕量, 不 wrap 会甩成负值)
        phi_corr = np.where(np.isfinite(phi_in), (phi_in - sp_used[:, None]) % 360.0, np.nan)
    else:
        # 直接 clip 到 0(威宁: φdp 值域 0~150°, wrap 会把"扣完略小于 0"的门甩成 ~354° 假相位)
        phi_corr = np.where(np.isfinite(phi_in),
                            np.maximum(phi_in - sp_used[:, None], 0.0), np.nan)
    phi_corr[~good] = np.nan

    if verbose:
        _v = start_phase[np.isfinite(start_phase)]
        msg = (f'  [系统相位/ρhv判据·{mode}] 连续{n_gates}个ρhv>{rho_min}'
               f'(Z≥{refl_min}): 命中 {n_found}/{start_phase.size} 条')
        if _v.size:
            msg += (f', 逐射线散布 p10={np.percentile(_v, 10):.1f} '
                    f'中位={np.median(_v):.1f} p90={np.percentile(_v, 90):.1f}°')
        if mode == 'per_ray':
            msg += f' | 没估到回退全局({global_phase:.2f}°)的 {n_fallback} 条'
        else:
            msg += f' -> 全局系统相位 {global_phase:.2f}°(全卷统一扣)'
        msg += (' | %s | stat=%s | wrap=%s'
                % (('不要求连续(sparse) 且需连续%d门锚' % int(require_run)) if (sparse and require_run is not None)
                   else ('不要求连续(sparse)' if sparse else '连续%d门' % n_gates),
                   stat, '取模360' if do_wrap else 'clip到0'))
        print(msg)
    return phi_corr, sp_used


def remove_system_phase_per_layer(radar, phi_raw, mask_array, rho_raw, refl=None,
                                  n_gates=50, rho_min=0.90, stat='mean',
                                  refl_min=None, verbose=False):
    """
    **整卷逐层**去系统相位 —— 2026-10-08 用户定稿: 链上与 notebook 统一用这个。

    为什么必须逐层, 而不是整卷调一次 remove_system_phase_by_rho:
      那个函数里"没估到基线的射线 -> 回退全体命中的中位数"。整卷一起跑时这个回退值是
      **跨层**的(实测 12:51: 整卷回退 = 94°, 而低层真实只有 2~15°), 会把大量射线过度扣除;
      逐层跑时回退值是该层自己的中位数, 合理得多。

    参数
      radar:  pyart Radar(用它的 sweep_start/end_ray_index 与 range)
      phi_raw / mask_array / rho_raw: 同 remove_system_phase_by_rho(掩膜外 nan / bool / nan)
      refl, refl_min: 参与估计的最低反射率(notebook 口径 = ref_raw 与 0.0)
      n_gates / rho_min / stat: 同 remove_system_phase_by_rho
    返回 (phi_out, sp_used)
      phi_out: (nrays, ngates) 扣完系统相位的 φdp, 掩膜外 = nan
      sp_used: (nrays,) 每条射线实际扣掉的值
    """
    phi_raw = np.asarray(phi_raw, dtype=float)
    nrays, ngates = phi_raw.shape
    phi_out = np.full((nrays, ngates), np.nan)
    sp_used = np.full(nrays, np.nan)
    si = radar.sweep_start_ray_index['data']
    ei = radar.sweep_end_ray_index['data']
    gs = radar.range['data']
    for sw in range(len(si)):
        a, b = int(si[sw]), int(ei[sw]) + 1
        if b <= a:
            continue
        sub, sp = remove_system_phase_by_rho(
            phi_raw[a:b], mask_array[a:b], rho_raw[a:b],
            n_gates=n_gates, rho_min=rho_min, stat=stat,
            gate_range=gs,
            refl=(None if refl is None else np.asarray(refl)[a:b]),
            refl_min=refl_min, verbose=False)
        phi_out[a:b] = sub
        sp_used[a:b] = sp
    if verbose:
        v = sp_used[np.isfinite(sp_used)]
        msg = ('[系统相位·逐层] %d 层, 命中 %d/%d 条' % (len(si), v.size, nrays))
        if v.size:
            msg += ', 扣掉的值 p10/中位/p90 = %.2f / %.2f / %.2f°' % tuple(
                np.percentile(v, [10, 50, 90]))
        print(msg)
    return phi_out, sp_used


if __name__ == '__main__':
    # 直接用 config 里的个例跑一遍, 看命中率与相位分布
    import os
    import sys

    import pycwr

    if __package__ in (None, ''):                      # 直接 python step03_system_phase.py 时补 ROOT
        _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)

    radar = pycwr.io.read_auto(C.FILE_PATH).ToPyartRadar()
    print(f'雷达: {radar.nsweeps} 层, nrays={radar.nrays}, ngates={radar.ngates}')
    start_phase, info = estimate_system_phase(radar)

    # 合成方式对比: 平均(默认) vs 中位数(看段内起伏大不大)
    sp_mean, info_m = estimate_system_phase(radar, stat='mean', verbose=False)
    sp_med, info_d = estimate_system_phase(radar, stat='median', verbose=False)
    d = sp_med - sp_mean
    print(f'mean 合成 : 命中 {info_m["n_found"]}/{radar.nrays} '
          f'相位中位={info_m.get("median", float("nan")):.2f}°')
    print(f'median 合成: 命中 {info_d["n_found"]}/{radar.nrays} '
          f'相位中位={info_d.get("median", float("nan")):.2f}°')
    print(f'两者差(median-mean) 中位={np.nanmedian(d):.3f}° '
          f'p90={np.nanpercentile(np.abs(d), 90):.3f}° '
          f'|差|>1° 占比={100.0*(np.abs(d) > 1.0).mean():.1f}%')

    # 限制搜索距离(避免在远距取到已含传播相位的 φdp)
    sp10, info10 = estimate_system_phase(radar, max_range_km=10, verbose=False)
    print(f'限 10 km 内  : 命中 {info10["n_found"]}/{radar.nrays} '
          f'({100.0*info10["n_found"]/radar.nrays:.1f}%), 相位中位={info10.get("median", float("nan")):.1f}°')
