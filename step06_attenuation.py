# -*- coding: utf-8 -*-
"""
QC/step06_attenuation.py — 第 6 部分: 衰减订正(唯一实现 = 用户口径)
========================================================================================
★ 2026-10-06: 旧的 Park2005 论文版 / ERAD 版已按用户要求删除(归档 _trash_20261006/)。
  实现与入口**合并在本文件** —— 只此一个模块、一个函数。

完整链路(用户口径, 与 pyart 官方 ZPHI 的差异只在"总相位差怎么取"):
  ① 系统相位: step03 的 ρhv 高值段法。cell1/威宁 稀疏回波必须放宽参数:
       n_gates=5 / rho_min=0.90 / refl_min=10 (默认 50门/0.97/≤10km 命中 0 条)。
       取**全局中位数**作为硬件基线(系统相位是常数, 不随射线变)。
       实测 11:09 帧 φ_sys=11.17°(与"雨区起点φ中位 11.38°"独立吻合)。
  ② 扣系统相位: φ_sub = max(φ_raw − φ_sys, 0)。
       ★ 不取模 %360 —— 威宁 φdp 值域 0~75°(非 0~360 卷绕), 取模会把"扣完略小于 0"
         的门甩成 ~354° 的假相位。直接 clip 到 0(累积相位本就不该为负)。
       ★ 先剔哨兵值: φdp 有一个 30.0 的填充值(占弱回波门 ~10%), 用 refl>=refl_min
         先置 nan, 否则系统相位估计会取到假的 30°。
  ③ zch_phi 双尺度滤波(轻 0.6 km / 重 1.5 km): 先 fix_phi_negative_jump(负段|负|>180
      且后接正 -> 加 360), 再 kdp_dual_filter 拿**重度平滑 φdp**(phidp_heavy)。
  ④ ★ 总相位差 Φmax = 末端最后 N_TAIL(默认 9) 个 ρhv>RHO_TAIL(默认 0.9) 的门的
       **滤波后 φdp 平均**(不取累积最大值)。命中不足则退回 pyart 的"末 6 门中位"。
       这是用户定死的口径: 总相位差只信**相关系数高(质量好)的门**, 不用 pyart 那
       "随便末 6 门"(会混进低 ρhv 脏门, 相位虚高)。实测中位偏差 -0.04° vs pyart +0.71°。
  ⑤ 沿距离分配: **完全复刻 pyart calculate_attenuation_zphi 的公式**(self_cons /
       I_indef / ah / pia 一行不改), 只把 phidp_max 的取法换成 ④。
       系数用 pyart 自动查表 X 波段 a_coef=0.31916 / beta=0.64884, **不手动改**。

为什么这样对(与 pyart 官方差异的根因, 会话中已验证):
  - pyart 官方 phidp_max = median(末端6个未掩膜门) 是**绝对值**, 把系统相位基线
    当成了衰减 => PIA 虚高(弱回波时次 14.7x, 扣相位后 4.3x)。
  - 扣系统相位后 pyart 的 Φmax 与真 ΔΦ 逐条一致(r=0.906, 回归斜率 0.837)。
  - 再换成"末端 ρhv>0.9 高质量门", 中位偏差降到 -0.04°(几乎无偏)。
  - 剩余差异几乎全在系数 a_coef 0.31916 vs Park2005 α=0.254 (1.26x), 属波段口径。

★ 只订正反射率: ZDR 不做差分衰减订正(cor_zdr 直通原始量, 只套 QC 掩膜)。

字段:
  spec_at_user  dB/km   比衰减 A_H(只在雨区有值)
  pia_user      dB      路径积分衰减(只在雨区有值)
  cor_z_user    dBZ     订正后反射率 = refl + PIA(雨区订正, 雨区外 = 原始值)
  cor_zdr_user  dB      ZDR 直通(原始 differential_reflectivity, 只套 QC 掩膜)
  (prepare_radar_products 末尾会别名成无后缀的 spec_at/pia/cor_z/cor_zdr)
"""
import numpy as np

from . import config as C
from . import step03_system_phase, zch_phi

# ================= 本方法的默认参数(要改就改这里 / config 的 ZPHI_USER_* 覆盖) =================
N_TAIL = 9           # 末端取几个高质量门
RHO_TAIL = 0.90      # 高质量门相关系数阈值
N_TAIL_FALLBACK = 6  # 命中不足时退回 pyart 的"末 N 门中位"
SP_N_GATES = 5       # 系统相位: 连续几个 ρhv 高值门(cell1 稀疏回波适配)
SP_RHO_MIN = 0.90    # 系统相位: ρhv 阈值
SP_REFL_MIN = 10.0   # 系统相位: 反射率下限(只取真实降水门, 剔哨兵值 30.0)
SP_STAT = 'median'   # 系统相位合成: 段内中位数
# 系统相位扣除口径(2026-10-08 新增 switch, 见 config.ZPHI_USER_SP_MODE):
#   'global'  = 逐射线估完之后取全体命中的中位数, 所有射线扣同一个值(2026-09-27 起的口径)
#   'per_ray' = 每条射线扣各自的估计值, 没估到的射线回退全局中位数
SP_MODE = 'global'
SP_DROP_SENTINEL = False   # True = 估计前把恰好 0.0 / 30.0 的门也判为无效(这两个是填充值)
# ★ 2026-10-08 新增: Φmax(=总相位差)候选门的额外门槛。见 config.ATTEN_TAIL_*。
#   背景(12:51 数据): 有一批 Z≈0(无回波) + ρhv=1.08~1.20(物理不可能) + φdp 恒为 104~109° 的
#   垃圾门, 因为满足 ρhv>0.9 而被选成 Φmax → PIA 假增 23~25 dB → cor_z 沿整条射线变成红杠。
TAIL_REFL_MIN = 10.0   # 候选门必须 Z ≥ 10 dBZ(雨区); 设 None 关闭
TAIL_RHO_MAX = None    # 候选门 ρhv ≤ 该值; (这批数据 ρhv 普遍 >1, 打开会大幅少命中, 慎用)
# Φmax 命中不足时走哪条回退:
#   'smooth'(默认, 2026-10-08 修) = 用同样门槛下"末端 N 个合格门"的**滤波 φdp 中位**
#   'corr'  (旧行为)             = pyart 式"末 N 个未掩膜门的**沿射线累积最大值**中位"
#   ★ 旧行为会把"无回波垃圾门(Z≈0/ρhv 1.2/φdp 恒定 109°)"的假相位顶成 Φmax → 假订正。
TAIL_FALLBACK = 'smooth'
# ★ 提速开关(2026-10-08): 复用链上第 4 步(KDP)已经算好的重度平滑 φdp(字段 phidp_heavy),
#   省掉 ZPHI 内部那次重复滤波(实测逐位相同, 省 ~40 s/帧)。
#   独立调用时默认 False(安全); pipeline 在"KDP 与 ZPHI 用同一份 φdp"时显式传 True。
REUSE_PHIDP = False


def _cfg(name, default):
    return getattr(C, 'ZPHI_USER_' + name, default)


def zphi_attenuation_correction(radar, gf=None, fzl=None,
                                refl_field='reflectivity',
                                phi_field='differential_phase',
                                rho_field='cross_correlation_ratio',
                                temp_field='temperature',
                                doc_gates=None,
                                n_tail=None, rho_tail=None,
                                tail_refl_min=None, tail_rho_max=None, tail_fallback=None,
                                reuse_phidp=None, phidp_field='phidp_heavy',
                                sp_n_gates=None, sp_rho_min=None,
                                sp_refl_min=None, sp_stat=None,
                                sp_mode=None, sp_drop_sentinel=None,
                                write_back=True, verbose=True):
    """
    【用户口径】ZPHI 衰减订正: 扣系统相位 + zch_phi 双尺度滤波 + 末端 ρhv 高质量门当总相位差。
    ★ 只订正反射率(ZDR 直通), 系数用 pyart 自动查表 X 波段值(不手动改)。

    参数
      gf:        GateFilter; 传入则订正掩膜 = gatefilter 排除门 ∪ 融化层以上门(与 pyart 同)
      fzl:       融化层高度(m)。None 时从 temperature 场反推(temp 首次 <=0 的高度)。
                 ★ 订正只到融化层为止, 与 pyart 的 get_mask_fzl 一致。
      doc_gates: 每条射线末端丢弃的门数, 默认 C.ZPHI_DOC(100 门 = 3 km)。
      n_tail / rho_tail: 末端取几个 ρhv>rho_tail 的门当总相位差(默认 9 / 0.90)。
      tail_refl_min:     ★ Φmax 的候选门必须 Z ≥ 这个值(默认 C.ATTEN_TAIL_REFL_MIN = 10 dBZ),
                         把"无回波但 ρhv 畸高、φdp 恒定"的垃圾门挡在外面; 设 None 关闭。
      tail_rho_max:      ★ Φmax 的候选门 ρhv ≤ 这个值(默认 C.ATTEN_TAIL_RHO_MAX = None 关闭;
                         这批数据 ρhv 普遍 >1, 打开会大幅减少命中, 慎用)。
      sp_*:      系统相位估计参数(step03), 默认 n_gates=5 / rho_min=0.90 / refl_min=10。
      write_back: 写回 spec_at_user / pia_user / cor_z_user / cor_zdr_user 字段。
      verbose:    打印诊断。

    返回 (pia_d, cor_z_d, phi_sys, info)
      pia_d / cor_z_d 为 (nrays, ngates) masked array [dB / dBZ];
      phi_sys 为全局系统相位(度); info 为 dict(命中数 / Φmax 分位等)。
    """
    n_tail = int(_cfg('N_TAIL', N_TAIL) if n_tail is None else n_tail)
    rho_tail = float(_cfg('RHO_TAIL', RHO_TAIL) if rho_tail is None else rho_tail)
    if tail_refl_min is None:
        tail_refl_min = _cfg('TAIL_REFL_MIN', TAIL_REFL_MIN)
    tail_refl_min = None if tail_refl_min is None else float(tail_refl_min)
    if tail_rho_max is None:
        tail_rho_max = _cfg('TAIL_RHO_MAX', TAIL_RHO_MAX)
    tail_rho_max = None if tail_rho_max is None else float(tail_rho_max)
    if tail_fallback is None:
        tail_fallback = str(_cfg('TAIL_FALLBACK', TAIL_FALLBACK)).lower()
    if tail_fallback not in ('smooth', 'corr'):
        raise ValueError("tail_fallback 只能是 'smooth' / 'corr', 收到 %r" % (tail_fallback,))
    if reuse_phidp is None:
        reuse_phidp = _cfg('REUSE_PHIDP', REUSE_PHIDP)
    sp_n_gates = int(_cfg('SP_N_GATES', SP_N_GATES) if sp_n_gates is None else sp_n_gates)
    sp_rho_min = float(_cfg('SP_RHO_MIN', SP_RHO_MIN) if sp_rho_min is None else sp_rho_min)
    sp_refl_min = float(_cfg('SP_REFL_MIN', SP_REFL_MIN) if sp_refl_min is None else sp_refl_min)
    sp_stat = _cfg('SP_STAT', SP_STAT) if sp_stat is None else sp_stat
    sp_mode = str(_cfg('SP_MODE', SP_MODE) if sp_mode is None else sp_mode).lower()
    if sp_mode not in ('global', 'per_ray'):
        raise ValueError("sp_mode 只能是 'global' / 'per_ray', 收到 %r" % (sp_mode,))
    sp_drop = bool(_cfg('SP_DROP_SENTINEL', SP_DROP_SENTINEL)
                   if sp_drop_sentinel is None else sp_drop_sentinel)
    doc = int(round(C.ZPHI_DOC if doc_gates is None else doc_gates))

    refl = np.ma.filled(radar.fields[refl_field]['data'], np.nan).astype(float)
    phi = np.ma.filled(radar.fields[phi_field]['data'], np.nan).astype(float)
    rho = np.ma.filled(radar.fields[rho_field]['data'], np.nan).astype(float)
    temp = np.ma.filled(radar.fields[temp_field]['data'], np.nan).astype(float)
    nrays, ngates = refl.shape
    range_m = radar.range['data'].astype(float)
    mask_ok = np.isfinite(phi) & np.isfinite(refl)
    # ★ 无 mask 副本(filled -20): pyart 内部做反射率算术必须用这个, 不能用 filled nan,
    #   否则 smooth_masked 遇 nan 整窗废掉 -> refl_linear 全 0 -> I[0]=0 -> ah 算不出。
    refl_nm = np.ma.filled(radar.fields[refl_field]['data'], -20.0).astype(float)

    # ---- 波段参数: 只补频率单位(GHz→Hz), 自动查表, 不手动改系数 ----
    from pyart.retrieve.echo_class import get_freq_band
    from pyart.correct.attenuation import _param_attzphi_table
    f_raw = float(np.asarray(radar.instrument_parameters['frequency']['data']).ravel()[0])
    f_hz = f_raw * 1e9 if f_raw < 1e9 else f_raw          # 9.304(GHz) -> Hz
    band = get_freq_band(f_hz)
    a_coef, beta, c_coef, d_coef = _param_attzphi_table()[band]

    # ---- ① 系统相位(step03 ρhv 高值段; 逐射线估计) ----
    # 先剔哨兵值 30.0: 只保留有真实降水回波的门
    #   ★ sp_drop_sentinel=True 时连 0.0 也一并判无效(这批数据里 0.0/30.0 都是填充值:
    #     全卷有效 φdp 门里 0.0 占 47.7%、30.0 占 16.1%, 而 _FillValue 写的是 -9999,
    #     读进来只按 NaN 掩膜, 所以它们会被当成真实相位)。
    phi_clean = np.where(refl >= sp_refl_min, phi, np.nan)
    if sp_drop:
        phi_clean = np.where((phi == 0.0) | (phi == 30.0), np.nan, phi_clean)
    sp_ray = step03_system_phase.per_ray_system_phase(
        rho, phi_clean, n_gates=sp_n_gates, rho_min=sp_rho_min, stat=sp_stat,
        gate_range=range_m, refl=refl, refl_min=sp_refl_min)
    n_sp = int(np.isfinite(sp_ray).sum())
    phi_sys = float(np.nanmedian(sp_ray)) if n_sp else 0.0

    # ---- ② 扣系统相位(不取模, clip>=0) ----
    #   'global'  : 全卷扣同一个常数(硬件相位是常数; 2026-09-27 起的口径)
    #   'per_ray' : 每条射线扣各自估计值, 没估到的回退全局中位数
    if sp_mode == 'per_ray':
        sp_use = np.where(np.isfinite(sp_ray), sp_ray, phi_sys)
        phi_sub = np.where(np.isfinite(phi),
                           np.maximum(phi - sp_use[:, None], 0.0), np.nan)
        n_sp_fallback = int(sp_ray.size - n_sp)
    else:
        sp_use = np.full(nrays, phi_sys)
        phi_sub = np.where(np.isfinite(phi), np.maximum(phi - phi_sys, 0.0), np.nan)
        n_sp_fallback = 0

    # ---- ③ zch_phi 双尺度滤波 -> 重度平滑 φdp ----
    # ★ 2026-10-08 提速: 链上第 4 步(KDP)已经算过同一份 φdp 的平滑结果并存成
    #   `phidp_heavy` 字段, 这里**逐位相同**, 没必要再算一遍(实测省 ~40 s/帧, 127->87 s)。
    #   reuse_phidp: True=用 radar.fields[phidp_field]; 或直接传 (nrays,ngates) 数组;
    #                False/None=照旧重算(独立调用时的安全默认)。
    ph_reuse = None
    if reuse_phidp is not None and not isinstance(reuse_phidp, (bool, np.bool_)):
        # 直接给数组: 掩膜数组要按 nan 取(不能用 np.asarray, 它会把掩膜下面的原始数据带出来)
        ph_reuse = np.ma.filled(np.ma.asarray(reuse_phidp), np.nan).astype(float)
    elif bool(reuse_phidp):
        _f = (radar.fields or {}).get(phidp_field)
        if _f is not None:
            ph_reuse = np.ma.filled(_f['data'], np.nan).astype(float)
    if ph_reuse is not None:
        ph = ph_reuse
        phi_smooth = np.where(np.isfinite(ph), np.maximum(ph, 0.0), np.nan)
    else:
        phi_fix = zch_phi.fix_phi_negative_jump(phi_sub)
        _kd, _kl, _kh, _pl, ph, _info = zch_phi.kdp_dual_filter(
            phi_fix, refl, range_m, verbose=False)
        phi_smooth = np.where(np.isfinite(ph), np.maximum(ph, 0.0), np.nan)

    # ---- 融化层掩膜(end_gate, 与 pyart 一致) ----
    if fzl is None:
        # 从 temperature 场反推: 每条射线第一个 temp<=0 的门高度(取中位数)
        _g0 = np.argmax(temp <= 0.0, axis=1)
        _has = (temp <= 0.0).any(axis=1)
        if _has.any():
            _alt = radar.gate_altitude['data']
            fzl = float(np.nanmedian(_alt[np.arange(nrays)[_has], _g0[_has]]))
        else:
            fzl = float(range_m[-1])
    from pyart.correct.attenuation import get_mask_fzl, _prepare_phidp
    mask_fzl, end_gate = get_mask_fzl(
        radar, fzl=fzl, doc=doc, min_temp=0.0, max_h_iso0=0.0,
        thickness=None, beamwidth=None, temp_field=None, iso0_field=None,
        temp_ref='fixed_fzl')

    # gatefilter 排除门 + 融化层以上门
    gf_excl = np.zeros((nrays, ngates), dtype=bool)
    if gf is not None:
        gf_excl = ~np.asarray(gf.gate_included)
    mask = gf_excl | mask_fzl

    # ---- ④ 总相位差 Φmax = 末端 N_TAIL 个 ρhv>RHO_TAIL 门的滤波后 φ 平均 ----
    # ★ 无 mask 副本(nan→0): _prepare_phidp 做 np.maximum.accumulate, 遇 nan 会一路传播
    #   成 nan -> init_refl 全 nan -> refl_linear 全 0。必须先把 nan 填 0。
    phi_sub_nm = np.where(np.isfinite(phi_sub), phi_sub, 0.0)
    rng_idx = np.arange(ngates)
    pmax = np.full(nrays, np.nan)
    pmax_fallback = np.zeros(nrays, dtype=bool)
    corr_phidp = _prepare_phidp(phi_sub_nm, mask_fzl)    # 累积最大值(反射率订正/fallback 用)
    for k in range(nrays):
        e = int(end_gate[k])
        if e < 8:
            continue
        tail = (rng_idx < e) & (rho[k] > rho_tail) & np.isfinite(phi_smooth[k])
        # ★ 2026-10-08: Φmax 只允许在**雨区**取值, 并排除 ρhv>1(物理不可能)的门。
        #   原因: 12:51 数据里有一批"Z≈0(无回波) + ρhv=1.08~1.20 + φdp 恒为 104~109°"的垃圾门,
        #   它们满足 ρhv>0.9 从而被选成 Φmax → PIA 假增 23~25 dB → cor_z 沿整条射线变红杠。
        if tail_refl_min is not None:
            tail &= np.isfinite(refl[k]) & (refl[k] >= tail_refl_min)
        if tail_rho_max is not None:
            tail &= (rho[k] <= tail_rho_max)
        tg = np.flatnonzero(tail)
        if tg.size >= n_tail:
            pmax[k] = float(np.mean(phi_smooth[k, tg[-n_tail:]]))
        else:
            # 命中不足 -> 回退分支。两种来源(见 TAIL_FALLBACK):
            #   'smooth' = 同样门槛下"末端 N 个合格门"的滤波 φdp 中位(2026-10-08 新增, 默认)
            #   'corr'   = pyart 式: 末 N 个未掩膜门的"沿射线累积最大值"中位(旧行为)
            # ★ 旧行为会把"无回波垃圾门(Z≈0 / ρhv 1.08~1.20 / φdp 恒定 104~109°)"的假相位
            #   顶成 Φmax → PIA 假增 23~25 dB → cor_z 沿整条射线变红杠。
            if tail_fallback == 'corr':
                six = np.flatnonzero(~mask[k, :e])[-N_TAIL_FALLBACK:]
                if six.size == N_TAIL_FALLBACK:
                    pmax[k] = float(np.median(corr_phidp[k, six]))
                    pmax_fallback[k] = True
            else:
                ok_g = np.isfinite(phi_smooth[k]) & np.isfinite(refl[k])
                if tail_refl_min is not None:
                    ok_g &= (refl[k] >= tail_refl_min)
                if tail_rho_max is not None:
                    ok_g &= (rho[k] <= tail_rho_max)
                six = np.flatnonzero(ok_g[:e])[-N_TAIL_FALLBACK:]
                if six.size == N_TAIL_FALLBACK:
                    pmax[k] = float(np.median(phi_smooth[k, six]))
                    pmax_fallback[k] = True

    # ---- ⑤ 复刻 pyart 循环(只换 phidp_max 取法) ----
    from pyart.correct.phase_proc import smooth_masked
    from scipy.integrate import cumulative_trapezoid
    init_refl = refl_nm + corr_phidp * a_coef
    dr = (range_m[1] - range_m[0]) / 1000.0
    # 平滑窗: 与 pyart 自适配一致(降水门中位数 // 4), 至少 3
    if gf is not None:
        _rain = mask_ok & ~gf_excl & (refl >= C.ATTEN_RAIN_REFL_MIN)
    else:
        _rain = mask_ok & (refl >= C.ATTEN_RAIN_REFL_MIN)
    _ng_rain = _rain.sum(axis=1)
    smooth = max(3, int(np.percentile(_ng_rain[_ng_rain > 0], 50) // 4)) if (_ng_rain > 0).any() else 3
    sm_refl = smooth_masked(init_refl, wind_len=smooth, min_valid=1, wind_type='mean')
    refl_linear = np.ma.power(10.0, 0.1 * beta * sm_refl).filled(fill_value=0)

    ah = np.ma.zeros((nrays, ngates), dtype='float64')
    pia = np.ma.zeros((nrays, ngates), dtype='float64')
    for ray in range(nrays):
        e = int(end_gate[ray])
        if e < 0 or e <= smooth:
            continue
        if not np.isfinite(pmax[ray]):
            continue
        ray_refl = refl_linear[ray, :e]
        self_cons = 10.0 ** (0.1 * beta * a_coef * pmax[ray]) - 1.0
        if self_cons <= 0.0:
            continue
        I = cumulative_trapezoid(0.46 * beta * dr * ray_refl[::-1])
        I = np.append(I, I[-1])[::-1]
        ah[ray, :e] = ray_refl * self_cons / (I[0] + self_cons * I)
        pia[ray, :-1] = cumulative_trapezoid(ah[ray, :]) * dr * 2.0
        pia[ray, -1] = pia[ray, -2]

    # ---- 订正: cor_z = refl + PIA(只雨区订正, 雨区外 = 原始值) ----
    pia_c = np.clip(np.ma.filled(pia, 0.0), 0.0, C.ATTEN_PIA_MAX)
    cor_z = refl + pia_c

    # mask 语义: cor_z 只 mask "原始 refl mask + gatefilter 排除门"; 融化层以上保持原值
    cor_z_mask = np.ma.getmaskarray(radar.fields[refl_field]['data']) | gf_excl
    rain = np.isfinite(refl) & np.isfinite(phi_sub) & (refl >= C.ATTEN_RAIN_REFL_MIN)
    pia_d = np.ma.masked_where(~rain, pia_c)
    cor_z_d = np.ma.minimum(np.ma.filled(cor_z, 0.0), C.ATTEN_COR_Z_MAX)
    cor_z_d = np.ma.masked_where(cor_z_mask, cor_z_d)

    if verbose:
        _pv = np.ma.filled(pia_d, np.nan)
        _pv = _pv[np.isfinite(_pv)]
        _pm = pmax[np.isfinite(pmax)]
        print(f'  [ZPHI·用户口径] 波段 "{band}" a_coef={a_coef:.5f} beta={beta:.5f} (自动查表)')
        print(f'  [ZPHI·用户口径] 系统相位 φ_sys={phi_sys:.2f}° ({sp_mode} 口径, step03 '
              f'{sp_n_gates}门/ρhv>{sp_rho_min}/Z≥{sp_refl_min}, 命中 {n_sp}/{nrays} 条)')
        if n_sp:
            _sv = sp_ray[np.isfinite(sp_ray)]
            print(f'  [ZPHI·用户口径] 逐射线基线散布 p10={np.percentile(_sv, 10):.2f} '
                  f'中位={np.median(_sv):.2f} p90={np.percentile(_sv, 90):.2f}° '
                  f'min={_sv.min():.2f} max={_sv.max():.2f}'
                  + (f' | 没估到回退全局的 {n_sp_fallback} 条' if sp_mode == 'per_ray' else ''))
        print(f'  [ZPHI·用户口径] 总相位差 = 末端{n_tail}个ρhv>{rho_tail}门平均: '
              f'命中 {int(np.isfinite(pmax).sum())}/{nrays} 条(退回末{N_TAIL_FALLBACK}门 '
              f'{int(pmax_fallback.sum())} 条); Φmax p50={np.median(_pm):.2f} '
              f'p90={np.percentile(_pm, 90):.2f} max={_pm.max():.2f}°')
        print(f'  [ZPHI·用户口径] PIA 均值={np.nanmean(_pv):.2f} '
              f'p99={np.percentile(_pv, 99):.2f} max={np.max(_pv):.2f} dB; '
              f'cor_z max={np.nanmax(np.ma.filled(cor_z_d, np.nan)):.1f} dBZ; '
              f'订正射线 {int((pia_c > 0).any(axis=1).sum())} 条')

    if write_back:
        radar.add_field('spec_at_user', {
            'data': np.ma.masked_where(~rain, ah), 'units': 'dB/km',
            'standard_name': 'specific_attenuation',
            'long_name': 'Specific attenuation (user: sys-phase + zch filter + rho-tail)'},
            replace_existing=True)
        radar.add_field('pia_user', {
            'data': pia_d, 'units': 'dB',
            'standard_name': 'path_integrated_attenuation',
            'long_name': 'Path integrated attenuation (user ZPHI)'},
            replace_existing=True)
        radar.add_field('cor_z_user', {
            'data': cor_z_d, 'units': 'dBZ',
            'standard_name': 'corrected_reflectivity',
            'long_name': 'Attenuation corrected reflectivity (user ZPHI)'},
            replace_existing=True)
        # ZDR 直通(不做差分订正, 与 park2005 同理由): 只套 QC 掩膜, 数值一个不改
        if 'differential_reflectivity' in radar.fields:
            zdr_raw = radar.fields['differential_reflectivity']['data']
            czdr = np.ma.masked_where(np.ma.getmaskarray(zdr_raw) | gf_excl,
                                      np.ma.filled(zdr_raw, 0.0))
            radar.add_field('cor_zdr_user', {
                'data': czdr, 'units': 'dB',
                'standard_name': 'differential_reflectivity',
                'long_name': 'ZDR passed through UNcorrected (user ZPHI)'},
                replace_existing=True)

    info = dict(phi_sys=phi_sys, sp_mode=sp_mode, sp_drop_sentinel=sp_drop,
                sp_ray=sp_ray, sp_ray_used=sp_use, n_sp_hit=n_sp,
                n_sp_fallback=n_sp_fallback,
                a_coef=a_coef, beta=beta, band=band,
                n_tail=n_tail, rho_tail=rho_tail, smooth=smooth, doc=doc,
                n_tail_hit=int(np.isfinite(pmax).sum()),
                n_tail_fallback=int(pmax_fallback.sum()),
                phidp_max=pmax, end_gate=end_gate,       # ★ 逐射线 Φmax 与末端门(排查虚假订正用)
                phidp_max_fallback=pmax_fallback,        # ★ 哪些射线的 Φmax 走了回退分支
                phidp_reused=bool(ph_reuse is not None))  # ★ 本次是否复用了 phidp_heavy
    return pia_d, cor_z_d, phi_sys, info


# 旧名别名(合并前 `zphi_attenuation_correction_user` 曾被导出), 指向同一个函数
zphi_attenuation_correction_user = zphi_attenuation_correction


__all__ = ['zphi_attenuation_correction',
           'zphi_attenuation_correction_user']
