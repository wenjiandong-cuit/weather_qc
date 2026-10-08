# -*- coding: utf-8 -*-
"""
QC/zch_phi.py — Φ_DP 双尺度滤波 + 最小二乘 KDP(Park et al. 2009 方案)
=========================================================================
★ 本模块是 `step04_kdp.py`(CSU Bringi) 的**并列替代实现**, 不替换它。
  `step04_kdp.py` 走的是 CSU 的 FIR 滤波 + 最小二乘斜率(单尺度 3 km);
  本模块走的是"**先平滑 φdp, 再对平滑后的 φdp 做最小二乘拟合**"的双尺度方案,
  两者可以同时跑、互相比对(KEEP_FIELDS 里名字不同, 不会打架)。

方法(截图原文口径):
    "我们采用两种方法平滑差分相位 Φ_DP, 即**轻度滤波**(2 km)和**重度滤波**(6 km),
     如 Park 等人 (2009) 所述。随后从滤波后 Φ_DP 的**最小二乘拟合** [求 K_DP]。"

    ⇒ 两个尺度各走一遍完整流程, 各自算一个 K_DP:
         轻度 (light, 0.6 km): φdp 滑动平均 0.6 km -> 在 0.6 km 窗上对**平滑后**的 φ 最小二乘拟合 -> K_DP^light
         重度 (heavy, 1.5 km): φdp 滑动平均 1.5 km -> 在 1.5 km 窗上对**平滑后**的 φ 最小二乘拟合 -> K_DP^heavy
       最后**按反射率强度选一个**(Ryzhkov & Zrnić 1996 的经典做法):
         Z >= Z_SWITCH_DBZ(默认 40 dBZ) -> 用**轻度**(强回波里小尺度结构要保住)
         Z <  Z_SWITCH_DBZ             -> 用**重度**(弱回波噪声大, 要长窗压噪)

    ★★ 窗长已按用户要求从 2/6 km 先改 1/3 km(2026-10-06)、再缩到 **0.6/1.5 km**(2026-10-07)。
       (短窗对噪声更敏感, 换来的是 K_DP 径向分辨率 —— 强回波核心的 KDP 梯度、差分相位
       对流羽流本来就需要亚公里级分辨率才刻画得出来。)
    ★★ 2026-10-08 **修正拟合输入**: 拟合必须喂**平滑后**的 φdp。此前(10-07 接线时)
       拟合窗里的累积和用的是**原始** φ, 平滑后的 φdp 只被当成 phidp_light/heavy 存下来
       —— 即"双尺度"实际只剩拟合窗长度不同, 平滑没进 KDP, 与上面的方法不符。
       现在 Σφ / Σxφ 都改用平滑值(见 `_smooth_and_fit` 第 ② 段)。

    径向分辨率: 轻 0.6 km / 重 1.5 km。★ 本雷达库长 30 m(威宁)/ 75 m(六枝、晴隆) ⇒
    窗长是**米**, 不是库数 —— 别把 1 km 写成 1 门。

★ 与论文/Ryzhkov 的差异(都是刻意的, 别"修"回去):
    1. 窗内**缺测门不参与**滑动平均与拟合(用有效门数归一), 而不是把整窗判为无效。
       X 波段弱回波里 φdp 常有孤立空洞, 整窗作废会丢大量可用数据。
       代价: 拟合点数不足时(见 MIN_VALID_GATES)仍会退化成 NaN。
    2. 不做**负 K_DP 截断**。经典方案允许轻估计出负值(弱回波里是正常噪声),
       强行 clamp 到 0 会把噪声压成"大面积 0", 反而不像真实衰减。
       需要非负输出时传 `nonneg=True`(默认 False)。
    3. **不做**自洽性约束(self-consistency)。pyart 的 LP(KDP 自带那套)才需要,
       最小二乘法用不上; 想要自洽约束请走 `pyart.correct.phase_proc_lp`。
    4. **不做相位解缠(np.unwrap)**, 也不做 0/360 卷绕修复 —— 用户在这批数据上
       从未遇到过折叠(见 `fix_phi_negative_jump` 的说明)。只在一种特定形态上
       做一次加 360 平移: **前段为负、后段为正, 且负值绝对值 > 180°**。

★ 为什么轻/重两个尺度都要留字段:
    Z 阈值切换在 40 dBZ 附近会**跳变** —— 阈值上下相邻门可能直接差好几个 °/km,
    时间序列上表现为"毛刺"。同时输出两个尺度(`kdp_light` / `kdp_heavy`),
    就可以事后画 `kdp_heavy - kdp_light` 看这个跳变有多大, 再决定要不要做过渡。

用法
----
    import QC
    QC.zch_phi.compute_kdp_dual(radar, phi_corr, ref_raw)      # 直接算 + 写字段
    # 或只要数组(自己后处理):
    kd, ph, info = QC.zch_phi.dual_kdp(phi_corr, ref_raw, range_m=radar.range['data'])
    kd, kdp_l, kdp_h = QC.zch_phi.kdp_dual_filter(phi_corr, range_m=radar.range['data'])

字段(全部 replace_existing, 可反复跑)
    kdp_dual     °/km   最终 K_DP(按 Z 阈值切换后的结果)—— **这是主产物**
    kdp_light    °/km   轻度(1 km)估计, 不做阈值切换
    kdp_heavy    °/km   重度(3 km)估计, 不做阈值切换
    phidp_light  °      轻度平滑后的 φdp —— 衰减订正里 ΔZ 用它
    phidp_heavy  °      重度平滑后的 φdp —— 衰减订正里 ΔZ 用**这个**(经典做法)

★ 供ZPHI 衰减订正用时应传 `phidp_heavy`: 经典关系 ΔZ = 0.04·(Φ_DP^(heavy) − Φ_DP^(sys))
  (Ryzhkov & Zrnić 1995) 用的就是长窗平滑值, 短窗会把 φdp 噪声放大进PIA。
  ⚠ 1/3 km 的窗比经典 2/6 km 短一截, 噪声会更大 —— 订正量若出现非单调的锯齿,
  优先怀疑窗长而不是去动 Φ_SYS。

★★ 2026-10-07 更新: 本模块**已经接进 pipeline 了** —— `QC/step04_kdp.py` 的
  `compute_kdp_bringi()`(函数名历史遗留)内部就是调本模块的 `kdp_dual_filter`,
  字段也是本模块的原生名(`kdp_dual` / `phidp_heavy` / `kdp_light` / `kdp_heavy` /
  `phidp_light`)。旧的 `kdp_bringi` / `phidp_bringi` 名字**已全部改名去掉**,
  下游(KEEP_FIELDS / FHC / 批处理)都已同步改成 `kdp_dual` / `phidp_heavy`。

直接跑这个文件 = 用合成数据自检(已知 K_DP 能否还原):
    conda run -n wradlib --no-capture-output python QC/zch_phi.py
"""

import numpy as np

try:
    from . import config as C
except ImportError:                       # 直接 `python QC/zch_phi.py` 跑自检时没有包上下文
    import os
    import sys
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    import config as C

# ===================== 本方法的默认参数(要改就改这里, 或调用时传参) =====================

LIGHT_FILTER_KM = 0.6    # 轻度滤波窗长(km)  ★ 2026-10-07 由 1.0 改为 0.6
HEAVY_FILTER_KM = 1.5    # 重度滤波窗长(km)  ★ 2026-10-07 由 3.0 改为 1.5
LIGHT_LS_KM = 0.6        # 轻度估计的最小二乘拟合窗长(km); 默认 = 滤波窗长
HEAVY_LS_KM = 1.5        # 重度估计的最小二乘拟合窗长(km); 默认 = 滤波窗长
Z_SWITCH_DBZ = 40.0      # 反射率阈值(dBZ): >= 用轻度, < 用重度(Ryzhkov & Zrnić 1996)
MIN_VALID_GATES = 4      # 窗内至少要有这么多**有效门**才给值(最小二乘至少要 2 点)
MIN_VALID_FRAC = 0.4     # 窗内有效门占比下限, 低于此整窗判为无效
NONNEG = False           # True = K_DP 负值截断到 0(默认不截, 见文件头说明第 2 条)
KDP_MAX = 45.0           # ★ 2026-10-07: K_DP 上限(°/km), 超过的**截断**到该值(不是置 NaN)。
                         #   X 波段暴雨 K_DP 顶天 15~20 °/km, 45 留了 2 倍余量;
                         #   用来压掉 φdp 台阶造成的假极值(cell1 实测曾到 309 °/km)。
                         #   None = 不设上限。
                         #   ★ 30 m 库长下 45 °/km 相当于单库 φdp 增量 ≤ 1.35°。
CHUNK_RAYS = 2048        # 分块处理射线数(控制内存峰值, 见 _kdp_core)
NEG_JUMP_THR = 180.0     # 负段绝对值阈值(度): 超过它且后面转正, 就给负段加 360
ZDR_FIELD = 'differential_reflectivity'
KDP_FIELD = 'specific_differential_phase'
RANGE_FIELD = 'range'


# ================================== 底层工具 ==================================

def _cfg(name, default):
    """读config.py 的 ZCH_* 覆盖值; 没定义就用本模块的默认值。"""
    return getattr(C, 'ZCH_' + name, default)


def fix_phi_negative_jump(phi, bad=None, thr=None, all_before=False):
    """
    逐射线修正"**前段为负、后段为正, 且负值绝对值 > 180°**"的形态: 给负段加 360°。

    phi:   (nrays, ngates) float, 无效为 nan(或等于 bad)
    bad:   坏值标记(nan 之外的那个, 通常是 C.ATTEN_BAD_FILL); None = 只认 nan
    thr:   判据阈值(度), 默认读 config.ZCH_NEG_JUMP_THR, 再兜底模块常量 180
    all_before: False(默认) = 只给**紧邻转折的那一段连续负值**加 360;
               True = 给转折点**之前所有**负值都加 360(不管中间夹了几段正)。

    ★ 为什么不做 np.unwrap(period=360):
      用户在这批数据(六枝/晴隆/威宁)上**从未遇到过 φdp 0/360 卷绕** ——
      φdp 沿距离是单调升的, 根本不会从 ~350° 掉回 ~10°。既然卷绕不存在, 做
      解缠就是纯粹的风险: np.unwrap 一旦遇到噪声造成的假折点, 会把整条射线
      之后的相位整体平移 ±360°, 而这条射线**本来就该有**的常数偏移也就跟着错了。
      所以解缠步骤整个去掉, 只保留下面这一条**形态明确、方向明确**的修正。

    ★ 规则怎么落成代码:
      Φ_DP 是累积相位, 物理上不会真出现 <-180° 的差分相位; 一旦看到"一段负值且
      |负值| > 180°", 只可能是**表示上的偏移**(相位基线/缠绕的残留), 而它后面
      又接了正值 ⇒ 这一段应当整体 +360° 归到正区间。用户原话:
          "如果 phi 以前面负, 后面正, 而且负数的绝对值大于 180,
           那就给负的加 360, 变成正的"

    ★ 只在**有效门序列**上判相邻(中间可能隔着 QC 空洞); 加完 360 后原样放回,
      坏值全程不动。
    """
    thr = float(_cfg('NEG_JUMP_THR', NEG_JUMP_THR) if thr is None else thr)
    phi = np.array(phi, dtype=np.float64, copy=True)
    ok = np.isfinite(phi)
    if bad is not None:
        ok &= (phi != bad)
    for r in range(phi.shape[0]):
        row_ok = ok[r]
        idx = np.flatnonzero(row_ok)
        if idx.size < 2:
            continue
        vals = phi[r, idx]
        neg = vals < 0.0
        if not neg.any() or neg.all():
            continue
        # 转折点: 严格"负 -> 正"(在有效门序列上的相邻对)。
        # ★ 用 (vals<0) & (vals>0) 而不是 (vals<0) & ~(vals<0) —— 后者把
        #   "负 -> 恰好 0" 也算成转折, 而用户规则要求的是后面接**正**值。
        cross = np.flatnonzero((vals[:-1] < 0.0) & (vals[1:] > 0.0))
        for j in cross:
            if abs(vals[j]) <= thr:
                continue                       # 负值幅度不够大, 不动(判据不成立)
            if all_before:
                lo = 0
            else:
                # 回退到紧邻 j 的那一段连续负值的起点
                lo = j
                while lo > 0 and vals[lo - 1] < 0.0:
                    lo -= 1
            seg = np.arange(lo, j + 1)
            m = vals[seg] < 0.0
            vals[seg[m]] += 360.0
        phi[r, idx] = vals
    if bad is not None:
        phi[phi == bad] = bad
    phi[~ok] = np.nan
    return phi


# 旧名保留: 外部/旧脚本可能还 import 它。★ 语义已变(不再是 np.unwrap), 别再当解缠用。
unwrap_phi_by_ray = fix_phi_negative_jump


def _half_gates(window_km, range_m, min_gates=None):
    """窗长(km) -> 半窗库数。★按**米**换算, 不按库数(库长两站不同)。"""
    gs = float(np.median(np.diff(range_m))) if len(range_m) > 1 else 1.0
    if gs <= 0:
        gs = float(range_m[1] - range_m[0]) if len(range_m) > 1 else 1.0
    if gs <= 0:
        raise ValueError('range_m 不是严格递增的, 算不出库长')
    half = int(round(float(window_km) * 1000.0 / gs / 2.0))
    if min_gates is not None:
        half = max(half, int(min_gates) // 2)
    return max(half, 1), gs


def _win_sum(cs, j, half):
    """从累积和里取 [j-half, j+half] 的窗和(越界自动截到数组范围)。"""
    lo = np.maximum(j - half, 0)
    hi = np.minimum(j + half + 1, cs.shape[1] - 1)
    return cs[:, hi] - cs[:, lo]


def _cumsum0(a):
    """沿库方向累积和, 并在最前面补一列 0。

    ★ 补零列是为了让窗和能统一写成 `cum[:, hi] - cum[:, lo]`(不做补零就得
      在 lo==0 时特判)。补完后列数 = 原列数 + 1, 列 k 的含义 = 原数组 [0, k) 的和,
      所以窗 [j-half, j+half] 对应 `cum[:, j+half+1] - cum[:, j-half]`。
    """
    return np.concatenate(
        [np.zeros((a.shape[0], 1), dtype=np.float64), np.cumsum(a, axis=1, dtype=np.float64)],
        axis=1)


def _smooth_and_fit(phi, x_km, half_f, half_ls, min_valid, min_frac):
    """
    核心: 滑动平均平滑 + 最小二乘拟合。**逐块处理**(见 CHUNK_RAYS 说明)。

    phi:  (n, ngates) 已做过负段修正的 φdp, 无效为 nan
    x_km: (ngates,)    距离(km)
    返回 (kdp, phidp_smooth, nvalid)
      kdp:(n, ngates) °/km, 拟合斜率; NaN = 无效
      phidp_smooth:      (n, ngates) °, 滑动平均后的 φdp
      nvalid:           (n, ngates) int, 窗内有效门数(诊断用)

    做法:
      平滑 = 窗内有效门的均值(不是 sum/n_window, 也不是遇 NaN 就整窗作废)
      拟合 = 对**平滑后**的 φdp 在滑动窗上做一元最小二乘, 斜率即 K_DP
            Σ 由累积和给出, 复杂度 O(n·ngates), 不随窗长线性增长
    """
    n, ngates = phi.shape
    kdp = np.full((n, ngates), np.nan)
    phs = np.full((n, ngates), np.nan)
    nva = np.zeros((n, ngates), dtype=np.int32)
    j = np.arange(ngates)
    need = max(int(min_valid), 2)

    for s in range(0, n, max(int(_cfg('CHUNK_RAYS', CHUNK_RAYS)), 1)):
        e = min(s + max(int(_cfg('CHUNK_RAYS', CHUNK_RAYS)), 1), n)
        p = phi[s:e]
        v = np.isfinite(p)
        # 有效值填 0、坏值填 0 —— 后面按有效门数归一
        p0 = np.where(v, p, 0.0)

        # --- 累积和(全部 float64) ---
        cs_n = _cumsum0(v.astype(np.float64))              # Σ1
        cs_y = _cumsum0(p0)                                # Σφ
        cs_x = _cumsum0(v * x_km[None, :])                 # Σx
        cs_xx = _cumsum0(v * (x_km ** 2)[None, :])         # Σx²
        cs_xy = _cumsum0(v * p0 * x_km[None, :])           # Σx·φ

        # --- ① 滑动平均: 平滑窗长 half_f ---
        nf = _win_sum(cs_n, j, half_f)
        sf = _win_sum(cs_y, j, half_f)
        good_f = (nf >= need) & (nf >= min_frac * (2 * half_f + 1))
        ph_f = np.where(good_f, sf / np.maximum(nf, 1.0), np.nan)

        # --- ② 最小二乘拟合: 拟合窗长 half_ls, 喂的是**平滑后**的 φ(ph_f) ---
        # ★ 2026-10-08 修正: 此前这一段的累积和直接复用了上面那 5 个**原始** φ 的累积和
        #   (cs_y / cs_xy 都是原始 φ 的), 于是"双尺度"只剩拟合窗长度不同 —— 平滑
        #   压根没进 KDP, 与 Park et al. (2009) 的方法(先平滑 φdp, 再对**平滑后**的
        #   φdp 做最小二乘)以及本文件头部/下面的说明都不符。
        #   合成数据实测(真值 1.5 °/km, σ=1.5° 噪声 + 15% 空洞): 用平滑后的 φ 拟合,
        #   KDP 偏差 std 0.529 -> 0.262 °/km; 旧写法与"原始 φ 拟合"逐点相同(最大差 0)。
        #   旧的原始累积和到这里用完即弃, 先释放再建平滑版的, 峰值内存不变。
        #   ★ 边界: 平滑窗在库序列两端被截断, 那里的平滑值是一侧平均, 平滑后的序列不再
        #     是直线 —— 拿它拟合会把斜率系统性拉小(实测线性 φdp 在门 0 处 K_DP 减半)。
        #     所以**拟合只吃"完整窗内"的平滑值**(interior), 端点那半个窗的 K_DP 直接缺测。
        #     ⚠ 只限拟合: phs(phidp_heavy)仍按原样返回, ZPHI 的末端 Φmax 要读到射线末端的
        #       平滑 φ, 不能因为边界缺测而丢掉(那里 end_gate/doc 通常在序列内部, 不受影响)。
        del cs_n, cs_y, cs_x, cs_xx, cs_xy, sf, good_f
        interior = (j >= half_f) & (j <= ngates - 1 - half_f)
        vs = np.isfinite(ph_f) & interior[None, :]
        ps = np.where(vs, ph_f, 0.0)
        cs_n = _cumsum0(vs.astype(np.float64))             # 平滑值的有效门数
        cs_y = _cumsum0(ps)                                # Σφ_smooth
        cs_x = _cumsum0(vs * x_km[None, :])                # Σx
        cs_xx = _cumsum0(vs * (x_km ** 2)[None, :])        # Σx²
        cs_xy = _cumsum0(vs * ps * x_km[None, :])          # Σx·φ_smooth

        nl = _win_sum(cs_n, j, half_ls)
        sl = _win_sum(cs_y, j, half_ls)
        sx = _win_sum(cs_x, j, half_ls)
        sxx = _win_sum(cs_xx, j, half_ls)
        sxy = _win_sum(cs_xy, j, half_ls)
        good_l = (nl >= need) & (nl >= min_frac * (2 * half_ls + 1))
        denom = nl * sxx - sx * sx
        # 分母≈0 ⇒ 窗内所有门距离几乎相同(只在极短窗+库长很大时可能), 判为无效
        with np.errstate(invalid='ignore', divide='ignore'):
            slope = np.where(denom > 0.0, (nl * sxy - sx * sl) / denom, np.nan)
        slope = np.where(good_l, slope, np.nan)

        kdp[s:e] = slope
        phs[s:e] = ph_f
        nva[s:e] = nf
    return kdp, phs, nva


def _kdp_core(phi, ref, range_m, light_filter_km, heavy_filter_km,
              light_ls_km, heavy_ls_km, z_switch_dbz, min_valid, min_frac,
              nonneg, chunk_rays, kdp_max=None):
    """
    双尺度 K_DP 的真正实现(数组层, 不碰 radar)。

    phi:      (nrays, ngates) **已扣系统相位**的 φdp(无效为 nan), 无效为 nan
    ref:      (nrays, ngates) 反射率 dBZ(只用于 40 dBZ 阈值切换), 无效为 nan
    range_m:  (ngates,)        距离(m), 必须严格递增
    返回 (kdp_dual, kdp_light, kdp_heavy, phidp_light, phidp_heavy, info)
    """
    phi = np.asarray(phi, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64) if ref is not None else None
    range_m = np.asarray(range_m, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError('phi 必须是 (nrays, ngates), 现在是 %r' % (phi.shape,))
    if range_m.shape[0] != phi.shape[1]:
        raise ValueError('range_m 长度 %d 与 phi 库数 %d 不一致'
                         % (range_m.shape[0], phi.shape[1]))
    x_km = range_m / 1000.0

    half_f_l, gs = _half_gates(light_filter_km, range_m, min_valid)
    half_f_h, _ = _half_gates(heavy_filter_km, range_m, min_valid)
    half_l_l, _ = _half_gates(light_ls_km, range_m, min_valid)
    half_l_h, _ = _half_gates(heavy_ls_km, range_m, min_valid)

    n = phi.shape[0]
    kdp_l = np.full(phi.shape, np.nan)
    kdp_h = np.full(phi.shape, np.nan)
    phl = np.full(phi.shape, np.nan)
    phh = np.full(phi.shape, np.nan)

    # 逐块跑(内存: 5 个 (chunk, ngates+1) float64 累积和, 2048×2000 约 160 MB)
    step = max(int(chunk_rays), 1)
    for s in range(0, n, step):
        e = min(s + step, n)
        p = phi[s:e]
        k1, p1, _ = _smooth_and_fit(p, x_km, half_f_l, half_l_l,
                                    min_valid, min_frac)
        k2, p2, _ = _smooth_and_fit(p, x_km, half_f_h, half_l_h,
                                    min_valid, min_frac)
        kdp_l[s:e], phl[s:e] = k1, p1
        kdp_h[s:e], phh[s:e] = k2, p2

    # --- 按反射率阈值切换 ---
    if ref is None or not np.isfinite(z_switch_dbz):
        kdp_d = kdp_h.copy()                # 没给反射率就只用重度(保守)
        use_light = np.zeros(phi.shape, bool)
    else:
        use_light = np.isfinite(ref) & (ref >= float(z_switch_dbz))
        kdp_d = np.where(use_light, kdp_l, kdp_h)

    if nonneg:
        kdp_d = np.where(np.isfinite(kdp_d), np.maximum(kdp_d, 0.0), np.nan)
        kdp_l = np.where(np.isfinite(kdp_l), np.maximum(kdp_l, 0.0), np.nan)
        kdp_h = np.where(np.isfinite(kdp_h), np.maximum(kdp_h, 0.0), np.nan)

    # ★ 2026-10-07: K_DP 上限截断(超过 -> 拉到上限)。
    #   用途: φdp 台阶(坏数据)会让最小二乘拟合冲出几百 °/km 的假极值,
    #         这些格点物理上不合理, 但位置可能落在真实强回波里, 直接剔掉不如截断保守。
    #   * 三个输出都截(kdp_dual / light / heavy), 免得诊断时对不上。
    #   ★ 同时把被截断格点的 φdp(kdp_dual 选中的那一路)清成 NaN —— 截断说明那里的
    #     φdp 台阶本来就坏, 平滑后的 φdp 不该再带着它往下游走(ZPHI 用 phidp_heavy)。
    #     清 phidp 用**该格点在 kdp_dual 里选中的那一路**的截断掩膜来判。
    n_clipped = 0
    n_phi_cleared = 0
    if kdp_max is not None and np.isfinite(kdp_max):
        kx = float(kdp_max)
        with np.errstate(invalid='ignore'):
            clip_l = np.isfinite(kdp_l) & (kdp_l > kx)
            clip_h = np.isfinite(kdp_h) & (kdp_h > kx)
            clip_d = np.isfinite(kdp_d) & (kdp_d > kx)
            n_clipped = int(clip_d.sum() + clip_l.sum() + clip_h.sum())
        kdp_d = np.where(np.isfinite(kdp_d), np.minimum(kdp_d, kx), np.nan)
        kdp_l = np.where(np.isfinite(kdp_l), np.minimum(kdp_l, kx), np.nan)
        kdp_h = np.where(np.isfinite(kdp_h), np.minimum(kdp_h, kx), np.nan)
        # φdp 跟随: kdp_dual 走轻度的那部分跟着 clip_l, 走重度的跟着 clip_h
        phl = np.where(clip_l, np.nan, phl)
        phh = np.where(clip_h, np.nan, phh)
        n_phi_cleared = int(clip_l.sum() + clip_h.sum())

    # 没有任何尺度给值的地方, 切换结果也该是 NaN(别让一侧有值另一侧没值时
    # np.where挑出一个"看起来有效"的组合)
    both_dead = ~np.isfinite(kdp_l) & ~np.isfinite(kdp_h)
    kdp_d[both_dead] = np.nan

    info = dict(gate_spacing_m=float(gs),
                light_filter_km=float(light_filter_km), heavy_filter_km=float(heavy_filter_km),
                light_ls_km=float(light_ls_km), heavy_ls_km=float(heavy_ls_km),
                light_gates=int(2 * half_f_l + 1), heavy_gates=int(2 * half_f_h + 1),
                light_ls_gates=int(2 * half_l_l + 1), heavy_ls_gates=int(2 * half_l_h + 1),
                z_switch_dbz=float(z_switch_dbz) if ref is not None else None,
                frac_light=float(use_light.mean()) if ref is not None else 0.0,
                n_light=int(np.isfinite(kdp_l).sum()),
                n_heavy=int(np.isfinite(kdp_h).sum()),
                n_dual=int(np.isfinite(kdp_d).sum()),
                n_neg_dual=int((np.isfinite(kdp_d) & (kdp_d < 0)).sum()),
                kdp_max=None if kdp_max is None else float(kdp_max),
                n_clipped=int(n_clipped),
                n_phi_cleared=int(n_phi_cleared))
    return kdp_d, kdp_l, kdp_h, phl, phh, info


# ================================ 数组层接口 ================================

def dual_kdp(phi_corr, ref=None, range_m=None, do_fixneg=True, verbose=True, **kw):
    """
    双尺度 K_DP(**数组层**, 不写 radar 字段)。负段修正 + 双尺度滤波 + 阈值切换。

    phi_corr: (nrays, ngates) 已扣系统相位的 φdp, 无效为 nan
    ref:      (nrays, ngates) 反射率 dBZ, 只用于 z_switch_dbz 切换; None = 全用重度
    range_m:  (ngates,) 距离(m); None 时按库长 30 m 等间距假设(⚠ 别乱用)
    do_fixneg: True = 先调 fix_phi_negative_jump("负段 |负|>180 且后接正值" -> 加 360)
    kw:       透传给 kdp_dual_filter(light_filter_km / heavy_filter_km /
               light_ls_km / heavy_ls_km / z_switch_dbz / min_valid_gates /
               min_valid_frac / nonneg / chunk_rays / kdp_max)
                ★ kdp_max: K_DP 上限(°/km), 超过的截断到该值; 不传=读 ZCH_KDP_MAX/KDP_MAX, 传 0=关闭
    返回 (kdp_dual, phidp_heavy, info) —— 只要一个 K_DP 时用这个。
    """
    phi_corr = np.asarray(phi_corr, dtype=np.float64)
    if range_m is None:
        gs = float(_cfg('KDP_GATE_SPACING', 30.0))
        range_m = np.arange(phi_corr.shape[1], dtype=np.float64) * gs
    range_m = np.asarray(range_m, dtype=np.float64)
    if do_fixneg:
        phi_corr = fix_phi_negative_jump(phi_corr)
    kd, kl, kh, pl, ph, info = kdp_dual_filter(phi_corr, ref, range_m,
                                               verbose=verbose, **kw)
    return kd, ph, info


def kdp_dual_filter(phi_corr, ref, range_m, light_filter_km=None, heavy_filter_km=None,
                    light_ls_km=None, heavy_ls_km=None, z_switch_dbz=None,
                    min_valid_gates=None, min_valid_frac=None, nonneg=None,
                    chunk_rays=None, kdp_max=None, verbose=True):
    """
    双尺度滤波 + 最小二乘 K_DP(**数组层, 输入应已做过负段修正**)。
    想要五个产物(两个 K_DP + 两个平滑 φdp)或想自己接后处理时用这个。

    默认参数全部取模块顶部的常量, 但**优先读 config.py 的 ZCH_* 覆盖值**
    (`_cfg`)。返回值见 `_kdp_core`。
    ★ kdp_max: K_DP 上限(°/km), 超过的**截断**到该值(不是置 NaN)。
               不传(None) = 用 config.py 的 ZCH_KDP_MAX(没定义则用模块常量 KDP_MAX)。
               ★ 想显式**关闭**上限, 传 0 或负数 或 False。
    """
    lf = _cfg('LIGHT_FILTER_KM', LIGHT_FILTER_KM) if light_filter_km is None else light_filter_km
    hf = _cfg('HEAVY_FILTER_KM', HEAVY_FILTER_KM) if heavy_filter_km is None else heavy_filter_km
    ll = lf if light_ls_km is None else light_ls_km
    hl = hf if heavy_ls_km is None else heavy_ls_km
    zs = _cfg('Z_SWITCH_DBZ', Z_SWITCH_DBZ) if z_switch_dbz is None else z_switch_dbz
    mv = _cfg('MIN_VALID_GATES', MIN_VALID_GATES) if min_valid_gates is None else min_valid_gates
    mf = _cfg('MIN_VALID_FRAC', MIN_VALID_FRAC) if min_valid_frac is None else min_valid_frac
    ng = _cfg('NONNEG', NONNEG) if nonneg is None else nonneg
    cr = _cfg('CHUNK_RAYS', CHUNK_RAYS) if chunk_rays is None else chunk_rays
    kx = _cfg('KDP_MAX', KDP_MAX) if kdp_max is None else kdp_max
    # 显式关闭: 0 / 负数 / False / NaN -> 不限
    if kx is False or kx == 0:
        kx = None
    elif kx is not None and (not np.isfinite(kx) or kx < 0):
        kx = None

    out = _kdp_core(phi_corr, ref, range_m, lf, hf, ll, hl, zs, mv, mf, ng, cr, kx)
    kd, kl, kh, pl, ph, info = out
    if verbose:
        print('  [ZCH-φ/KDP双尺度] 库长 %.0f m | 滤波窗 轻%.1fkm(%d库) 重%.1fkm(%d库) '
              '| 拟合窗 轻%d库 重%d库'
              % (info['gate_spacing_m'], lf, info['light_gates'], hf, info['heavy_gates'],
                 info['light_ls_gates'], info['heavy_ls_gates']))
        print('  [ZCH-φ/KDP双尺度] Z 阈值 %s dBZ -> 走轻度占 %.1f%%| 有值: 轻 %d / 重 %d '
              '/ 最终 %d (负值 %d) | K_DP 上限 %s'
              % ('无' if info['z_switch_dbz'] is None else '%.0f' % info['z_switch_dbz'],
                 100.0 * info['frac_light'], info['n_light'], info['n_heavy'],
                 info['n_dual'], info['n_neg_dual'],
                 '不限' if kx is None else '%.1f °/km' % kx))
        if info.get('n_clipped', 0):
            print('  [ZCH-φ/KDP双尺度] ★ 上限截断命中 %d 个格点(极端假值已拉回 %.1f °/km), '
                  '对应 φdp 清空 %d 个格点'
                  % (info['n_clipped'], kx, info.get('n_phi_cleared', 0)))
    return kd, kl, kh, pl, ph, info


# ============================ 对象层入口(写 radar) ============================

def compute_kdp_dual(radar, phi_corr=None, ref=None, phi_field=None, ref_field=None,
                     range_field=None, do_fixneg=True, write_fields=True,
                     verbose=True, **kw):
    """
    ★ 主入口: 对一个 pyart Radar 算双尺度 K_DP, 并写回字段。

    phi_corr: (nrays, ngates) **已扣系统相位**的 φdp(QC/step03 的输出)。
              None 时自己从 `phi_field`(默认 differential_phase)取, 并把 QC 掩膜
              以外的置 nan —— ⚠ 那条链上系统相位已不再估计扣除(见 pipeline.py L78),
              所以自己取字段这条路**没有扣系统相位**, KDP 的常数偏移不要紧,
              但送ZPHI 用之前请确认相位基线。
    ref:      (nrays, ngates) 反射率 dBZ, 只用于 40 dBZ 阈值切换。
              None 时从 `ref_field`(默认 'reflectivity')取。
    do_fixneg: True = 先调 fix_phi_negative_jump 修"负段 |负|>180 且后接正值"(默认 True)
    写出字段: kdp_dual / kdp_light / kdp_heavy / phidp_light / phidp_heavy

    返回 (kdp_dual, info); info['kdp_dual'] / info['kdp_light'] / info['kdp_heavy'] /
    info['phidp_light'] / info['phidp_heavy'] 是对应的 masked array(便于直接画图)。
    """
    phi_field = phi_field or _cfg('PHI_FIELD', 'differential_phase')
    ref_field = ref_field or _cfg('REF_FIELD', 'reflectivity')
    range_field = range_field or _cfg('RANGE_FIELD', RANGE_FIELD)

    if phi_corr is None:
        if phi_field not in radar.fields:
            print('  [警告] radar 里没有 %s 字段 -> 跳过 ZCH双尺度 KDP' % (phi_field,))
            return None, {}
        phi_corr = np.ma.filled(radar.fields[phi_field]['data'], np.nan).astype(float)
        # QC 掩膜(有 gatefilter 的话)以外的置 nan
        if 'reflectivity' in radar.fields:
            m = np.ma.getmaskarray(radar.fields['reflectivity']['data'])
            if m.shape == phi_corr.shape:
                phi_corr = np.where(~m, phi_corr, np.nan)
    if ref is None:
        if ref_field in radar.fields:
            ref = np.ma.filled(radar.fields[ref_field]['data'], np.nan).astype(float)
        else:
            ref = None
            if verbose:
                print('  [提示] radar 里没有 %s 字段 -> 阈值切换只用重度估计'
                      % (ref_field,))
    if range_field in getattr(radar, 'fields', {}):
        range_m = radar.fields[range_field]['data'].astype(float)
    else:
        range_m = radar.range['data'].astype(float)

    if do_fixneg:
        phi_corr = fix_phi_negative_jump(phi_corr)

    kd, kl, kh, pl, ph, info = kdp_dual_filter(
        np.asarray(phi_corr, dtype=float), ref, range_m, verbose=verbose, **kw)

    def _ma(a):
        return np.ma.masked_invalid(a)

    if write_fields:
        _lf = _cfg('LIGHT_FILTER_KM', LIGHT_FILTER_KM)
        _hf = _cfg('HEAVY_FILTER_KM', HEAVY_FILTER_KM)
        _zs = _cfg('Z_SWITCH_DBZ', Z_SWITCH_DBZ)
        radar.add_field('kdp_dual', {
            'data': _ma(kd), 'units': 'degrees/km',
            'standard_name': 'specific_differential_phase',
            'long_name': ('KDP dual-scale least-squares (Park et al. 2009): '
                          'light %.1f km / heavy %.1f km, switched at %.0f dBZ'
                          % (_lf, _hf, _zs)),
            'valid_min': -20.0, 'valid_max': 30.0,
        }, replace_existing=True)
        radar.add_field('kdp_light', {
            'data': _ma(kl), 'units': 'degrees/km',
            'standard_name': 'specific_differential_phase',
            'long_name': 'KDP lightly filtered least-squares (%.1f km)' % _lf,
            'valid_min': -20.0, 'valid_max': 30.0,
        }, replace_existing=True)
        radar.add_field('kdp_heavy', {
            'data': _ma(kh), 'units': 'degrees/km',
            'standard_name': 'specific_differential_phase',
            'long_name': 'KDP heavily filtered least-squares (%.1f km)' % _hf,
            'valid_min': -20.0, 'valid_max': 30.0,
        }, replace_existing=True)
        radar.add_field('phidp_light', {
            'data': _ma(pl), 'units': 'degrees',
            'standard_name': 'differential_propagation_phase',
            'long_name': 'PhiDP lightly smoothed (%.1f km)' % _lf,
            'valid_min': -360.0, 'valid_max': 3600.0,
        }, replace_existing=True)
        radar.add_field('phidp_heavy', {
            'data': _ma(ph), 'units': 'degrees',
            'standard_name': 'differential_propagation_phase',
            'long_name': ('PhiDP heavily smoothed (%.1f km) -- use this one for ZPHI'
                          % _hf),
            'valid_min': -360.0, 'valid_max': 3600.0,
        }, replace_existing=True)
        if verbose:
            print('  [ZCH-φ/KDP双尺度] 已写字段: kdp_dual / kdp_light / kdp_heavy / '
                  'phidp_light / phidp_heavy')

    info['kdp_dual'] = _ma(kd)
    info['kdp_light'] = _ma(kl)
    info['kdp_heavy'] = _ma(kh)
    info['phidp_light'] = _ma(pl)
    info['phidp_heavy'] = _ma(ph)
    return kd, info


def kdp_threshold_jump(radar, kdp_field='kdp_dual', ref_field='reflectivity',
                       z_switch_dbz=None, range_m=None, verbose=True):
    """
    诊断: 量一下 40 dBZ 阈值切换处K_DP 的**跳变幅度**(同射线相邻门之差的最大值)。

    这是双尺度方案最需要注意的副作用 —— 阈值上下相邻门可能差好几个 °/km,
    在时间序列/剖面图上表现为**毛刺**。函数返回
      (max_jump, n_jump, p95_jump) 以及跳变最大的那条射线的门号 idx。
    跳变大 ⇒把 Z_SWITCH_DBZ 提高(阈值取更高的强度, 让轻度只在强回波核心用),
    或者干脆只用kdp_heavy。★ 别靠调滤波窗长来治跳变, 那是两个独立的问题。
    """
    if kdp_field not in radar.fields:
        print('  [警告] radar 里没有 %s 字段' % (kdp_field,))
        return np.nan, 0, np.nan, 0
    zs = _cfg('Z_SWITCH_DBZ', Z_SWITCH_DBZ) if z_switch_dbz is None else float(z_switch_dbz)
    kd = np.ma.filled(radar.fields[kdp_field]['data'], np.nan).astype(float)
    ref = (np.ma.filled(radar.fields[ref_field]['data'], np.nan).astype(float)
           if ref_field in radar.fields else None)
    if ref is None:
        print('  [提示] 没有 %s -> 无法定位阈值线, 改量整条射线的最大相邻差' % (ref_field,))
        cross = np.ones_like(kd, dtype=bool)
    else:
        # 跨过阈值线的那一对门(一个 < zs、一个 >= zs)
        hi = ref >= zs
        cross = np.zeros_like(kd, dtype=bool)
        cross[:, :-1] = (hi[:, :-1] != hi[:, 1:])
    if not cross.any():
        return 0.0, 0, 0.0, -1
    d = np.abs(np.diff(kd, axis=1))
    jj, ii = np.where(cross[:, :-1])
    if jj.size == 0:
        return 0.0, 0, 0.0, -1
    vals = d[jj, ii]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 0, 0.0, -1
    k = int(np.argmax(vals))
    if verbose:
        print('  [ZCH-φ阈值跳变] %s 在 Z=%.0f dBZ 处相邻门差: max %.3f / p95 %.3f / '
              'n=%d | 最坏 射线#%d 门#%d'
              % (kdp_field, zs, float(vals.max()), float(np.percentile(vals, 95)),
                 vals.size, jj[k], ii[k]))
    return float(vals.max()), int(vals.size), float(np.percentile(vals, 95)), jj[k]


# ================================== 自检 ==================================

def _self_test(verbose=True):
    """
    合成数据自检: 造已知 K_DP 的 φdp, 看能不能还原。
    覆盖: (a) 窗长换算对不对  (b) 轻/重两尺度差异  (c) 负段加 360 规则
          (d) NaN 空洞能不能跳过  (e) 40 dBZ 阈值切换有没有生效
    """
    gs = 30.0
    ng = 2000
    nr = 40
    rng = np.arange(ng, dtype=float) * gs
    x_km = rng / 1000.0

    # (a)(b) 纯线性: KDP = 1.2 °/km, 起点相位 25°
    truth = 1.2
    phi = 25.0 + truth * x_km[None, :]
    phi = np.repeat(phi, nr, axis=0)
    ref = np.full((nr, ng), 45.0)          # 全部 >= 40 dBZ -> 应该全走轻度

    kd, kl, kh, pl, ph, info = kdp_dual_filter(phi, ref, rng, verbose=verbose)
    a1 = float(np.nanmedian(kl[np.isfinite(kl)]))
    a2 = float(np.nanmedian(kh[np.isfinite(kh)]))

    def _gates(_info):
        """从 info 里取(滤波半窗, 拟合半窗)的轻度版本 —— 给自检 c 直接调核心用。"""
        return (_info['light_gates'] // 2, _info['light_ls_gates'] // 2)

    if verbose:
        print('  [自检a] 窗长: 轻 %d 库(%.1f km) 重 %d 库(%.1f km) 库长 %.0f m'
              % (info['light_gates'], info['light_gates'] * gs / 1000.0,
                 info['heavy_gates'], info['heavy_gates'] * gs / 1000.0,
                 info['gate_spacing_m']))
        print('  [自检b] 线性 φdp(KDP=%.2f): 轻度 %.4f (偏 %.2e) / 重度 %.4f (偏 %.2e)'
              % (truth, a1, a1 - truth, a2, a2 - truth))
    ok_b = abs(a1 - truth) < 1e-6 and abs(a2 - truth) < 1e-6

    # (c) 负段加 360 规则: 前段负(绝对值 > 180) + 后段正-> 负段应被拉回正区间
    #    构造: 真实 KDP=2.5 的直线φ = 5 + 2.5x (全程 5°~155°, 正区间),
    #    把**前 8 km 整体减 360** -> 该段变成 -355°~-335°(负且 |负|>180),
    #    而第 8 km 之后仍是正的 ⇒ 正好命中"前负后正 + 负的绝对值>180"。
    #    ★ 转折点(第 n_neg 门)的负值必须|.| > 180 规则才会触发 —— 构造时先算一遍:
    #      起点相位选5° 就是要保证前 8 km 的负值全在 -360~-335 之间。
    #★ 这里**不再造 0/360 卷绕** —— 用户明确说这批数据没折叠, 卷绕用例已删除。
    #    ⇒ 顺带把一条经验固化: 卷绕会让折点附近出假**低** K_DP(窗内 -360 阶跃),
    #      断言要用 max|dev| 而不是 max(dev)。这条现在只作注释留档。
    kdp_true = 2.5
    phi2 = np.repeat((5.0 + kdp_true * x_km[None, :]), nr, axis=0)
    n_neg = int(8 * 1000.0 / gs)             # 前 8 km 打成负段
    broken = phi2.copy()
    broken[:, :n_neg] -= 360.0# -> 前段变成-355~-335
    fixed = fix_phi_negative_jump(broken)
    seg_before = broken[0, :n_neg]
    seg_after = fixed[0, :n_neg]
    n_added = int(np.sum(seg_after - seg_before))
    all_pos = bool(np.all(seg_after > 0.0)) and bool(np.all(fixed[0, n_neg:] > 0.0))
    # 整条射线的**常数偏移**变了, 但**斜率**必须还原(最小二乘对平移不敏感)
    raw_kd = _smooth_and_fit(broken, x_km,
                             *_gates(info), MIN_VALID_GATES, MIN_VALID_FRAC)[0]
    fix_kd = _smooth_and_fit(fixed, x_km,
                             *_gates(info), MIN_VALID_GATES, MIN_VALID_FRAC)[0]

    def _dev(a):
        v = a[np.isfinite(a)]
        return float(np.abs(v - kdp_true).max()) if v.size else np.nan

    dev_raw, dev_fix = _dev(raw_kd), _dev(fix_kd)
    a3 = float(np.nanmedian(fix_kd[np.isfinite(fix_kd)]))
    if verbose:
        print('  [自检c] 负段加 360: 前 %d 库(%d km)原值 %.1f~%.1f° -> 修正后 %.1f~%.1f°'
              ' (加 360 共 %d 次, 全正=%s)'
              % (n_neg, n_neg * gs / 1000.0, seg_before.min(), seg_before.max(),
                 seg_after.min(), seg_after.max(), n_added, all_pos))
        print('         max|偏差|(真 KDP=%.2f): 修正前 %.4f -> 修正后 %.2e| 整体中位偏差 %+.2e'
              % (kdp_true, dev_raw, dev_fix, a3 - kdp_true))
    ok_c = (n_added > 0 and all_pos and dev_fix < 1e-6 and abs(a3 - kdp_true) < 1e-6
            and dev_raw > 1.0)

    # (d) 随机挖 20% 空洞 + 加噪声
    rngn = np.random.default_rng(42)
    phi3 = np.repeat((60.0 + 1.5 * x_km[None, :]), nr, axis=0)
    phi3 = phi3 + rngn.normal(0, 0.3, phi3.shape)
    hole = rngn.random(phi3.shape) < 0.20
    phi3 = np.where(hole, np.nan, phi3)
    kd3, kl3, kh3, _, _, i3 = kdp_dual_filter(phi3, ref, rng, verbose=False)
    a4 = float(np.nanmedian(kh3[np.isfinite(kh3)]))
    if verbose:
        print('  [自检d] 20%% 空洞 + σ=0.3° 噪声(真 KDP=1.5): 重度 %.4f (偏 %+.4f), 有值率 %.1f%%'
              % (a4, a4 - 1.5, 100.0 * i3['n_heavy'] / phi3.size))
    ok_d = abs(a4 - 1.5) < 0.05 and i3['n_heavy'] > 0.5 * phi3.size

    # (e) 阈值切换: 一半射线 45 dBZ / 一半 20 dBZ -> 轻度、重度各占一半
    ref4 = np.full((nr, ng), 20.0)
    ref4[:nr // 2] = 45.0
    kd4, kl4, kh4, _, _, i4 = kdp_dual_filter(phi, ref4, rng, verbose=False)
    if verbose:
        print('  [自检e] 阈值切换: 前半 45dBZ/ 后半 20dBZ -> 走轻度占比 %.1f%% (期望 50.0%%)'
              % (100.0 * i4['frac_light']))
    ok_e = abs(i4['frac_light'] - 0.5) < 0.01

    allok = all([ok_b, ok_c, ok_d, ok_e])
    if verbose:
        print('  [自检] %s' % ('全部通过' if allok else '有未通过项'))
    return allok


if __name__ == '__main__':
    import os
    import sys

    if __package__ in (None, ''):
        _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
    print('=== QC/zch_phi.py 自检 ===')
    _self_test()