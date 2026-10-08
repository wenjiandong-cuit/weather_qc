# -*- coding: utf-8 -*-
"""
QC/config.py — 雷达资料处理链的**全部可调参数**(集中在这里改)
=================================================================
处理链顺序(见 QC/pipeline.py):
    读取 → QC 门控(QC/step02_gatefilter.py) → 逐射线系统相位(QC/step03_system_phase.py)
    → KDP/φdp(QC/step04_kdp.py) → 温度场(QC/step05_temperature.py)
    → ZPHI 衰减订正(QC/step06_attenuation.py) → FHC 分类(QC/step07_classification.py)
    组合反射率: QC/step08_composite.py   出图: QC/step09_plotting.py

运行时想临时改某个阈值(不改文件):
    import QC.config as C
    C.BAD_TILT_FRAC = 1.1        # gatefilter 内部实时读取 C.*, 立即生效
"""

import numpy as np   # 仅用于 np.inf 等常量

# ================= 输入文件(仅 __main__ / 单例调试用) =================
# ★ 盘符自适应: 走项目根 paths.py(移动硬盘 F:/E:/… 都行), 不再写死盘符。
try:
    from paths import p as _p
except ImportError:                     # 没有包上下文时(直接 python config.py)自己补 sys.path
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from paths import p as _p

FILE_PATH = _p('hail_growth_cases', '2025_weining', 'weining_pa_radar_base', 'ZWN01',
               '20250509', 'Z_RADR_I_ZWN01_20250509140600_O_DOR-XPD-CAP-FMT.BIN.zip')
SOUNDING_PATH = _p('hail_growth_cases', '2025_weining', 'weining_pa_radar_base', 'ZWN01',
                   'sounding', '20250509.txt')

# ================= 处理参数(可调) =================
# --- QC/杂波 ---
QC_RHOHV_MIN = 0.75        # ρhv 下限(低于视为非气象/杂波)
QC_RHOHV_MAX = None        # ρhv(相关系数)上限 —— **2026-10-08 用户要求移除, 现为 None(不启用)**
                           #   置成数值即可重新启用(例如 1.15)。背景留档:
                           #   这批数据 54.7% 的门 >1.0、20.3% 恰好 =1.200(尖峰),
                           #   1.0~1.2 之间是连续尾巴; 开上限会削掉有回波门的 33%、
                           #   并把"连续 50 门"这种判据切碎(整卷连续命中 483 -> 335)。
QC_Z_MIN = 0.0            # 反射率下限(弱噪声剔除)

# ---- 链上系统相位口径(2026-10-08 用户定: 链上要用处理好的 my_phi) ----
# pipeline 会: 逐层去系统相位 -> 写 radar 字段 my_phi -> KDP 与 ZPHI 都用它。
#   ★ 逐层(而不是整卷一次)是因为"没估到回退全局中位数"会被别的层污染(实测回退 94° vs 低层真实 2~15°)。
PIPE_SYSPHASE = True          # False = 回到"不扣系统相位"(phi_corr = phi_raw, ZPHI 用原始 φdp)
PIPE_SYSPHASE_STAT = 'max'    # 段内合成方式(与 notebook 一致; 'mean'/'median'/'p5' 等亦可)
PIPE_SYSPHASE_N_GATES = 50    # 前多少个连续合格门(或 sparse 时的最少门数)
PIPE_SYSPHASE_RHO_MIN = 0.90  # 合格门 ρhv 下限
PIPE_SYSPHASE_REFL_MIN = 0.0  # 合格门反射率下限(0 = 只要 QC 通过即可)
# 链上是否复用 KDP 已经算好的 phidp_heavy 给 ZPHI(实测逐位相同, 省 ~40 s/帧 = 30%)。
#   仅当 KDP 与 ZPHI 用同一份 φdp(即 PIPE_SYSPHASE=True)时才安全; pipeline 会自动判断
#   (两条件同时满足才复用)。
ZPHI_USER_REUSE_PHIDP = True
QC_HAIL_KEEP_Z = 45.0      # 冰雹核: Z>=45 且 0.5<=ρhv<QC_RHOHV_MIN 保留(冰雹ρhv偏低)
QC_HAIL_RHOHV_MIN = 0.5
CLUTTER_ELEV_MAX = 90.0     # 静止地物判定只作用于低仰角
CLUTTER_RANGE_KM = 15.0
CLUTTER_VEL_MAX = 0.75     # m/s, 静止地物速度近 0
CLUTTER_SW_MAX = 0.75      # m/s, 静止地物谱宽小
CLUTTER_Z_MIN = 0.0
CLUTTER_Z_MAX = 50.0
REFL_POWER_DIFF_DB = 6.0      # 近距离内 reflectivity 明显高于 total_power(去基线后)视为近场/地物伪回波
REFL_POWER_DIFF_RANGE_KM = 8.0  # 该规则只在 8 km 内生效: 中远距 refl>power 多为衰减订正产物, 不能剔
BAD_TILT_GAP_DB = 20.0          # 整层损坏判定: reflectivity - total_power 的异常下限
BAD_TILT_FRAC = 0.20            # 整层损坏判定: 该层有效回波中 gap>BAD_TILT_GAP_DB 的比例阈值
BAD_TILT_MIN_RANGE_KM = 3.0     # 整层损坏判定只统计 3km 外的门
# --- 雷达附近"同色大值区"(数值几乎相同的连通大块)整层剔除 ---
PLATEAU_CHECK = True            # 总开关
PLATEAU_Z_MIN = 40.0            # 只看 Z>=此值的门
PLATEAU_RANGE_KM = 20.0         # 只看雷达附近此距离内
PLATEAU_BIN_DB = 0.5            # 数值分箱宽度(判"同一个数值/颜色")
PLATEAU_PEAK_FRAC = 0.45        # 众数箱占比阈值
PLATEAU_MIN_BLOB = 200          # 最大连通块门数阈值
PLATEAU_MIN_RAYS = 20           # 最大连通块跨射线数阈值
DESPECKLE_PHI_SIZE = 20    # φdp 椒盐斑点抑制窗口
DESPECKLE_Z_SIZE = 9       # 反射率椒盐斑点抑制窗口

# --- 第 8 条: pyart "moment + texture" 门控(沿射线纹理; 见 step02_gatefilter.py) ---
# 阈值全部用 pyart 自带默认值(wind_size=7 / min_rhv=0.6 / max_textphi=20 /
# max_textrhv=0.30 / max_textzdr=2.85 / max_textrefl=8.0), 这里不再另设参数;
# 要调就直接给函数传 pyart 的同名参数: moment_texture_gatefilter(radar, wind_size=11)

# --- 订正保护(数值安全阀) ---
# ⚠ ATTEN_PIA_MAX 已按用户决定**取消限高**(2026-09-12)。
#   原因: 实测该封顶只影响 0.06% 的门(937/1602705), 不封顶后 cor_z 最大仅 76.7 dBZ,
#   仍在物理合理范围内; 而物理已三重验证(Park2005 闭式解 PIA=α·Δφ 恒成立)。
#   用人均可调的线去裁剪真实强降水衰减, 会系统性**低估降水强度**。
#   若日后需要重新启用, 把下面这行改回数值(如 15.0)即可, 代码路径仍然支持。
ATTEN_PIA_MAX = np.inf     # 双程积分衰减上限(dB); np.inf = 不设限
ATTEN_COR_Z_MAX = 75.0     # 订正后反射率上限(dBZ) — 保留为最终产品绝对天花板(本个例未触发)

# --- ★ 只订正反射率, ZDR 不做衰减订正(2026-09-14 用户决定; 差分订正代码已整段删除) ---
# Park2005 式(9)(12) 的差分订正(A_DP / PIDA -> cor_zdr)已从
# step06_attenuation_park2005.py 删除, **不再有开关**, 也没有 ATTEN_CORRECT_ZDR /
# ATTEN_PIDA_MAX / ATTEN_COR_ZDR_MAX 这几个常量。
#   ⇒ cor_zdr = 原始 differential_reflectivity 直通(只套 QC 掩膜), 全域可用。
# 为什么删(实测发现): 式(9)(12) 的 PIDA 与式(1) 的 PIA 共用同一个雨区窗口(温度>0 且
#   refl≥阈值), 订正版 cor_zdr **只在雨区有值、融化层以上整片缺测**; 而 ZDR 柱恰恰长在
#   融化层以上 -> 指标(7) 判据① 永远 0 格点。实测威宁 20250509 单个体扫 6200 万门里
#   订正版 cor_zdr 只剩 331 门有效, 原始 ZDR 有 6.8 万门(遍布所有仰角)。
#   X 波段冰相区的差分衰减本身很小, 而反射率衰减的影响大得多 -> 订正只留给 Z_H。
#   ⇒ 连带好处: cor_zdr 字段名不变, 下游(指标/FHC/出图/别名)一行都不用改。
# 报警线(不裁剪数值, 仅统计供人工核查): PIA 超过此值的门数会写进 verbose 输出
ATTEN_PIA_WARN = 15.0      # dB, 仅作告警统计, 不影响订正结果
# ⚠ ATTEN_REL_PHIDP 已删除(2026-09-12): 它原来只控制 pyart 版是否做 φdp 零基线对齐,
#   而 pyart 版已按用户要求删除。现在的唯一实现(Park2005)只用 ΔΦ_DP,
#   零基线自动抵消, 不需要这个开关。
# ★ 2026-09-25: 链上**也不再调用** align_phidp_baseline 产出 phidp_rel 了 ——
#   那一步对结果无影响(空操作), 且没有任何消费者, 只白占约 300 MB 常驻内存。
#   函数定义仍留在 step06_attenuation_park2005.py 备用(换回 pyart 式实现时才需要)。

# --- 雨区判据(Park2005 版的积分窗口 r1..r0) ---
# 背景: 仅靠"温度>0°C"判断雨区会偏松 —— 温度场由探空外推, 低仰角算出的 0°C 层
# 高度常超出雷达量程, 于是整条射线都被当成雨区。
# ⚠ 但后续三重验证表明: 这**不会制造虚假 Δφ** —— 大 Δφ 射线全部位于真实降水柱内
#   (窗口跨度/真雨跨度 = 1.04, 尾部 0/108 条无回波, 原始 Zmax 中位 59 dBZ),
#   且原始 φdp / 扣系统相位 / Bringi 滤波 / KDP 积分四路读数一致。
#   故此处判据只用于"收紧窗口到真实降水段", 不是修 bug。
ATTEN_RAIN_REFL_MIN = 10.0    # 雨区门的最小反射率(dBZ), 低于视为无雨
ATTEN_RAIN_TAIL_GAP = 5       # 从窗口尾部回退: 允许连续无回波的门数(容错)
ATTEN_RAIN_MIN_GATES = 10     # 修整后窗口至少要有这么多有效门, 否则整条射线跳过

# --- 系统相位/逐射线起始相位 ---
# ⚠ SYSPHASE_NWIN 已删除(2026-09-12): 它只服务**旧法**(前 N 门中位数, 那个文件已删除),
#   现在的唯一实现 QC/step03_system_phase.py 用的是 ρhv 判据
#   (前 30 个连续 ρhv>0.97 门的 φdp 平均, 参数写在该文件顶部 RHO_MIN/N_GATES/STAT)。
#   注意: 现在的 step03_system_phase.py 是**重命名后的 ρhv 判据实现**(原 相位估计.py),
#   与已删除的那个同名旧法文件无关。
#   (前 30 个连续 ρhv>0.97 门的 φdp 平均, 参数写在该文件顶部 RHO_MIN/N_GATES/STAT)。
SYSPHASE_FALLBACK_PCT = 10.0  # 无有效门射线 / 全局兜底所用的分位数
# --- 系统相位 ρhv 判据(step03)可调参数(2026-09-27 用户定) ---
# 注意 step03 文件顶部的 RHO_MIN/N_GATES 是"未传参时的默认"；这里由 pipeline 显式覆盖：
SYSPHASE_RHO_MIN = 0.95       # ρhv 阈值(用户定 0.95)
SYSPHASE_REFL_MIN = 15.0      # 反射率下限(dBZ)：只取有真实回波的门做系统相位(滤晴空噪声)

# --- KDP(Bringi)参数 ---
# ⚠⚠ 2026-10-07 起 **已被弃用**: 第 4 步(step04_kdp)已从 CSU Bringi 换成 zch_phi
#    双尺度方案, 下面这几个参数**不再有任何消费方**, 保留仅为老脚本兼容/留档。
#    现在 KDP 的参数在 QC/zch_phi.py 顶部(轻 0.6 km / 重 1.5 km / Z_SWITCH_DBZ=40 等),
#    可用 config 里的 ZCH_* 覆盖(如 ZCH_Z_SWITCH_DBZ)。
KDP_THSD = 10.0            # [弃用] Bringi KDP 计算的 φdp 标准差阈值
KDP_NFILTER = 1            # [弃用]
KDP_GATE_SPACING = 30.0    # [弃用] 库长已由 radar.range['data'] 现取
KDP_STD_GATE = 55          # [弃用]
KDP_FIR_WINDOW = 3.0       # [弃用] FIR 滤波窗口

# --- zch_phi 双尺度 KDP(step04 现用实现; 不写用 zch_phi.py 顶部默认值) ---
# 轻度 0.6 km / 重度 1.5 km, 对**平滑后**的 φdp 做最小二乘拟合, 按 Z>=40 dBZ 切换。
# 要改就取消注释并调整(名字是 ZCH_ + zch_phi.py 里的常量名):
# ZCH_LIGHT_FILTER_KM = 0.6
# ZCH_HEAVY_FILTER_KM = 1.5
# ZCH_Z_SWITCH_DBZ = 40.0
# ZCH_NONNEG = False       # 是否把负 KDP 截到 0
# ZCH_KDP_MAX = 45.0       # ★ KDP 上限(°/km): 超过的截断到该值(不置 NaN)。0 = 关闭上限。
#                          #   30 m 库长下 45 °/km = 单库 φdp 增量 ≤ 1.35°。
#                          #   压掉 φdp 台阶造成的假极值(cell1 ray2030 实测曾到 309)。

# ★★ 衰减订正实现(2026-10-06 起唯一 = 用户口径; Park2005/ERAD 已删除)
#   'user' = 用户口径(扣系统相位 + zch_phi 双尺度滤波 + 末端 ρhv>0.9 取 9 门平均
#            当总相位差, 再复刻 pyart 公式算 PIA), 见 step06_attenuation.py
#            ★ 系数用 pyart 自动查表 X 波段 a_coef=0.31916, 不手动改。
# 旧的 Park2005 论文版 / ERAD 版已于 2026-10-06 删除(归档 _trash_20261006/)。
ATTEN_BACKEND = 'user'

# --- ZPHI 衰减订正参数 ---
# doc: 每条射线末端丢弃的门数(Bringi2001/Park2005 用于抑制远端弱信号或折叠)。
ZPHI_DOC = 100.0           # doc: 100门 = 3.0km(库长 30 m) —— 2026-09-22 由 125门(3.75km)收紧到 3.0km
# ⚠ 末端实际总退门数 = ATTEN_RAIN_TAIL_GAP(5门) + ZPHI_DOC, 即 105 门 = 3.15km。
#   若要"净退 3.00km", 把本值改成 95.0。
ZPHI_SMOOTH_WINDOW = 42    # φdp 平滑窗口 —— 原为 pyart 版参数, 现已无人使用(保留仅供诊断脚本)
ATTEN_BAD_FILL = -32768.0  # 无效门填充值(供 CSU/外部算法使用)

# --- 用户口径 ZPHI(step06_attenuation.py, 2026-10-06 新增) ---
# 总相位差 = 末端最后 ZPHI_USER_N_TAIL 个 ρhv>ZPHI_USER_RHO_TAIL 的门的滤波后 φdp 平均。
# 系统相位估计复用 step03(ρhv 高值段), 但 cell1/威宁 回波稀疏, 必须放宽参数:
#   ZPHI_USER_SP_N_GATES=5 / ZPHI_USER_SP_RHO_MIN=0.90 / ZPHI_USER_SP_REFL_MIN=10
#   (step03 默认 50门/0.97 在 cell1 命中 0 条; 且 φdp 有哨兵值 30.0, 需 refl>=10 剔掉)。
ZPHI_USER_N_TAIL = 9           # 末端取几个高质量门
ZPHI_USER_RHO_TAIL = 0.90      # 高质量门相关系数阈值
ZPHI_USER_SP_N_GATES = 5       # 系统相位: 连续几个 ρhv 高值门
ZPHI_USER_SP_RHO_MIN = 0.90    # 系统相位: ρhv 阈值
ZPHI_USER_SP_REFL_MIN = 10.0   # 系统相位: 反射率下限(只取真实降水门)
# ★★ 系统相位扣除口径(2026-10-08 新增)：
#   'global'  = 逐射线估完后取全体命中射线的中位数, 全卷扣同一个常数(2026-09-27 起的口径);
#   'per_ray' = 每条射线扣各自的估计值, 没估到的射线回退全局中位数。
#   ⚠ 实测(威宁 20250520110900): 逐射线的"差异"主要不是硬件相位差, 而是
#     ① 填充值 0.0(占有效 φdp 门 47.7%) / 30.0(16.1%) 被当成相位;
#     ② "前 5 个连续 ρhv>0.9 且 Z≥10 门"在不同射线上落到不同距离 —— 13.75° 那层
#        落到对流核里, 逐射线基线中位 129.5°(27 条全在 127~131°), 而低层弱回波
#        (Z 10~15, ρhv>0.95)独立验证的硬件基线是 **11.9°**。
#     ⇒ 想用 'per_ray' 必须同时开 ZPHI_USER_SP_DROP_SENTINEL, 并先看打印出来的散布。
ZPHI_USER_SP_MODE = 'global'
# True = 估计系统相位前, 把恰好等于 0.0 / 30.0 的门也判为无效(它们在这批数据里是填充值)。
#   注意: 开了之后全局中位数会明显变化(实测整卷 11.88° -> 92.69°), 因为剩下的多是
#   已经累积了传播相位的对流核门 —— 所以这个开关是**排查用**的, 别默认开。
ZPHI_USER_SP_DROP_SENTINEL = False

# --- 论文版 ZPHI 自洽约束法(Park et al. 2005, JTECH 22:1621-1632, X 波段) ---
# 论文 Table 1(X 波段 9.375 GHz, 0/15/30°C × 3 种雨滴形状的统计):
#   A_H  = a·Z_H^b      a=1.367e-4, b=0.780 (均值)     ← 式(5); 注意 a 不参与 Z_H 订正
#   A_H  = α·K_DP^c     α=0.254(0.139~0.335), c=1.143 ← 式(6); α 参与式(3)
#   A_DP = γ·A_H^d      γ=0.139(0.114~0.174), d=1.134 ← 式(9); ★本模块不实现(差分订正已删)
# 实现见 step06_attenuation_park2005.zphi_attenuation_correction_park2005
PARK_B = 0.779             # A-Z 关系指数 b(论文 15°C 值; Table1 均值 0.780)
# (原 PARK_BETA / PARK_D —— 式(9) 的 γ 与 d —— 已随 ZDR 差分订正一起删除, 不再需要)
# A_H-K_DP 系数 α[dB/度]: 式(3) 的 scn 必须写成 10^(0.1·b·α·ΔΦdp)-1。
# α 在 X 波段随雨滴形状/温度变化很大(0.139~0.335, 标准差 28%), 论文明确指出
# "固定 α 会给 A_H 廓线带来误差", 因此自洽法的核心是**逐射线搜最优 α**(式7/8)。
PARK_A = 0.254             # α 的论文 Table 1 均值(固定 α 模式使用)
# α 自洽搜索区间(论文 Table 1 的物理范围)
PARK_A_MIN = 0.139         # 论文 Table 1 最小值
PARK_A_MAX = 0.335         # 论文 Table 1 最大值
PARK_A_NGRID = 61          # 自洽搜索的 α 网格点数(0.139~0.335 步长约 0.0033)
# α 模式开关:
#   True  = 逐射线自洽搜索(最贴论文式7/8)
#   False = 固定用 PARK_A(论文 Table 1 均值 0.254), 即 Testud(2000) 定系数做法
# 实测(威宁 20250509 整卷)在默认搜索下端点占比 98.4% —— φdp 廓线平坦时 α 不可辨识,
# 搜索退化为"固定端点值"(且上下界混用会造成 2.4 倍跳变)。因此**默认关闭搜索**,
# 改用论文统计均值; 想复现搜索行为就把下面改成 True。
PARK_ADAPTIVE_ALPHA = False

# --- VIL 垂直累积液态水含量(step10_metrics.metric_vil, 2026-09-13 新增) ---
# 公式(Greene & Clark 1972): M = 3.44e-6 * Z^(4/7) [g/m³], VIL = Σ M̄·Δh / 1000 [kg/m²]
# 做法: 先把体扫网格化到笛卡尔坐标(pyart.map.grid_from_radars), 再逐列梯形积分。
# ★ 区域(2026-09-15 用户明确: "区域是和 VIL 相同的区域"): VIL 与指标(7) ZDR 柱**共用
#   同一个 box_mask**(all_metrics_ginput 里 ginput 点两次: 左上角 + 右下角 -> 定出一个矩形
#   -> build_box_mask_3d 出一个掩膜 -> 两条链各自网格化但都拿它), 且**统计只用框内格点**:
#     · VIL:  metric_vil 在 compute_vil_field 第 5.5 步映射出的 box2d 里取 max/mean;
#     · ZDR 柱: 判据① 同样只取 box2d 内的格点。
#   网格化的水平包络比框大一圈(每边 +VIL_GRID_MARGIN_M, 默认 2 km), 那圈只为**边缘列的
#   插值**服务, 不进任何指标 —— 否则 20km×20km 的框会被撑成 24km×24km(面积 +44%), 外圈
#   的弱回波会把 mean VIL 系统性拉低, 也会让两个指标不在"同一区域"里。
# ⚠ 因此下面 VIL_GRID_* 与 ZDRCOL_GRID_* 必须保持一致(现在都是 250 m / 150 m / 15 km /
#   margin 2 km), 改一个就顺手把另一个也改了 —— 否则两条链的网格不再重合。
VIL_Z_CAP_DBZ = 55.0      # 雹電容: 超过该 dBZ 按该值算(防冰雹区 Z^(4/7) 非线性放大); None=不封顶
VIL_Z_MIN_DBZ = 0.0       # 弱回波下限: 低于该 dBZ 的层按无回波(M=0)处理
VIL_GRID_DZ_M = 250.0     # 网格垂直间距 (m) —— 雷达库长 30 m, 250 m 约为中距离垂直采样尺度
VIL_GRID_DXY_M = 150.0    # 网格水平间距 (m) —— 雷达库长 30 m, 取 5 倍; 更细则网格化明显变慢
VIL_Z_TOP_M = 15000.0     # 积分顶高 (m)
VIL_GRID_MARGIN_M = 2000.0  # 网格范围在框选范围外扩的余量 (m, 防边缘列缺数据)

# --- ZDR 柱(step10_metrics.metric_zdr_column, 2026-09-14 新增) ---
# ★★ 论文原文口径(逐条核对过, 见 Zhao et al. 2025 Sect. 2.4 + 3.4 + Fig. 3/4) ★★
#   · 网格: **0.25 km 水平 / 500 m 垂直, 从 500 m 到 20 km 共 40 层**(Fig. 3 Step 1);
#     Py-ART Barnes2 网格化; 单格体积 0.03125 km³。
#   · 判据①(阈值): 从"融化层下 1 km"到**混合相区顶**, ZDR >= 1.5 dB 置 1;
#     (Fig.3 Step 2 原话: "from low level to the mixed-phase region"。本链 2026-09-14
#      起也设了这个上界, 默认取 **-20 °C 等温线高度** —— 见 ZDRCOL_MIXED_PHASE_TOP_C;
#      论文未给该高度的数值, -20 °C 是 mixed-phase region 最常见的上界约定。)
#     ★ 0 °C 层与这条 -20 °C 线都**优先查探空原始廓线**(指标函数传 snd=read_sounding(...)),
#       **不要**绕道 radar 的插值 temperature 场 —— 那个场的上限被雷达库点封住, 浅体扫够不着高层。
#   · 判据②(柱状): 用 **0 °C 层以下**的逻辑矩阵"向上约束", 去掉不在向上连续柱内的格点。
#     Fig.3 Step 2/3 显示这是**同列垂直连续**(successive columnar, Step 3 的 y−x<=0
#     就是逐列相邻两层比较) —— **2026-10-06 已按论文改成逐层 `mask[k] &= mask[k-1]`** ✔
#     (此前是 6-邻域三维连通, 它含"同层左右相邻", 柱子水平错位时会多留格点 → 柱体偏胖;
#      实测 cell1 前 6 帧柱体格点 199→104、柱高最多矮 500 m)。
#   · 判据③(梯度): 自 **0 °C 层到网格顶(uppermost limit height = 20 km)** 逐相邻两层,
#     ZDR **向上非增**才保留 —— 论文 Step 3 原式 y−x <= 0(**等于 0 也保留**),
#     本链 **2026-09-14 起已改成 <=**(ZDRCOL_GRAD_TOL_DB=0 时与论文原式逐字等同) ✔。
#   · 柱体 = 上述格点里 **0 °C 层及以上**的部分(Fig. 4 的白点正好从 0 °C 层起,
#     融化层以下那截只是判据②的锚, 不算柱体) —— 与本实现一致;
#   · **柱"高度" = 从融化层往上数到柱内最高层的层数 n × dz**(Sect. 3.4 明写),
#     即 `depth_m` = `n_layers` × `ZDRCOL_GRID_DZ_M`。**2026-10-06 起 `depth_m` 就按这个
#     离散口径算**(层数 × 层高, 必然是 dz 的整数倍), 旧的"柱顶海拔 − 0 °C 层高度"那个
#     连续差已废。**`height_m`(柱顶海拔 AGL)已删除**, 只留 `depth_m` + `n_layers`。
#     (Fig. 2c: 17:24 时白点占 2 层 -> 高 1 km ✔ —— 注意这与"海拔"是两回事)
#   · 体积 = 柱内格点数 × 单格体积(论文按固定 0.03125 km³ 量化)。
# ⚠ 2026-10-06 统计表口径变更: `zdrcol_height_m` 这一列现在装的是**层数×层高**
#   (0.25~2 km 量级), 不再是柱顶海拔(5~8 km 量级) ⇒ **与 2026-10-06 之前跑出来的
#   cell1~cell5 统计表不可比**, 要么全量重跑要么另起列名。
# ⚠ 本链默认网格是 150 m / 250 m / 顶 15 km(与 VIL 那套一致), **与论文不同**:
#   高度量化(250 m vs 500 m)、体积最小单元、顶部截断都不同。要与论文数值直接对比,
#   就把下面 ZDRCOL_GRID_DZ_M 改 500.0、ZDRCOL_GRID_DXY_M 改 250.0、
#   ZDRCOL_Z_TOP_M 改 20000.0(论文层高 500 m, 从 500 m 起算)。
# 做法(Zhao et al. 2025, ACP, "3D mapping columns"): 网格化 ZDR 后逐格判定三条
#   ① ZDR >= ZDRCOL_ZDR_MIN_DB, 且高度落在 [融化层 - ZDRCOL_BELOW_FZL_M, 混合相区顶];
#   ② 柱状形态: 必须与"融化层以下 1 km ~ 融化层"这段下部高值区连通(向上约束);
#   ③ 融化层以上 ZDR 向上非增(相邻两层之差 <= 0, 等于 0 也算通过; 尺度分选指纹)。
# 三条都满足、**且位于融化层以上**的格点集合 = ZDR 柱 -> 算柱顶/柱底/体积。
#   (定义上柱子必须在 0°C 层以上: 融化层以下那截只是判据②的"锚", 不算柱体本身)
#
# ⚠ 用哪个 ZDR 场(2026-09-14 实测发现的关键点):
#   2026-09-14 起衰减订正**只订反射率**(见上面那节), cor_zdr 就是原始 ZDR 的直通
#   (只套 QC 掩膜), 全域可用 -> 判据① 在融化层以上也有格点, 指标(7) 可以直接跑。
#   历史坑(留档): 在这之前 cor_zdr 是 Park2005 的雨区产物(温度>0 + 有回波 + φdp 有效),
#   融化层以上缺测 —— 实测威宁 20250509 一个体扫 6200 万门里只有 331 门有效,
#   于是判据① 在融化层以上永远 0 格点、指标(7) 恒为"无柱"。
#   两个字段现在数值一致, 填哪个都行; 保留本开关是为了显式记录用的是哪一路。
ZDRCOL_ZDR_FIELD = 'cor_zdr'  # 用于判定的 ZDR 字段: 'cor_zdr'(= 原始 ZDR 直通, 全域可用)
                              # 或 'differential_reflectivity'(源字段, 数值相同)
ZDRCOL_ZDR_MIN_DB = 1.5      # ZDR 阈值 (dB); 论文取 1.5(对应柱内雨滴 >2mm), 老文献常用 1.0
ZDRCOL_BELOW_FZL_M = 1000.0  # 判定范围从融化层以下多少米起算 (论文: 1 km)
ZDRCOL_MIXED_PHASE_TOP_C = -20.0  # ★2026-09-14 新增: 判据① 的**上界** = 混合相区顶,
                              # 取该等温线的高度。★优先查**探空原始廓线**
                              # (isotherm_height(snd=snd)) —— 探空到 30 km 以上, 必然找得到;
                              # 没传 snd 才退回"雷达 temperature 场反推", 那个场的上限被
                              # 雷达库点封住, 浅体扫可能够不着(此时退回不设上界并告警)。
                              # -20 = mixed-phase region 最常见的上界约定(论文没给数值);
                              # 改 -40 就取更低; 填 None = 不设上界(旧行为, 判到网格顶)。
                              # 调用时也可用 z_upper_m=<米> 直接给高度, 或 z_upper_m=None 关掉。
ZDRCOL_GRID_DZ_M = 250.0     # 网格垂直间距 (m) —— 与 VIL 网格保持一致
ZDRCOL_GRID_DXY_M = 150.0    # 网格水平间距 (m) —— 与 VIL 网格保持一致
ZDRCOL_Z_TOP_M = 15000.0     # 网格顶高 (m)
ZDRCOL_GRID_MARGIN_M = 2000.0  # 网格范围在框选范围外扩的余量 (m)
ZDRCOL_USE_CONNECTIVITY = True    # 是否启用判据②(柱状 = **逐列垂直连续**, 论文 successive
                                  # columnar 口径: 从种子板层往上逐层 mask[k] &= mask[k-1],
                                  # 无横向自由度)。False = 只按阈值(易误判, 会把悬空高值
                                  # 当柱子)。★ 2026-10-06 由 6-邻域三维连通改成逐列, 见上。
ZDRCOL_USE_NEG_GRADIENT = True    # 是否启用判据③ (融化层以上 ZDR 向上非增, 论文 y−x<=0)
ZDRCOL_GRAD_TOL_DB = 0.0          # 判据③ 的容差 (dB): 要求相邻两层"下降幅度至少"这么多。
                                  # 0(默认) = 只要求不上升(<= 0), 与论文 y−x <= 0 逐字一致;
                                  # 调大 = 更严(必须比下层低 at least 这个量), 用来滤噪声。
ZDRCOL_MIN_VOXELS = 1             # 少于该格点数的连通块丢弃 (去小碎块, 1 = 不过滤;
                                  # ★ 论文没有这一步, 保持 1 = 不过滤, 才是论文口径)
ZDRCOL_REF_MIN_DBZ = None         # 反射率下限 (dBZ); None = 不设(论文做法, 保住初生期弱回波)
