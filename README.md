# QC —— 雷达资料处理流程(按部分拆分, 文件名带序号)

一条处理链, 一个部分一个文件。**文件名开头的 `stepNN_` 就是执行顺序** —— 打开目录一眼就能
找到"要改的那一步"。**参数集中**在 `QC/config.py`(**无序号**), **顺序编排**在
`QC/pipeline.py`(**无序号**), 各步之间只通过字段/数组传递, 可以单独调用、单独替换。

## 文件 ↔ 处理部分

| 顺序 | 文件 | 负责部分 | 主要函数 | 产出字段 |
|---|---|---|---|---|
| — | `QC/config.py` | 全部可调参数 | — | — |
| 1 | `QC/step01_sounding.py` | 探空读取 | `read_sounding()` | dict: `heights_m / temps_C / fzl` |
| 2 | `QC/step02_gatefilter.py` | QC 门控 | `build_qc_gatefilter()` / `build_qc_gatefilter_basic()` | `GateFilter` (`gate_included`) |
| 3 | `QC/step03_system_phase.py` | 系统相位 · **唯一实现**(ρhv 判据: 前 30 个连续 ρhv>0.97 门的 φdp **平均**) | `remove_system_phase_by_rho()` / `estimate_system_phase()` | `phi_corr` |
| 4 | `QC/step04_kdp.py` | φdp 平滑 + KDP(CSU Bringi) | `compute_kdp_dual()` / `mask_reflectivity()` | `kdp_dual`, `phidp_heavy` |
| 5 | `QC/step05_temperature.py` | 温度场(探空插值到库点) | `add_temperature_field()` | `temperature` |
| 6 | `QC/step06_attenuation.py` | 衰减订正**入口**(转发层, 无实现) | — | — |
| 6 | `QC/step06_attenuation_park2005.py` | 衰减订正 · **唯一实现**(链上只跑前一个函数) | `zphi_attenuation_correction_park2005()` / `align_phidp_baseline()`(2026-09-25 起链上不再调用) | `spec_at_park`, `pia_park`, `cor_z_park`, `cor_zdr_park`, `alpha_park` |
| 7 | `QC/step07_classification.py` | FHC 水凝物分类 | `add_fhc_field()` / `fhc_counts()` | `FH` (**1..10**: 9=冰雹, 7/8=低/高密度霰, 10=大滴) |
| 8 | `QC/step08_composite.py` | 组合反射率(CR) | `composite_radar()` / `composite_reflectivity()` / `composite_reflectivity_fast()` | `CR` / `CR_<源字段>` / `(comp_z, comp_alt)` |
| 9 | `QC/step09_plotting.py` | 出图(统一色标 + Science 风格) | `plot_cr()` / `plot_ppi()` / `plot_composite()` / **`plot_cr_compare()`** | PNG / PDF / SVG / 图 |
| 10 | `QC/step10_metrics.py` | **指标提取**(框选 → 掩膜 → 指标) | `select_box_ginput()` / `build_box_mask_3d()` / `metric_max_z()` / `metric_zmax_height()` / `metric_echo_top()` / **`metric_vil()`**(VIL) / **`metric_zdr_column()`**(ZDR 柱) / **`all_metrics_series()` + `metric_trends()`**(指标(8) 变化速率) / `all_metrics()` / `print_metrics()` | dict / 数值 |
| — | `QC/plotstyle.py` | **绘图风格与色标** | `ref_cmap_params()` / `science_style()` / `add_latlon_grid()` / `bold_colorbar()` | — |
| — | `QC/pipeline.py` | 编排 + **FHC 独立入口** | `prepare_radar_products()` / `process_radar()` / **`run_fhc()`** / **`process_radar_fhc()`** / `cr_from_file()` | 以上全部 |
| — | `QC/fast_texture.py` | 沿射线纹理**快速版**(step02 自动装上) | `install_fast_texture()` / `texture_along_ray_fast()` / `uninstall_fast_texture()` / `fast_texture_status()` | — |

`QC/__init__.py` 把函数和参数再导出, 所以可以 `from QC import composite_radar, plot_cr`。

> **2026-09-22 移植说明**: `QC/fast_texture.py` 从 `cell_program_v2` 移植过来 —— 第 8 条
> moment+texture 门控的沿射线纹理改走"跳过空窗"快速版, 实测 41.7 s → 3.2 s(约 13 倍),
> 与 pyart 原版**逐值、逐掩码一致**(已对拍 wind_size=5/7/9/11)。`import QC.step02_gatefilter`
> 时自动装上, 不用手动调用; 环境变量 `QC_FAST_TEXTURE=0` 可关回 pyart 原版。

> **2026-09-12 合并说明**: 原先 `core/renwu.py` 里的**指标提取层**与 `core/csu_function.py`
> 的**绘图风格**都已搬进本包(分别是 `step10_metrics.py` 和 `plotstyle.py`)。现在**只跟
> `QC/` 一个文件夹打交道**就够了, 不需要 `core/`。`core/renwu.py` 降级为纯转发层
> (老脚本 `from renwu import ...` 照旧可用)。

## FHC 已从 process_radar 剥离(2026-09-13 新增)

FHC(模糊逻辑粒子识别)现在是**独立函数**, `process_radar` 本身不再做分类:

```python
import QC
radar = QC.process_radar(FILE, SOUNDING)   # 不含 FHC, 不加载 csu_radartools
radar = QC.run_fhc(radar)                  # 需要时再单独补 FHC(就地添加 FH 字段)

# 或者一步到位(等价旧的默认行为):
radar = QC.process_radar_fhc(FILE, SOUNDING)
cr_radar, radar = QC.cr_from_file(FILE, SOUNDING)          # 默认含 FHC
cr_radar, radar = QC.cr_from_file(FILE, SOUNDING, no_fhc=True)   # 不要 FHC
```

- `process_radar` 默认**不做** FHC —— 更快、不依赖 csu_radartools。
- 代价: 此时 radar 里**没有 `FH` 字段**, 所以**指标(5)(冰雹/霰数量)用不了**
  (`metric_hail_graupel_count` 会打印警告并返回 0, 不会报错); 调一次
  `QC.run_fhc(radar)` 即可补上。
- `process_radar` 的旧参数 `no_fhc` 仍被接受(向后兼容), 但已不起作用;
  显式传 `no_fhc=False` 只会打印一条提示。

## 字段名说明: 为什么产出里同时有 `cor_z` 和 `cor_z_park`

衰减订正**全项目只有一个实现**: 论文版 Park2005 (`step06_attenuation_park2005.py`),
它自己的字段带 `_park` 后缀。为了让指标函数、出图、老脚本直接可用,
`prepare_radar_products` 末尾会把 `_park` 那套**别名**成无后缀的名字:

| Park2005 产物 | 别名 | 说明 |
|---|---|---|
| `spec_at_park` | `spec_at` | 同一份 ndarray, 不额外占内存 |
| `pia_park` | `pia` | 同上 |
| `cor_z_park` | `cor_z` | 同上(指标函数默认就用 `cor_z`) |
| `cor_zdr_park` | `cor_zdr` | 同上 |

> **2026-09-12: pyart 官方版已删除。** 原先并存的
> `step06_attenuation_pyart.py`(调 pyart `calculate_attenuation_zphi`)已按用户要求移除,
> 备份在 `.workbuddy/backup/2026-09-12-pyart-atten/`。同时删掉了只服务 pyart 版的
> 开关 `C.ATTEN_REL_PHIDP`(`ZPHI_SMOOTH_WINDOW` 保留但已无代码使用)。
> 老名字 `zphi_attenuation_correction` 仍可 import, 现在指向本项目的唯一实现。


导出是**惰性**的(PEP 562 模块级 `__getattr__`): `import QC` 本身不加载 numpy / pyart /
cartopy, 只有真正取到某个函数时才 import 它所在的子模块。所以只想要参数时写
`import QC.config as C` 或 `from QC.config import KDP_THSD`, 代价最低; 依赖缺失也不会
让整个包不可用, 只影响实际用到的那一层。子模块名(`QC.config` / `QC.step08_composite` 等)
照旧可以当属性取。

## 为什么是这个顺序

1. **QC 在前**: ρhv/弱回波/地物/坏层没剔掉, 后面的相位与 KDP 统计会被污染。
2. **系统相位必须在平滑前扣**: X 波段相控阵系统相位随仰角/温度漂移(每层 9~35°)。
   先在原始 φdp 上逐射线扣除起点, 再做 Bringi FIR 平滑; 顺序颠倒会把残留起点
   平滑进整条廓线, ZPHI 会把整层订正过头(实测 `cor_z` 能到 90 dBZ+)。
3. **温度场在衰减订正前**: ZPHI 需要 `fzl`(融化层)和 `temperature` 场判相态。
4. **FHC 最后**: 用的是**订正后**的 `cor_z`/`cor_zdr`。
5. **上限保护**: `ATTEN_COR_Z_MAX / ATTEN_COR_ZDR_MAX` 兜底, 防止相位残留导致
   90 dBZ 级虚假订正。
   ⚠ **`ATTEN_PIA_MAX` 已于 2026-09-12 取消限高(改为 `np.inf`)**。原因: 实测该封顶
   只影响 0.06% 的门(937/1602705), 不封顶后 `cor_z` 最大 76.7 dBZ(仍被
   `ATTEN_COR_Z_MAX=75` 兜住); 而物理已三重验证正确(原始 φdp 自身涨幅 = 扣系统相位
   = Bringi 滤波 = KDP 积分), 人为裁剪会低估强降水衰减。
   现改用 `ATTEN_PIA_WARN = 15.0` 作**报警线**: verbose 输出超标门数, 不裁剪数值。
   要恢复限高, 把 `config.py` 里那行改回数值即可, 代码路径不变。

## 关于 φdp 零基线对齐(`align_phidp_baseline`)

> **2026-09-25 现行状态: 链上已不再调用它**(`QC/pipeline.py` 里那两行已删除)。
> 原因: 本项目唯一实现(Park2005)**只用 ΔΦ_DP = Φ(r0) − Φ(r1)**, 逐射线常数基线在相减时
> 自动抵消 ⇒ 这一步对结果而言是**空操作**; 而它产出的 `phidp_rel` 全库没有任何消费者
> (不在 `KEEP_FIELDS` 里、不进 `metric_cache`, 指标/出图/FHC 都不读, 跑完 run 段就随 radar
> 释放), 却要白占约 **300 MB 常驻内存**(62 层 × 269 rays × 2000 gates 的新建 float64 数组
> 加 mask)。函数定义保留在 `step06_attenuation_park2005.py`, 需要时手动调即可。
> 下面记录的是当初为 **pyart 版**(已删除)写的理由, 留作背景 —— 若日后换回 pyart
> 那类实现, 这一段就是**必须做的**(附实测表)。

pyart 的 `calculate_attenuation_zphi` **不是只用 Δφdp** —— 它在两处把 φdp 的
**绝对值**当 Δφdp 用:

```python
corr_phidp = _prepare_phidp(phidp, mask_fzl)            # mask 门填 0, 再 cummax
init_refl_correct = refl + corr_phidp * a_coef          # ① 绝对值进 Z 预校正
...
phidp_max = np.median(ray_phase_shift[last_six_good])
self_cons_number = 10.0 ** (0.1 * beta * a_coef * phidp_max) - 1.0   # ② 绝对值进指数项
```

它靠 `_prepare_phidp` 里 `phidp < 0 -> 0` 隐式假定"φdp 起点是 0"。所以系统相位
只要留残差 δ(逐射线常数), ①每 1° 给 Z 多算 `a_coef` dB(9.31 GHz 下 0.319 dB),
②让自洽数乘 `10^(0.1·β·a) ≈ 1.05`; 而 `phidp<0` 被截平 ⇒ **误差单向偏高, 不会
像 KDP 那样在导数里自动抵消**。

`align_phidp_baseline()` 逐射线把 φdp 减到"参与计算的门上的最小值"
(φdp 单调上升 ⇒ 最小值即起点相位; 用 min 比取首门更抗近端噪声尖峰), 写回
`phidp_rel`, 再把它喂给 ZPHI。这样"绝对值"精确等于"Δφdp", δ 自动抵消。

当年实测(62 层整卷, 人为给 φdp 加常数残差, pyart 版):

| 情形 | cor_z 有回波均值 | PIA 均值 | PIA 触顶门数 |
|---|---|---|---|
| 不对齐, 无残差 | 32.281 | 0.741 | 4301 |
| 不对齐, +20° 残差 | 35.644 (**+2.82**) | 4.095 | 25365 |
| **对齐**, +20° 残差 | **32.366** | 0.825 | 3992 |
| **对齐**, 无残差 | **32.366** | 0.825 | 3992 |
| **对齐**, −20° 残差 | **32.366** | 0.825 | 3992 |

对齐后三种残差给出**完全相同**的结果 ⇒ 只依赖 Δφdp。
(原开关 `C.ATTEN_REL_PHIDP` 随 pyart 版一并删除。)

## 衰减订正: Park et al. (2005) 自洽约束法(`zphi_attenuation_correction_park2005`)

**这是本项目唯一的衰减订正实现**(`step06_attenuation_park2005.py`),
按 Park et al. (2005, JTECH 22:1621–1632) 的式(1)(3)(4)(9)(12) **独立实现**,
不调 pyart。与 pyart 版(Gu et al. 2011 的 base-10 形式)同源——都是 Testud et al. (2000)
的 ZPHI 自洽约束法, 但**系数体系不同**(下表留作与文献对比的记录):

| | pyart 版(已删除) | 本项目实现 park2005 |
|---|---|---|
| 反射率线性化 | `10^(0.1·β·Z)`, β=0.64884 | `Z^b`, b=0.779 |
| 自洽数指数项 | `0.1·β·a_coef·φdp_max`, =0.0207/度 | `0.1·b·a·ΔΦdp`, =0.0249/度(a=`C.PARK_A`) |
| 约束相位 | `φdp_max`(零基线对齐后≈ΔΦdp) | `ΔΦdp = Φdp(r0) − Φdp(r1)` |
| A_DP 关系 | `c·A_H^d`, c=0.15917, d=1.0804 | `β·A_H^d`, β=0.139, d=1.13 |

核心公式(式 3, **分子必须乘自洽数**, 且自洽数的指数里**必须带 A_H-K_DP 系数 a**):

```
A_H(r) = Z'(r)^b · [10^(0.1·b·a·ΔΦdp) − 1] / ( I(r1,r0) + [10^(0.1·b·a·ΔΦdp) − 1]·I(r,r0) )
I(r1,r0) = 0.46·b·∫_{r1}^{r0} Z'(s)^b ds ;  I(r,r0) = 0.46·b·∫_{r}^{r0} Z'(s)^b ds
```

系数在 `C.PARK_B / C.PARK_BETA / C.PARK_D / C.PARK_A`(默认 0.779 / 0.139 / 1.134 /
**0.254**, 前三个是论文 Table 1 的 X 波段均值, `PARK_A` 是 α 的 Table 1 均值)。
产出字段带 `_park` 后缀(`spec_at_park / pia_park / cor_z_park / cor_zdr_park`),
链里会别名成无后缀的 `spec_at / pia / cor_z / cor_zdr`(见上文"字段名说明")。

### α 怎么取(自洽搜索 vs 固定值)

论文的核心是**逐射线自洽搜索最优 α**(式 7/8), 代码完整支持, 用
`C.PARK_ADAPTIVE_ALPHA = True` 打开。但**默认是固定 α = 0.254**(论文 Table 1 均值), 因为:

> ⚠ 式(8) 的 Σ|Φ_cal−Φ_DP| 靠 **φdp 廓线的形状**区分 α。当实测 φdp 沿射线接近直线
> (弱降水/短雨区/已平滑)时, 所有 α 给出的 Φ_cal 形状几乎相同(末端恒 ≈Δφ),
> 误差随 α 单调 → 最优点被推到搜索区间端点。**实测(威宁 20250509)端点占比 100%。**
> 这是该方法的固有性质(φdp 平坦时 α 不可辨识), 不是代码 bug。

| 模式 | 怎么用 | 特点 |
|---|---|---|
| 固定 α=0.254(默认) | 什么都不用改 | 稳定可复现; 即 Testud(2000) 定系数做法, 论文第 2 节批评其偏差 |
| 逐射线自洽搜索 | `C.PARK_ADAPTIVE_ALPHA = True` | 最贴论文式 7/8; 但 φdp 平坦时退化, verbose 会报端点占比 |
| 单条射线指定 | 调用时传 `a_phi=0.3` | 显式固定该值 |


**只订正零度层以下**(必须): `below_fzl = 温度场 temp > 0°C` 限定订正范围, 积分
`I(r,r0)`、约束量 `ΔΦdp` 都只在冻结层以下的雨区门 [r1, r0] 上取。零度层以上的
冰相门**保持原始 `reflectivity` 不订正**(`cor_z_park` 在零度层以上 = `refl`, 差严格
为 0), 只有 gatefilter 排除的门和原始 refl 已 mask 的门才 mask 掉。衰减量字段
(`spec_at_park / pia_park / cor_zdr_park`) 只定义在雨区, 零度层以上 mask。

实测(62 层整卷 ZWN03 202607260708): A_H 随 Z 单调增, 40–50 dBZ 档中位 ~0.46 dB/km、
50–60 dBZ ~1.6 dB/km(符合 `A_H = a·Z^b` 的 X 波段物理量级)。

### 2026-09-11 修正(两个问题, 之前"park 订正比 ZPHI 多"的原因)

> 下表左列的"pyart"仅作对照基准(当时两版并存), pyart 版本身已于 2026-09-12 删除。

| | 修好前 | 修好后 |
|---|---|---|
| 自洽数指数 | `0.1·b·Δφdp` = 0.0779/度(**少了 a**) | `0.1·b·a·Δφdp` = 0.0249/度 |
| Δφdp=30° 的 scn | 216(pyart 3.18) | 4.57 |
| 单门订正量 | 最大 **+58.9 dB** | ≤ `ATTEN_PIA_MAX`(当时 15 dB) |
| 雨区 PIA 均值(6 层) | 2.36~3.59 dB | **1.28 dB**(pyart 当时 1.98) |
| `cor_z` max | 74.4 dBZ | **62.4 dBZ**(pyart 当时 61.3) |

1. **自洽数少了 A_H-K_DP 系数 a**: 指数因此大 1/a≈3.1 倍, scn 从 O(1) 爆到 10²~10⁴,
   分母里 `I(r1,r0)` 被完全压住 → `A_H` 退化成"只由 Z 廓线决定的上限"、Δφdp 约束失效 →
   订正系统性偏大 ~2 倍。量纲核对: 小 scn 极限下 `PIA≈(2/(0.46b))·scn`, 物理上
   `PIA=a·Δφdp` ⇒ 指数应为 `0.1·a·b≈0.022/度`, 与 pyart 的 `0.1·β·a_coef=0.0207` 一致。
   因子替换实验(同一雨区): 只换指数 3.59→1.29 dB(重合 pyart), 只换 Z 项 3.59→3.65 dB(无影响)。
   传 `a_phi=1.0` 可复现修好前的旧行为, 仅用于对比。
2. **`cor_z` 用了未封顶的 PIA**: 旧代码 `cor_z = refl + PIA` 里 PIA 没封顶(只封了 `pia_park`
   字段), 于是单门能被抬到 +59 dB、`cor_z` 顶到 74.4 dBZ、CR 上 ΔCR 最大 +42.8 dB。
   现在改为**先用封顶增量订正**: `pia_c = clip(PIA, 0, ATTEN_PIA_MAX)` → `cor_z = refl + pia_c`
   (`pida_c` 同理, 上限 `ATTEN_PIDA_MAX`), 增量在雨区外为 0, 所以**零度层以上仍严格等于原始
   `reflectivity`**(已回归验证: 差值 max 0.000000)。

### 关于 `align_phidp_baseline`(2026-09-25 起链上已不调用)

它逐射线把 φdp 拉到 0 起点, 产出 `phidp_rel`。**对 Park2005 而言这一步不是数学必需的**
(论文只用 ΔΦ_DP, 逐射线常数基线自动抵消), 而且那个字段没有任何消费者 —— 已在
2026-09-25 从 `QC/pipeline.py` 的链上删除(理由与内存占用见上面"关于 φdp 零基线对齐"一节)。
函数定义保留, 只在两种场合才需要它:
· 想单独画/核查"从 0 起跳"的相位剖面时手动调一次;
· 若日后要换回 pyart 那类"把 φdp 绝对值当 Δφ 用"的实现, 这一步是必须的。

> 注: 参数 `C.ATTEN_REL_PHIDP` 已随 pyart 版一起删除(它只控制 pyart 版是否对齐)。

## 用法

```python
import QC

# 方式 A: 从文件一路到底(含退折叠; 默认**不含 FHC** —— 2026-09-13 起分类已剥离)
radar = QC.process_radar(BIN_PATH, SOUNDING_PATH)
radar = QC.process_radar_fhc(BIN_PATH, SOUNDING_PATH)     # 要 FH 字段用这个(一步到位)
#   或事后补: radar = QC.run_fhc(radar)

# 方式 B: 已有 radar 对象(批量画图用这个, 省 I/O)
snd = QC.read_sounding(SOUNDING_PATH)
radar = QC.prepare_radar_products(radar, snd)

# 方式 C: 指标提取(框选 -> 掩膜 -> 指标)
lon1, lat1, lon2, lat2 = QC.select_box_ginput(radar, sweep=0, field='cor_z')  # 鼠标点图
box = QC.build_box_mask_3d(radar, lon1, lat1, lon2, lat2)
QC.print_metrics(QC.all_metrics(radar, box, field='cor_z',
                                fhc=('FH' in radar.fields)))
```

> 无图形界面时(批量/服务器)跳过框选, 直接给定四个经纬度:
> `box = QC.build_box_mask_3d(radar, 103.5, 27.5, 105.0, 26.5)`

## 调参

统一改 `QC/config.py`(各模块内部实时读取 `C.*`, 改文件或运行时改都立即生效):

```python
import QC.config as C
C.BAD_TILT_FRAC = 1.1        # 例如临时关掉整层损坏剔除
C.ATTEN_PIA_MAX = 30.0       # 放开 PIA 上限
```

## 指标(4) VIL 与 指标(7) ZDR 柱(都走笛卡尔网格化)

两者都先把体扫网格化到笛卡尔坐标(`pyart.map.grid_from_radars`, Barnes2 + `dist_beam`
影响半径), 但 **各网格化各的, 互不相干**(2026-09-14 用户要求: VIL 里不要带 ZDR 的东西):

- VIL 的网格**只装 `cor_z`** -> 存出来的 `vilgrid_<CASE>_<时刻>.nc` 字段只有
  `cor_z` / `VIL` / `ROI`, 不含 `cor_zdr`、不含 `ZDRCOL` 掩膜;
- ZDR 柱自己网格化一次(只装 `cor_zdr`), 那份 Grid 默认只在内存里(要留就单独写, 见下):

```python
m, res = QC.metric_zdr_column(radar, box, return_field=True)
grid_col = res['grid']          # ZDR 柱自己的网格(含 cor_zdr + ZDRCOL 掩膜)
```

网格分辨率由 `C.VIL_GRID_DZ_M`(默认 250 m)/ `C.VIL_GRID_DXY_M`(默认 150 m)/
`C.ZDRCOL_GRID_*` 决定(两者默认取同一套值)。代价是网格化多做一遍 —— 那是本链最耗时
的一步; 想省时间可以显式传 `grid=` 让 ZDR 柱复用某个已有 Grid(但别把 VIL 那份传进去,
否则 ZDR 字段又混进 VIL 产物了)。

```python
import QC
box = QC.build_box_mask_3d(radar, 103.5, 27.5, 105.0, 26.5)

# 推荐把探空一起传进去: 0 °C 层用探空 fzl, 判据① 上界(-20 °C)直接查探空原始廓线
snd = QC.read_sounding(SOUNDING_PATH)

# (4) VIL: Greene & Clark 1972 逐列积分, 55 dBZ 雹电容封顶
vil_max, vil_mean = QC.metric_vil(radar, box)

# (7) ZDR 柱: Zhao et al. (2025) "3D mapping columns" 法
m = QC.metric_zdr_column(radar, box, snd=snd)
# 进统计表的只有三项(2026-09-14 用户要求):
# -> zdrcol_max_zdr_db(柱内最大 ZDR) / zdrcol_height_m(柱顶高度) / zdrcol_depth_m(柱顶距0°C层)
#    柱底/体积/格点数/水平位置等仍在 res 里(return_field=True), 只是不写表

# 想换阈值 / 手给融化层 / 只按阈值判(不做柱状约束):
QC.metric_zdr_column(radar, box, zdr_min_db=1.0, z0c_m=snd['fzl'],
                     use_connectivity=False)
```

**ZDR 柱的三条判据**(全部满足才算柱内格点, 参数都在 `config.py` 的 `ZDRCOL_*`):

1. **阈值**: `ZDR >= ZDRCOL_ZDR_MIN_DB`(默认 **1.5 dB**, 论文取法; 老文献常用 1.0),
   且高度在 `[0°C 层 - ZDRCOL_BELOW_FZL_M, 混合相区顶]`(下界默认从融化层以下 1 km 起算;
   **上界 2026-09-14 新增** = `ZDRCOL_MIXED_PHASE_TOP_C`(默认 **-20 °C**)那条等温线的
   高度 —— 论文 Fig.3 Step 2 写的是 "from low level to the mixed-phase region")。
   想直接给高度就传 `z_upper_m=<米>`; 想关掉上界就传 `z_upper_m=None`;
2. **柱状形态**(`ZDRCOL_USE_CONNECTIVITY`): 必须与"融化层以下 1 km ~ 融化层"这段
   下部高值区**连通** —— 排除混合相区里悬空的高 ZDR 假信号(三体散射、退偏振条纹、扁冰晶);
3. **向上非增**(`ZDRCOL_USE_NEG_GRADIENT`): 融化层以上相邻两层 ZDR 之差 `<= 0`
   (等于 0 也算通过, 与论文 Fig.3 Step 3 的 `y − x <= 0` 逐字一致; 尺度分选指纹)。
   容差 `ZDRCOL_GRAD_TOL_DB`(默认 0)可调大 = 要求"至少降这么多", 用来滤噪声。

**0 °C 层与 -20 °C 层的高度都优先查探空原始廓线** —— 给 `metric_zdr_column` /
`all_metrics*` 传 `snd=QC.read_sounding(...)` 即可(工作.py 就是这么做的, 探空只读一次、
处理链与指标共用同一份, `read_sounding()` 收到 dict 会原样返回)。

> 为什么要这样: `radar.fields['temperature']` 是探空按**雷达库点高度**插值出来的,
> 它的有效高度上限 = `min(探空顶, 雷达库点最高)` —— 浅体扫、或框选的区域离雷达近/远
> 导致最高门只有几 km 时, 约 7~8 km 的 -20 °C 就落在场外。而探空原始廓线一般到
> 30 km 以上, 一定覆盖得到。**"给了探空"不等于"温度场有那个高度"**, 所以别绕道温度场。

- **0 °C 层**: 优先 `snd['fzl']`(探空原始廓线算的融化层, 与衰减订正用的那条**同源**),
  其次才从 `radar.fields['temperature']` 反推(`QC.zero_degree_height()`, 按高度分箱取
  中位温度再插值找穿越点)。论文用的就是探空融化层, 并在讨论里指出"融化层在上升气流核里
  会被抬高"是该法的一个已知偏差。
- **-20 °C 层**(判据① 上界): `QC.isotherm_height(radar, -20.0, snd=snd)` ——
  `zero_degree_height` 就是 `temp_c=0` 的薄包装。
  **万一真找不到**(既没传 `snd`、温度场又没覆盖到), 它**不会硬凑一个值**: `strict=True`
  返回 None, 指标(7) 随即**退回"不设上界"并打印告警** —— 判据① 宁可少一个约束,
  也绝不能拿一个假上界(例如"廓线顶")把真实存在的柱子截短。

> ### ⚠ 与论文原口径的差异(逐条核对过: Zhao et al. 2025 Sect. 2.4 + 3.4 + Fig. 3/4)
> **算法三步与论文一致**(阈值 1.5 dB / 从融化层下 1 km 起 / 用 0 °C 层以下向上约束 /
> 0 °C 层到网格顶逐层非增 / 柱体只算 0 °C 层及以上), 但有几处不同:
>
> | 项 | 论文 | 本链 |
> |---|---|---|
> | 网格 | **0.25 km 水平 / 500 m 垂直 / 500 m–20 km 共 40 层** | 150 m / 250 m / 0–15 km(VIL 同套, 更细) |
> | **柱"高度"** | **从融化层往上数层数 n × 0.5 km**(相对融化层) | 表里 `height_m` 是**柱顶海拔 AGL**; 论文口径 = `depth_m` |
> | 体积 | 格点数 × 0.03125 km³(固定单格) | 格点数 × 实际格点体积 |
> | 判据①上界 | Fig.3 明写"到 mixed-phase region"(**未给高度**) | ✔ 已对齐: 默认取 **-20 °C 等温线**(`ZDRCOL_MIXED_PHASE_TOP_C`) |
> | 判据②实现 | Fig.3 Step 2/3 = **同列垂直**连续(successive columnar; y−x≤0 也是逐列比) | ✗ **未对齐**: scipy 6-邻域**三维**连通 + 锚定 [z0c−1 km, z0c](水平错位时会多留格点) |
> | 判据③等号 | `y − x <= 0`(**等于 0 也保留**) | ✔ 已对齐(2026-09-14): 改成 `<=`, `grad_tol_db=0` 时与论文原式逐字等同 |
>
> **比"柱高"时要拿 `zdrcol_depth_m`(距 0 °C 层)去对论文**, 不是 `zdrcol_height_m`。
> 想跟论文数值直接可比: 把 `ZDRCOL_GRID_DZ_M/ZDRCOL_GRID_DXY_M/ZDRCOL_Z_TOP_M`
> 改成 `500/250/20000`。

柱体掩膜落在 **ZDR 柱自己那份 Grid** 上: `compute_zdr_column_field()` 会把柱子作为字段
`ZDRCOL`(0/1)挂到它用的网格上(`attach_to_grid=True` 默认)。**它不在 `vilgrid_*.nc` 里**
—— VIL 产物按用户要求保持干净。要看柱子/存掩膜, 直接拿 `res['grid']`:

```python
import pyart
m, res = QC.metric_zdr_column(radar, box, return_field=True)
d = pyart.graph.GridMapDisplay(res['grid'])
d.plot_latitudinal_level('ZDRCOL', y_index=20)   # 柱子在哪一层、长什么样

# 想单独留一份柱体网格:
pyart.io.write_grid(r'D:\...\zdrcol_grid_20250509_153000.nc', res['grid'])
```

> 注意: 显式传了 `grid=` 时, `dz_m/dxy_m/z_top_m/margin_m` 四个分辨率参数会被忽略
> —— 网格已经定好了, 只有阈值/0°C 层/判据开关仍然生效。

## 组合反射率(CR): 计算 + 落成 radar + 一行出图

**平时就用这两行**(结果既写回原 radar, 又给你一个能直接画的单层 radar):

```python
import QC

cr_radar, comp_z, comp_alt = QC.composite_radar(radar, field='cor_z')   # 算 + 落成 radar
QC.plot_cr(cr_radar)                                                    # 一行出图
```

- `cr_radar` 是**只含 ref_sweep 那一层 + CR 字段**的 slim radar, 可直接交给 pyart:

  ```python
  import pyart
  pyart.graph.RadarMapDisplay(cr_radar).plot_ppi_map(field='CR', sweep=0)
  ```

- `comp_z` / `comp_alt` 是二维数组 `(n_az, ngates)`: 组合反射率 / 最大值所在高度。
- 同时写回**原 radar** 的字段: `field='cor_z'` → `radar.fields['CR_cor_z']`;
  `field='reflectivity'` → `radar.fields['CR']`。不想要就 `write_back=False`。

出图(存图给 `out_path`, 不给就直接显示; 降雹点用黑色五角星):

```python
QC.plot_cr(cr_radar, out_path=r'D:\...\cr.png',
           hail_location=[[104.02, 27.15], [104.19, 27.23]], vmin=0, vmax=70)
```

**从文件一步到 CR 图**:

```python
cr_radar, radar = QC.cr_from_file(BIN_PATH, SOUNDING_PATH)
QC.plot_cr(cr_radar, out_path=r'D:\...\cr.png', hail_location=[[104.02, 27.15]])
```

### 需要更细的控制时(底层接口)

```python
# 只要数组, 不想写回字段
comp_z, comp_alt = QC.composite_reflectivity(radar, field='cor_z', write=False)
print(QC.composite_stats(comp_z, comp_alt))

# 快速版(批量出图: 方位角一致时整卷一次 fmax 归约)
comp, az0 = QC.composite_reflectivity_fast(radar, gatefilter=gf, field='reflectivity')

# 自己接底层工具
QC.store_composite_field(radar, comp_z, source_field='cor_z')   # 单独写回
QC.build_cr_radar(radar, comp_z, ref_sweep=0)                   # 二维 CR -> 单层 radar
```

- 精确版: 方位角线性插值对齐(`scipy.interpolate.interp1d`) → max + 对应高度 → `fill_azimuth_gaps` 补缺
- 快速版: 方位角一致时 `np.fmax.reduce` 一次归约整卷; 不一致时 `align_azimuth_nearest` 最近邻对齐
- 出图不传 `out_path` 就直接 `plt.show()`(notebook 用), 传了就存图并关闭

## Nature 风格: 未订正 CR vs 衰减订正后 CR 对比图

一图回答一个问题: **ZPHI 衰减订正把组合反射率改了多少、改在哪里**。

```python
import QC
fig, axs = QC.plot_cr_compare(radar, ref_sweep=0,
                              uncorrected='reflectivity', corrected='cor_z',
                              out_path=r'D:\...\cr_compare.png',
                              emit=('png', 'pdf', 'svg'))
```

三栏的设计(每栏一个论证角色, 不是重复画三遍):

| panel | 画什么 | 角色 |
|---|---|---|
| a | CR(来自 `reflectivity`, 未订正) | 基准: 衰减把强回波压低了 |
| b | CR(来自 `cor_z`, 衰减订正后) | 处理结果 |
| c | ΔCR = b − a | **主 panel**: 订正量的空间分布 |

- a/b **共用同一 0~70 dBZ 色标**(共用才能直接比高低); c 单独一根**关于 0 对称**的发散色标。
- **ΔCR 只在两侧都有真实数据的门上算**, 不参与 `fill_azimuth_gaps` 补缺 —— c 里的空白
  是"无有效数据", 不是被插值污染出来的假差值。这是差值图必须守住的一点。
- 版面按 **Nature 双栏 183 mm**(`fig_width_in=7.2`)由 extent 纵横比反推画布高度, 三栏绘图区
  严格等宽等高; 文字保持可编辑(`pdf.fonttype=42` / `svg.fonttype=none`)。
- 每次出图自动跑 **nature-figure 的多 panel 绘图区对齐门**(量最终渲染矩形 → 校验等宽等高
  + 报告 JSON/SVG)。找不到技能脚本只提示不报错; 投稿前用 `qa_strict=True` 卡住导出。

### 多版本并排对比(`plot_cr_compare_multi`)

要一次比较**多个衰减订正方案**(如 pyart ZPHI / Park2005 修好版 / Park2005 旧版)时用这个:

```python
fig, axs = QC.plot_cr_compare_multi(
    radar, ref_sweep=0, lang='en', verbose=True,
    corrected_fields=[('ZPHI (pyart)',       'cor_z'),
                      ('Park2005 (fixed)',   'cor_z_park'),
                      ('Park2005 (pre-fix)', 'cor_z_park_old')],
    hail_location=[[104.02, 27.15]],
    out_path=None)            # None=显示; 给路径=按 png/pdf/svg 导出
```

- **第一行**: 未订正 CR + 各订正版 CR(共用 0~70 dBZ 色标, 可直接比高低)
- **第二行**: 各订正版的 ΔCR = 订正 − 未订正(共用关于 0 对称的色标, 红=订正抬升),
  与上一行对应的订正列**逐列对齐**(左起第一列留空, 因为未订正没有 Δ)
- 每个面板左上角写该版本 `max dBZ / n`(CR)或 `mean / p95 / max dB`(ΔCR);
  `verbose=True` 时终端再打印一张所有版本的数值表(含 ΔCR 的 mean/p50/p95/max)
- `show_delta_row=False` 只出第一行 CR; `delta_vmax` 可固定 Δ 色标半幅
- 实现见 `QC/step09_plotting.py`; 用法示例见 `notebooks/QC_test.ipynb` 的作图章节
  (原 `notebooks/atten_correction_compare.ipynb` 是 pyart-vs-Park2005 对比用的, 已于 2026-09-12 随 pyart 版一并删除)

常用参数: `show_delta=False`(只出两栏) / `ref_sweep=3`(换基准层) / `delta_vmax=10`(固定 Δ 色标)
/ `lang='zh'`(中文标题) / `fill_gaps=False`(不补缺) / `fig_width_in=3.5`(Nature 单栏 89 mm)。

单独导出/审计: `QC.save_pub_fig(fig, path, emit=('png','pdf','svg'), dpi=600)` 和
`QC.nature_panel_alignment(fig, json_out=..., row_groups=[['a','b','c']])`。

QC 门控有两个入口:

```python
QC.build_qc_gatefilter(radar)                       # 完整(整层坏层/同色平台/近距 refl-tp/椒盐 全开)
QC.build_qc_gatefilter_basic(radar,                 # 批量出图用(快)
                             whole_layer=False)     #   whole_layer=True 可去掉环形假回波
```

> 第 8 条 `moment_texture_gatefilter(radar, **kwargs)` 就是 pyart 自带
> `pyart.filters.moment_and_texture_based_gate_filter` 的直接调用(沿射线纹理: ρhv 下限 +
> φdp/ρhv/ZDR/反射率的纹理上限), 已并入上面两个入口; 阈值都用 pyart 默认值, 要调就传 pyart
> 同名参数。整卷 62 层约 +35 s, 想快就别在批量循环里用它。

## 分部分测试 notebook

`notebooks/QC_test.ipynb` —— 按同一条链的顺序逐段跑(step01→step07), 每段都能单独看统计、单独改参数,
最后还有"分步 vs 一步到位"的一致性对比、参数一览、PPI / 组合反射率出图。

- 生成脚本: `diagnostics/_make_qc_nb.py`(改示例路径/内容后重跑即可, 保证与代码同步)
- 冒烟测试: `diagnostics/_smoke_qc_nb.py`(把 `# [PLOT]` 开头的绘图 cell 换成 `pass` 后真实执行其余 cell)

## 与旧代码的兼容

`core/renwu.py` 已降级为**纯转发层**(2026-09-12): 它把 QC 里的函数/参数原样再导出,
所以老的 `from renwu import read_sounding, prepare_radar_products, _get_sweep_indices`
全部照旧可用, 不用改脚本。**新代码请直接用 `QC`, 不要再 `from renwu import`。**

- 处理流程 + 指标: `QC/step01..step10`(**都在这一个包里**)
- 绘图风格与色标: `QC/plotstyle.py`(原 `core/csu_function.py`, 内容一致)
- 组合反射率: `QC/step08_composite.py`;出图: `QC/step09_plotting.py`
- 指标提取: `QC/step10_metrics.py`(原 `core/renwu.py` 里的指标函数)
- 批量出图: `products/`
- 相控阵循环 PPI: `scripts/loop_plot.py`

> `core/csu_function.py`、`notebooks/csu_function.py` 与 `QC/plotstyle.py` 三者内容一致。
> 以后**改绘图风格只改 `QC/plotstyle.py`**; 另两份是历史副本, 新代码不再引用。

## 改名历史

原先 `QC/` 里的步骤文件没有序号(`sounding.py` / `gatefilter.py` / … / `composite.py` /
`plotting.py`), 2026-09-11 统一加上 `stepNN_` 前缀(见
`.workbuddy/backup/2026-09-11-steps/`)。**包级导入不受影响**(`from QC import xxx`),
只有直接写子模块路径的老代码要改成 `from QC.step08_composite import ...`。
