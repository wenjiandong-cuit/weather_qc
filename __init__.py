# -*- coding: utf-8 -*-
"""
QC —— 雷达资料处理流程(按部分拆分的模块包)
=================================================================
一条链, 一个部分一个文件; **文件名带序号**, 打开目录就是执行顺序:

  stepNN_文件名              部分                        主要函数/字段
  ----------------------    ------------------------    --------------------------------------
  QC/step01_sounding.py     1 探空读取                  read_sounding() -> fzl/层结
  QC/step02_gatefilter.py   2 QC 门控                   build_qc_gatefilter() -> GateFilter
                                                       (内含 pyart moment+texture 沿射线纹理门控
                                                       moment_texture_gatefilter())
  QC/step03_system_phase.py            3 系统相位(唯一实现, ρhv判据) remove_system_phase_by_rho()
                                                       判据: 前 30 个连续 ρhv>0.97 门的 φdp 平均
                                                       estimate_system_phase()
                                                       find_first_rho_run()
                                                       ★ 旧法(前 40 门中位数, 同名文件)已删除
  QC/step04_kdp.py          4 φdp 平滑 / KDP             compute_kdp_bringi() -> kdp_dual
                                                       mask_reflectivity()
                                                       ★2026-10-07 起内部改用 QC/zch_phi
                                                       双尺度方案(字段名 kdp_dual 不变)
  QC/zch_phi.py            — φdp 双尺度滤波/KDP        compute_kdp_dual() -> kdp_dual
                                                       (轻 0.6 km / 重 1.5 km, 对平滑后 φdp
                                                        最小二乘, 按 Z 阈值切换;
                                                        Park et al. 2009 方案;
                                                        ★step04 现用它实现)
  QC/step05_temperature.py  5 温度场                     add_temperature_field() -> temperature
  QC/step06_attenuation.py  6 衰减订正【唯一实现】      zphi_attenuation_correction()
                                                       扣系统相位 + zch_phi 双尺度滤波
                                                       + 末端 ρhv>0.9 取 9 门平均当总相位差
                                                       产出 spec_at_user / pia_user /
                                                         cor_z_user / cor_zdr_user
                                                       ★ 旧 Park2005 / ERAD 版已删除
                                                         (2026-10-06, 归档 _trash_20261006)
  QC/step07_classification.py 7 FHC 水凝物分类           add_fhc_field() -> FH
  QC/step08_composite.py    8 组合反射率                 composite_radar() / composite_reflectivity()
  QC/step09_plotting.py     9 出图                       plot_ppi() / plot_cr() / plot_composite()
                                                       plot_cr_compare()(未订正 vs 订正 对比图)
                                                       save_pub_fig() / nature_panel_alignment()
  QC/step10_metrics.py     10 指标提取(框选→掩膜→指标)   select_box_ginput() / build_box_mask_3d()
                                                       metric_max_z() / metric_echo_top() /
                                                       metric_echo_count() / all_metrics()
                                                       (2026-09-12 从 core/renwu.py 搬进来)

  非步骤: QC/config.py(全部可调参数) / QC/pipeline.py(编排) / QC/plotstyle.py(绘图风格色标,
                                                                            原 core/csu_function.py)
          / QC/fast_texture.py(沿射线纹理快速版, step02 自动装上; 2026-09-22 移植)

用法(全部等价, 想用哪个用哪个):
    import QC

    # ① 全流程: 文件 -> 处理好的 radar(**不含 FHC**, 2026-09-13 起分类已剥离为独立函数)
    radar = QC.process_radar('xxx.BIN.zip', 'sounding.txt')
    #    需要 FHC 水凝物分类(FH 字段)时, 二选一:
    radar = QC.run_fhc(radar)                          # 对处理好的 radar 补分类
    radar = QC.process_radar_fhc('xxx.BIN.zip', 'sounding.txt')   # 或一步到位

    # ② 已有 radar, 只跑处理链(省 I/O)
    radar = QC.prepare_radar_products(radar, snd)

    # ③ 组合反射率: 算完直接给你一个能画图的 radar
    cr_radar, comp_z, comp_alt = QC.composite_radar(radar, field='cor_z')
    QC.plot_cr(cr_radar)                    # 一行出图

    # ④ 从文件一步到 CR 图
    cr_radar, radar = QC.cr_from_file('xxx.BIN.zip', 'sounding.txt')
    QC.plot_cr(cr_radar, out_path=r'...\\cr.png', hail_location=[[104.02, 27.15]])

    # ⑤ 指标提取(框选 -> 掩膜 -> 指标)
    lon1, lat1, lon2, lat2 = QC.select_box_ginput(radar, sweep=0, field='cor_z')
    box = QC.build_box_mask_3d(radar, lon1, lat1, lon2, lat2)
    QC.print_metrics(QC.all_metrics(radar, box, field='cor_z'))
    #  无图形界面时直接给四个经纬度:
    box = QC.build_box_mask_3d(radar, 103.5, 27.5, 105.0, 26.5)

    # ⑥ 第二步的 QC 门控里已含 pyart moment+texture(沿射线纹理); 想单独要一个就用
    gf = QC.build_qc_gatefilter(radar)
    mtf_gf = QC.moment_texture_gatefilter(radar)        # 也可以单独跑, 参数同 pyart

    # ⑦ 衰减订正(唯一实现 = 用户口径, 产出字段带 _user 后缀)
    #    链路: 扣系统相位 + zch_phi 双尺度滤波(轻0.6km/重1.5km) + 末端ρhv>0.9取9门平均
    #          当总相位差, 复刻 pyart 公式算 PIA; 系数用 pyart 自动查表 X 波段 0.31916。
    #    字段: 产出 spec_at_user / pia_user / cor_z_user / cor_zdr_user,
    #    链里会自动别名成无后缀的 spec_at / pia / cor_z / cor_zdr。
    #    ★ 只订正反射率: ZDR **不做**差分衰减订正, cor_zdr 是原始 ZDR 直通(2026-09-14 起;
    #      原因见 config.py 那节说明)。
    #    参数集中在 config 的 ZPHI_USER_* (N_TAIL / RHO_TAIL / SP_N_GATES / ...)。

    # ⑧ 系统相位: 链里用的是 ρhv 判据(每条射线取前 30 个连续 ρhv>0.97 的门, 取 φdp 平均)
    sp, info = QC.estimate_system_phase(radar)          # 单独估一个看看

    # ⑨ 绘图风格(原 core/csu_function.py 的工具)
    QC.science_style(); cmap, norm = QC.ref_cmap_params()

    import QC.config as C          # 只拿参数(不加载 numpy/pyart/cartopy)
    from QC.config import KDP_THSD

【字段名】prepare_radar_products 会把
`cor_z_user / cor_zdr_user / pia_user / spec_at_user` **别名**成无后缀的
`cor_z / cor_zdr / pia / spec_at`(同一份 ndarray, 不额外占内存), 好让指标、出图、
老脚本直接可用。
★ 衰减订正**唯一实现** = 用户口径(step06_attenuation, 实现与入口已合并):
  扣系统相位 + zch_phi 双尺度滤波 + 末端 ρhv>0.9 取 9 门平均当总相位差, 复刻 pyart
  公式算 PIA; 系数用 pyart 自动查表 X 波段 a_coef=0.31916。
  旧的 Park2005 论文版 / ERAD 版已于 2026-10-06 删除(归档 _trash_20261006/);
  pyart 官方版已于 2026-09-12 删除。
★ 其中**只有反射率被订正**: `cor_z` = 原始 refl + PIA; `cor_zdr` = 原始
`differential_reflectivity` **直通**(只套 QC 掩膜, 不做式(9)(12) 差分订正,
2026-09-14 决定, 原因见 QC/config.py)。

【惰性导入】本文件用 PEP 562 的模块级 __getattr__ 做按需加载 —— 取到哪个名字才去 import
对应的子模块。因此 `import QC` 本身**不会**拉起 numpy / pyart / cartopy, 也不会因为
某个可选依赖缺失就让整个包不可用; 只有真正用到某个函数时才加载它所在的那一层。
子模块名(QC.config / QC.plotstyle / QC.step06_attenuation_park2005 等)照旧可以直接当属性取。

【合并历史 2026-09-12】原先 core/renwu.py 的指标层与 core/csu_function.py 的绘图风格
都搬进本包(step10_metrics.py / plotstyle.py)。**现在只跟 QC/ 一个文件夹打交道就够了。**
core/renwu.py 降级为纯转发层, 老脚本 `from renwu import ...` 仍然可用。
"""

import importlib as _importlib

__version__ = '1.1'

# 名字 -> 所在子模块。顺序决定 __all__ 的排列顺序, 改动请保持分组。
_EXPORTS = {
    '.step01_sounding': ('read_sounding',),
    '.step02_gatefilter': ('build_qc_gatefilter', 'build_qc_gatefilter_basic',
                           'moment_texture_gatefilter'),
    # 非步骤: 沿射线纹理"跳过空窗"快速版 —— step02 的 moment+texture 门控自动用它
    '.fast_texture': ('texture_along_ray_fast', 'install_fast_texture',
                      'uninstall_fast_texture', 'fast_texture_status',
                      'fast_texture_status_lines'),
    '.step03_system_phase': ('remove_system_phase_by_rho', 'remove_system_phase_per_layer',
                             'estimate_system_phase',
                             'per_ray_system_phase', 'find_first_rho_run',
                             'valid_gate_mask', 'combine_phase', 'max_run_length',
                             'PHI_SENTINELS'),
    '.step04_kdp': ('compute_kdp_bringi', 'mask_reflectivity'),
    # 非步骤: Φ_DP 双尺度滤波 + 最小二乘 K_DP(Park et al. 2009 方案)
    # —— step04_kdp.py(Bringi) 的**并列替代**, 不替换它; 链上未接线, 要用显式调
    '.zch_phi': ('compute_kdp_dual', 'dual_kdp', 'kdp_dual_filter',
                 'unwrap_phi_by_ray', 'kdp_threshold_jump'),
    '.step05_temperature': ('add_temperature_field',),
    '.step06_attenuation': ('zphi_attenuation_correction',
                            'zphi_attenuation_correction_user'),
    '.step07_classification': ('add_fhc_field', 'fhc_counts',
                               'FHC_CATEGORIES', 'FHC_HAIL', 'FHC_GRAUPEL'),
    '.step08_composite': ('composite_radar', 'build_cr_radar',
                          'composite_reflectivity', 'composite_reflectivity_fast',
                          'align_azimuth_nearest', 'fill_azimuth_gaps', 'sweep_indices',
                          'composite_stats', 'store_composite_field', 'composite_field_name'),
    '.step09_plotting': ('plot_cr', 'plot_ppi', 'plot_composite',
                         'plot_cr_compare', 'plot_cr_compare_multi',
                         'save_pub_fig', 'nature_panel_alignment'),
    '.step10_metrics': ('select_box_ginput', 'select_box_drag', 'build_box_mask_3d',
                        'metric_max_z', 'metric_zmax_height',
                        'metric_echo_top', 'metric_echo_count',
                        'metric_echo_volume', 'metric_hail_graupel_count',
                        'compute_vil_field', 'metric_vil', 'save_vil_nc',
                        'save_grid_nc', 'read_grid_nc', 'make_vil_radar',
                        'isotherm_height', 'zero_degree_height',
                        'compute_zdr_column_field',
                        'metric_zdr_column',
                        'save_box', 'load_box', 'load_boxes',
                        'select_and_save_box', 'all_metrics_series_from_file',
                        'save_metrics_series', 'load_metrics_series',
                        'save_metrics_excel',
                        'all_metrics_ginput', 'all_metrics_series_ginput',
                        'all_metrics_series', 'metric_trends', 'max_trend',
                        'print_trends',
                        'all_metrics', 'print_metrics'),
    '.plotstyle': ('ref_cmap_params', 'science_style', 'add_latlon_grid',
                   'bold_colorbar', 'bold_title', 'cjk_font_family',
                   'add_field_to_radar_object',
                   'two_panel_plot', 'adjust_fhc_colorbar_for_pyart',
                   'print_fields', 'calculate_angle'),
    '.pipeline': ('prepare_radar_products', 'process_radar', 'process_radar_fhc',
                  'run_fhc', 'cr_from_file'),
    '.config': ('FILE_PATH', 'SOUNDING_PATH',
                'QC_RHOHV_MIN', 'QC_RHOHV_MAX', 'QC_Z_MIN', 'QC_HAIL_KEEP_Z', 'QC_HAIL_RHOHV_MIN',
                'CLUTTER_ELEV_MAX', 'CLUTTER_RANGE_KM', 'CLUTTER_VEL_MAX', 'CLUTTER_SW_MAX',
                'CLUTTER_Z_MIN', 'CLUTTER_Z_MAX',
                'REFL_POWER_DIFF_DB', 'REFL_POWER_DIFF_RANGE_KM',
                'BAD_TILT_GAP_DB', 'BAD_TILT_FRAC', 'BAD_TILT_MIN_RANGE_KM',
                'PLATEAU_CHECK', 'PLATEAU_Z_MIN', 'PLATEAU_RANGE_KM', 'PLATEAU_BIN_DB',
                'PLATEAU_PEAK_FRAC', 'PLATEAU_MIN_BLOB', 'PLATEAU_MIN_RAYS',
                'DESPECKLE_PHI_SIZE', 'DESPECKLE_Z_SIZE',
                'ATTEN_PIA_MAX', 'ATTEN_PIA_WARN', 'ATTEN_COR_Z_MAX',
                'ATTEN_BACKEND',
                'PARK_A',
                'PARK_B',
                'SYSPHASE_FALLBACK_PCT',
                'KDP_THSD', 'KDP_NFILTER', 'KDP_GATE_SPACING', 'KDP_STD_GATE',
                'ZPHI_DOC', 'ZPHI_SMOOTH_WINDOW', 'ATTEN_BAD_FILL',
                'ZPHI_USER_N_TAIL', 'ZPHI_USER_RHO_TAIL',
                'ZPHI_USER_SP_MODE', 'ZPHI_USER_SP_DROP_SENTINEL',
                'VIL_Z_CAP_DBZ', 'VIL_Z_MIN_DBZ', 'VIL_GRID_DZ_M',
                'VIL_GRID_DXY_M', 'VIL_Z_TOP_M', 'VIL_GRID_MARGIN_M',
                'ZDRCOL_ZDR_FIELD', 'ZDRCOL_ZDR_MIN_DB', 'ZDRCOL_BELOW_FZL_M',
                'ZDRCOL_MIXED_PHASE_TOP_C', 'ZDRCOL_GRID_DZ_M',
                'ZDRCOL_GRID_DXY_M', 'ZDRCOL_Z_TOP_M', 'ZDRCOL_GRID_MARGIN_M',
                'ZDRCOL_USE_CONNECTIVITY', 'ZDRCOL_USE_NEG_GRADIENT',
                'ZDRCOL_GRAD_TOL_DB', 'ZDRCOL_MIN_VOXELS', 'ZDRCOL_REF_MIN_DBZ'),
}

_ATTR2MOD = {}
for _mod, _names in _EXPORTS.items():
    for _name in _names:
        _ATTR2MOD[_name] = _mod

# 子模块属性表(QC.step08_composite / QC.plotstyle / QC.step06_attenuation_park2005 …):
# 以 _EXPORTS 的键为主, 再补上"只作为模块用、函数不在上面导出表里"的那些。
_EXTRA_SUBMODS = (
    'step07_classification', 'pipeline', 'config', 'plotstyle', 'step10_metrics',
    'wjd_cmaps',          # 除反射率外的色标(ZDR/KDP/相关系数), 导入即注册
    # 老名字: core/csu_function.py 的内容已并入 QC/plotstyle.py;
    # 这里保留 QC.csu_function 子模块属性, 让 `from QC import csu_function` 仍可用
    # (旧脚本里的 `import csu_function as cs` 的替代写法)
    'csu_function',
)
_SUBMODS = {m.lstrip('.'): m for m in _EXPORTS}
for _m in _EXTRA_SUBMODS:
    _SUBMODS.setdefault(_m, '.' + _m)
del _mod, _names, _name, _m

__all__ = [n for _names in _EXPORTS.values() for n in _names]


def __getattr__(name):
    """按需加载: QC.xxx 或 from QC import xxx 时才 import 对应子模块。"""
    if name in _SUBMODS:                      # QC.config / QC.step08_composite 这类子模块属性
        mod = _importlib.import_module(_SUBMODS[name], __name__)
        globals()[name] = mod
        return mod
    mod_name = _ATTR2MOD.get(name)
    if mod_name is None:
        raise AttributeError('module %r has no attribute %r' % (__name__, name))
    value = getattr(_importlib.import_module(mod_name, __name__), name)
    globals()[name] = value                   # 缓存, 后续直接命中, 不再走 __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_ATTR2MOD) | set(_SUBMODS))
