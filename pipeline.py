# -*- coding: utf-8 -*-
"""
QC/pipeline.py — 处理流程编排(把各部分按顺序串起来)
=================================================================
三个入口:
    prepare_radar_products(radar, snd)   —— 在**已读好的** radar 上跑完整条链
                                            (products/ 批量画图走这个, 省 I/O)
    process_radar(file, sounding)        —— 从文件开始: 读探空 → 读雷达
                                            → prepare_radar_products(**不含 FHC**)
    cr_from_file(file, sounding, no_fhc=False)
                                         —— 上面全部 + 组合反射率, 直接返回可绘图的
                                            cr_radar(单层) 和 radar(多层)

顺序(每一步的产物是下一步的输入, 不要随意调换):
    QC 门控 → 逐射线系统相位(平滑前, ρhv 判据) → KDP/φdp → 温度场 → ZPHI 衰减订正

★ 2026-10-07: 链上 KDP 已从 CSU Bringi 换成 **zch_phi 双尺度方案**
  (轻 0.6 km / 重 1.5 km 平滑 φdp -> 对**平滑后**的 φdp 最小二乘拟合, 按 40 dBZ 阈值切换),
  实现全在 QC/zch_phi.py, step04 只做接线。字段名仍是 kdp_dual / phidp_heavy
  (FHC 与精简缓存按此名取), 另附带 kdp_dual / phidp_heavy 等原生名。

FHC(模糊逻辑粒子识别)已于 2026-09-13 从 process_radar **剥离**, 改为独立函数:
    run_fhc(radar)                       —— 对处理好的 radar 就地添加 FH 字段
    process_radar_fhc(file, sounding)    —— 一步到"处理 + FHC"(替代旧的 process_radar 默认行为)
想一步到位(处理 + FHC)就用 process_radar_fhc / cr_from_file(no_fhc=False);
只想快出衰减订正结果、不依赖 csu_radartools, 就用 process_radar —— 此时 radar 里
没有 FH 字段, 指标(5)(冰雹/霰数量)用不了, 需要时再补 QC.run_fhc(radar) 即可。
"""

import time

import numpy as np

import pyart
import pycwr

from .step06_attenuation import zphi_attenuation_correction
from .step02_gatefilter import build_qc_gatefilter
from .step03_system_phase import remove_system_phase_per_layer
from .step04_kdp import compute_kdp_bringi, mask_reflectivity
from .step01_sounding import read_sounding
from .step08_composite import composite_radar
from .step03_system_phase import remove_system_phase_by_rho
from .step05_temperature import add_temperature_field
from . import config as C

# FHC(模糊逻辑粒子识别)独立入口 —— 见 run_fhc / process_radar_fhc:
#   add_fhc_field 依赖 csu_radartools, 而"不做 FHC"的流程不该被它拖累。
#   所以这里**不在模块级 import step07**, 而是在 run_fhc 里按需惰性加载。


def prepare_radar_products(radar, snd, verbose=True):
    """
    共用处理链(62层/单层通吃), 在 radar 对象上就地添加/替换:
      QC mask → 逐射线系统相位(平滑前, ρhv 判据) → Bringi KDP/φdp → 温度场
      → ZPHI 衰减订正 → cor_z / cor_zdr / pia / spec_at
      (★ 2026-09-25 起链上**不再**做 φdp 零基线对齐 —— 对本实现是空操作, 见下方注释)
    系统相位用 QC/step03_system_phase.py 的 ρhv 判据(前 30 个连续 ρhv>0.97 的门, 取 φdp 平均);
    这是本项目唯一实现 —— 早先那个同名旧法(前 40 门中位数)已删除。
    参数: radar = pyart Radar(含 reflectivity/differential_phase/...), snd = read_sounding() 结果
    返回: radar(带 kdp_dual, phidp_heavy, temperature, spec_at, pia, cor_z, cor_zdr)

    关于字段名(重要):
      衰减订正是**论文版 Park2005**(本项目唯一实现, 不调 pyart), 它自己的字段带 `_park`
      后缀(spec_at_park / pia_park / cor_z_park / cor_zdr_park)。
      本函数末尾会把它们**别名**成无后缀的 spec_at / pia / cor_z / cor_zdr —— 数据是同一份
      ndarray(不额外占内存), 只是换个名字, 好让指标函数、出图、老脚本直接可用。
    """
    t0 = time.time()

    # --- QC ---
    gf = build_qc_gatefilter(radar, verbose=verbose)
    mask_array = gf.gate_included

    # --- 取原始数据 ---
    ref_raw = radar.fields['reflectivity']['data'].data.copy()
    phi_raw = radar.fields['differential_phase']['data'].data.copy()
    rho_raw = radar.fields['cross_correlation_ratio']['data'].data.copy()
    ref_raw[~mask_array] = np.nan
    phi_raw[~mask_array] = np.nan
    rho_raw[~mask_array] = np.nan

    # --- ★ 系统相位(2026-10-08 用户要求: 链上要用处理好的 my_phi) ---
    #   逐层扣掉(不是整卷一次), 结果写进 radar 字段 `my_phi`, 之后 **KDP 与 ZPHI 都用它**。
    #   为什么逐层: 见 step03.remove_system_phase_per_layer 的说明(整卷一次时"回退值"会跨层污染)。
    #   开关: config.PIPE_SYSPHASE(False = 回到"不扣", phi_corr = phi_raw)。
    if getattr(C, 'PIPE_SYSPHASE', True):
        my_phi, sp_used = remove_system_phase_per_layer(
            radar, phi_raw, mask_array, rho_raw, refl=ref_raw,
            n_gates=getattr(C, 'PIPE_SYSPHASE_N_GATES', 50),
            rho_min=getattr(C, 'PIPE_SYSPHASE_RHO_MIN', 0.90),
            stat=getattr(C, 'PIPE_SYSPHASE_STAT', 'max'),
            refl_min=getattr(C, 'PIPE_SYSPHASE_REFL_MIN', 0.0),
            verbose=verbose)
        radar.add_field('my_phi', {'data': np.ma.masked_where(~mask_array, my_phi),
                                   'units': 'deg',
                                   'long_name': 'phiDP after per-sweep system phase removal'},
                        replace_existing=True)
        phi_corr = my_phi
        phi_field_for_zphi = 'my_phi'
    else:
        phi_corr = phi_raw
        phi_field_for_zphi = 'differential_phase'

    # --- KDP(Bringi) ---
    kd_lin, dp_lin = compute_kdp_bringi(radar, phi_corr, ref_raw, verbose=verbose)

    # --- reflectivity 掩膜(-30 填充, 供 zphi/画图) ---
    mask_reflectivity(radar, ref_raw)

    # --- 温度场 ---
    fzl_for_corr = add_temperature_field(radar, snd, verbose=verbose)

    # --- 衰减订正(唯一实现 = 用户口径; 2026-10-06 起 Park2005/ERAD 已删除) ---
    # 产出 `_user` 后缀字段, 下游统一用无后缀别名。
    #   链路: 扣系统相位 + zch_phi 双尺度滤波 + 末端ρhv>0.9取9门平均当总相位差,
    #         复刻 pyart 公式算 PIA; 系数用 pyart 自动查表 X 波段 0.31916。
    # ★ 只订正反射率: cor_zdr_user 是原始 differential_reflectivity 的直通(只套 QC 掩膜),
    #   论文式(9)(12) 的差分订正代码已删除 —— 订正版只在雨区有值、融化层以上缺测,
    #   会让指标(7) ZDR 柱恒判无柱(见 config.py 说明)。
    zphi_attenuation_correction(radar, gf, fzl_for_corr,
                                phi_field=phi_field_for_zphi, verbose=verbose,
                                # ★ 提速: KDP 刚用同一份 φdp 算过平滑结果(phidp_heavy),
                                #   逐位相同, ZPHI 直接复用(省 ~40 s/帧)。两个开关都要开才复用:
                                #   PIPE_SYSPHASE(保证两份 φdp 相同) 与 ZPHI_USER_REUSE_PHIDP。
                                reuse_phidp=(bool(getattr(C, 'PIPE_SYSPHASE', True))
                                             and bool(getattr(C, 'ZPHI_USER_REUSE_PHIDP', True))))

    # --- 把衰减订正的产物另起一套"无后缀"名字 ---
    # 底层实现(用户口径)字段带 `_user` 后缀, 指标函数 / 出图 / 老脚本都默认找无后缀的
    # cor_z / cor_zdr …, 所以这里补一份别名(数据是同一份 ndarray, 不额外占内存)。
    _alias = {'spec_at_user': 'spec_at', 'pia_user': 'pia',
              'cor_z_user': 'cor_z', 'cor_zdr_user': 'cor_zdr'}
    for _src, _dst in _alias.items():
        if _src in radar.fields and _dst not in radar.fields:
            radar.fields[_dst] = radar.fields[_src]
    if verbose:
        print('  已将衰减订正产物(%s)别名到: %s'
              % (C.ATTEN_BACKEND,
                 ', '.join(d for s, d in _alias.items() if d in radar.fields)))

    # --- 统一 QC mask 应用到速度/ρhv ---
    mask = gf.gate_included
    for fname in ['velocity', 'cross_correlation_ratio']:
        if fname in radar.fields:
            radar.fields[fname]['data'] = np.ma.masked_where(~mask, radar.fields[fname]['data'])

    if verbose:
        print(f'  衰减订正完成, 处理耗时 {time.time()-t0:.1f}s')
    return radar


def run_fhc(radar, verbose=True):
    """
    【独立入口】FHC 模糊逻辑水凝物分类(CSU_HIDRO / csu_fhc_summer)。
    已从 process_radar 剥离 —— 想要 FH 字段时**单独调用**本函数:

        radar = QC.process_radar(FILE, SOUNDING)   # 先跑 QC/温度/衰减订正链
        radar = QC.run_fhc(radar)                  # 需要时再补 FHC(就地添加 FH 字段)

    前置条件: radar 已跑完 prepare_radar_products(必须有 cor_z / cor_zdr /
    kdp_dual / temperature 字段 —— 分类用的是**衰减订正后**的量)。
    依赖 csu_radartools, 只在真正调用时才加载。
    返回添加了 FH 字段的 radar。
    """
    from .step07_classification import add_fhc_field   # 惰性: 不做 FHC 就不加载
    if verbose: print('\n--- FHC 模糊逻辑分类 ---')
    return add_fhc_field(radar, verbose=verbose)


def process_radar(file_path, sounding_path, verbose=True, no_fhc=True):
    """
    雷达处理流程(62层体积扫描, 不 extract_sweeps), **不含 FHC**:
      读探空 → 读雷达 → QC → 逐射线系统相位(平滑前, ρhv 判据)
      → KDP → 温度 → ZPHI 衰减订正(φdp 零基线对齐 + 物理上限)

    参数
      no_fhc: 兼容旧签名的**废弃参数**(2026-09-13 起 FHC 已剥离, 本函数不再做分类)。
              显式传 no_fhc=False 只会打印一条提示, 行为不变 —— 要 FH 字段请改用
              process_radar_fhc() 或对返回值调 run_fhc(radar)。

    返回处理后的 radar 对象:
      含 cor_z, cor_zdr, kdp_dual, phidp_heavy, temperature, pia 等字段,
      **没有 FH 字段**(需要冰雹/霰分类时: radar = QC.run_fhc(radar) 补上)。
    """
    if no_fhc is False and verbose:
        print('  [提示] FHC 已从 process_radar 剥离(2026-09-13), 本次不做分类;\n'
              '         需要 FH 字段请改用 process_radar_fhc() 或事后调 run_fhc(radar)。')

    t0 = time.time()

    # --- 读取探空 ---
    if verbose: print('\n--- 读取探空 ---')
    snd = read_sounding(sounding_path)

    # --- 读取雷达(快速通道: fast_pa 向量化解码 + to_pyart 桥接, 与 pycwr 逐位一致) ---
    if verbose: print('\n--- 读取雷达(快速通道) ---')
    try:
        from .io import fast_read
        radar = fast_read.read_pa_radar(file_path, fields=fast_read.FAST_FIELDS_RUN,
                                        verbose=verbose)
    except Exception as e:
        # 快速通道对个别非常规格式失手时, 回退 pycwr 原路, 保证处理不中断
        if verbose: print(f'  [!] 快速读取失败({type(e).__name__}: {e}) -> 回退 pycwr 慢读')
        cwr_radar = pycwr.io.read_auto(file_path)
        radar = cwr_radar.ToPyartRadar()
    if verbose: print(f'  仰角层数: {radar.nsweeps}, nrays: {radar.nrays}, ngates: {radar.ngates}')

    # --- 共用处理链: QC → 逐射线系统相位(平滑前) → KDP → 温度 → 衰减订正 ---
    if verbose: print('\n--- QC / KDP / 系统相位 / 温度 / 衰减订正 ---')
    prepare_radar_products(radar, snd, verbose=verbose)

    if verbose: print(f'\n=== 处理完成, 总耗时 {time.time()-t0:.1f}s ===')
    return radar


def process_radar_fhc(file_path, sounding_path, verbose=True):
    """
    【一步到位版】process_radar 的全部流程 + FHC 模糊逻辑分类
    (等价于旧的 process_radar(no_fhc=False) 默认行为):

        radar = QC.process_radar_fhc(FILE, SOUNDING)   # 返回的 radar 含 FH 字段

    返回含 cor_z / cor_zdr / kdp_dual / temperature / FH 等字段的 radar。
    """
    radar = process_radar(file_path, sounding_path, verbose=verbose)
    return run_fhc(radar, verbose=verbose)


def cr_from_file(file_path, sounding_path, field='cor_z', ref_sweep=0,
                 fill_gaps=True, write_back=True, verbose=True, no_fhc=False):
    """
    从雷达文件一步到"组合反射率 radar"(最省事的入口):

        读探空 → 读文件 → 完整处理链(QC→系统相位→KDP→温度→衰减)
        → (no_fhc=False 时) FHC → 组合反射率(精确版)

    参数
      file_path / sounding_path: 雷达基数据文件 / 探空文件
      field:      组合用的字段, 默认 'cor_z'(衰减订正后); 也可 'reflectivity'
      ref_sweep:  CR 基准层(默认 0)
      write_back: 是否把 CR 也写回**处理好的原 radar**(默认 True)
      no_fhc:     跳过 FHC 分类(默认 False = 处理后调 run_fhc 补 FH 字段;
                  True 则完全不做分类, 不依赖 csu_radartools)
    返回
      (cr_radar, radar)
        cr_radar : 单层 slim radar(含 CR 字段), 直接 QC.plot_cr(cr_radar)
        radar    : 完整处理后的多层 radar(字段齐全, 可继续做指标/画其它图)

    用法
      import QC
      cr_radar, radar = QC.cr_from_file(FILE, SOUNDING)
      QC.plot_cr(cr_radar, out_path=r'D:\\...\\cr.png', hail_location=[[104.02, 27.15]])
    """
    radar = process_radar(file_path, sounding_path, verbose=verbose)
    if not no_fhc:
        radar = run_fhc(radar, verbose=verbose)
    if verbose: print('\n--- 组合反射率 ---')
    cr_radar, _comp_z, _comp_alt = composite_radar(
        radar, field=field, ref_sweep=ref_sweep, fill_gaps=fill_gaps,
        write_back=write_back, verbose=verbose)
    return cr_radar, radar
