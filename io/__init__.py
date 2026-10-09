# -*- coding: utf-8 -*-
"""
QC/io — 读入/写出: 探空、雷达文件解码、pycwr->pyart 转换、缓存读写。
=================================================================
本子包是 2026-09-21 目录重组时按 pyart 的"领域分包"方式切出来的
(pyart: io/ correct/ filters/ retrieve/ graph/ util/ map/)。

★ 对外用法完全不变: 老脚本照旧 `import QC; QC.process_radar(...)`,
  所有名字都由 QC/__init__.py 的惰性导出层统一暴露, 不需要知道文件搬到哪了。

★ 读取入口(2026-10-08): 都只认**全路径**
    from QC.io import read_base_data         # 基数据 .bin.zip -> pyart Radar
    from QC.io import read_qc_data           # QC 产物 .pkl.gz -> pyart Radar

    radar = read_base_data(r'F:\\cell1\\file\\radar_01\\Z_RADR_I_ZWN01_20250520110900_O_DOR-XPD-CAP-FMT.bin.zip')
    radar = read_qc_data(r'F:\\cell1\\file\\processed_qc\\radar_01\\20250520110900.pkl.gz')

  ★ 旧名字 read_pa_radar / load_qc_radar 保留为**别名**, 老脚本照旧能跑。
"""
from .fast_read import read_base_data, read_pa_radar
from .load_qc_radar import read_qc_data, load_qc_radar, dict_to_radar

# 对外主推的名字(旧名字仍可直接 import, 只是不进 __all__)
__all__ = ['read_base_data', 'read_qc_data', 'dict_to_radar']
