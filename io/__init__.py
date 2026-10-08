# -*- coding: utf-8 -*-
"""
QC/io — 读入/写出: 探空、雷达文件解码、pycwr->pyart 转换、缓存读写。
=================================================================
本子包是 2026-09-21 目录重组时按 pyart 的"领域分包"方式切出来的
(pyart: io/ correct/ filters/ retrieve/ graph/ util/ map/)。

★ 对外用法完全不变: 老脚本照旧 `import QC; QC.process_radar(...)`,
  所有名字都由 QC/__init__.py 的惰性导出层统一暴露, 不需要知道文件搬到哪了。
"""
