# -*- coding: utf-8 -*-
"""
QC/fast_texture.py —— 沿射线纹理的"跳过空窗"快速版(与 pyart 原版**逐位等价**)
================================================================================
pyart 原版: pyart.util.sigmath.texture_along_ray(radar, var, wind_size=7)

    for timestep in range(16678):                       # 一条射线一次
        ray = np.ma.std(rolling_window(fld[i, :], 7), 1)  # 7 点滑窗标准差
        tex[i, 3:-3] = ray
        tex[i, 0:3]  = np.ones(3) * ray[0]              # 左边缘照抄第 1 个窗
        tex[i, -3:]  = np.ones(3) * ray[-1]             # 右边缘照抄最后 1 个窗

为什么它慢
----------
威宁这卷数据 **97.6% 的门是空值(NaN)**。而窗口里只要夹着一个 NaN,
标准差必然还是 NaN —— 也就是 **97.8% 的滑窗是白算的**。
实测(16678 根 × 2000 库 × 4 个场): 原版 41.7 s。

快速版怎么快
------------
1) 用前缀和(一次累加, 不写循环)定位"窗口内一个 NaN 都没有"的窗口 —— 只占 2.2%;
2) 只对这些窗口调 np.ma.std;
3) 其余位置直接填 NaN / 标记为空(和原版算出来的一模一样), 然后照抄原版那三段写入。

实测 41.7 s -> 3.2 s(约 13 倍), 且**逐位一致**(数值、NaN 位置、掩码、连被掩格子底下的
脏数据都相同)。

为什么连"脏数据"也要复刻
------------------------
原版边缘那两行是 `np.ones(3) * ray[0]`。当 ray[0] 是"被掩掉的值"(np.ma.masked,
它的底层 data 是常数 1.0)时, 乘出来塞进 tex 的就是 **mask=True 且 data=1.0**。
实测这种格子正好 100026 个。门控只看"空不空", 所以这些脏数据不影响结果 ——
但既然本项目的验收标准是"逐位一致", 这里就一起复刻了。

用法
----
    import QC
    QC.install_fast_texture()          # 一般在 import QC.step02_gatefilter 时已自动装上
    QC.texture_along_ray_fast(radar, 'reflectivity')   # 单独调也可以
    QC.uninstall_fast_texture()        # 还原 pyart 原版(验证对拍时用)
    print(QC.fast_texture_status())

开关(环境变量)
--------------
    QC_FAST_TEXTURE=0          关掉快速版, 用 pyart 原版(对拍/排查时用)
    QC_FAST_TEXTURE_CHUNK=1024 一批处理多少根射线(默认 1024; 小一点省内存)

★ 本文件不动 pyart 源码一个字节 —— 只在运行时把 pyart 里那个函数名换成快速版。
  注意 gatefilter.py 里是 `from ..util import texture_along_ray`(已经把函数名绑到
  自己的模块全局了), 所以**必须**改 pyart.filters.gatefilter 里的那个名字,
  只改 sigmath 里的定义对它无效。
"""

import os
import time

import numpy as np

# ------------------------------------------------------------------ 模块级配置
_ENV = os.environ.get('QC_FAST_TEXTURE', '1').strip().lower()
ENABLE = _ENV not in ('0', 'false', 'no', 'off')
CHUNK = int(os.environ.get('QC_FAST_TEXTURE_CHUNK', '1024') or 1024)

# 被我们替换过的原函数 {模块名.函数名: 原函数}
_ORIGINALS = {}
_INSTALLED = False

# 运行统计(只用于报日志, 不影响计算)
STATS = {'calls': 0, 'sec': 0.0, 'rows': 0, 'win_total': 0, 'win_used': 0}

_rolling_window = None


def _rw():
    """懒加载 pyart 的 rolling_window(避免 import QC 时就拉起 pyart)。"""
    global _rolling_window
    if _rolling_window is None:
        from pyart.util.sigmath import rolling_window
        _rolling_window = rolling_window
    return _rolling_window


def _call_original(radar, var, wind_size):
    """退化情形(窗口比一行还长 / 空数组 / 非浮点)交回 pyart 原版, 保证行为完全一致。"""
    for key in ('pyart.util.sigmath.texture_along_ray',
                'pyart.util.texture_along_ray',
                'pyart.filters.gatefilter.texture_along_ray'):
        if key in _ORIGINALS:
            return _ORIGINALS[key](radar, var, wind_size)
    from pyart.util.sigmath import texture_along_ray as _orig
    return _orig(radar, var, wind_size)


# =============================================================================
# 快速版(与 pyart 原版逐位等价)
# =============================================================================
def texture_along_ray_fast(radar, var, wind_size=7, chunk=None):
    """
    沿射线纹理 —— pyart.util.sigmath.texture_along_ray 的快速等价版。

    参数/返回与原版完全一致(radar.fields[var]['data'] 的 2 维沿射线滑窗标准差,
    长度不足 half_wind 的两端照抄边缘窗口)。
    多出的 chunk 只是分批大小, 不影响结果。
    """
    rw = _rw()
    t0 = time.perf_counter()

    half_wind = int((wind_size - 1) / 2)
    fld = radar.fields[var]['data']
    arr = np.asarray(fld)
    if arr.ndim != 2 or arr.shape[1] < wind_size or arr.shape[0] <= 0 \
            or arr.dtype.kind != 'f':
        return _call_original(radar, var, wind_size)

    if chunk is None or chunk <= 0:
        chunk = CHUNK

    nray, ngate = arr.shape
    nwin = ngate - wind_size + 1
    tex = np.ma.zeros(fld.shape)

    # 与 pyart 完全相同的取数方式: as_strided 会把掩码丢掉, 拿到的是底层数据
    # (被掩的位置是 NaN) —— 这里必须一致, 否则"哪些窗口是空窗"会判断错。
    raw = np.array(fld, copy=False, subok=False)

    used = 0
    for s in range(0, nray, chunk):
        e = min(s + chunk, nray)
        blk = fld[s:e, :]

        # --- ① 前缀和: 一减就知道每个窗口里有几个非 NaN ---
        fin = ~np.isnan(np.array(raw[s:e], copy=False, subok=False))
        P = np.zeros((fin.shape[0], ngate + 1), dtype=np.int32)
        np.cumsum(fin, axis=1, dtype=np.int32, out=P[:, 1:])
        ok = (P[:, wind_size:] - P[:, :nwin]) == wind_size      # (m, nwin) bool

        # --- ② 只对这些窗口真算标准差; 其余保持 NaN ---
        rayd = np.full(ok.shape, np.nan, dtype=np.float64)
        if ok.any():
            sub = np.ma.std(rw(blk, wind_size)[ok], axis=-1)
            rayd[ok] = np.asarray(sub.data)
            # 理论上不会发生(有限值的方差不会为负), 真发生就退回"被掩"处理,
            # 与原版 np.ma.sqrt 的域检查结果一致。
            extra = np.ma.getmaskarray(sub)
            if extra.any():
                ok = ok.copy()
                ok[np.nonzero(ok)[0][extra]] = False
            used += int(ok.sum())
        ray = np.ma.array(rayd, mask=~ok, copy=False)

        # --- ③ 与原版逐字相同的三段写入 ---
        tex[s:e, half_wind:-half_wind] = ray
        if half_wind:
            ones = np.ones((e - s, half_wind))
            left = ones * ray[:, :1]
            # 原版 np.ones(3) * ray[0]: ray[0] 被掩时, np.ma.masked 的底层 data 是 1.0,
            # 乘出来塞进 tex 的就是 data=1.0 + mask=True。这里照抄, 连脏数据都一致。
            left.data[np.ma.getmaskarray(ray[:, :1])[:, 0]] = 1.0
            tex[s:e, 0:half_wind] = left
            right = ones * ray[:, -1:]
            right.data[np.ma.getmaskarray(ray[:, -1:])[:, 0]] = 1.0
            tex[s:e, -half_wind:] = right

    STATS['calls'] += 1
    STATS['sec'] += time.perf_counter() - t0
    STATS['rows'] += nray
    STATS['win_total'] += nray * nwin
    STATS['win_used'] += used
    return tex


# =============================================================================
# 装上 / 卸下
# =============================================================================
def install_fast_texture(verbose=False):
    """把 pyart 里的 texture_along_ray 换成快速版(幂等)。返回是否已生效。"""
    global _INSTALLED
    if not ENABLE:
        return False
    if _INSTALLED:
        return True

    import pyart.util.sigmath as _sigmath
    import pyart.util as _util
    import pyart.filters.gatefilter as _gatefilter

    hit = 0
    for mod in (_sigmath, _util, _gatefilter):
        cur = getattr(mod, 'texture_along_ray', None)
        if cur is None or cur is texture_along_ray_fast:
            continue
        _ORIGINALS[mod.__name__ + '.texture_along_ray'] = cur
        setattr(mod, 'texture_along_ray', texture_along_ray_fast)
        hit += 1
    _INSTALLED = hit > 0
    if _INSTALLED and verbose:
        print('  [QC] 沿射线纹理: 已启用快速版(跳过空窗, 与原版逐位等价)')
    return _INSTALLED


def uninstall_fast_texture():
    """还原成 pyart 原版(验证对拍时用)。返回还原了几个。"""
    global _INSTALLED
    hit = 0
    for key, orig in list(_ORIGINALS.items()):
        mod_name, _, attr = key.rpartition('.')
        try:
            mod = __import__(mod_name, fromlist=[attr])
        except Exception:
            continue
        if getattr(mod, attr, None) is texture_along_ray_fast:
            setattr(mod, attr, orig)
            hit += 1
    _ORIGINALS.clear()
    _INSTALLED = False
    return hit


def fast_texture_status():
    """当前状态, 给日志/排查用。"""
    d = dict(installed=bool(_INSTALLED), enabled=bool(ENABLE), chunk=int(CHUNK),
             patched=sorted(_ORIGINALS), **STATS)
    if d['win_total']:
        d['win_used_pct'] = round(100.0 * d['win_used'] / d['win_total'], 2)
    return d


def fast_texture_status_lines():
    """fast_texture_status() 的人话版(几行字符串)。"""
    d = fast_texture_status()
    if not d['enabled']:
        return ['  沿射线纹理: 快速版已按环境变量关闭, 走 pyart 原版']
    if not d['installed']:
        return ['  沿射线纹理: 尚未装上快速版(还没跑到 QC 门控)']
    out = ['  沿射线纹理: 快速版已生效 —— 已替换 %d 处 (%s)'
           % (len(d['patched']), ', '.join(d['patched']))]
    if d['calls']:
        out.append('    已算 %d 个场 / %d 根射线, 占 %.2f s; '
                   '滑窗 %d 个里只真算了 %d 个 (%.2f%%)'
                   % (d['calls'], d['rows'], d['sec'], d['win_total'],
                      d['win_used'], d.get('win_used_pct', 0.0)))
    return out
