# -*- coding: utf-8 -*-
"""
fast_pa.py —— 威宁 X 波段相控阵(PA / DOR-XPD-CAP-FMT) 体扫的"快速读取器"

【它解决什么】
    pycwr.io.read_auto(fp).ToPyartRadar() 对单个体扫要 **38.5 s**。
    实测(2025-05-20 11:09 ZWN01, 16678 根 x 2000 库 x 12 个矩)这 38.5 s 花在:
        逐射线 Python 循环解析           11.5 s   (200136 次 _decode_moment_payload)
        把 12 个小数组拼成 12 个大矩阵     4.0 s   (每个 127 MB)
        建 PRD 对象(含 62 次经纬度投影)   11.7 s   (其中 aeqd 投影 8.3 s)
        再导出成 pyart Radar             11.0 s
    同一份 1.6 GB 数据被**物化三遍**, 峰值内存 4.8 GB, 实测缺页 485 万次。
    而真正的磁盘 IO(解压 421 MB)只要 1.1~2.4 s —— 所以**慢的不是硬盘**。

【为什么能快】
    PA 文件的射线是**严格定长**的( _probe8 已核验 ):
        · 文件头区 = 416 + 640*BeamNumber(8) + 256*CutNumber(62) = 21408 字节
        · 之后每根射线 26512 字节 = 128(射线头) + 12*(32(矩头) + 2000*库宽)
        · 数据区 442167136 / 26512 = 16678 根, **整除无余数**
        · 抽样 201 根(横跨全卷): 12 个矩的 DataType/顺序/Scale/Offset 全部一致
    于是可以用**一个 numpy 结构化 dtype 一次性 frombuffer 解析整卷**,
    把 pycwr 那 16678 x 12 次 Python 层循环压成 1 次 C 层循环。

【本模块提供】
    read_pa(path, fields=..., ray_limit=...)    -> dict (纯 numpy, 不 import pyart)
    parse_header(path)                          -> dict (只读文件头, 极快)
    CLI:
        python fast_pa.py <file> --info                 看文件头
        python fast_pa.py <file> --selfcheck 1500       与"逐射线慢解码"逐位比对
        python fast_pa.py <file> --bench                整卷计时 + 峰值内存
        python fast_pa.py <file> --fields reflectivity,velocity

【注意】
    · 只依赖 numpy + 标准库 -> **可以直接用环境目录里的 python.exe 跑**, 不触发渲染崩溃问题。
      但一旦要 import pyart / matplotlib, 必须先补 PATH (见本文件顶部 _ensure_env_dll_path)。
    · 本模块**只读文件, 不写任何东西**, 可以放心对正在跑的数据目录使用。
"""

from __future__ import annotations

import ctypes
import datetime
import gc
import os
import struct
import sys
import time
import zipfile

import numpy as np

# ---------------------------------------------------------------------------
# 0. 需要 pyart 时, 先把 conda 环境的 DLL 目录补进 PATH
#    (否则 freetype/agg 解析到错版本, C 层直接 __fastfail, Python 抓不到)
# ---------------------------------------------------------------------------
def _ensure_env_dll_path():
    env = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    if os.path.basename(env).startswith('envs'):
        pass
    else:
        # sys.executable 形如 <root>\envs\wradlib\python.exe -> 取 envs\wradlib
        pass
    cands = [env, os.path.join(env, 'Library', 'bin'),
             os.path.join(env, 'Library', 'mingw-w64', 'bin'),
             os.path.join(env, 'Library', 'usr', 'bin'),
             os.path.join(env, 'DLLs')]
    os.environ['PATH'] = os.pathsep.join(cands + [os.environ.get('PATH', '')])
    for d in cands:
        try:
            if os.path.isdir(d):
                os.add_dll_directory(d)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1. 格式常量
# ---------------------------------------------------------------------------
FIXED_HDR_SIZE = 416
SITE_CFG_POS = 32
TASK_CFG_POS = 160
BEAM_CFG_POS = 416
BEAM_CFG_SIZE = 640
CUT_CFG_SIZE = 256
RADIAL_HDR_SIZE = 128
MOMENT_HDR_SIZE = 32
VALID_CODE_MIN = 5                    # <5 视为特殊值(0阈值下/1距离折叠/2未扫描/3未知/4保留)
REFLECTIVITY_FIELDS = ('dBT', 'dBZ', 'Zc')

# pycwr dtype_PA.flag2Product 里我们关心的部分
FLAG2PRODUCT = {1: 'dBT', 2: 'dBZ', 3: 'V', 4: 'W', 5: 'SQI', 6: 'CPA', 7: 'ZDR',
                8: 'LDR', 9: 'CC', 10: 'PhiDP', 11: 'KDP', 12: 'CP', 13: 'FLAG',
                14: 'HCL', 15: 'CF', 16: 'SNRH', 17: 'SNRV', 32: 'Zc', 33: 'Vc',
                34: 'Wc', 35: 'ZDRc', 0: 'Flag'}

# 矩名 -> pyart 里的字段名。
#   ★ 2026-09-15: 之前自己维护 12 条, 漏了 SNRV / Wc / SQI / CPA / LDR / HCL / CF,
#     导致 ngates_per_moment 之类的推导里冒出矩名当键（'SNRV' / 'Wc'）。
#     现在先放 14 条本地兜底(覆盖本链会遇到的全部矩), 再**懒加载** pycwr 那份权威表覆盖。
#   ★ 注意: pycwr 会连带 import pyart+matplotlib —— 那个导入在缺 DLL 时是 C 层崩溃,
#     try/except 抓不住。所以这里是**懒加载**(首次调用 _sync_moment_map() 时才 import),
#     保证"只读文件"的 fast_pa 自己跑时不碰 pycwr。
MOMENT2PYART = {'dBT': 'total_power', 'dBZ': 'reflectivity', 'V': 'velocity',
                'W': 'spectrum_width', 'ZDR': 'differential_reflectivity',
                'CC': 'cross_correlation_ratio', 'PhiDP': 'differential_phase',
                'KDP': 'specific_differential_phase',
                'SNRH': 'horizontal_signal_noise_ratio',
                'SNRV': 'vertical_signal_noise_ratio',
                'Zc': 'corrected_reflectivity', 'Vc': 'corrected_velocity',
                'Wc': 'spectrum_width_corrected',
                'ZDRc': 'corrected_differential_reflectivity'}
PYART2MOMENT = {v: k for k, v in MOMENT2PYART.items()}

_MAP_SYNCED = False


def _sync_moment_map():
    """
    首次使用时把 pycwr 的 CINRAD_field_mapping(29 条) 并进来, 作为权威表。

    设计取舍: fast_pa 的本职是"只依赖 numpy 的快速读取", 所以绝不能在 import 时就
    去拉 pycwr(它会连带拉 pyart+matplotlib)。这里做成懒加载 + 只在导入成功时覆盖,
    失败(没装 pycwr / DLL 不全)就继续用上面那 14 条本地表 —— 本链用到的矩它都覆盖了。
    """
    global _MAP_SYNCED
    if _MAP_SYNCED:
        return
    _MAP_SYNCED = True
    try:
        from pycwr.configure.default_config import CINRAD_field_mapping as m
    except Exception:
        return
    MOMENT2PYART.update(m)
    PYART2MOMENT.update({v: k for k, v in m.items()})

# 射线头 (128 字节, 严格 packed)
RADIAL_HDR_DTYPE = np.dtype([
    ('RadialState', '<i4'), ('SpotBlank', '<i4'), ('SequenceNumber', '<i4'),
    ('RadialNumber', '<i4'), ('ElevationNumber', '<i4'),
    ('Azimuth', '<f4'), ('Elevation', '<f4'),
    ('Seconds', '<i8'), ('MicroSeconds', '<i4'),
    ('LengthOfData', '<i4'), ('MomentNumber', '<i4'),
    ('ScanBeamIndex', '<i2'), ('HorizontalEstimatedNoise', '<i2'),
    ('VerticalEstimatedNoise', '<i2'), ('PRFFLAG', '<u4'),
    ('Reserved04', 'V70')])

# 矩头 (32 字节)
MOMENT_HDR_DTYPE = np.dtype([
    ('DataType', '<i4'), ('Scale', '<i4'), ('Offset', '<i4'),
    ('BinLength', '<i2'), ('Flags', '<i2'), ('Length', '<i4'), ('Reserved05', 'V12')])

# 波束配置块 (640 字节) —— 只要 SubPulseBandWidth
BEAM_CFG_DTYPE = np.dtype([
    ('BeamIndex', '<i4'), ('BeamType', '<i4'), ('SubPulseNumber', '<i4'),
    ('TxBeamDirection', '<f4'), ('TxBeamWidthH', '<f4'), ('TxBeamWidthV', '<f4'),
    ('TxBeamGain', '<f4'), ('Reserved00', 'V100'),
    ('SubPulseStrategy', '<i4'), ('SubPulseModulation', '<i4'),
    ('SubPulseFrequency', '<f4'), ('SubPulseBandWidth', '<f4'),
    ('SubPulseWidth', '<i4'), ('Reserved01', 'V492')])

# 扫描层配置块 (256 字节)
CUT_CFG_DTYPE = np.dtype([
    ('CutIndex', '<i2'), ('TxBeamIndex', '<i2'), ('Elevation', '<f4'),
    ('TxBeamGain', '<f4'), ('RxBeamWidthH', '<f4'), ('RxBeamWidthV', '<f4'),
    ('RxBeamGain', '<f4'), ('ProcessMode', '<i4'), ('WaveForm', '<i4'),
    ('N1_PRF_1', '<f4'), ('N1_PRF_2', '<f4'), ('N2_PRF_1', '<f4'), ('N2_PRF_2', '<f4'),
    ('UnfoldMode', '<i4'), ('Azimuth', '<f4'), ('StartAngle', '<f4'),
    ('EndAngle', '<f4'), ('AngleResolution', '<f4'), ('ScanSpeed', '<f4'),
    ('LogResolution', '<f4'), ('DopplerResolution', '<f4'),
    ('MaximumRange', '<i4'), ('MaximumRange2', '<i4'), ('StartRange', '<i4'),
    ('Sample_1', '<i4'), ('Sample_2', '<i4'), ('PhaseMode', '<i4'),
    ('AtmosphericLoss', '<f4'), ('NyquistSpeed', '<f4'),
    ('MomentsMask', '<i8'), ('MomentsSizeMask', '<i8'), ('MiscFilterMask', '<i4'),
    ('SQIThreshold', '<f4'), ('SIGThreshold', '<f4'), ('CSRThreshold', '<f4'),
    ('LOGThreshold', '<f4'), ('CPAThreshold', '<f4'), ('PMIThreshold', '<f4'),
    ('DPLOGThreshold', '<f4'), ('ThresholdsReserved', 'V4'),
    ('dBTMask', '<i4'), ('dBZMask', '<i4'), ('Velocity', '<i4'),
    ('SpectrumWidthMask', '<i4'), ('ZDRMask', '<i4'), ('MaskResvered', 'V12'),
    ('ScanSync', 'V4'), ('Direction', '<i4'),
    ('GroundClutterClassifierType', '<i2'), ('GroundClutterFilterType', '<i2'),
    ('GroundClutterFilterNotchWidth', '<i2'), ('GroundClutterFilterWindow', '<i2'),
    ('Reserved', 'V44')])

assert RADIAL_HDR_DTYPE.itemsize == RADIAL_HDR_SIZE, RADIAL_HDR_DTYPE.itemsize
assert MOMENT_HDR_DTYPE.itemsize == MOMENT_HDR_SIZE
assert BEAM_CFG_DTYPE.itemsize == BEAM_CFG_SIZE, BEAM_CFG_DTYPE.itemsize
assert CUT_CFG_DTYPE.itemsize == CUT_CFG_SIZE, CUT_CFG_DTYPE.itemsize


# ---------------------------------------------------------------------------
# 2. 内存 / 时间小工具
# ---------------------------------------------------------------------------
class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


def _proc_mem():
    """(当前工作集 MB, 峰值工作集 MB, 缺页次数)

    ★ 2026-09-18 修: 原来调 `ctypes.windll.psapi.GetProcessMemoryInfo`, 在 Win10/11 上
      **静默返回 0**(不报错也不填结构体) -> 所有"峰值内存"都打印 0 MB。
      必须用 kernel32 的 K32GetProcessMemoryInfo, 并显式写 argtypes/restype。
    """
    import ctypes.wintypes as wt

    f = ctypes.windll.kernel32.K32GetProcessMemoryInfo
    f.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), wt.DWORD]
    f.restype = wt.BOOL
    p = _PROCESS_MEMORY_COUNTERS()
    p.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
    if not f(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb):
        return 0.0, 0.0, 0
    return p.WorkingSetSize / 1048576.0, p.PeakWorkingSetSize / 1048576.0, p.PageFaultCount


def available_ram_gb():
    class MS(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1073741824.0, m.ullTotalPhys / 1073741824.0


# ---------------------------------------------------------------------------
# 3. 打开文件 (zip / bz2 / gz / 裸文件)
# ---------------------------------------------------------------------------
def _open_payload(path):
    """返回 (fileobj, 解压后总字节, 关闭函数)"""
    with open(path, 'rb') as fh:
        magic = fh.read(4)
    if magic[:2] == b'PK':
        z = zipfile.ZipFile(path, 'r')
        members = [m for m in z.infolist() if not m.is_dir()]
        if len(members) != 1:
            z.close()
            raise ValueError('zip 里应当只有 1 个成员, 实际 %d 个' % len(members))
        return z.open(members[0], 'r'), members[0].file_size, z.close
    if magic[:3] == b'BZh':
        import bz2
        f = bz2.BZ2File(path, 'rb')
        return f, _stream_size(f), f.close
    if magic[:2] == b'\x1f\x8b':
        import gzip
        f = gzip.GzipFile(path, 'rb')
        return f, _stream_size(f), f.close
    f = open(path, 'rb')
    return f, os.path.getsize(path), f.close


def _stream_size(f):
    pos = f.tell()
    f.seek(0, os.SEEK_END)
    n = f.tell()
    f.seek(pos)
    return n


# ---------------------------------------------------------------------------
# 4. 只读文件头
# ---------------------------------------------------------------------------
def parse_header(path):
    """只读文件头 + 第 0 根的矩模板, 毫秒级。"""
    f, total, close = _open_payload(path)
    try:
        fixed = f.read(FIXED_HDR_SIZE)
        if len(fixed) < FIXED_HDR_SIZE:
            raise ValueError('文件头不足 416 字节')
        magic, major, minor = struct.unpack_from('<ihh', fixed, 0)
        if fixed[:4] != b'RSTM':
            raise ValueError('不是 RSTM 开头的 PA 文件: %r' % fixed[:4])
        site = fixed[SITE_CFG_POS:SITE_CFG_POS + 40]
        lat, lon, height, ground, freq, bwh, bwv = struct.unpack_from('<ffiifff', fixed, 72)
        pol, scan_type, beam_num, cut_num, ray_order = struct.unpack_from('<iiiii', fixed, 320)

        beam_buf = f.read(BEAM_CFG_SIZE * beam_num)
        cut_buf = f.read(CUT_CFG_SIZE * cut_num)
        beam_cfg = np.frombuffer(beam_buf, dtype=BEAM_CFG_DTYPE)
        cut_cfg = np.frombuffer(cut_buf, dtype=CUT_CFG_DTYPE)

        radial_buf = f.read(RADIAL_HDR_SIZE)
        rh = RADIAL_HDR_DTYPE_val(radial_buf)
        moment_num = int(rh['MomentNumber'])

        head = RADIAL_HDR_SIZE
        moments = []
        pos = head
        for _ in range(moment_num):
            mb = f.read(MOMENT_HDR_SIZE)
            mh = MOMENT_HDR_DTYPE_val(mb)
            body = int(mh['Length'])
            f.read(body)          # ★ 必须跳过该矩的数据体, 否则下一根矩头会读错位
            moments.append(dict(DataType=int(mh['DataType']),
                                Scale=int(mh['Scale']), Offset=int(mh['Offset']),
                                BinLength=int(mh['BinLength']), Length=body))
            pos += MOMENT_HDR_SIZE + body

        hdr_area = FIXED_HDR_SIZE + BEAM_CFG_SIZE * beam_num + CUT_CFG_SIZE * cut_num
        ray_bytes = pos
        nray, rem = divmod(total - hdr_area, ray_bytes)
        for m in moments:
            m['name'] = FLAG2PRODUCT.get(m['DataType'])
            m['nbins'] = m['Length'] // m['BinLength']
            m['pyart'] = MOMENT2PYART.get(m['name'])

        return dict(path=path, file_size=total, uncompressed=total,
                    magic=magic, version=(major, minor),
                    site_code=site[:8].split(b'\x00')[0].decode('utf-8', 'ignore'),
                    site_name=site[8:40].split(b'\x00')[0].decode('utf-8', 'ignore'),
                    latitude=lat, longitude=lon, altitude=height, ground=ground,
                    frequency_mhz=freq, beam_width=(bwh, bwv),
                    polarization=pol, scan_type=scan_type,
                    beam_number=beam_num, cut_number=cut_num, ray_order=ray_order,
                    header_area=hdr_area, ray_bytes=ray_bytes, nrays=nray,
                    remainder=rem, fixed_stride=(rem == 0),
                    moments=moments, cut_cfg=cut_cfg, beam_cfg=beam_cfg)
    finally:
        close()


def RADIAL_HDR_DTYPE_val(buf):
    return np.frombuffer(buf, dtype=RADIAL_HDR_DTYPE, count=1)[0]


def MOMENT_HDR_DTYPE_val(buf):
    return np.frombuffer(buf, dtype=MOMENT_HDR_DTYPE, count=1)[0]


def _unpack(dt, buf):           # 占位, 保持上面写法可读
    return np.frombuffer(buf, dtype=dt, count=1)[0], dt.itemsize


# ---------------------------------------------------------------------------
# 5. 主读取
# ---------------------------------------------------------------------------
def read_pa(path, fields=None, ray_limit=None, exact=True, check_layout=True,
            verbose=False):
    """
    快速读取一个 PA 体扫。

    path        : .zip / .bz2 / .bin 路径
    fields      : None=全解; 或可迭代, 元素可用"矩名"(dBZ/ZDR/...) 或
                  "pyart 名"(reflectivity/velocity/...)。**指定后只解这些** ——
                  cr 阶段只要 reflectivity, 用这个能省一大半时间和内存。
    ray_limit   : 只读前 N 根(调试用)
    exact       : True = 与 pycwr 逐位一致(经 float64 中转);
                  False = 全程 float32(略省内存/时间, 末位可能有 1 ulp 差)
    check_layout: 校验整卷矩顺序一致(抽 200 根), 显著不一致则报错

    返回 dict, 关键键:
        fields            {pyart名: (nrays, 该矩自己的 nbins) float32}
        fields_moment     {矩名: 同上}
        nbins_by_moment   {矩名: 该矩的库数}   —— 威宁 12 个矩库数一致
        ngates_per_moment {pyart名: 该矩的库数}
        azimuth, elevation, seconds, microseconds,
        radial_number, elevation_number, radial_state, sequence_number
        range, ngates, nrays, nsweeps, sweep_start, sweep_end
        latitude, longitude, altitude, frequency_ghz, nyquist, unambiguous_range
        sitename, scan_type, moments, timings

    ★ ngates = 全部矩里的**最大**库数(威宁各矩一致, 所以它就等于每一矩的库数)。
      每个矩都按**它自己的 nbins** 出数组, 不强行铺到统一列宽;
      威宁版要求各矩库数一致, 遇到不一致会在 fast_read.to_pyart() 处直接报错。
    """
    t_all = time.time()
    timings = {}
    _sync_moment_map()          # 首次调用时并入 pycwr 的权威矩名映射表(懒加载)

    if fields is not None:
        want = set()
        for f in fields:
            want.add(PYART2MOMENT.get(f, f))
    else:
        want = None

    t0 = time.time()
    hdr = parse_header(path)
    timings['header'] = time.time() - t0
    if not hdr['fixed_stride']:
        raise ValueError('射线不是定长(余 %d 字节), 本快速读取器不适用 → 请回退 pycwr'
                         % hdr['remainder'])

    hdr_area = hdr['header_area']
    ray_bytes = hdr['ray_bytes']
    moments = hdr['moments']
    ngates = max(m['nbins'] for m in moments)
    total_rays = hdr['nrays']
    nrays = total_rays if ray_limit is None else min(int(ray_limit), total_rays)
    read_bytes = nrays * ray_bytes

    # ---- 读数据区 ----
    f, _total, close = _open_payload(path)
    try:
        f.seek(hdr_area)
        t0 = time.time()
        raw = f.read(read_bytes)
        timings['decompress_read'] = time.time() - t0
        if len(raw) != read_bytes:
            raise ValueError('数据区读取不足: 期望 %d, 实得 %d' % (read_bytes, len(raw)))
    finally:
        close()

    # ---- 整卷结构化视图 ----
    t0 = time.time()
    items = [('rhdr', 'V%d' % RADIAL_HDR_SIZE)]      # ★ 别忘了每根开头那 128 字节射线头
    for i, m in enumerate(moments):
        items.append(('h%d' % i, 'V%d' % MOMENT_HDR_SIZE))
        items.append(('d%d' % i, {1: 'u1', 2: 'u2'}[m['BinLength']], m['nbins']))
    ray_dtype = np.dtype(items)
    if ray_dtype.itemsize != ray_bytes:
        raise ValueError('结构化 dtype 尺寸 %d != 射线长度 %d'
                         % (ray_dtype.itemsize, ray_bytes))
    arr = np.frombuffer(raw, dtype=ray_dtype, count=nrays)

    rh_view = np.frombuffer(raw, dtype=np.dtype(
        {'names': ['h'], 'formats': [RADIAL_HDR_DTYPE], 'offsets': [0],
         'itemsize': ray_bytes}), count=nrays)['h']
    timings['build_view'] = time.time() - t0

    # ---- 布局校验(抽样) ----
    if check_layout:
        t0 = time.time()
        step = max(1, nrays // 200)
        idx = np.arange(0, nrays, step)
        mnum = rh_view['MomentNumber'][idx]
        lod = rh_view['LengthOfData'][idx]
        if not np.all(mnum == len(moments)):
            raise ValueError('抽样发现矩个数不一致: %s' % np.unique(mnum))
        if not np.all(lod == ray_bytes - RADIAL_HDR_SIZE):
            raise ValueError('抽样发现 LengthOfData 不一致: %s' % np.unique(lod))
        # 逐个矩头: 顺序/参数是否与模板一致
        _nb_mismatch = 0
        for r in idx[:40]:
            base = int(r) * ray_bytes
            p = base + RADIAL_HDR_SIZE
            for m in moments:
                mh = np.frombuffer(raw, dtype=MOMENT_HDR_DTYPE, count=1, offset=p)[0]
                if int(mh['BinLength']) != m['BinLength']:
                    _nb_mismatch += 1
                if (int(mh['DataType']) != m['DataType'] or int(mh['Scale']) != m['Scale']
                        or int(mh['Offset']) != m['Offset']
                        or int(mh['BinLength']) != m['BinLength']):
                    continue
                p += MOMENT_HDR_SIZE + int(mh['Length'])
            if p != base + ray_bytes:
                raise ValueError('第 %d 根的矩区长度对不上' % r)
        if _nb_mismatch:
            print('[fast_pa] 注意: 抽样中有 %d 个矩头参数与首根模板不同, '
                  '已按"不改模板"处理(未改任何矩的库数)' % _nb_mismatch)
        timings['check_layout'] = time.time() - t0

    # ---- 逐矩解码 ----
    #   ★ 按**每个矩自己的 nbins** 出数组（威宁 12 个矩库数一致, 所以实际就是同一个宽度）。
    #     不强行铺到统一列宽 —— 铺列宽(以及"各矩库数不一致就报错")是
    #     fast_read.to_pyart() 的活。
    t0 = time.time()
    fields_moment = {}
    for i, m in enumerate(moments):
        if want is not None and m['name'] not in want:
            continue
        col = arr['d%d' % i]
        raw_i = np.ascontiguousarray(col, dtype=np.int32)
        valid = raw_i >= VALID_CODE_MIN
        if exact:
            out = (raw_i - m['Offset']) / float(m['Scale'])    # float64
            out[~valid] = np.nan
            fields_moment[m['name']] = out.astype(np.float32)
        else:
            out = (raw_i - m['Offset']).astype(np.float32)
            out /= np.float32(m['Scale'])
            out[~valid] = np.nan
            fields_moment[m['name']] = out
        del raw_i, valid, out
        gc.collect()
    timings['decode_fields'] = time.time() - t0
    nbins_by_moment = {m['name']: int(m['nbins']) for m in moments}

    # ---- 几何 / 时间 / 扫描层 ----
    t0 = time.time()
    elevation = np.asarray(rh_view['Elevation'], dtype=np.float32)
    elevation = np.where(elevation > 180, elevation - 360, elevation).astype(np.float32)
    azimuth = np.asarray(rh_view['Azimuth'], dtype=np.float32)
    seconds = np.asarray(rh_view['Seconds'], dtype=np.int64)
    microseconds = np.asarray(rh_view['MicroSeconds'], dtype=np.int64)
    elevation_number = np.asarray(rh_view['ElevationNumber'], dtype=np.int32)
    radial_number = np.asarray(rh_view['RadialNumber'], dtype=np.int32)
    radial_state = np.asarray(rh_view['RadialState'], dtype=np.int32)

    sweep_start, sweep_end = _build_sweep_indices(
        elevation_number, radial_number, radial_state, hdr['cut_number'], nrays,
        strict=(ray_limit is None))

    cut = hdr['cut_cfg']
    doppler_res = float(cut['DopplerResolution'][0])
    rng = _range_axis(doppler_res, ngates)
    nyquist = np.asarray(cut['NyquistSpeed'][:len(sweep_start)], dtype=np.float32)
    unamb = np.asarray(cut['MaximumRange'][:len(sweep_start)], dtype=np.float32)
    timings['geometry'] = time.time() - t0

    lat = hdr['latitude']
    lon = hdr['longitude']
    alt = float(hdr['altitude'])
    freq_ghz = float(hdr['frequency_mhz']) / 1000.0
    scan_type = {0: 'ppi', 1: 'ppi', 2: 'rhi', 5: 'rhi',
                 3: 'sector', 4: 'sector'}.get(int(hdr['scan_type']), 'other')

    timings['total'] = time.time() - t_all
    return dict(
        path=path, header=hdr, moments=moments,
        fields={MOMENT2PYART.get(k, k): v for k, v in fields_moment.items()},
        fields_moment=fields_moment,
        nbins_by_moment=nbins_by_moment,
        ngates_per_moment={MOMENT2PYART.get(k, k): n for k, n in nbins_by_moment.items()},
        azimuth=azimuth, elevation=elevation, seconds=seconds,
        microseconds=microseconds, radial_number=radial_number,
        elevation_number=elevation_number, radial_state=radial_state,
        range=rng, ngates=ngates, nrays=nrays, nsweeps=int(len(sweep_start)),
        sweep_start=sweep_start, sweep_end=sweep_end,
        latitude=lat, longitude=lon, altitude=alt, frequency_ghz=freq_ghz,
        nyquist=nyquist, unambiguous_range=unamb,
        sitename=hdr['site_name'], scan_type=scan_type,
        timings=timings, raw_bytes=read_bytes)


def _range_axis(resolution, length):
    """与 pycwr PA2NRadar._range_axis 完全一致"""
    if resolution == 0:
        return np.linspace(30, 30 * length, length, dtype=np.float32)
    return np.linspace(resolution, resolution * length, length, dtype=np.float32)


def _build_sweep_indices(elev_number, radial_number, radial_state, nsweeps, nrays,
                         strict=True):
    """与 pycwr PABaseData._build_sweep_indices 同一套判定。
    strict=False 时不校验层数与 CutNumber 是否相符(只读前缀时必然对不上)。"""
    if nrays == 0:
        return np.array([], np.int32), np.array([], np.int32)
    elev_starts = [0]
    state_starts = [0]
    de = np.diff(elev_number) != 0
    elev_starts += list(np.nonzero(de)[0] + 1)
    st = (radial_number[1:] == 1) | ((radial_state[1:] == 0) | (radial_state[1:] == 3))
    state_starts += list(np.nonzero(st)[0] + 1)
    elev_starts = np.unique(np.asarray(elev_starts, dtype=np.int32))
    state_starts = np.unique(np.asarray(state_starts, dtype=np.int32))
    if state_starts.size == nsweeps:
        starts = state_starts
    elif elev_starts.size == nsweeps:
        starts = elev_starts
    elif not strict:
        starts = state_starts if state_starts.size >= elev_starts.size else elev_starts
    else:
        raise ValueError('扫描层切分对不上 CutNumber: 声明 %d, 仰角法 %d, 状态法 %d'
                         % (nsweeps, elev_starts.size, state_starts.size))
    ends = np.concatenate((starts[1:] - 1, np.array([nrays - 1], dtype=np.int32)))
    return starts.astype(np.int32), ends.astype(np.int32)


# ---------------------------------------------------------------------------
# 6. 慢速参考解码 (只用于 selfcheck, 逻辑 = pycwr PAFile._parse_radial_single)
# ---------------------------------------------------------------------------
def slow_decode(path, nrays, moments):
    """按 pycwr 的逐射线/逐矩方式解码前 nrays 根, 用于比对。"""
    f, _total, close = _open_payload(path)
    try:
        hdr = parse_header(path)
        f.seek(hdr['header_area'])
        out = {m['name']: np.empty((nrays, m['nbins']), dtype=np.float32) for m in moments}
        for r in range(nrays):
            rh_buf = f.read(RADIAL_HDR_SIZE)
            rh = RADIAL_HDR_DTYPE_val(rh_buf)
            for i in range(int(rh['MomentNumber'])):
                mh_buf = f.read(MOMENT_HDR_SIZE)
                mh = MOMENT_HDR_DTYPE_val(mh_buf)
                length = int(mh['Length'])
                data = f.read(length)
                name = FLAG2PRODUCT.get(int(mh['DataType']))
                if name is None:
                    continue
                binlen = int(mh['BinLength'])
                bins = np.frombuffer(data, dtype={1: np.uint8, 2: np.uint16}[binlen])
                raw = bins.astype(np.int32)
                dec = np.where(raw >= VALID_CODE_MIN,
                               (raw - int(mh['Offset'])) / int(mh['Scale']), np.nan)
                out[name][r, :] = dec.astype(np.float32)
        return out
    finally:
        close()


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
def _hr(ch='-'):
    print(ch * 78)


def _cmd_info(path):
    h = parse_header(path)
    _hr('=')
    print('PA 文件体检: %s' % os.path.basename(path))
    _hr('=')
    print('磁盘文件 %.1f MB  ->  解出 %.1f MB'
          % (os.path.getsize(path) / 1048576.0, h['uncompressed'] / 1048576.0))
    print('站点 %s / %s' % (h['site_code'], h['site_name']))
    print('位置 lat=%.5f lon=%.5f alt=%d m  频率 %.0f MHz'
          % (h['latitude'], h['longitude'], h['altitude'], h['frequency_mhz']))
    print('ScanType=%d  BeamNumber=%d  CutNumber=%d' %
          (h['scan_type'], h['beam_number'], h['cut_number']))
    print('文件头区 %d 字节;  单根射线 %d 字节;  共 %d 根 (余 %d) → %s'
          % (h['header_area'], h['ray_bytes'], h['nrays'], h['remainder'],
             '严格定长, 可整卷向量化' if h['fixed_stride'] else '不定长!'))
    _hr()
    print('%-8s %-10s %-8s %-6s %-6s %-6s %s' %
          ('矩', 'DataType', 'BinLen', 'nbins', 'Scale', 'Offset', 'pyart 名'))
    for m in h['moments']:
        print('%-8s %-10d %-8d %-6d %-6d %-6d %s' %
              (m['name'], m['DataType'], m['BinLength'], m['nbins'],
               m['Scale'], m['Offset'], m['pyart']))
    return h


def _cmd_selfcheck(path, n, fields=None):
    """与 pycwr 式的逐射线慢解码逐位比对(只比前 n 根, 省内存)。"""
    av, tot = available_ram_gb()
    print('可用内存 %.2f GB / %.2f GB   比对前 %d 根' % (av, tot, n))
    h = parse_header(path)
    keep = [m for m in h['moments'] if fields is None or m['name'] in fields]
    print('参与比对的矩: %s' % [m['name'] for m in keep])

    print('--- 慢速参考(pycwr 同款逐射线解码) ---')
    t0 = time.time()
    ref = slow_decode(path, n, h['moments'])
    t_slow = time.time() - t0
    print('  %.2f s' % t_slow)

    del ref
    gc.collect()
    print('--- 本模块快速读取 ---')
    t0 = time.time()
    got = read_pa(path, fields=None, ray_limit=n)
    t_fast = time.time() - t0
    print('  %.2f s   明细: %s' % (t_fast, {k: round(v, 3) for k, v in got['timings'].items()}))
    _hr()
    print('%-8s %-14s %-10s %-10s %s' % ('矩', 'NaN 掩码一致', '最大绝对差', '有效点数', '结论'))
    allok = True
    # 慢速重算一次(上面 del 了), 但为了省内存只重算需要的矩
    ref2 = slow_decode(path, n, [m for m in h['moments'] if m['name'] in
                                 {x['name'] for x in keep}])
    for m in keep:
        a = ref2[m['name']]
        b = got['fields_moment'][m['name']]
        mask_ok = np.array_equal(np.isnan(a), np.isnan(b))
        both = ~np.isnan(a) & ~np.isnan(b)
        maxdiff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        ok = mask_ok and maxdiff == 0.0
        allok &= ok
        print('%-8s %-14s %-10.3g %-10d %s'
              % (m['name'], '是' if mask_ok else '★否', maxdiff, int(both.sum()),
                 'OK 逐位一致' if ok else '★不一致'))
    _hr()
    print('慢速(逐射线)外推整卷 ≈ %.1f s' % (t_slow / n * h['nrays']))
    print('快速(向量化)外推整卷 ≈ %.1f s' % (t_fast / n * h['nrays']))
    print('pycwr 实测锚点           = 38.5 s')
    print('自检结论: %s' % ('全部逐位一致 ✔' if allok else '★存在差异, 不可替换'))
    return allok


def _cmd_bench(path, fields=None):
    av, tot = available_ram_gb()
    print('可用内存 %.2f GB / %.2f GB' % (av, tot))
    m0, pk0, pf0 = _proc_mem()
    t0 = time.time()
    got = read_pa(path, fields=fields)
    dt = time.time() - t0
    m1, pk1, pf1 = _proc_mem()
    h = got['header']
    _hr('=')
    print('整卷读取: %s' % os.path.basename(path))
    _hr('=')
    print('根数 %d / 库数 %d / 层数 %d;  数据区 %.1f MB'
          % (got['nrays'], got['ngates'], got['nsweeps'], got['raw_bytes'] / 1048576.0))
    for k, v in got['timings'].items():
        print('  %-18s %7.2f s' % (k, v))
    print('  %-18s %7.2f s' % ('合计', dt))
    print('峰值工作集 %.0f MB (增量 %.0f MB);  缺页 +%d'
          % (pk1, pk1 - pk0, pf1 - pf0))
    print('解出的字段: %s' % sorted(got['fields']))
    if fields is None:
        tot_mb = sum(v.nbytes for v in got['fields'].values()) / 1048576.0
        print('字段数组合计 %.0f MB' % tot_mb)
    print('对比锚点: pycwr read_auto().ToPyartRadar() = 38.5 s, 峰值约 4842 MB')
    return got


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = argv[1]
    if not os.path.exists(path):
        print('文件不存在: %s' % path)
        return 1
    n = 1500
    fields = None
    for i, a in enumerate(argv):
        if a == '--rays' and i + 1 < len(argv):
            n = int(argv[i + 1])
        if a == '--fields' and i + 1 < len(argv):
            fields = [x.strip() for x in argv[i + 1].split(',') if x.strip()]
    if '--info' in argv:
        _cmd_info(path)
    if '--selfcheck' in argv:
        _cmd_selfcheck(path, n, fields)
    if '--bench' in argv:
        _cmd_bench(path, fields)
    if not any(a.startswith('--') for a in argv[1:]):
        _cmd_info(path)
        _cmd_bench(path, fields)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
