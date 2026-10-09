# -*- coding: utf-8 -*-
"""
QC/io/load_qc_radar.py — 读"QC 处理产物"(.pkl.gz) -> pyart Radar
=================================================================
产物文件里存的**不是** pyart Radar, 而是一个 dict:
    {'fields':   {矩名: (nrays, ngates) 数组},
     'geometry': {range_km, azimuth, elevation, fixed_angle,
                  sweep_start, sweep_end, latitude, longitude,
                  altitude, nsweeps, nrays, ngates}}

本模块把它还原成正常的 pyart.core.Radar, 之后画 PPI / 网格化 / 喂 pydda 都能直接用。

用法(只认**全路径**, 2026-10-08 用户要求)
----------------------------------------
    from QC.io import read_qc_data
    radar = read_qc_data(r'F:\\cell1\\file\\processed_qc\\radar_01\\20250520110900.pkl.gz')

    radar.nsweeps, radar.nrays, radar.ngates
    sorted(radar.fields)     # ['cor_z','cor_zdr','velocity','kdp_dual',...,'my_phi']

(旧名字 load_qc_radar 保留为别名, 老的 notebook / 脚本照旧能用)

字段是**通用遍历**的: 产物里有什么就还原什么 —— 2026-10-08 新增的 `my_phi`
不需要改这里就能读到。

两个坑(已在本模块里处理):
  1. geometry['range_km'] 名字带 km, **实测单位是米**(30, 60, ..., 44160)。
  2. pyart 的 radar.nrays 取自 len(time['data']), 不是方位角长度;
     必须把 time 造成长度 = nrays 的数组, 否则 add_field 之类的形状检查会挂。
"""
import gzip
import os
import pickle

import numpy as np
import pyart


def _stamp_to_units(name14):
    """把 14 位时间戳变成 pyart 的 time units 字符串。"""
    if name14 and len(name14) == 14 and name14.isdigit():
        return ('seconds since %s-%s-%sT%s:%s:%sZ'
                % (name14[0:4], name14[4:6], name14[6:8],
                   name14[8:10], name14[10:12], name14[12:14]))
    return 'seconds since 1970-01-01T00:00:00Z'


def dict_to_radar(d, stamp=None):
    """{'fields','geometry'} -> pyart.core.Radar。字段通用遍历, 有什么还原什么。"""
    g = d['geometry']
    nrays = int(g['nrays'])
    ngates = int(g['ngates'])
    nsweeps = int(g['nsweeps'])
    rng = np.asarray(g['range_km'], dtype=np.float64)          # 注意: 已是米

    def md(data, units='', long_name=''):
        return {'data': np.asarray(data), 'units': units, 'long_name': long_name}

    fields = {}
    for name, arr in d['fields'].items():
        fields[name] = {
            'data': np.ma.masked_invalid(np.asarray(arr, dtype=np.float32)),
            'units': '', 'long_name': name, '_FillValue': None,
        }

    return pyart.core.Radar(
        time={'data': np.zeros(nrays, dtype='float64'),
              'units': _stamp_to_units(stamp),
              'calendar': 'gregorian', 'standard_name': 'time'},
        _range=md(rng, 'm', 'Range from radar'),
        fields=fields,
        metadata={},
        scan_type='ppi',
        latitude=md([g['latitude']], 'degrees_north', 'Latitude'),
        longitude=md([g['longitude']], 'degrees_east', 'Longitude'),
        altitude=md([g['altitude']], 'm', 'Altitude'),
        sweep_number=md(np.arange(nsweeps), '', 'sweep number'),
        sweep_mode=md(['ppi'] * nsweeps, '', 'sweep mode'),
        fixed_angle=md(np.asarray(g['fixed_angle'], np.float32), 'degrees', 'fixed angle'),
        sweep_start_ray_index=md(np.asarray(g['sweep_start'], np.int32), '', 'start ray index'),
        sweep_end_ray_index=md(np.asarray(g['sweep_end'], np.int32), '', 'end ray index'),
        azimuth=md(np.asarray(g['azimuth'], np.float32), 'degrees', 'azimuth'),
        elevation=md(np.asarray(g['elevation'], np.float32), 'degrees', 'elevation'),
    )


def read_qc_data(path):
    """
    读一个 QC 产物 .pkl.gz -> pyart.core.Radar。

    参数
      path: .pkl.gz 的**全路径**, 例如
            r'F:\\cell1\\file\\processed_qc\\radar_01\\20250520110900.pkl.gz'
    返回
      pyart.core.Radar(文件里若存的直接就是 Radar, 则原样返回)
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError('文件不存在: ' + path)
    stamp = os.path.basename(path).split('.')[0]
    with gzip.open(path, 'rb') as f:
        d = pickle.load(f)

    # 万一哪天这里直接存的就是 Radar, 原样返回
    if isinstance(d, pyart.core.Radar) or (hasattr(d, 'fields') and hasattr(d, 'nsweeps')):
        return d
    if isinstance(d, dict) and 'fields' in d and 'geometry' in d:
        return dict_to_radar(d, stamp=stamp)
    raise TypeError('不认识的内容: %s' % type(d))


# ★ 2026-10-08 用户定的对外名字: read_qc_data(全路径读 QC 产物)。
#   load_qc_radar 保留为别名, 不破坏已有调用(notebook / 脚本 / F:\dda 转发层)。
load_qc_radar = read_qc_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        p = sys.argv[1]
    else:
        p = r'F:\cell1\file\processed_qc\radar_01\20250520110900.pkl.gz'
    r = read_qc_data(p)
    print('文件     :', p)
    print('type     :', type(r).__module__ + '.' + type(r).__name__)
    print('nsweeps/nrays/ngates: %d / %d / %d' % (r.nsweeps, r.nrays, r.ngates))
    print('station  : %.4fN %.4fE %.0fm'
          % (r.latitude['data'][0], r.longitude['data'][0], r.altitude['data'][0]))
    print('time     :', r.time['units'])
    print('fields   :', sorted(r.fields))
    print('gate lat : %.4f ~ %.4f' % (np.nanmin(r.gate_latitude['data']),
                                      np.nanmax(r.gate_latitude['data'])))
