import numpy as np
import pandas as pd
import xarray as xr

from mdx2.io import loadobj, saveobj
from mdx2.mdx2.command_line.map import run_map

import sys
# prevent local mdx2/ directory from shadowing mdx2 package import
if '' in sys.path:
    sys.path.remove('')

##VISUALIZATION


#Specify filepath
Crystal  = loadobj('mdx2/split_00/geometry.nxs','crystal')
Symmetry = loadobj('mdx2/split_00/geometry.nxs','symmetry')

def hkl2s(h,k,l):
    """Compute the magnitude of s from Miller indices."""
    UB = Crystal.ub_matrix
    s = UB @ np.stack((h,k,l))
    return np.sqrt(np.sum(s*s,axis=0))

def isInteger(h):
    return (h.round()-h).abs() < 0.00001

def isReflection(h,k,l):
    return isInteger(h) & isInteger(k) & isInteger(l) & Symmetry.is_reflection(h,k,l)

def calc_stats(fn):
    # load the tables and convert to pandas dataframe
    tab = loadobj(fn,'hkl_table')
    tab.s = hkl2s(tab.h,tab.k,tab.l)
    df = tab.to_frame()
    df = df[~isReflection(df['h'],df['k'],df['l'])]
    df = df.set_index(['h','k','l']).sort_index()
    s_bins = pd.cut(df['s'],np.linspace(0,.8,81))
    df_isoavg = df.groupby(s_bins).agg({
        's':'mean',
        'intensity':['mean','std'],
        'intensity_error':'mean'})

    dfh = tab.to_frame()
    dfh = dfh[isReflection(dfh['h'],dfh['k'],dfh['l'])]
    dfh = dfh.set_index(['h','k','l']).sort_index()
    s_bins = pd.cut(dfh['s'],np.linspace(0,.8,81))
    dfh_isoavg = dfh.groupby(s_bins).agg({
        's':'mean',
        'intensity':['mean','std'],
        'intensity_error':'mean'})
    df_out = pd.DataFrame({
        's':df_isoavg['s']['mean'],
        'non-halo':df_isoavg['intensity']['mean'],
        'halo':dfh_isoavg['intensity']['mean'],
        })
    df_out = df_out.set_index('s')
    return df_out

def merged_subplots(dsets=['all','crystal1','crystal2']):
    for dset in dsets:
        df_stats = calc_stats(f'mdx2/merged_{dset}.nxs')
        nh = df_stats['non-halo'].dropna()
        x = nh.keys().values
        y = nh.values
        Imax = np.max(y)
        tab = loadobj(f'mdx2/merged_{dset}.nxs','hkl_table')
        tab.isoavg = np.interp(hkl2s(tab.h,tab.k,tab.l),x,y/Imax)
        tab.intensity/=Imax
        tab.intensity_error/=Imax
        tab.intensity -= tab.isoavg
        saveobj(tab,f'merged_{dset}_sub.nxs','hkl_table')

def create_slices(geom,limits):
    limit_labels = ['hmin', 'hmax', 'kmin', 'kmax', 'lmin', 'lmax']
    limits_dict = dict(zip(limit_labels, limits))
    signals = ['isoavg', 'intensity', 'intensity', 'intensity']
    outfiles = ['slice_all_iso.nxs', 'slice_all.nxs', 'slice_crystal1.nxs', 'slice_crystal2.nxs']

    args = []

    for outfile, signal in zip(outfiles, signals):
        outfile_dict = limits_dict.copy()
        outfile_dict['geom'] = geom
        outfile_dict['signal'] = signal
        outfile_dict['outfile'] = outfile
        args.append(outfile_dict)

    for i in args:
        run_map(**i)

def map2xr(m):
    return xr.DataArray(
        data=m.signal.nxvalue,
        dims=m.axes,
        coords=dict(h=m.h.nxvalue, k=m.k.nxvalue, l=m.l.nxvalue)
    )


## MAP STATISTICS
def load_refl(fn):
    t1 = loadobj(fn,'hkl_table')
    t1.s = hkl2s(t1.h,t1.k,t1.l)
    df1 = t1.to_frame()
    df1h = df1[isReflection(df1['h'],df1['k'],df1['l'])].set_index(['h','k','l'])
    df1 = df1[~isReflection(df1['h'],df1['k'],df1['l'])].set_index(['h','k','l'])
    return df1, df1h

def calc_map_stats(df):
    # divide into 100 equal sized bins
    s_bins = pd.cut(df['s'],np.linspace(0.03,.8,78))

    # compute statistics in each bin
    df_isoavg = df.groupby(s_bins).agg({
        's':'mean',
        'intensity':['mean','std'],
        'intensity_error':'mean'})
    df_isoavg.columns = df_isoavg.columns.to_flat_index()
    df_isoavg = df_isoavg.rename(columns={
        ('s', 'mean'): "s",
        ('intensity','std'):'standard deviation',
        ('intensity','mean'):'mean',
        ('intensity_error','mean'):'average measurement error',
        })
    df_isoavg = df_isoavg.set_index('s')
    return df_isoavg