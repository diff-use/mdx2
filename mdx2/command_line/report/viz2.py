import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from nexusformat.nexus import nxload
from mdx2.mdx2.command_line.report1._functions import create_slices, map2xr
import sys

# prevent local mdx2/ directory from shadowing mdx2 package import
if '' in sys.path:
    sys.path.remove('')

def run():
    #Will have to specify directory
    if [''] not in os.listdir():
        geom = input('Enter geom: ')
        limits = input('Enter limits: ')
        create_slices(geom=geom, limits=limits)

    imsub1 = nxload('slice_crystal1.nxs')['/entry/intensity']
    imsub2 = nxload('slice_crystal2.nxs')['/entry/intensity']

    a = map2xr(imsub1).isel(l=2)
    b = map2xr(imsub2).isel(l=2)
    d = xr.concat((a,b,b-a),pd.Index(['1','2','2-1'],name="Crystal"))
    fh = d[:,330:391,240:301].plot(x='h',y='k',col='Crystal',vmin=-.04,vmax=.04,cmap='bwr')
    [ax.set_aspect('equal') for ax in fh.axs.flatten()]
    fh.fig.savefig('viz2_plot.png', dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    run()
