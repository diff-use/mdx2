import os
import matplotlib.pyplot as plt
from nexusformat.nexus import nxload
from mdx2.mdx2.command_line.report1._functions import create_slices, map2xr
import sys

# prevent local mdx2/ directory from shadowing mdx2 package import
if '' in sys.path:
    sys.path.remove('')

def run(geom=None, limits=None):
    #Will have to specify directory
    if not set(['slice_all_iso.nxs', 'slice_all.nxs', 'slice_crystal1.nxs', 'slice_crystal2.nxs']).issubset(os.listdir()):
        create_slices(geom=geom, limits=limits)

    imav = nxload('slice_all_isoavg.nxs')['/entry/isoavg']
    imsub = nxload('slice_all.nxs')['/entry/intensity']

    im = imsub + imav

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))

    map2xr(im).isel(l=2).plot(x='h',y='k',ax=ax1,vmin=0,vmax=1.1,cmap='viridis')
    map2xr(imsub).isel(l=2).plot(x='h',y='k',ax=ax2,vmin=-.05,vmax=.05,cmap='bwr')
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    fig.savefig('viz1_plot.png', dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    run()