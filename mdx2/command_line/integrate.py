"""
Integrate counts in an image stack on a Miller index grid
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from simple_parsing import ArgumentParser, field

from mdx2.data import HKLTable
from mdx2.utils import (
    loadobj,
    nxload,  # mask is too big to read all at once?
    saveobj,
)


@dataclass
class Parameters:
    """Options for integrating counts on a Miller index grid"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    data: str = field(positional=True)  # NeXus data file containing image_series
    mask: Optional[str]  # NeXus data file containing mask
    subdivide: Tuple[int, int, int] = (1, 1, 1)  # subdivisions of the Miller index grid
    max_spread: float = 1.0  # maximum angular spread (degrees) for binning partial observations
    nproc: int = 1  # number of parallel processes
    outfile: str = "integrated.nxs"  # name of the output NeXus file


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Parameters, dest="parameters")
    opts = parser.parse_args(args)
    return opts.parameters


def run_integrate(params):
    """Run the integrate script"""
    geom = params.geom
    data = params.data
    outfile = params.outfile
    nproc = params.nproc
    ndiv = params.subdivide
    max_degrees = params.max_spread
    maskfile = params.mask

    MI = loadobj(geom, "miller_index")
    IS = loadobj(data, "image_series")

    if maskfile is not None:
        # MA = loadobj(maskfile,'mask')
        # mask = MA.data
        nxs = nxload(maskfile)  # <-- loadobj fails if the array is too large. weird
        mask = nxs.entry.mask.signal  # nxfield

    else:
        mask = None

    # if opts.limits is not None:
    #    lim = opts.limits
    # else:
    #    lim = None

    def intchunk(sl):
        ims = IS[sl]
        if mask is not None:
            tab = ims.index(MI, mask=mask[sl].nxdata)  # added nxdata to deal with NXfield wrapper
        else:
            tab = ims.index(MI)
        tab.ndiv = ndiv
        return tab.bin(count_name="pixels")

    if nproc == 1:
        T = []  # list of tables
        print("Looping through chunks")
        for ind, sl in enumerate(IS.chunk_slice_iterator()):
            T.append(intchunk(sl))
            print(f"  binned chunk {ind}")
    else:
        with Parallel(n_jobs=nproc, verbose=10) as parallel:
            T = parallel(delayed(intchunk)(sl) for sl in IS.chunk_slice_iterator())

    print(f"Summing partial observations over {len(T)} chunks")
    df = HKLTable.concatenate(T).to_frame()  # .set_index(['h','k','l'])

    df["tmp"] = df["phi"] / df["pixels"]
    delta_phi = df.tmp - df.groupby(["h", "k", "l"])["tmp"].transform("min")
    df["n"] = np.floor(delta_phi / max_degrees)
    df = df.drop(columns=["tmp"])

    df = df.groupby(["h", "k", "l", "n"]).sum()

    # compute mean positions in the scan
    df["phi"] = df["phi"] / df["pixels"]
    df["iy"] = df["iy"] / df["pixels"]
    df["ix"] = df["ix"] / df["pixels"]

    print(f"  binned from {np.sum([len(t) for t in T])} to {len(df)} voxels")

    print(f"Saving table of integrated data to {outfile}")

    hkl_table = HKLTable.from_frame(df)
    hkl_table.ndiv = ndiv  # lost in conversion to/from dataframe

    hkl_table.h = hkl_table.h.astype(np.float32)
    hkl_table.k = hkl_table.k.astype(np.float32)
    hkl_table.l = hkl_table.l.astype(np.float32)
    hkl_table.phi = hkl_table.phi.astype(np.float32)
    hkl_table.ix = hkl_table.ix.astype(np.float32)
    hkl_table.iy = hkl_table.iy.astype(np.float32)
    hkl_table.seconds = hkl_table.seconds.astype(np.float32)
    hkl_table.counts = hkl_table.counts.astype(np.int32)
    hkl_table.pixels = hkl_table.pixels.astype(np.int32)

    saveobj(hkl_table, outfile, name="hkl_table", append=False)

    print("done!")


def run(args=None):
    """Run the integrate script"""
    params = parse_arguments(args=args)
    run_integrate(params)


if __name__ == "__main__":
    run()
