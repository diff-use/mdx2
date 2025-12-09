"""
Integrate counts in an image stack on a Miller index grid
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import ArgumentParser, field

from mdx2.command_line import with_logging
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
    mask: Optional[str] = None  # NeXus data file containing mask
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

    logger.info("Loading geometry and image data...")
    MI = loadobj(geom, "miller_index")
    IS = loadobj(data, "image_series")

    if maskfile is not None:
        logger.info("Loading mask...")
        # Use nxload directly instead of loadobj because loadobj fails for very large arrays
        nxs = nxload(maskfile)
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

    slices = list(IS.chunk_slice_iterator())
    with Parallel(n_jobs=nproc, verbose=10) as parallel:
        backend_name = parallel._backend.__class__.__name__
        logger.info(
            "Integrating {} image chunks using {} processes (backend: {})...",
            len(slices),
            nproc,
            backend_name,
        )
        T = parallel(delayed(intchunk)(sl) for sl in slices)

    logger.info("Summing partial observations over {} chunks...", len(T))
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

    voxels_before = np.sum([len(t) for t in T])
    voxels_after = len(df)
    logger.info("Binned from {} to {} voxels", voxels_before, voxels_after)

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
    logger.info("Integration completed successfully")


@with_logging()
def run(args=None):
    """Run the integrate script"""
    params = parse_arguments(args=args)
    logger.info(params)
    run_integrate(params)


if __name__ == "__main__":
    run()
