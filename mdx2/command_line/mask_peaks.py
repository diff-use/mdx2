"""
Create a peak mask for in an image stack
"""

import logging
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from simple_parsing import ArgumentParser, field  # pip install simple-parsing

from mdx2.command_line import configure_logging
from mdx2.geometry import GridData
from mdx2.utils import loadobj, saveobj

logger = logging.getLogger(__name__)


@dataclass
class Parameters:
    """Options for creating a peak mask"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    data: str = field(positional=True)  # NeXus data file containing image_series
    peaks: str = field(positional=True)  # NeXus data file containing peak_model and peaks
    sigma_cutoff: float = 3.0  # contour level for drawing the peak mask
    outfile: str = "mask.nxs"  # name of the output NeXus file
    nproc: int = 1  # number of parallel processes
    bragg: bool = False  # create a Bragg peak mask instead


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Parameters, dest="parameters")
    opts = parser.parse_args(args)
    return opts.parameters


def run_mask_peaks(params):
    """Run the mask peaks script"""
    geom = params.geom
    data = params.data
    peaks = params.peaks
    outfile = params.outfile
    sigma_cutoff = params.sigma_cutoff
    nproc = params.nproc
    bragg = params.bragg

    MI = loadobj(geom, "miller_index")
    Symm = loadobj(geom, "symmetry")
    IS = loadobj(data, "image_series")
    GP = loadobj(peaks, "peak_model")
    P = loadobj(peaks, "peaks")

    # initialize the mask using Peaks
    mask = np.zeros(IS.shape, dtype="bool")

    def maskchunk(sl):
        MIdense = MI.regrid(IS.phi[sl[0]], IS.iy[sl[1]], IS.ix[sl[2]])
        H = np.round(MIdense.h)
        K = np.round(MIdense.k)
        L = np.round(MIdense.l)
        dh = MIdense.h - H
        dk = MIdense.k - K
        dl = MIdense.l - L
        isrefl = Symm.is_reflection(H, K, L)
        return isrefl & GP.mask(dh, dk, dl, sigma_cutoff=sigma_cutoff)

    # loop over phi values
    print(f"masking peaks with sigma above threshold: {sigma_cutoff}")

    if nproc == 1:
        for ind, sl in enumerate(IS.chunk_slice_iterator()):
            print(f"indexing chunk {ind}")
            mask[sl] = maskchunk(sl)
    else:
        with Parallel(n_jobs=nproc, verbose=10) as parallel:
            masklist = parallel(delayed(maskchunk)(sl) for sl in IS.chunk_slice_iterator())
        for msk, sl in zip(masklist, IS.chunk_slice_iterator()):
            mask[sl] = msk  # <-- note, this copy step could be avoided with shared mem

    if bragg:
        print("inverting mask to retain Bragg peaks")
        mask = np.logical_not(mask)
    else:
        print("masking count threshold peaks")
        P.to_mask(IS.phi, IS.iy, IS.ix, mask_in=mask)

    # add peaks
    print(f"Saving mask to {outfile}")

    maskobj = GridData((IS.phi, IS.iy, IS.ix), mask)
    saveobj(maskobj, outfile, name="mask", append=False)

    print("done!")


def run(args=None):
    """Run the mask peaks script"""
    configure_logging(filename="mdx2.mask_peaks.log")
    params = parse_arguments(args=args)
    run_mask_peaks(params)


if __name__ == "__main__":
    run()
