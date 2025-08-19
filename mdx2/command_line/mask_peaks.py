"""
Create a peak mask for in an image stack
"""

import argparse

import numpy as np
from joblib import Parallel, delayed

from mdx2.geometry import GridData
from mdx2.utils import loadobj, saveobj


def parse_arguments(args=None):
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("geom", help="NeXus file w/ miller_index")
    parser.add_argument("data", help="NeXus file w/ image_series")
    parser.add_argument("peaks", help="NeXus file w/ peak_model and peaks")
    parser.add_argument(
        "--sigma_cutoff", metavar="SIGMA", default=3, type=float, help="sigma value to draw the peak mask"
    )
    parser.add_argument("--outfile", default="mask.nxs", help="name of the output NeXus file")
    parser.add_argument("--nproc", type=int, default=1, metavar="N", help="number of parallel processes")
    parser.add_argument("--bragg", action="store_true", help="create a Bragg peak mask instead")

    params = parser.parse_args(args)

    return params


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
    params = parse_arguments(args=args)
    run_mask_peaks(params)


if __name__ == "__main__":
    run()
