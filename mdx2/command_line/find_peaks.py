"""
Find and analyze peaks in an image stack
"""

import argparse

import numpy as np

from mdx2.data import Peaks
from mdx2.geometry import GaussianPeak
from mdx2.utils import loadobj, saveobj


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("geom", help="NeXus file w/ miller_index")
    parser.add_argument("data", help="NeXus file w/ image_series")
    parser.add_argument(
        "--count_threshold",
        metavar="THRESH",
        required=True,
        type=float,
        help="pixels with counts above threshold are flagged as peaks",
    )
    parser.add_argument(
        "--sigma_cutoff", metavar="SIGMA", default=3, type=float, help="\for outlier rejection in Gaussian peak fitting"
    )
    parser.add_argument("--outfile", default="peaks.nxs", help="name of the output NeXus file")
    parser.add_argument("--nproc", type=int, default=1, metavar="N", help="number of parallel processes")
    params = parser.parse_args(args)
    return params


def run_find_peaks(params):
    """Run the find peaks script"""
    geom = params.geom
    data = params.data
    count_threshold = params.count_threshold
    sigma_cutoff = params.sigma_cutoff
    outfile = params.outfile
    nproc = params.nproc

    MI = loadobj(geom, "miller_index")
    IS = loadobj(data, "image_series")

    print(f"finding pixels with counts above threshold: {count_threshold}")
    P = IS.find_peaks_above_threshold(count_threshold, nproc=nproc)

    print("indexing peaks")
    h, k, l = MI.interpolate(P.phi, P.iy, P.ix)
    dh = h - np.round(h)
    dk = k - np.round(k)
    dl = l - np.round(l)

    print("fitting Gaussian peak model")
    GP, is_outlier = GaussianPeak.fit_to_points(dh, dk, dl, sigma_cutoff=sigma_cutoff)

    Outliers = Peaks(P.phi[is_outlier], P.iy[is_outlier], P.ix[is_outlier])

    print(f"{np.sum(is_outlier)} peaks were rejected as outliers")
    print("GaussianPeak model:", GP)

    print(f"Saving results to {outfile}")

    saveobj(GP, outfile, name="peak_model", append=False)
    saveobj(P, outfile, name="peaks", append=True)
    saveobj(Outliers, outfile, name="outliers", append=True)

    print("done!")


def run(args=None):
    """Run the find peaks script"""
    params = parse_arguments(args=args)
    run_find_peaks(params)


if __name__ == "__main__":
    run()
