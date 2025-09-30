"""
Find and analyze peaks in an image stack
"""

import logging
from dataclasses import dataclass

import numpy as np
from simple_parsing import ArgumentParser, field  # pip install simple-parsing

from mdx2.command_line import configure_logging
from mdx2.data import Peaks
from mdx2.geometry import GaussianPeak
from mdx2.utils import loadobj, saveobj

logger = logging.getLogger(__name__)


@dataclass
class Parameters:
    """Options for finding peaks in an image stack"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    data: str = field(positional=True)  # NeXus data file containing image_series
    count_threshold: float  # pixels with counts above threshold are flagged as peaks
    sigma_cutoff: float = 3.0  # for outlier rejection in Gaussian peak fitting
    outfile: str = "peaks.nxs"  # name of the output NeXus file
    nproc: int = 1  # number of parallel processes


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Parameters, dest="parameters")
    opts = parser.parse_args(args)
    return opts.parameters


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
    configure_logging(filename="mdx2.find_peaks.log")
    params = parse_arguments(args=args)
    run_find_peaks(params)


if __name__ == "__main__":
    run()
