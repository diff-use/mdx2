"""Fit a Gaussian peak model to strong pixels"""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from simple_parsing import field  # pip install simple-parsing

from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.geometry import GaussianPeak
from mdx2.io import loadobj, saveobj


@dataclass
class Parameters:
    """Options for finding peaks in an image stack"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    strong: str = field(positional=True)  # NeXus data file containing strong_pixels
    sigma_cutoff: float = 3.0  # for outlier rejection in Gaussian peak fitting
    outfile: str = "peaks.nxs"  # name of the output NeXus file

    def __post_init__(self):
        """Validate sigma_cutoff parameter"""
        if self.sigma_cutoff <= 0:
            raise ValueError(f"sigma_cutoff must be > 0, got {self.sigma_cutoff}")


def fit_peaks(miller_index, strong_pixels, sigma_cutoff):
    tab = strong_pixels.index(miller_index)
    h, k, l = tab.h, tab.k, tab.l
    dh = h - np.round(h)
    dk = k - np.round(k)
    dl = l - np.round(l)

    logger.info("Fitting Gaussian peak model...")
    peak_model, is_outlier = GaussianPeak.fit_to_points(dh, dk, dl, sigma_cutoff=sigma_cutoff)
    logger.info("Rejected {} outliers (sigma cutoff: {})", np.sum(is_outlier), sigma_cutoff)
    logger.info("Peak model r0: {}", peak_model.r0)

    # Compute principal axes of error ellipsoid using SVD
    # sigma = U @ diag(s) @ V.T, where U contains the principal axes
    # and s contains the semi-axis lengths
    U, s, _Vt = np.linalg.svd(peak_model.sigma)

    logger.info("Error ellipsoid semi-axis lengths: {}", s)
    logger.info("Error ellipsoid principal axis 1: {}", U[:, 0])
    logger.info("Error ellipsoid principal axis 2: {}", U[:, 1])
    logger.info("Error ellipsoid principal axis 3: {}", U[:, 2])

    return peak_model


def run_fit_peaks(params):
    """Run the fit peaks script"""
    geom = params.geom
    strong = params.strong
    outfile = params.outfile
    sigma_cutoff = params.sigma_cutoff

    logger.info("Loading geometry and strong pixels data...")
    miller_index = loadobj(geom, "miller_index")
    strong_pixels = loadobj(strong, "strong_pixels")

    peak_model = fit_peaks(miller_index, strong_pixels, sigma_cutoff)

    logger.info("Saving peak model to {}...", outfile)
    saveobj(peak_model, outfile, name="peak_model")
    logger.info("Peak finding completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_fit_peaks))

if __name__ == "__main__":
    run()
