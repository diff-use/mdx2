"""
Find and analyze peaks in an image stack
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import field  # pip install simple-parsing

from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.data import Peaks
from mdx2.geometry import GaussianPeak
from mdx2.io import loadobj, saveobj
from mdx2.stats import find_peaks


@dataclass
class Parameters:
    """Options for finding peaks in an image stack"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    data: str = field(positional=True)  # NeXus data file containing image_series
    count_threshold: Optional[float] = None  # pixels with counts above threshold are flagged as peaks
    sigma_cutoff: float = 3.0  # for outlier rejection in Gaussian peak fitting
    outfile: str = "peaks.nxs"  # name of the output NeXus file
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)

    def __post_init__(self):
        """Validate sigma_cutoff parameter"""
        if self.sigma_cutoff <= 0:
            raise ValueError(f"sigma_cutoff must be > 0, got {self.sigma_cutoff}")


def run_find_peaks(params):
    """Run the find peaks script"""
    geom = params.geom
    data = params.data
    count_threshold = params.count_threshold
    sigma_cutoff = params.sigma_cutoff
    outfile = params.outfile
    nproc = params.nproc
    p_value = 1e-6  # NOTE: does this need to be a CLI option?

    logger.info("Loading geometry and image data...")
    MI = loadobj(geom, "miller_index")
    IS = loadobj(data, "image_series")

    logger.info("Finding pixels above threshold: {}", count_threshold)

    if count_threshold is not None:
        logger.info("Finding pixels above threshold: {}", count_threshold)
    else:
        logger.info("Estimating count thresholds dynamically for each data chunk")

    # Pre-compile JIT functions before forking if using multiprocessing
    if count_threshold is None and nproc > 1:
        dummy_data = np.array([0, 1, 2, 3, 4, 5], dtype=IS.data.dtype)
        # Trigger JIT compilation
        _ = find_peaks(dummy_data, threshold=count_threshold, p_value=p_value, bin_size=None)
        logger.info("Pre-compiled find_peaks JIT functions for multiprocessing")

    # Find peaks in parallel
    def peaksearch(sl):
        ims = IS[sl]
        peak_mask = find_peaks(ims.data, threshold=count_threshold, p_value=p_value, bin_size=None)
        peaks = Peaks.where(peak_mask, ims.phi, ims.iy, ims.ix)
        if peaks.size:
            return peaks

    slices = list(IS.chunk_slice_iterator())
    logger.info("Searching for peaks in {} image chunks (requested n_jobs: {})...", len(slices), nproc)
    with Parallel(n_jobs=nproc, verbose=10) as parallel:
        log_parallel_backend(parallel)
        peaklist = parallel(delayed(peaksearch)(sl) for sl in slices)
    logger.info("Peak search completed")

    peaks_nonempty = [p for p in peaklist if p is not None]
    if not peaks_nonempty:
        if count_threshold is not None:
            raise ValueError(
                f"No peaks found above threshold {count_threshold}. Try lowering the threshold or check your data."
            )
        else:
            raise ValueError(
                "No peaks found using dynamic threshold. Try setting a manual threshold or checking your data."
            )

    P = Peaks.stack(peaks_nonempty)
    logger.info("Found {} peak pixels", P.size)

    logger.info("Indexing peaks...")
    h, k, l = MI.interpolate(P.phi, P.iy, P.ix)
    dh = h - np.round(h)
    dk = k - np.round(k)
    dl = l - np.round(l)

    logger.info("Fitting Gaussian peak model...")
    GP, is_outlier = GaussianPeak.fit_to_points(dh, dk, dl, sigma_cutoff=sigma_cutoff)
    logger.info("Rejected {} outliers (sigma cutoff: {})", np.sum(is_outlier), sigma_cutoff)
    logger.info("Peak model r0: {}", GP.r0)

    # Compute principal axes of error ellipsoid using SVD
    # sigma = U @ diag(s) @ V.T, where U contains the principal axes
    # and s contains the semi-axis lengths
    U, s, _Vt = np.linalg.svd(GP.sigma)

    logger.info("Error ellipsoid semi-axis lengths: {}", s)
    logger.info("Error ellipsoid principal axis 1: {}", U[:, 0])
    logger.info("Error ellipsoid principal axis 2: {}", U[:, 1])
    logger.info("Error ellipsoid principal axis 3: {}", U[:, 2])

    Outliers = Peaks(P.phi[is_outlier], P.iy[is_outlier], P.ix[is_outlier])

    logger.info("Saving peaks to {}...", outfile)
    saveobj(GP, outfile, name="peak_model", append=False)
    saveobj(P, outfile, name="peaks", append=True)
    saveobj(Outliers, outfile, name="outliers", append=True)
    logger.info("Peak finding completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_find_peaks))

if __name__ == "__main__":
    run()
