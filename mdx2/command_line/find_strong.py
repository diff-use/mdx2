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
from mdx2.data import SparseImageSeries
from mdx2.io import loadobj, saveobj
from mdx2.processing import find_peaks


@dataclass
class Parameters:
    """Options for finding strong pixels in an image stack"""

    data: str = field(positional=True)  # NeXus data file containing image_series
    count_threshold: Optional[float] = None  # pixels with counts above threshold are flagged as peaks
    outfile: str = "strong.nxs"  # name of the output NeXus file
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)


def run_find_strong(params):
    """Run the find strong pixels in the data"""
    data = params.data
    count_threshold = params.count_threshold
    outfile = params.outfile
    nproc = params.nproc
    p_value = 1e-6  # NOTE: does this need to be a CLI option?

    logger.info("Loading the image data...")
    image_series = loadobj(data, "image_series")

    if count_threshold is not None:
        logger.info("Finding pixels above threshold: {}", count_threshold)
    else:
        logger.info("Estimating count thresholds dynamically for each data chunk")

    # Pre-compile JIT functions before forking if using multiprocessing
    if count_threshold is None and nproc > 1:
        dummy_data = np.array([0, 1, 2, 3, 4, 5], dtype=image_series.data.dtype)
        # Trigger JIT compilation
        _ = find_peaks(dummy_data, threshold=count_threshold, p_value=p_value, bin_size=None)
        logger.info("Pre-compiled find_peaks JIT functions for multiprocessing")

    # Find peaks in parallel
    def peaksearch(sl):
        ims = image_series[sl]  # slicing reads the data --> InMemoryImageSeries
        peak_mask = find_peaks(ims.data, threshold=count_threshold, p_value=p_value, bin_size=None)
        indices = np.argwhere(peak_mask)
        values = np.zeros(shape=indices.shape[0], dtype=ims.data.dtype)
        indices += np.array([sl[0].start, sl[1].start, sl[2].start])
        return indices, values

    slices = list(image_series.chunk_slice_iterator())
    logger.info("Searching for peaks in {} image chunks (requested n_jobs: {})...", len(slices), nproc)
    with Parallel(n_jobs=nproc, verbose=10) as parallel:
        log_parallel_backend(parallel)
        results = parallel(delayed(peaksearch)(sl) for sl in slices)
    logger.info("Strong pixel search completed")

    indices = np.vstack([res[0] for res in results])
    values = np.concatenate([res[1] for res in results])

    logger.info("found {} strong pixels", indices.shape[0])

    sparse_image_series = SparseImageSeries(
        image_series.phi,
        image_series.iy,
        image_series.ix,
        indices,
        values,
        image_series.exposure_times,
    )
    logger.info("Saving strong pixels to {}...", outfile)
    saveobj(sparse_image_series, outfile, "strong_pixels")

    logger.info("Finding strong pixels completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_find_strong))

if __name__ == "__main__":
    run()
