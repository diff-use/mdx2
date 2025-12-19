"""
Import data and geometry using dxtbx and save NeXus files

This function combines mdx2.import_data and mdx2.find_peaks in a single step
"""

# todo: combine import_data and find_peaks using HDRHistogram for threshold estimation
# todo: implement automatic reindexing option for import_geometry
# todo: implement reflection data import option, save as reflections.nxs
# todo: implement dynamic mask for mdx2.mask_peaks, rather than saving a static mask as numexpr string.
# todo: handle background data import in the same function?

# should the objects all be in an output file called "imported.nxs"?
# in this first version, I'll ignore most of these features and just combine mdx2.import_data and mdx2.find_peaks

from dataclasses import dataclass
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import field

from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.data import ImageSeries, Peaks
from mdx2.dxtbx_machinery import ImageSet
from mdx2.io import nxload, saveobj
from mdx2.processing import find_peaks


@dataclass
class DataParameters:
    """Options for data compression and storage"""

    outfile: str = "data.nxs"  # name of the output NeXus file
    chunks: Optional[tuple[int, int, int]] = field(
        default=None, help="chunking for compression (frames, y, x). Use -1 for default"
    )
    datastore: str = "datastore"  # folder for storing source datasets


@dataclass
class PeakParameters:
    """Options for peak finding"""

    threshold: Optional[float] = None  # pixels with counts above threshold are flagged as peaks
    bin_size: tuple[int, int, int] = (-1, 50, 50)  # region size for dynamic thresholding (frames, y, x), -1=default
    outfile: str = "peaks.nxs"  # name of the output NeXus file


@dataclass
class Parameters:
    """Options for importing x-ray image data"""

    expt: str = field(positional=True)  # experiments file, such as from dials.import
    data: DataParameters = field(default_factory=DataParameters)
    peaks: PeakParameters = field(default_factory=PeakParameters)
    nproc: int = 1  # number of parallel processes


def run_mdx2_import(params):
    exptfile = params.expt
    chunks = params.data.chunks
    nproc = params.nproc
    data_outfile = params.data.outfile
    datastore = params.data.datastore
    peaks_outfile = params.peaks.outfile
    threshold = params.peaks.threshold
    bin_size = params.peaks.bin_size

    image_series = ImageSeries.from_expt(exptfile)
    iset = ImageSet.from_file(exptfile)

    if chunks is not None:
        # override the default chunking
        # if any of the chunks dimensions is <=0, use the default for that dimension
        default_chunks = image_series.data.chunks
        chunks = tuple(c if c > 0 else d for c, d in zip(chunks, default_chunks))
        image_series.data.chunks = chunks
        logger.info(f"Using chunking {chunks} for data compression")

    if bin_size is not None:
        # override the default background region
        # if any of the region dimensions is <=0, use the default for that dimension
        default_bin_size = image_series.data.chunks
        bin_size = tuple(b if b > 0 else d for b, d in zip(bin_size, default_bin_size))
        logger.info(f"Using bin_size {bin_size} for background estimation")

    logger.info("Creating virtual dataset structure...")
    image_series.save(
        data_outfile,
        virtual=True,
        source_directory=datastore,
    )

    # trigger JIT for find_peaks if threshold is not provided
    if threshold is None:
        dummy_data = np.array([0, 1, 2, 3, 4, 5], dtype=iset.dtype)
        _ = find_peaks(dummy_data, bin_size=None)

    def write_stack_find_peaks(istart, istop, filename, datapath):
        source = nxload(filename, "r+")[datapath]
        data = iset.read_stack(istart, istop)
        peak_mask = find_peaks(data, threshold=threshold, bin_size=bin_size)
        # find the coordinates of the peaks, and add istart to the first dimension
        peaks = np.argwhere(peak_mask) + np.array([istart, 0, 0])
        source[:, :, :] = data  # write the data stack to the file.
        return peaks

    slices = [sl for sl in image_series.chunk_slice_along_axis(0)]

    # Access virtual dataset information through public API
    # Reload the ImageSeries to ensure we have the updated virtual dataset
    image_series_reloaded = ImageSeries.load(data_outfile)
    files = image_series_reloaded.virtual_source_files
    vpath = image_series_reloaded.virtual_dataset_path

    # These edge cases should not happen if virtual datasets are implemented correctly, but check anyway
    if len(files) != len(slices):
        raise RuntimeError(f"Virtual dataset mismatch: {len(slices)=} vs {len(files)=}")
    if len(set(files)) != len(files):
        raise RuntimeError("Virtual_source_files contains duplicates, cannot proceed with import.")

    # Properties will raise RuntimeError if internal API has changed
    # If data is not a virtual dataset, they return None
    if files is None or vpath is None:
        raise RuntimeError(
            f"Expected virtual dataset in {data_outfile}, but virtual dataset information is not available."
        )

    logger.info("Writing {} image batches (requested n_jobs: {})...", len(slices), nproc)
    with Parallel(n_jobs=nproc, verbose=10) as parallel:
        log_parallel_backend(parallel)
        hotpixelslist = parallel(
            delayed(write_stack_find_peaks)(sl.start, sl.stop, fn, vpath) for sl, fn in zip(slices, files)
        )
    logger.info("Image data writing completed")

    hotpixels = np.vstack(hotpixelslist)
    peaks = Peaks(image_series.phi[hotpixels[:, 0]], image_series.iy[hotpixels[:, 1]], image_series.ix[hotpixels[:, 0]])
    logger.info("Found {} peak pixels", peaks.size)

    # save the peaks to a file
    logger.info("Saving peaks to {}...", peaks_outfile)
    saveobj(peaks, peaks_outfile, name="peaks")

    logger.info("Image data import completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_mdx2_import))


if __name__ == "__main__":
    run()
