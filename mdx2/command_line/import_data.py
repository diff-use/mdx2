"""
Import x-ray image data using the dxtbx machinery
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from simple_parsing import field, subgroups

from mdx2.command_line import log_parallel_backend, make_argument_parser, with_logging, with_parsing
from mdx2.data import LazyImageSeries, SparseImageSeries, VirtualImageSeries
from mdx2.dxtbx_machinery import ImageSet
from mdx2.io import nxload, saveobj
from mdx2.processing import find_peaks


@dataclass
class Processing:
    pass


@dataclass
class NoProcessing(Processing):
    """No per-image processing performed"""

    pass


@dataclass
class StrongPixelsProcessing(Processing):
    """Optional detection and masking of strong pixels"""

    mask: bool = True  # remove strong pixels from the image data by masking them out
    bin_size: tuple[int, int, int] = (-1, 50, 50)  # region size for dynamic thresholding (frames, y, x), -1=default


@dataclass
class Parameters:
    """Options for importing x-ray image data"""

    expt: str = field(positional=True)  # experiments file, such as from dials.import
    chunks: Optional[Tuple[int, int, int]] = field(
        default=None, help="chunking for compression (frames, y, x). Use -1 for default"
    )
    outfile: str = "data.nxs"  # name of the output NeXus file
    nproc: int = 1  # number of parallel processes (or 1 for sequential, -1 for all CPUs, -N for all but N+1)
    datastore: str = "datastore"  # folder for storing source datasets
    processing: Processing = subgroups(
        {"none": NoProcessing, "strong_pixels": StrongPixelsProcessing},
        default="strong_pixels",
    )


def run_import_data(params):
    exptfile = params.expt
    chunks = params.chunks
    nproc = params.nproc
    outfile = params.outfile
    datastore = params.datastore
    process_strong_pixels = isinstance(params.processing, StrongPixelsProcessing)

    if process_strong_pixels:
        mask_strong = params.processing.mask
        bin_size = params.processing.bin_size

    logger.info("Loading experiment metadata...")
    image_series = LazyImageSeries.from_expt(exptfile)
    iset = ImageSet.from_file(exptfile)
    logger.info("Image data shape (phi, iy, ix): {}", image_series.shape)

    if chunks is not None:
        # override the default chunking
        # if any of the chunks dimensions is <=0, use the default for that dimension
        default_chunks = image_series.data.chunks
        chunks = tuple(c if c > 0 else d for c, d in zip(chunks, default_chunks))
        image_series.data.chunks = chunks
        logger.info("Using chunking: {}", chunks)

    if process_strong_pixels and bin_size is not None:
        # override the default background region
        # if any of the region dimensions is <=0, use the default for that dimension
        default_bin_size = image_series.data.chunks
        bin_size = tuple(b if b > 0 else d for b, d in zip(bin_size, default_bin_size))
        logger.info(f"Using bin_size {bin_size} for background estimation")

    logger.info("Creating virtual dataset structure...")
    virtual_image_series = VirtualImageSeries.create(
        image_series,
        outfile,
        source_directory=datastore,
    )

    slices = [sl for sl in virtual_image_series.chunk_slice_along_axis(0)]

    # Access virtual dataset information through public API
    files = virtual_image_series.virtual_source_files
    vpath = virtual_image_series.virtual_dataset_path

    # These edge cases should not happen if virtual datasets are implemented correctly, but check anyway
    if len(files) != len(slices):
        raise RuntimeError(f"Virtual dataset mismatch: {len(slices)=} vs {len(files)=}")
    if len(set(files)) != len(files):
        raise RuntimeError("Virtual_source_files contains duplicates, cannot proceed with import.")

    if not process_strong_pixels:
        # original behavior of mdx2.import_data

        def write_stack(istart, istop, filename, datapath):
            source = nxload(filename, "r+")[datapath]
            data = iset.read_stack(istart, istop)
            source[:, :, :] = data

        logger.info("Writing {} image batches (requested n_jobs: {})...", len(slices), nproc)
        with Parallel(n_jobs=nproc, verbose=10) as parallel:
            log_parallel_backend(parallel)
            parallel(delayed(write_stack)(sl.start, sl.stop, fn, vpath) for sl, fn in zip(slices, files))
        logger.info("Image data writing completed")

    else:
        # trigger JIT compilation before the fork
        dummy_data = np.array([0, 1, 2, 3, 4, 5], dtype=iset.dtype)
        _ = find_peaks(dummy_data, bin_size=None)

        def write_stack_find_strong(istart, istop, filename, datapath):
            source = nxload(filename, "r+")[datapath]
            data = iset.read_stack(istart, istop)
            peak_mask = find_peaks(data, bin_size=bin_size)
            # find the coordinates of the peaks, and add istart to the first dimension
            indices = np.argwhere(peak_mask)
            if mask_strong:
                values = data[indices[:, 0], indices[:, 1], indices[:, 2]]
                data[indices[:, 0], indices[:, 1], indices[:, 2]] = iset.maskval
            else:
                values = np.zeros(shape=indices.shape[0], dtype=data.dtype)
            source[:, :, :] = data  # write the data stack to the file.
            indices += np.array([istart, 0, 0])  # add the offset
            return indices, values

        logger.info("Writing {} image batches (requested n_jobs: {})...", len(slices), nproc)
        with Parallel(n_jobs=nproc, verbose=10) as parallel:
            log_parallel_backend(parallel)
            results = parallel(
                delayed(write_stack_find_strong)(sl.start, sl.stop, fn, vpath) for sl, fn in zip(slices, files)
            )
        logger.info("Image data writing completed")

        indices = np.vstack([res[0] for res in results])
        values = np.concatenate([res[1] for res in results])  # vstack? or concatenate?

        logger.info("found {} strong pixels", indices.shape[0])

        sparse_image_series = SparseImageSeries(
            virtual_image_series.phi,
            virtual_image_series.iy,
            virtual_image_series.ix,
            indices,
            values,
            virtual_image_series.exposure_times,
        )

        logger.info("Saving strong pixels to {}...", outfile)
        saveobj(sparse_image_series, outfile, "strong_pixels", append=True)

    logger.info("Image data import completed successfully")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_import_data))


if __name__ == "__main__":
    run()
