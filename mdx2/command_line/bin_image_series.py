"""
Bin down an image stack
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from simple_parsing import ArgumentParser, field

from mdx2.command_line import configure_logging
from mdx2.utils import (
    loadobj,
    nxload,  # mask is too big to read all at once?
    saveobj,
)

logger = logging.getLogger(__name__)


@dataclass
class Parameters:
    """Options for binning a series of images in the phi and xy directions"""

    data: str = field(positional=True)  # NeXus data file containing the image_series
    bins: Tuple[int, int, int] = field(positional=True)  # number per bin in each direction (frames, y, x)
    mask: Optional[str]  # name of NeXus file containing the mask
    valid_range: Optional[Tuple[int, int]]  # minimum and maximum valid data values
    outfile: str = "binned.nxs"  # name of the output NeXus file
    nproc: int = 1  # number of parallel processes


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Parameters, dest="parameters")
    opts = parser.parse_args(args)
    return opts.parameters


def run_bin_image_series(params):
    """Run the binning algorithm"""
    data = params.data
    bins = params.bins
    outfile = params.outfile
    valid_range = params.valid_range
    nproc = params.nproc
    maskfile = params.mask

    image_series = loadobj(data, "image_series")

    if maskfile is not None:
        # MA = loadobj(maskfile,'mask')
        # mask = MA.data
        nxs = nxload(maskfile)  # <-- loadobj fails if the array is too large. weird
        mask = nxs.entry.mask.signal  # nxfield
    else:
        mask = None

    binned = image_series.bin_down(bins, valid_range=valid_range, nproc=nproc, mask=mask)

    print(f"saving to file: {outfile}")
    nxs = saveobj(binned, outfile, name="binned_image_series")


def run(args=None):
    """Run the binning script"""
    configure_logging(filename="mdx2.bin_image_series.log")
    params = parse_arguments(args=args)
    run_bin_image_series(params)


if __name__ == "__main__":
    run()
