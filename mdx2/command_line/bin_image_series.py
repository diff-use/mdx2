"""
Bin down an image stack
"""

import argparse

from mdx2.utils import (
    loadobj,
    nxload,  # mask is too big to read all at once?
    saveobj,
)


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data", help="NeXus data file containing the image_series")
    parser.add_argument("bins", nargs=3, type=int, metavar="N", help="number per bin in each direction (frames, y, x)")
    parser.add_argument("--outfile", default="binned.nxs", help="name of the output NeXus file")
    parser.add_argument("--valid_range", nargs=2, type=int, metavar="N", help="minimum and maximum valid data values")
    parser.add_argument("--nproc", type=int, default=1, metavar="N", help="number of parallel processes")
    parser.add_argument("--mask", help="name of NeXus file containing the mask")
    params = parser.parse_args(args)

    return params


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
    params = parse_arguments(args=args)
    run_bin_image_series(params)


if __name__ == "__main__":
    run()
