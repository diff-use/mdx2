"""
Import x-ray image data using the dxtbx machinery
"""

import argparse

from mdx2.data import ImageSeries
from mdx2.dxtbx_machinery import ImageSet
from mdx2.utils import saveobj


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("expt", help="experiments file, such as from dials.import")
    parser.add_argument("--outfile", default="data.nxs", help="name of the output NeXus file")
    parser.add_argument("--chunks", nargs=3, type=int, metavar="N", help="chunking for compression (frames, y, x)")
    parser.add_argument("--nproc", type=int, default=1, metavar="N", help="number of parallel processes")
    params = parser.parse_args(args)
    return params


def run_import_data(params):
    exptfile = params.expt
    chunks = params.chunks
    nproc = params.nproc
    outfile = params.outfile

    image_series = ImageSeries.from_expt(exptfile)
    iset = ImageSet.from_file(exptfile)

    if chunks is not None:
        image_series.data.chunks = tuple(chunks)

    saveobj(image_series, outfile, name="image_series")

    nbatches = image_series.data.chunks[0]

    if nproc == 1:
        iset.read_all(image_series.data, nbatches)
    else:
        iset.read_all_parallel(image_series.data, nbatches, nproc)

    print("done!")


def run(args=None):
    """Run the import data script"""
    params = parse_arguments(args=args)
    run_import_data(params)


if __name__ == "__main__":
    run()
