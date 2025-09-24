"""
Import x-ray image data using the dxtbx machinery
"""

from dataclasses import dataclass

from joblib import Parallel, delayed
from simple_parsing import ArgumentParser, field

from mdx2.data import ImageSeries
from mdx2.dxtbx_machinery import ImageSet
from mdx2.utils import nxload, saveobj


@dataclass
class Parameters:
    """Options for importing x-ray image data"""

    expt: str = field(positional=True)  # experiments file, such as from dials.import
    chunks: tuple[int, int, int] = None  # chunking for compression (frames, y, x)
    outfile: str = "data.nxs"  # name of the output NeXus file
    nproc: int = 1  # number of parallel processes
    datastore: str = None  # folder for storing source datasets


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Parameters, dest="parameters")
    opts = parser.parse_args(args)
    return opts.parameters


def run_import_data(params):
    exptfile = params.expt
    chunks = params.chunks
    nproc = params.nproc
    outfile = params.outfile
    datastore = params.datastore

    image_series = ImageSeries.from_expt(exptfile)
    iset = ImageSet.from_file(exptfile)

    if chunks is not None:
        # override the default chunking
        image_series.data.chunks = tuple(chunks)

    if datastore is None:
        saveobj(image_series, outfile, name="image_series")

        nbatches = image_series.data.chunks[0]

        if nproc == 1:
            iset.read_all(image_series.data, nbatches)
        else:
            iset.read_all_parallel(image_series.data, nbatches, nproc)
    else:  # use a virtual NXfield linking to source files in datastore/
        nxobj = image_series.save(
            outfile,
            virtual=True,
            source_directory=datastore,
        )

        def write_stack(istart, istop, filename, datapath):
            source = nxload(filename, "r+")[datapath]
            data = iset.read_stack(istart, istop)
            source[:, :, :] = data

        slices = [sl for sl in image_series.chunk_slice_along_axis(0)]
        files = nxobj.data._vfiles

        with Parallel(n_jobs=nproc, verbose=1) as parallel:
            parallel(delayed(write_stack)(sl.start, sl.stop, fn, nxobj.data._vpath) for sl, fn in zip(slices, files))

    print("done!")


def run(args=None):
    """Run the import data script"""
    params = parse_arguments(args=args)
    run_import_data(params)


if __name__ == "__main__":
    run()
