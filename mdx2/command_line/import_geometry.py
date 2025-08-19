"""
Import experimental geometry using the dxtbx machinery
"""

import argparse

import mdx2.geometry as geom
from mdx2.utils import saveobj


def parse_arguments(args=None):
    """Parse command line arguments for the import geometry script."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("expt", help="dials experiments file, such as refined.expt")
    parser.add_argument(
        "--sample_spacing",
        nargs=3,
        metavar=("PHI", "IY", "IX"),
        type=int,
        default=[1, 10, 10],
        help="inverval between samples in degrees or pixels",
    )
    parser.add_argument("--outfile", default="geometry.nxs", help="name of the output NeXus file")
    params = parser.parse_args(args)
    return params


def run_import_geometry(params):
    """Run the import geometry script with the given parameters"""
    exptfile = params.expt
    spacing_phi_px = tuple(params.sample_spacing)
    spacing_px = spacing_phi_px[1:]
    outfile = params.outfile

    print("Computing miller index lookup grid")
    miller_index = geom.MillerIndex.from_expt(
        exptfile,
        sample_spacing=spacing_phi_px,
    )

    print("Computing geometric correction factors")
    corrections = geom.Corrections.from_expt(
        exptfile,
        sample_spacing=spacing_px,
    )

    print("Gathering space group info")
    symmetry = geom.Symmetry.from_expt(exptfile)

    print("Gathering unit cell info")
    crystal = geom.Crystal.from_expt(exptfile)

    print(f"Saving geometry to {outfile}")

    saveobj(crystal, outfile, name="crystal", append=False)
    saveobj(symmetry, outfile, name="symmetry", append=True)
    saveobj(corrections, outfile, name="corrections", append=True)
    saveobj(miller_index, outfile, name="miller_index", append=True)

    print("done!")


def run(args=None):
    """Run the import geometry script"""
    params = parse_arguments(args=args)
    run_import_geometry(params)


if __name__ == "__main__":
    run()
