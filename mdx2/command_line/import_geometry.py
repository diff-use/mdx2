"""
Import experimental geometry using the dxtbx machinery
"""

from dataclasses import dataclass

from simple_parsing import ArgumentParser, field

import mdx2.geometry as geom
from mdx2.utils import saveobj


@dataclass
class Parameters:
    """Options for importing experimental geometry"""

    expt: str = field(positional=True)  # dials experiments file, such as refined.expt
    sample_spacing: tuple[int, int, int] = (1, 10, 10)  # inverval in degrees or pixels (phi, iy, ix)
    outfile: str = "geometry.nxs"  # name of the output NeXus file


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Parameters, dest="parameters")
    opts = parser.parse_args(args)
    return opts.parameters


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
