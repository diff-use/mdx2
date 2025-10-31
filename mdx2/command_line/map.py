"""
Create a map from data in an hkl table
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from simple_parsing import ArgumentParser, field

from mdx2.command_line import configure_logging
from mdx2.data import HKLTable
from mdx2.geometry import GridData
from mdx2.utils import loadobj, saveobj

logger = logging.getLogger(__name__)


@dataclass
class Parameters:
    """Options for creating an array from an hkl table"""

    geom: str = field(positional=True)  # NeXus file containing symmetry and crystal
    hkl: str = field(positional=True)  # NeXus file containing hkl_table
    symmetry: bool = True  # apply symmetry operators
    limits: Tuple[float, float, float, float, float, float] = (0, 10, 0, 10, 0, 10)
    """limits for the hkl grid (hmin, hmax, kmin, kmax, lmin, lmax)"""
    signal: str = "intensity"  # column in hkl_table to map
    outfile: str = "map.nxs"  # name of the output NeXus file


# NOTE: should perhaps change so that limits is a required argument


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Parameters, dest="parameters")
    opts = parser.parse_args(args)
    return opts.parameters


def run_map(params):
    """Run the map script"""
    hkl = params.hkl
    geom = params.geom
    outfile = params.outfile
    apply_symmetry = params.symmetry
    signal = params.signal
    hmin, hmax, kmin, kmax, lmin, lmax = params.limits

    T = loadobj(hkl, "hkl_table")
    Symmetry = loadobj(geom, "symmetry")  # used only if symmetry flag is set
    ndiv = T.ndiv

    Hmin = np.round(hmin * ndiv[0]).astype(int)
    Hmax = np.round(hmax * ndiv[0]).astype(int)
    Kmin = np.round(kmin * ndiv[1]).astype(int)
    Kmax = np.round(kmax * ndiv[1]).astype(int)
    Lmin = np.round(lmin * ndiv[2]).astype(int)
    Lmax = np.round(lmax * ndiv[2]).astype(int)

    h_axis = np.arange(Hmin, Hmax + 1) / ndiv[0]
    k_axis = np.arange(Kmin, Kmax + 1) / ndiv[1]
    l_axis = np.arange(Lmin, Lmax + 1) / ndiv[2]

    print("map region:")
    print(f"  h from {h_axis[0]} to {h_axis[-1]} ({h_axis.size} grid points)")
    print(f"  k from {k_axis[0]} to {k_axis[-1]} ({k_axis.size} grid points)")
    print(f"  l from {l_axis[0]} to {l_axis[-1]} ({l_axis.size} grid points)")

    print("generating Miller index array")
    h, k, l = np.meshgrid(h_axis, k_axis, l_axis, indexing="ij")

    Tgrid = HKLTable(h.ravel(), k.ravel(), l.ravel(), ndiv=ndiv)

    if apply_symmetry:
        print("mapping Miller indices to asymmetric unit")
        Tgrid = Tgrid.to_asu(Symmetry)

    print(f"looking up {signal} in data table")
    # lookup in the table
    data = T.lookup(Tgrid.h, Tgrid.k, Tgrid.l, signal).reshape(h.shape)

    print("preparing output array")
    G = GridData((h_axis, k_axis, l_axis), data, axes_names=["h", "k", "l"])
    saveobj(G, outfile, name=signal, append=False)

    print("done!")


def run(args=None):
    """Run the map script"""
    configure_logging(filename="mdx2.map.log")
    params = parse_arguments(args=args)
    run_map(params)


if __name__ == "__main__":
    run()
