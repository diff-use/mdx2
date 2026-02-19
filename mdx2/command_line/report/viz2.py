"""Visualization helper: generate comparison slice plots for two crystals."""

from dataclasses import dataclass

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from nexusformat.nexus import nxload
from mdx2.mdx2.command_line.report._functions import create_slices, map2xr
import sys

from mdx2.command_line import make_argument_parser, with_logging, with_parsing

# prevent local mdx2/ directory from shadowing mdx2 package import
if "" in sys.path:
    sys.path.remove("")


def run_viz2():
    #Will have to specify directory
    if [""] not in os.listdir():
        geom = input("Enter geom: ")
        limits = input("Enter limits: ")
        create_slices(geom=geom, limits=limits)

    imsub1 = nxload("slice_crystal1.nxs")["/entry/intensity"]
    imsub2 = nxload("slice_crystal2.nxs")["/entry/intensity"]

    a = map2xr(imsub1).isel(l=2)
    b = map2xr(imsub2).isel(l=2)
    d = xr.concat((a, b, b - a), pd.Index(["1", "2", "2-1"], name="Crystal"))
    fh = d[:, 330:391, 240:301].plot(
        x="h", y="k", col="Crystal", vmin=-0.04, vmax=0.04, cmap="bwr"
    )
    [ax.set_aspect("equal") for ax in fh.axs.flatten()]
    fh.fig.savefig("viz2_plot.png", dpi=150, bbox_inches="tight")


@dataclass
class Parameters:
    """Options for generating the viz2 comparison plot."""

    # Currently interactive via stdin; no CLI options yet.
    pass


def run_viz2(params: Parameters):
    """Entry point used by the command-line wrapper."""
    _run_viz2()


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_viz2))


if __name__ == "__main__":
    run()
