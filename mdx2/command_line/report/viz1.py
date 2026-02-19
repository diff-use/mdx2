"""Visualization helper: generate a slice map figure from NeXus files."""

from dataclasses import dataclass
from typing import Optional, Tuple

import os
import matplotlib.pyplot as plt
from nexusformat.nexus import nxload
from mdx2.mdx2.command_line.report._functions import create_slices, map2xr
import sys

from mdx2.command_line import make_argument_parser, with_logging, with_parsing

# prevent local mdx2/ directory from shadowing mdx2 package import
if "" in sys.path:
    sys.path.remove("")


def run_viz1(geom: Optional[str] = None, limits: Optional[Tuple[int, int, int, int, int, int]] = None):
    #Will have to specify directory
    if not {
        "slice_all_iso.nxs",
        "slice_all.nxs",
        "slice_crystal1.nxs",
        "slice_crystal2.nxs",
    }.issubset(os.listdir()):
        create_slices(geom=geom, limits=limits)

    imav = nxload("slice_all_isoavg.nxs")["/entry/isoavg"]
    imsub = nxload("slice_all.nxs")["/entry/intensity"]

    im = imsub + imav

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    map2xr(im).isel(l=2).plot(x="h", y="k", ax=ax1, vmin=0, vmax=1.1, cmap="viridis")
    map2xr(imsub).isel(l=2).plot(
        x="h", y="k", ax=ax2, vmin=-0.05, vmax=0.05, cmap="bwr"
    )
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    fig.savefig("viz1_plot.png", dpi=150, bbox_inches="tight")


@dataclass
class Parameters:
    """Options for generating the viz1 slice map figure."""

    geom: Optional[str] = None
    limits: Optional[Tuple[int, int, int, int, int, int]] = None


def run_viz1(params: Parameters):
    """Entry point used by the command-line wrapper."""
    _run_viz1(geom=params.geom, limits=params.limits)


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_viz1))


if __name__ == "__main__":
    run()