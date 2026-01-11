"""Generate image masks for integration

Examples:

mdx2.define_mask peaks.nxs --contour_level 3.0 --exclude True --outfile mask.nxs

"""

import re
from dataclasses import dataclass

from loguru import logger
from simple_parsing import field

from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.geometry import DynamicMask, GaussianPeakMask, ResolutionMask
from mdx2.io import loadobj, saveobj


@dataclass
class Parameters:
    """A 3D gaussian, e.g. from mdx2.fit_peaks"""

    geom: str = field(positional=False)  # file containing the crystal (required for d_min) and symmetry objects
    peaks: str = field(positional=True)  # file containing the peak_model (required)
    contour_level: float = 3.0  # contour level of the peak mask (standard deviations)
    exclude_peaks: bool = True  # True means pixels within the contour are excluded, otherwise polarity is flipped
    d_min: float = None  # data beyond resolution d_min are masked
    outfile: str = "mask.nxs"

    def __post_init__(self):
        """Validate contour_level parameter"""
        if self.contour_level <= 0:
            raise ValueError(f"contour_level must be > 0, got {self.contour_level}")
        if self.d_min is not None and self.geom is None:
            raise ValueError("If d_min is specified, geom must also be provided")
        if self.d_min is not None and self.d_min <= 0:
            raise ValueError(f"d_min must be > 0, got {self.d_min}")


def run_define_mask(params):
    """Create the mask definition and write to a file"""

    logger.info("Loading peak model from {}...", params.peaks)
    peak_model = loadobj(params.peaks, "peak_model")

    logger.info("Creating peak mask with contour level {}...", params.contour_level)
    mask = GaussianPeakMask(peak_model.r0, peak_model.sigma, params.contour_level)

    symmetry = loadobj(params.geom, "symmetry")
    if symmetry.reflection_conditions:  # remove forbidden reflections from the mask
        logger.info("Applying reflection conditions {}...", symmetry.reflection_conditions)
        expr = re.sub(r"(?<!\w)(h|k|l)(?!\w)", r"floor(\1 + 0.5)", symmetry.reflection_conditions)
        is_refl = DynamicMask(expr, "h", "k", "l")
        mask = mask & is_refl

    if not params.exclude_peaks:  #  flip the peak mask
        mask = ~mask  # include everything that is not a peak

    if params.d_min is not None:  # apply the resolution mask
        logger.info("reading crystal from {} for d_min mask...", params.geom)
        crystal = loadobj(params.geom, "crystal")
        rmask = ResolutionMask(crystal.ub_matrix, params.d_min)
        mask = rmask | mask  # combine the masks

    logger.info("Saving mask to {}...", params.outfile)
    saveobj(mask, params.outfile, name="mask")


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_define_mask))


if __name__ == "__main__":
    run()
