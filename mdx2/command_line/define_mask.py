"""Generate image masks for integration"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger
from simple_parsing import subgroups

from mdx2.command_line import make_argument_parser, with_logging, with_parsing
from mdx2.io import loadobj, saveobj


@dataclass
class MaskType:
    pass


@dataclass
class GaussianPeakMask(MaskType):
    """A 3D gaussian, e.g. from mdx2.fit_peaks"""

    peaks: str  # file containing the peak_model (required)
    contour_level: float = 3.0  # contour level of the peak mask (standard deviations)
    exclude: bool = True  # True means pixels within the contour are excluded, otherwise polarity is flipped

    def __post_init__(self):
        """Validate contour_level parameter"""
        if self.contour_level <= 0:
            raise ValueError(f"contour_level must be > 0, got {self.contour_level}")


@dataclass
class ResolutionMask(MaskType):
    """Remove peaks as a function of resolution"""

    geom: str  # file containing the crystal (required)
    d_min: Optional[float] = None  # highest resolution to include
    d_max: Optional[float] = None
    exclude: bool = False  # False means only include pixels within the resolution range

    def __post_init__(self):
        """Either d_min or d_max (or both) must be set, and both must be greater than zero and in correct order"""
        if self.d_min is None and self.d_max is None:
            raise ValueError("Either d_min or d_max must be set")
        if self.d_min is not None and self.d_min <= 0:
            raise ValueError(f"Expected d_min > 0, got {self.d_min}")
        if self.d_max is not None and self.d_max <= 0:
            raise ValueError(f"Expected d_max > 0, got {self.d_max}")
        if self.d_max is not None and self.d_min is not None and self.d_min >= self.d_max:
            raise ValueError(f"Expected d_max > d_min, got d_max={self.d_max}, d_min={self.d_min}")


class Parameters:
    """Options for creating a peak mask"""

    outfile: str = "mask.nxs"  # name of the output NeXus file
    masking: MaskType = subgroups(
        {"gaussian_peak": GaussianPeakMask, "resolution": ResolutionMask},
        default="gaussian_peak",
    )


def define_resolution_mask(params):
    crystal = loadobj(params.geom, "crystal")
    mask_expressions = []
    if params.d_min is not None:
        pass # create the d_min mask expression and append to mask_expressions
    if params.d_max is not None:
        pass # create the d_max mask expression and append to mask_expressions
    if len(mask_expressions) == 2:
        expr = f"({}) & ({})".format(*mask_expressions)
    elif len(mask_expressions) == 1:
        expr = mask_expressions[0]
    else:
        raise RuntimeError("No resolution conditions were defined") # should never happen if using dataclass to validate
    if params.exclude: # the area between the resolution cutoffs is excluded (e.g. for removing ice rings)
        expr = f"~({expr})"
    raise NotImplementedError("Resolution mask not yet implemented")
    return expr


def define_gaussian_peak_mask(params):
    peak_model = loadobj(params.peaks, "peak_model")
    raise NotImplementedError("Resolution mask not yet implemented")


def run_define_mask(params):
    """Create the mask definition and write to a file"""
    outfile = params.outfile

    if isinstance(params.masking, GaussianPeakMask):
        maskobj = define_gaussian_peak_mask(params.masking)
    elif isinstance(params.masking, ResolutionMask):
        maskobj = define_resolution_mask(params.masking)
    else:
        raise ValueError(f"Unknown masking parameter type {params.masking}")

    saveobj(maskobj, outfile, name="mask")

    logger.info("Saving mask to {}...", outfile)


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(with_logging()(run_define_mask))


if __name__ == "__main__":
    run()
