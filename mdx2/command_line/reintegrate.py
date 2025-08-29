"""
Reintegrate on a different grid, applying corrections, scaling, and merging symmetry-equivalent observations.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from simple_parsing import ArgumentParser, field

from mdx2.data import HKLTable, HKLGrid
from mdx2.utils import (
    loadobj,
    nxload,  # mask is too big to read all at once?
    saveobj,
)


@dataclass
class Parameters:
    """Options for integrating counts on a Miller index grid"""

    geom: str = field(positional=True)  # NeXus data file containing miller_index
    data: str = field(positional=True)  # NeXus data file containing image_series
    mask: Optional[str] = None  # NeXus data file containing mask
    subdivide: Tuple[int, int, int] = (1, 1, 1)  # subdivisions of the Miller index grid
    scale: Optional[str] = None  # NeXus file with scaling model
    background: Optional[str] = None  # NeXus file with background binned_image_series
    split: Optional[str] = field(choices=["randomHalf", "weightedRandomHalf", "Friedel"])
    """also merge data into separate columns based on splitting criteria"""
    nproc: int = 1  # number of parallel processes
    outfile: str = "reintegrated.nxs"  # name of the output data


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Parameters, dest="parameters")
    opts = parser.parse_args(args)
    return opts.parameters


def run_reintegrate(params):
    # TODO: implement splitting
    # TODO: what if scaling models are missing from the nexus file?

    miller_index = loadobj(params.geom, "miller_index")
    image_series = loadobj(params.data, "image_series")
    corrections = loadobj(params.geom, "corrections")
    crystal = loadobj(params.geom, "crystal")
    symmetry = loadobj(params.geom, "symmetry")
    background = loadobj(params.background, "binned_image_series") if params.background else None
    scaling_model = loadobj(params.scale, "scaling_model") if params.scale else None
    absorption_model = loadobj(params.scale, "absorption_model") if params.scale else None
    offset_model = loadobj(params.scale, "offset_model") if params.scale else None
    detector_model = loadobj(params.scale, "detector_model") if params.scale else None
    mask = nxload(params.mask).entry.mask.signal if params.mask else None

    def intchunk(sl):
        ims = image_series[sl]
        if mask is not None:
            tab = ims.index(miller_index, mask=mask[sl].nxdata)  # added nxdata to deal with NXfield wrapper
        else:
            tab = ims.index(miller_index)
        if len(tab) == 0:
            return None  # signal that no pixels were integrated
        tab.ndiv = params.subdivide
        tab = tab.bin(count_name="pixels")
        tab.phi /= tab.pixels
        tab.iy /= tab.pixels
        tab.ix /= tab.pixels
        return tab

    def calc_corrections(tab):
        UB = crystal.ub_matrix
        s = UB @ np.stack((tab.h, tab.k, tab.l))
        tab.s = np.sqrt(np.sum(s * s, axis=0))
        correction_factors = corrections.interpolate(tab.iy, tab.ix)
        solid_angle = correction_factors["solid_angle"]
        solid_angle *= correction_factors["attenuation"]
        solid_angle *= correction_factors["polarization"]
        solid_angle *= correction_factors["efficiency"]
        tab.multiplicity = tab.pixels * correction_factors["d3s"] * np.prod(tab.ndiv) / np.linalg.det(UB)
        b = scaling_model.interp(tab.phi) if scaling_model else 1.0
        a = absorption_model.interp(tab.ix, tab.iy, tab.phi) if absorption_model else 1.0
        d = detector_model.interp(tab.ix, tab.iy) if detector_model else 1.0
        c = offset_model.interp(tab.s, tab.phi) if offset_model else 0.0
        bg_rate = background.interpolate(tab.phi, tab.iy, tab.ix) if background else 0.0
        tab.scale = tab.seconds * solid_angle * a * b * d
        tab.background_counts = tab.seconds * (bg_rate + c * a * d * solid_angle)
        tab = tab.to_asu(symmetry)
        del tab.phi, tab.iy, tab.ix, tab.seconds, tab.s, tab.op
        return tab

    with Parallel(n_jobs=params.nproc, verbose=10, return_as="generator_unordered") as parallel:
        slices = list(image_series.chunk_slice_iterator())
        tab_chunk = parallel(delayed(intchunk)(sl) for sl in slices)
        grid = None
        for tab in tab_chunk:
            if tab is None:
                continue
            corrected_table = calc_corrections(tab)
            if grid is None:
                grid = HKLGrid.from_table(corrected_table)
            else:
                grid.accumulate_from_table(corrected_table, resize=True)

    if grid is None:
        raise ValueError("No valid data found")

    print(f"Saving table of integrated data to {params.outfile}")
    hkl_table = grid.to_table(sparse=True)
    hkl_table.h = hkl_table.h.astype(np.float32)
    hkl_table.k = hkl_table.k.astype(np.float32)
    hkl_table.l = hkl_table.l.astype(np.float32)
    hkl_table.counts = hkl_table.counts.astype(np.int32)
    hkl_table.background_counts = hkl_table.background_counts.astype(np.int32)
    hkl_table.scale = hkl_table.scale.astype(np.float32)
    hkl_table.multiplicity = hkl_table.multiplicity.astype(np.float32)
    saveobj(hkl_table, params.outfile, name="hkl_table", append=False)
    print("done!")


def run(args=None):
    """Run the correct script"""
    params = parse_arguments(args=args)
    run_reintegrate(params)


if __name__ == "__main__":
    run()
