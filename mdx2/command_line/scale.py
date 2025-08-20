"""
Fit scaling model to unmerged corrected intensities
"""

import os
from dataclasses import dataclass

import numpy as np
from simple_parsing import ArgumentGenerationMode, ArgumentParser, NestedMode, field

from mdx2.data import HKLTable
from mdx2.scaling import BatchModelRefiner, ScaledData
from mdx2.utils import loadobj, saveobj


@dataclass
class ScalingModelParameters:
    """Options to control refinement of the scaling model"""

    enable: bool = True  # include smooth scale factor vs. phi
    alpha: float = 1.0  # amount to rescale the default smoothness (regularization parameter)
    dphi: float = 1.0  # spacing of phi control points in degrees
    niter: int = 10  # maximum iterations in refinement
    x2tol: float = 1.0e-4  # maximum change in x2 to stop refinement early
    outlier: float = 10.0  # standard error cutoff for outlier rejection after refinement


@dataclass
class OffsetModelParameters:
    """Options to control refinement of the offset model"""

    enable: bool = False  # include smooth offset vs. resolution and phi
    alpha_x: float = 1.0  # smoothness vs. s (resolution), multiplies the regularization parameter
    alpha_y: float = 1.0  # smoothness vs. phi, multiplies the regularization parameter
    alpha_min: float = 0.001  # deviation from offset.min, multiplies the regularization parameter
    min: float = 0.0  # minimum value of offset
    dphi: float = 2.5  # spacing of phi control points in degrees
    ns: int = 31  # number of s (resolution) control points
    niter: int = 5  # maximum iterations in refinement
    x2tol: float = 1.0e-3  # maximum change in x2 to stop refinement early
    outlier: float = 5.0  # standard error cutoff for outlier rejection after refinement


@dataclass
class DetectorModelParameters:
    """Options to control refinement of the detector model"""

    enable: bool = False  # include smooth scale vs. detector xy position
    alpha: float = 1.0  # smoothness vs. xy position: multiplies the regularization parameter
    nx: int = 200  # number of grid control points in the x direction
    ny: int = 200  # number of grid control points in the y direction
    niter: int = 5  # maximum iterations in refinement
    x2tol: float = 1.0e-3  # maximum change in x2 to stop refinement early
    outlier: float = 5.0  # standard error cutoff for outlier rejection after refinement


@dataclass
class AbsorptionModelParameters:
    """Options to control refinement of the absorption model"""

    enable: bool = True  # include smooth scale vs. detector xy position and phi
    alpha_xy: float = 10.0  # smoothness vs. xy position: multiplies the regularization parameter
    alpha_z: float = 1.0  # smoothness vs. phi: multiplies the regularization parameter
    nx: int = 20  # number of grid control points in the x direction
    ny: int = 20  # number of grid control points in the y direction
    dphi: float = 5.0  # spacing of phi control points in degrees
    niter: int = 5  # maximum iterations in refinement
    x2tol: float = 1.0e-4  # maximum change in x2 to stop refinement early
    outlier: float = 5.0  # standard error cutoff for outlier rejection after refinement


@dataclass
class Parameters:
    """Options for refining a scaling model to unmerged corrected intensities"""

    hkl: str = field(positional=True, nargs="+")  # NeXus file(s) containing hkl_table
    scaling: ScalingModelParameters
    absorption: AbsorptionModelParameters
    detector: DetectorModelParameters
    offset: OffsetModelParameters
    outfile: str = field(nargs="*")
    """name of the output NeXus file(s). If omitted, will attempt a sensible name such as scales.nxs"""
    mca2020: bool = False
    """shortcut for --scaling.enable True --offset.enable True --detector.enable True --absorption.enable True"""


def generate_default_outfiles(infiles):
    """Generate default output file names based on input file names.

    - If the input files are in different directories, returns a list of scales.nxs in each directory.
    - If the input files are in the same directory, returns a single scales.nxs file with a unique postfix
    based on the input file names.
    - If the input files have a common pattern, returns a list of scales_<postfix>.nxs
    where <postfix> is derived from the input file names.
    - If the input files do not match any of these criteria, returns None.
    """
    dirs = [os.path.dirname(fn) for fn in infiles]
    if len(set(dirs)) == len(dirs):  # dirs are unique
        return [os.path.join(d, "scales.nxs") for d in dirs]
    if len(set(dirs)) == 1:  # dirs are identical
        roots = [os.path.splitext(os.path.split(fn)[-1])[0] for fn in infiles]
        if all(["_" in root for root in roots]):
            postfix = [root.split("_")[-1] for root in roots]
            if len(set(postfix)) == len(postfix):  # postfixes are unique
                return [os.path.join(dirs[0], f"scales_{pf}.nxs") for pf in postfix]
    return None


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = ArgumentParser(
        description=__doc__,
        argument_generation_mode=ArgumentGenerationMode.NESTED,
        nested_mode=NestedMode.WITHOUT_ROOT,
    )
    parser.add_arguments(Parameters, dest="parameters")
    opts = parser.parse_args(args)

    if opts.parameters.mca2020:
        opts.parameters.scaling.enable = True
        opts.parameters.detector.enable = True
        opts.parameters.absorption.enable = True
        opts.parameters.offset.enable = True

    if opts.parameters.outfile is None:
        opts.parameters.outfile = generate_default_outfiles(opts.parameters.hkl)
        if opts.parameters.outfile is None:
            raise SystemExit("unable to auto-generate output file names from input name pattern")

    return opts.parameters


def run_scale(params):
    """Run the scale algorithm"""
    hkl = params.hkl
    outfile = params.outfile
    scaling_enable = params.scaling.enable
    offset_enable = params.offset.enable
    absorption_enable = params.absorption.enable
    detector_enable = params.detector.enable
    scaling_alpha = params.scaling.alpha
    scaling_dphi = params.scaling.dphi
    scaling_niter = params.scaling.niter
    scaling_x2tol = params.scaling.x2tol
    scaling_outlier = params.scaling.outlier
    offset_alpha_x = params.offset.alpha_x
    offset_alpha_y = params.offset.alpha_y
    offset_alpha_min = params.offset.alpha_min
    offset_min = params.offset.min
    offset_dphi = params.offset.dphi
    offset_ns = params.offset.ns
    offset_niter = params.offset.niter
    offset_x2tol = params.offset.x2tol
    offset_outlier = params.offset.outlier
    detector_alpha = params.detector.alpha
    detector_nx = params.detector.nx
    detector_ny = params.detector.ny
    detector_niter = params.detector.niter
    detector_x2tol = params.detector.x2tol
    detector_outlier = params.detector.outlier
    absorption_alpha_xy = params.absorption.alpha_xy
    absorption_alpha_z = params.absorption.alpha_z
    absorption_nx = params.absorption.nx
    absorption_ny = params.absorption.ny
    absorption_dphi = params.absorption.dphi
    absorption_niter = params.absorption.niter
    absorption_x2tol = params.absorption.x2tol
    absorption_outlier = params.absorption.outlier

    # load data into a giant table
    tabs = []

    for n, fn in enumerate(hkl):
        tmp = loadobj(fn, "hkl_table")
        tmp.batch = n * np.ones_like(tmp.op)
        tabs.append(tmp)

    hkl = HKLTable.concatenate(tabs)

    print("Grouping redundant observations")
    (h, k, l), index_map, counts = hkl.unique()

    S = ScaledData(
        hkl.intensity,
        hkl.intensity_error,
        index_map,
        phi=hkl.phi,
        s=hkl.s,
        ix=hkl.ix,
        iy=hkl.iy,
        batch=hkl.batch,
    )

    MR = BatchModelRefiner(S)

    if scaling_enable:
        MR.add_scaling_models(
            dphi=scaling_dphi,
        )
    if offset_enable:
        MR.add_offset_models(
            dphi=offset_dphi,
            ns=offset_ns,
        )
    if absorption_enable:
        MR.add_absorption_models(
            dphi=absorption_dphi,
            nix=absorption_nx,
            niy=absorption_ny,
        )
    if detector_enable:
        MR.add_detector_model(
            nix=detector_nx,
            niy=detector_ny,
        )

    print(f"applying scale factors")
    MR.apply()
    print(f"merging")
    Im, sigmam, counts = MR.data.merge()

    old_x2 = 1e6  # initialize to some large number

    if scaling_enable:
        print("optimizing scale vs. phi (b)")
        for j in range(scaling_niter):
            print(f"  iteration {j + 1} of {scaling_niter}")
            print(f"    fitting the model")
            x2 = MR.bfit(Im, scaling_alpha)
            print(f"    current x2: {x2}")
            print(f"    applying scale factors")
            MR.apply()
            print(f"    merging")
            Im, sigmam, counts = MR.data.merge()
            if old_x2 - x2 < scaling_x2tol:
                print(f"    change in x2 less than tolerance of {scaling_x2tol}, stopping")
                break
            old_x2 = x2
        nout = MR.data.mask_outliers(Im, scaling_outlier)
        print(f"removed {nout} outliers > {scaling_outlier} sigma")

    if scaling_enable and offset_enable:
        print("optimizing scale and offset vs. phi and resolution (b/c)")
        for j in range(offset_niter):
            print(f"  iteration {j + 1} of {offset_niter}")
            print(f"    fitting the model")
            MR.cfit(
                Im,
                offset_alpha_x,
                offset_alpha_y,
                offset_alpha_min,
                offset_min,
            )  # 1,1,.1,min_c=0
            x2 = MR.bfit(Im, scaling_alpha)
            print(f"    current x2: {x2}")
            print(f"    applying scale factors")
            MR.apply()
            print(f"    merging")
            Im, sigmam, counts = MR.data.merge()
            if old_x2 - x2 < offset_x2tol:
                print(f"    change in x2 less than tolerance of {offset_x2tol}, stopping")
                break
            old_x2 = x2
        nout = MR.data.mask_outliers(Im, offset_outlier)
        print(f"removed {nout} outliers > {offset_outlier} sigma")

    if offset_enable:
        print("optimizing offset vs. phi and resolution (c)")
        for j in range(offset_niter):
            print(f"  iteration {j + 1} of {offset_niter}")
            print(f"    fitting the model")
            x2 = MR.cfit(
                Im,
                offset_alpha_x,
                offset_alpha_y,
                offset_alpha_min,
                offset_min,
            )  # 1,1,.1,min_c=0
            print(f"    current x2: {x2}")
            print(f"    applying scale factors")
            MR.apply()
            print(f"    merging")
            Im, sigmam, counts = MR.data.merge()
            if old_x2 - x2 < offset_x2tol:
                print(f"    change in x2 less than tolerance of {offset_x2tol}, stopping")
                break
            old_x2 = x2
        nout = MR.data.mask_outliers(Im, offset_outlier)
        print(f"removed {nout} outliers > {offset_outlier} sigma")

    if detector_enable:
        print("optimizing scale vs. detector position (d)")
        for j in range(detector_niter):
            print(f"  iteration {j + 1} of {detector_niter}")
            print(f"    fitting the model")
            x2 = MR.dfit(Im, detector_alpha)
            print(f"    current x2: {x2}")
            print(f"    applying scale factors")
            MR.apply()
            print(f"    merging")
            Im, sigmam, counts = MR.data.merge()
            if old_x2 - x2 < detector_x2tol:
                print(f"    change in x2 less than tolerance of {detector_x2tol}, stopping")
                break
            old_x2 = x2
        nout = MR.data.mask_outliers(Im, detector_outlier)
        print(f"removed {nout} outliers > {detector_outlier} sigma")

    if absorption_enable:
        print("optimizing scale vs. detector position and phi (a)")
        for j in range(absorption_niter):
            print(f"  iteration {j + 1} of {absorption_niter}")
            print(f"    fitting the model")
            x2 = MR.afit(Im, absorption_alpha_xy, absorption_alpha_z)
            print(f"    current x2: {x2}")
            print(f"    applying scale factors")
            MR.apply()
            print(f"    merging")
            Im, sigmam, counts = MR.data.merge()
            if old_x2 - x2 < absorption_x2tol:
                print(f"    change in x2 less than tolerance of {absorption_x2tol}, stopping")
                break
            old_x2 = x2
        nout = MR.data.mask_outliers(Im, absorption_outlier)
        print(f"removed {nout} outliers > {absorption_outlier} sigma")

    print("finished refining")

    for model_refiner, fn in zip(MR._batch_refiners, outfile):
        models = dict(
            scaling_model=model_refiner.scaling.model,
            detector_model=model_refiner.detector.model,
            absorption_model=model_refiner.absorption.model,
            offset_model=model_refiner.offset.model,
        )

        file_created = False
        for name, model in models.items():
            if model is not None:
                saveobj(model, fn, name=name, append=file_created)
                file_created = True

    print("done!")


def run(args=None):
    """Run the scale script"""
    params = parse_arguments(args=args)
    run_scale(params)


if __name__ == "__main__":
    run()
