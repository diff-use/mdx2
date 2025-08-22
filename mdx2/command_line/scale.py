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
class PrescaleParameters:
    """Options for initial quick refinement of models to a subset of data and/or isotropically averaged data"""

    enable: bool = False  # include pre-scaling refinement
    subsample: int = 10  # subsample factor for pre-scaling refinement
    isotropic: bool = True  # use isotropically averaged data for pre-scaling refinement
    scaling: bool = True  # include scaling model in pre-scaling refinement if scaling.enable is True
    offset: bool = False  # include offset model in pre-scaling refinement if offset.enable is True
    detector: bool = False  # include detector model in pre-scaling refinement if detector.enable is True
    absorption: bool = True  # include absorption model in pre-scaling refinement if absorption.enable is True


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
    prescale: PrescaleParameters
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


def mask_outliers(MR, outlier):
    """Mask outliers"""
    print(f"    applying scale factors")
    MR.apply()
    print(f"    merging")
    Im, sigmam, counts = MR.data.merge()
    nout = MR.data.mask_outliers(Im, outlier)
    print(f"removed {nout} outliers > {outlier} sigma")


def refine_offset_model(MR, offset_params):
    """Refine the offset model"""
    old_x2 = 1e6  # initialize to some large number
    for j in range(offset_params.niter):
        print(f"  iteration {j + 1} of {offset_params.niter}")
        print(f"    applying scale factors")
        MR.apply()
        print(f"    merging")
        Im, sigmam, counts = MR.data.merge()

        print(f"    fitting the model")
        x2 = MR.cfit(
            Im,
            offset_params.alpha_x,
            offset_params.alpha_y,
            offset_params.alpha_min,
            offset_params.min,
        )  # 1,1,.1,min_c=0
        print(f"    current x2: {x2}")
        if old_x2 - x2 < offset_params.x2tol:
            print(f"    change in x2 less than tolerance of {offset_params.x2tol}, stopping")
            break
        old_x2 = x2


def refine_scaling_model(MR, scaling_params):
    """Refine the scaling model"""
    old_x2 = 1e6  # initialize to some large number
    for j in range(scaling_params.niter):
        print(f"  iteration {j + 1} of {scaling_params.niter}")
        print(f"    applying scale factors")
        MR.apply()
        print(f"    merging")
        Im, sigmam, counts = MR.data.merge()
        print(f"    fitting the model")
        x2 = MR.bfit(Im, scaling_params.alpha)
        print(f"    current x2: {x2}")
        if old_x2 - x2 < scaling_params.x2tol:
            print(f"    change in x2 less than tolerance of {scaling_params.x2tol}, stopping")
            break
        old_x2 = x2


def refine_scaling_and_offset_model(MR, scaling_params, offset_params):
    """Refine the scaling and offset model"""
    old_x2 = 1e6  # initialize to some large number
    for j in range(offset_params.niter):
        print(f"  iteration {j + 1} of {offset_params.niter}")
        print(f"    applying scale factors")
        MR.apply()
        print(f"    merging")
        Im, sigmam, counts = MR.data.merge()
        print(f"    fitting the model")
        MR.cfit(
            Im,
            offset_params.alpha_x,
            offset_params.alpha_y,
            offset_params.alpha_min,
            offset_params.min,
        )  # 1,1,.1,min_c=0
        x2 = MR.bfit(Im, scaling_params.alpha)
        print(f"    current x2: {x2}")
        if old_x2 - x2 < offset_params.x2tol:
            print(f"    change in x2 less than tolerance of {offset_params.x2tol}, stopping")
            break
        old_x2 = x2


def refine_detector_model(MR, detector_params):
    """Refine the detector model"""
    old_x2 = 1e6  # initialize to some large number
    for j in range(detector_params.niter):
        print(f"  iteration {j + 1} of {detector_params.niter}")
        print(f"    applying scale factors")
        MR.apply()
        print(f"    merging")
        Im, sigmam, counts = MR.data.merge()
        print(f"    fitting the model")
        x2 = MR.dfit(Im, detector_params.alpha)
        print(f"    current x2: {x2}")
        if old_x2 - x2 < detector_params.x2tol:
            print(f"    change in x2 less than tolerance of {detector_params.x2tol}, stopping")
            break
        old_x2 = x2


def refine_absorption_model(MR, absorption_params):
    """Refine the absorption model"""
    old_x2 = 1e6  # initialize to some large number
    for j in range(absorption_params.niter):
        print(f"  iteration {j + 1} of {absorption_params.niter}")
        print(f"    applying scale factors")
        MR.apply()
        print(f"    merging")
        Im, sigmam, counts = MR.data.merge()
        print(f"    fitting the model")
        x2 = MR.afit(Im, absorption_params.alpha_xy, absorption_params.alpha_z)
        print(f"    current x2: {x2}")
        if old_x2 - x2 < absorption_params.x2tol:
            print(f"    change in x2 less than tolerance of {absorption_params.x2tol}, stopping")
            break
        old_x2 = x2


def load_data_for_scaling(*hkl_files, subsample=None, merge_isotropic=False):
    """Load data from hkl files into a single HKLTable"""
    tabs = []
    for n, fn in enumerate(hkl_files):
        tmp = loadobj(fn, "hkl_table")
        if subsample is not None:
            tmp = tmp[::subsample]
        tmp.batch = n * np.ones_like(tmp.op)
        tabs.append(tmp)

    hkl = HKLTable.concatenate(tabs)

    print(f"loaded {len(hkl)} reflections from {len(hkl_files)} files")

    if merge_isotropic:
        raise NotImplementedError("isotropic averaging not implemented yet")
    else:
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

    nsingletons = S.mask_singletons()
    if nsingletons > 0:
        print(f"masked {nsingletons} singletons (reflections with only one observation)")

    return S


def run_scale(params):
    """Run the scale algorithm"""

    if params.prescale.enable and params.prescale.subsample > 1:
        subsample = params.prescale.subsample
    else:
        subsample = None

    S = load_data_for_scaling(*params.hkl, subsample=subsample)

    MR = BatchModelRefiner(S)

    if params.scaling.enable:
        MR.add_scaling_models(
            dphi=params.scaling.dphi,
        )
    if params.offset.enable:
        MR.add_offset_models(
            dphi=params.offset.dphi,
            ns=params.offset.ns,
        )
    if params.absorption.enable:
        MR.add_absorption_models(
            dphi=params.absorption.dphi,
            nix=params.absorption.nx,
            niy=params.absorption.ny,
        )
    if params.detector.enable:
        MR.add_detector_model(
            nix=params.detector.nx,
            niy=params.detector.ny,
        )

    if params.scaling.enable:
        print("optimizing scale vs. phi (b)")
        refine_scaling_model(MR, params.scaling)
        mask_outliers(MR, params.scaling.outlier)

    if params.scaling.enable and params.offset.enable:
        print("optimizing scaling and offset models together (b,c)")
        refine_scaling_and_offset_model(MR, params.scaling, params.offset)
        mask_outliers(MR, params.offset.outlier)

    if params.offset.enable:
        print("optimizing offset vs. phi and resolution (c)")
        refine_offset_model(MR, params.offset)
        mask_outliers(MR, params.offset.outlier)

    if params.detector.enable:
        print("optimizing scale vs. detector position (d)")
        refine_detector_model(MR, params.detector)
        mask_outliers(MR, params.detector.outlier)

    if params.absorption.enable:
        print("optimizing scale vs. detector position and phi (a)")
        refine_absorption_model(MR, params.absorption)
        mask_outliers(MR, params.absorption.outlier)

    print("finished refining")

    for model_refiner, fn in zip(MR._batch_refiners, params.outfile):
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
