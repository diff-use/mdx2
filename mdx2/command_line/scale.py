"""
Fit scaling model to unmerged corrected intensities
"""

import argparse
import os

import numpy as np

from mdx2.data import HKLTable
from mdx2.scaling import BatchModelRefiner, ScaledData
from mdx2.utils import loadobj, saveobj


def parse_arguments(args=None):
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("hkl", nargs="+", help="NeXus file(s) with hkl_table")
    parser.add_argument(
        "--mca2020",
        action="store_true",
        help="shortcut for --scaling.enable True --offset.enable True --detector.enable True --absorption.enable True",
    )
    parser.add_argument(
        "--outfile",
        nargs="+",
        help="name of the output NeXus file(s). "
        "If not specified, will attempt a sensible name such as scales.nxs for a single input file",
    )

    scaling_group = parser.add_argument_group(title="Scaling parameters")
    scaling_group.add_argument(
        "--scaling.enable",
        dest="scaling_enable",
        default=True,
        metavar="TF",
        help="include smooth scale factor vs. phi",
    )
    scaling_group.add_argument(
        "--scaling.alpha",
        dest="scaling_alpha",
        default=1.0,
        type=float,
        metavar="ALPHA",
        help="amount to rescale the default smoothness (regularization parameter)",
    )
    scaling_group.add_argument(
        "--scaling.dphi",
        dest="scaling_dphi",
        default=1.0,
        type=float,
        metavar="DEGREES",
        help="spacing of phi control points in degrees",
    )
    scaling_group.add_argument(
        "--scaling.niter",
        dest="scaling_niter",
        default=10,
        type=int,
        metavar="N",
        help="maximum iterations in refinement",
    )
    scaling_group.add_argument(
        "--scaling.x2tol",
        dest="scaling_x2tol",
        default=1e-4,
        type=float,
        metavar="TOL",
        help="maximum change in x2 to stop refinement early",
    )
    scaling_group.add_argument(
        "--scaling.outlier",
        dest="scaling_outlier",
        default=10,
        type=float,
        metavar="NSIGMA",
        help="standard error cutoff for outlier rejection after refinement",
    )

    offset_group = parser.add_argument_group(title="Offset parameters")
    offset_group.add_argument(
        "--offset.enable",
        dest="offset_enable",
        default=False,
        metavar="TF",
        help="include smooth offset vs. resolution and phi",
    )
    offset_group.add_argument(
        "--offset.alpha_x",
        dest="offset_alpha_x",
        default=1.0,
        type=float,
        metavar="ALPHA",
        help="smoothness vs. s (resolution): multiplies the regularization parameter",
    )
    offset_group.add_argument(
        "--offset.alpha_y",
        dest="offset_alpha_y",
        default=1.0,
        type=float,
        metavar="ALPHA",
        help="smoothness vs. phi -- multiplies the regularization parameter",
    )
    offset_group.add_argument(
        "--offset.alpha_min",
        dest="offset_alpha_min",
        default=0.001,
        type=float,
        metavar="ALPHA",
        help="deviation from offset.min: multiplies the regularization parameter",
    )
    offset_group.add_argument(
        "--offset.min", dest="offset_min", default=0.0, type=float, metavar="VAL", help="minimum value of offset"
    )
    offset_group.add_argument(
        "--offset.dphi",
        dest="offset_dphi",
        default=2.5,
        type=float,
        metavar="DEGREES",
        help="spacing of phi control points in degrees",
    )
    offset_group.add_argument(
        "--offset.ns",
        dest="offset_ns",
        default=31,
        type=int,
        metavar="N",
        help="number of s (resolution) control points",
    )
    offset_group.add_argument(
        "--offset.niter", dest="offset_niter", default=5, type=int, metavar="N", help="maximum iterations in refinement"
    )
    offset_group.add_argument(
        "--offset.x2tol",
        dest="offset_x2tol",
        default=1e-3,
        type=float,
        metavar="TOL",
        help="maximum change in x2 to stop refinement early",
    )
    offset_group.add_argument(
        "--offset.outlier",
        dest="offset_outlier",
        default=5,
        type=float,
        metavar="NSIGMA",
        help="standard error cutoff for outlier rejection after refinement",
    )

    detector_group = parser.add_argument_group(title="Detector parameters")
    detector_group.add_argument(
        "--detector.enable",
        dest="detector_enable",
        default=False,
        metavar="TF",
        help="include smooth scale vs. detector xy position",
    )
    detector_group.add_argument(
        "--detector.alpha",
        dest="detector_alpha",
        default=1.0,
        type=float,
        metavar="ALPHA",
        help="smoothness vs. xy position: multiplies the regularization parameter",
    )
    detector_group.add_argument(
        "--detector.nx",
        dest="detector_nx",
        default=200,
        type=float,
        metavar="N",
        help="number of grid control points in the x direction",
    )
    detector_group.add_argument(
        "--detector.ny",
        dest="detector_ny",
        default=200,
        type=float,
        metavar="N",
        help="number of grid control points in the y direction",
    )
    detector_group.add_argument(
        "--detector.niter",
        dest="detector_niter",
        default=5,
        type=int,
        metavar="N",
        help="maximum iterations in refinement",
    )
    detector_group.add_argument(
        "--detector.x2tol",
        dest="detector_x2tol",
        default=1e-3,
        type=float,
        metavar="TOL",
        help="maximum change in x2 to stop refinement early",
    )
    detector_group.add_argument(
        "--detector.outlier",
        dest="detector_outlier",
        default=5,
        type=float,
        metavar="NSIGMA",
        help="standard error cutoff for outlier rejection after refinement",
    )

    absorption_group = parser.add_argument_group(title="Absorption parameters")
    absorption_group.add_argument(
        "--absorption.enable",
        dest="absorption_enable",
        default=False,
        metavar="TF",
        help="include smooth scale vs. detector xy position and phi",
    )
    absorption_group.add_argument(
        "--absorption.alpha_xy",
        dest="absorption_alpha_xy",
        default=10.0,
        type=float,
        metavar="ALPHA",
        help="smoothness vs. xy position: multiplies the regularization parameter",
    )
    absorption_group.add_argument(
        "--absorption.alpha_z",
        dest="absorption_alpha_z",
        default=1.0,
        type=float,
        metavar="ALPHA",
        help="smoothness vs. phi: multiplies the regularization parameter",
    )
    absorption_group.add_argument(
        "--absorption.nx",
        dest="absorption_nx",
        default=20,
        type=float,
        metavar="N",
        help="number of grid control points in the x direction",
    )
    absorption_group.add_argument(
        "--absorption.ny",
        dest="absorption_ny",
        default=20,
        type=float,
        metavar="N",
        help="number of grid control points in the y direction",
    )
    absorption_group.add_argument(
        "--absorption.dphi",
        dest="absorption_dphi",
        default=5.0,
        type=float,
        metavar="DEGREES",
        help="spacing of phi control points in degrees",
    )
    absorption_group.add_argument(
        "--absorption.niter",
        dest="absorption_niter",
        default=5,
        type=int,
        metavar="N",
        help="maximum iterations in refinement",
    )
    absorption_group.add_argument(
        "--absorption.x2tol",
        dest="absorption_x2tol",
        default=1e-4,
        type=float,
        metavar="TOL",
        help="maximum change in x2 to stop refinement early",
    )
    absorption_group.add_argument(
        "--absorption.outlier",
        dest="absorption_outlier",
        default=5,
        type=float,
        metavar="NSIGMA",
        help="standard error cutoff for outlier rejection after refinement",
    )

    params = parser.parse_args(args)

    if params.mca2020:
        params.scaling_enable = True
        params.detector_enable = True
        params.absorption_enable = True
        params.offset_enable = True

    def generate_default_outfiles(infiles):
        dirs = [os.path.dirname(fn) for fn in infiles]
        if len(set(dirs)) == len(dirs):  # dirs are unique
            return [os.path.join(d, "scales.nxs") for d in dirs]
        if len(set(dirs)) == 1:  # dirs are identical
            roots = [os.path.splitext(os.path.split(fn)[-1])[0] for fn in infiles]
            if all(["_" in root for root in roots]):
                postfix = [root.split("_")[-1] for root in roots]
                if len(set(postfix)) == len(postfix):  # postfixes are unique
                    return [os.path.join(dirs[0], f"scales_{pf}.nxs") for pf in postfix]

    if params.outfile is None:
        params.outfile = generate_default_outfiles(params.hkl)
        if params.outfile is None:
            raise ValueError("unable to auto-generate output file names from input name pattern")

    return params


def run_scale(params):
    """Run the scale algorithm"""
    hkl = params.hkl
    outfile = params.outfile
    scaling_enable = params.scaling_enable
    offset_enable = params.offset_enable
    absorption_enable = params.absorption_enable
    detector_enable = params.detector_enable
    scaling_alpha = params.scaling_alpha
    scaling_dphi = params.scaling_dphi
    scaling_niter = params.scaling_niter
    scaling_x2tol = params.scaling_x2tol
    scaling_outlier = params.scaling_outlier
    offset_alpha_x = params.offset_alpha_x
    offset_alpha_y = params.offset_alpha_y
    offset_alpha_min = params.offset_alpha_min
    offset_min = params.offset_min
    offset_dphi = params.offset_dphi
    offset_ns = params.offset_ns
    offset_niter = params.offset_niter
    offset_x2tol = params.offset_x2tol
    offset_outlier = params.offset_outlier
    detector_alpha = params.detector_alpha
    detector_nx = params.detector_nx
    detector_ny = params.detector_ny
    detector_niter = params.detector_niter
    detector_x2tol = params.detector_x2tol
    detector_outlier = params.detector_outlier
    absorption_alpha_xy = params.absorption_alpha_xy
    absorption_alpha_z = params.absorption_alpha_z
    absorption_nx = params.absorption_nx
    absorption_ny = params.absorption_ny
    absorption_dphi = params.absorption_dphi
    absorption_niter = params.absorption_niter
    absorption_x2tol = params.absorption_x2tol
    absorption_outlier = params.absorption_outlier

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
