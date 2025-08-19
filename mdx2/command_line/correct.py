"""
Apply corrections to integrated data
"""

import argparse

import numpy as np

from mdx2.utils import loadobj, saveobj


def parse_arguments(args=None):
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("geom", help="NeXus file with miller_index")
    parser.add_argument("hkl", help="NeXus file with hkl_table")
    parser.add_argument("--background", help="NeXus file with background map")
    parser.add_argument("--attenuation", metavar="TF", default=True, help="apply attenuation correction?")
    parser.add_argument("--efficiency", metavar="TF", default=True, help="apply efficiency correction?")
    parser.add_argument("--polarization", metavar="TF", default=True, help="apply polarization correction?")
    parser.add_argument("--lorentz", metavar="TF", default=False, help="apply Lorentz correction?")
    parser.add_argument(
        "--p1", action="store_true", help="map Miller indices to asymmetric unit for P1 (Friedel symmetry only)"
    )
    parser.add_argument("--outfile", default="corrected.nxs", help="name of the output NeXus file")
    params = parser.parse_args(args)

    # fix argparse ~bug where booleans are given as strings
    for p in ["attenuation", "efficiency", "polarization", "lorentz"]:
        if getattr(params, p) in ["True", "true", "T", "t"]:
            setattr(params, p, True)
        else:
            setattr(params, p, False)

    return params


def run_correct(params):
    """Run the correct script"""
    hkl = params.hkl
    geom = params.geom
    background = params.background
    outfile = params.outfile
    p1 = params.p1
    attenuation = params.attenuation
    efficiency = params.efficiency
    polarization = params.polarization
    lorentz = params.lorentz

    T = loadobj(hkl, "hkl_table")

    # hack to work with older versions
    if "_ndiv" in T.__dict__:
        T.ndiv = T._ndiv
        del T._ndiv

    Corrections = loadobj(geom, "corrections")
    Crystal = loadobj(geom, "crystal")

    if p1:
        print("ignoring space group information and using P1 symmetry only")
        Symmetry = None
    else:
        Symmetry = loadobj(geom, "symmetry")

    UB = Crystal.ub_matrix

    # computing scattering vector magnitude
    print("calculating scattering vector magnitude (s)")
    s = UB @ np.stack((T.h, T.k, T.l))
    T.s = np.sqrt(np.sum(s * s, axis=0))

    # map h,k,l to asymmetric unit
    print("mapping Miller indices to the asymmetric unit")
    T = T.to_asu(Symmetry)

    # apply corrections to intensities

    Cinterp = Corrections.interpolate(T.iy, T.ix)

    count_rate = T.counts / T.seconds
    count_rate_error = np.sqrt(T.counts) / T.seconds

    if background is not None:
        Bkg = loadobj(background, "binned_image_series")
        bkg_count_rate = Bkg.interpolate(T.phi, T.iy, T.ix)
        print("subtracting background from count rate")
        count_rate = count_rate - bkg_count_rate

    solid_angle = Cinterp["solid_angle"]

    if attenuation:
        print("correctin solid angle for attenuation")
        solid_angle *= Cinterp["attenuation"]
    if efficiency:
        print("correcting solid angle for efficiency")
        solid_angle *= Cinterp["efficiency"]
    if polarization:
        print("correcting solid angle for polarization")
        solid_angle *= Cinterp["polarization"]

    print("computing the swept reciprocal space volume fraction (rs_volume)")
    T.rs_volume = T.pixels * Cinterp["d3s"] / np.linalg.det(UB)

    print("computing intensity and intensity_error")
    T.intensity = count_rate / solid_angle
    T.intensity_error = count_rate_error / solid_angle

    if lorentz:
        T.intensity *= T.rs_volume
        T.intensity_error *= T.rs_volume

    # remove some unnecessary columns
    del T.counts
    del T.seconds
    del T.pixels

    # save some disk space
    T.h = T.h.astype(np.float32)
    T.k = T.k.astype(np.float32)
    T.l = T.l.astype(np.float32)
    T.s = T.s.astype(np.float32)
    T.intensity = T.intensity.astype(np.float32)
    T.intensity_error = T.intensity_error.astype(np.float32)
    T.ix = T.ix.astype(np.float32)
    T.iy = T.iy.astype(np.float32)
    T.phi = T.phi.astype(np.float32)
    T.rs_volume = T.rs_volume.astype(np.float32)
    T.n = T.n.astype(np.int32)
    T.op = T.op.astype(np.int32)

    saveobj(T, outfile, name="hkl_table", append=False)

    print("done!")


def run(args=None):
    """Run the correct script"""
    params = parse_arguments(args=args)
    run_correct(params)


if __name__ == "__main__":
    run()
