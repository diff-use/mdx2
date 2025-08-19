from mdx2.command_line.bin_image_series import parse_arguments as bin_image_series_parse_arguments
from mdx2.command_line.find_peaks import parse_arguments as find_peaks_parse_arguments
from mdx2.command_line.import_data import parse_arguments as import_data_parse_arguments
from mdx2.command_line.import_geometry import parse_arguments as import_geometry_parse_arguments
from mdx2.command_line.integrate import parse_arguments as integrate_parse_arguments
from mdx2.command_line.mask_peaks import parse_arguments as mask_peaks_parse_arguments
from mdx2.command_line.merge import parse_arguments as merge_parse_arguments
from mdx2.command_line.scale import parse_arguments as scale_parse_arguments


def test_import_data_parse_arguments():
    """Test the import data command line argument parsing."""
    args = ["test.expt", "--outfile", "test.nxs", "--chunks", "1", "10", "10", "--nproc", "4"]
    params = import_data_parse_arguments(args=args)

    assert params.expt == "test.expt"
    assert params.outfile == "test.nxs"
    assert tuple(params.chunks) == (1, 10, 10)
    assert params.nproc == 4


def test_import_geometry_parse_arguments():
    """Test the import geometry command line argument parsing."""
    args = ["test.expt", "--sample_spacing", "1", "10", "10", "--outfile", "geometry.nxs"]
    params = import_geometry_parse_arguments(args=args)

    assert params.expt == "test.expt"
    assert tuple(params.sample_spacing) == (1, 10, 10)
    assert params.outfile == "geometry.nxs"


def test_find_peaks_parse_arguments():
    """Test the find peaks command line argument parsing."""
    args = [
        "geometry.nxs",
        "data.nxs",
        "--count_threshold",
        "1000",
        "--sigma_cutoff",
        "3.0",
        "--outfile",
        "peaks.nxs",
        "--nproc",
        "2",
    ]
    params = find_peaks_parse_arguments(args=args)

    assert params.geom == "geometry.nxs"
    assert params.data == "data.nxs"
    assert params.count_threshold == 1000.0
    assert params.sigma_cutoff == 3.0
    assert params.outfile == "peaks.nxs"
    assert params.nproc == 2


def test_mask_peaks_parse_arguments():
    """Test the mask peaks command line argument parsing."""
    args = [
        "geometry.nxs",
        "data.nxs",
        "peaks.nxs",
        "--sigma_cutoff",
        "3.0",
        "--outfile",
        "mask.nxs",
        "--nproc",
        "4",
        "--bragg",
    ]
    params = mask_peaks_parse_arguments(args=args)

    assert params.geom == "geometry.nxs"
    assert params.data == "data.nxs"
    assert params.peaks == "peaks.nxs"
    assert params.sigma_cutoff == 3.0
    assert params.outfile == "mask.nxs"
    assert params.nproc == 4
    assert params.bragg is True


def test_bin_image_series_parse_arguments():
    """Test the bin image series command line argument parsing."""
    args = [
        "data.nxs",
        "2",
        "50",
        "30",
        "--outfile",
        "binned_data.nxs",
        "--valid_range",
        "0",
        "1000",
        "--nproc",
        "4",
    ]
    params = bin_image_series_parse_arguments(args=args)

    assert params.data == "data.nxs"
    assert tuple(params.bins) == (2, 50, 30)
    assert params.outfile == "binned_data.nxs"
    assert tuple(params.valid_range) == (0, 1000)
    assert params.nproc == 4


def test_merge_parse_arguments():
    """Test the merge command line argument parsing."""

    args = (
        "hkl1.nxs hkl2.nxs --scale scale1.nxs scale2.nxs --outfile merged.nxs --outlier 3.0"
        " --split randomHalf --geometry geometry.nxs --no-scaling --no-offset --no-absorption --no-detector"
    ).split()
    params = merge_parse_arguments(args=args)
    assert params.hkl == ["hkl1.nxs", "hkl2.nxs"]
    assert params.scale == ["scale1.nxs", "scale2.nxs"]
    assert params.outfile == "merged.nxs"
    assert params.outlier == 3.0
    assert params.split == "randomHalf"
    assert params.geometry == "geometry.nxs"
    assert params.no_scaling is True
    assert params.no_offset is True
    assert params.no_absorption is True
    assert params.no_detector is True


def test_integrate_parse_arguments():
    """Test the integrate command line argument parsing."""
    args = [
        "geom.nxs",
        "data.nxs",
        "--mask",
        "mask.nxs",
        "--subdivide",
        "2",
        "2",
        "2",
        "--max_spread",
        "1.5",
        "--outfile",
        "integrated.nxs",
        "--nproc",
        "4",
    ]
    params = integrate_parse_arguments(args=args)

    assert params.geom == "geom.nxs"
    assert params.data == "data.nxs"
    assert params.mask == "mask.nxs"
    assert tuple(params.subdivide) == (2, 2, 2)
    assert params.max_spread == 1.5
    assert params.outfile == "integrated.nxs"
    assert params.nproc == 4


def test_scale_parse_arguments():
    """Test the scale command line argument parsing.

    Check whether parse_arguments from mdx2.command_line.scale works correctly.
    """
    args = (
        "crystal1/integrated.nxs crystal2/integrated.nxs --absorption.enable True --absorption.nx 10"
        " --absorption.ny 10 --absorption.dphi 15.0 --absorption.niter 5 --absorption.x2tol 0.01"
        " --absorption.outlier 3.0"
    ).split()
    params = scale_parse_arguments(args=args)
    assert params.hkl == ["crystal1/integrated.nxs", "crystal2/integrated.nxs"]
    assert params.absorption_enable is True
    assert params.absorption_nx == 10
    assert params.absorption_ny == 10
    assert params.absorption_dphi == 15.0
    assert params.absorption_niter == 5
    assert params.absorption_x2tol == 0.01
    assert params.absorption_outlier == 3.0
    assert params.outfile == ["crystal1/scales.nxs", "crystal2/scales.nxs"]
