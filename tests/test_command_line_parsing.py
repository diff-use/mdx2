import pytest

from mdx2.command_line.bin_image_series import parse_arguments as bin_image_series_parse_arguments
from mdx2.command_line.find_peaks import parse_arguments as find_peaks_parse_arguments
from mdx2.command_line.import_data import parse_arguments as import_data_parse_arguments
from mdx2.command_line.import_geometry import parse_arguments as import_geometry_parse_arguments
from mdx2.command_line.integrate import parse_arguments as integrate_parse_arguments
from mdx2.command_line.mask_peaks import parse_arguments as mask_peaks_parse_arguments
from mdx2.command_line.merge import parse_arguments as merge_parse_arguments
from mdx2.command_line.scale import parse_arguments as scale_parse_arguments


@pytest.mark.parametrize(
    "args,expected,raises",
    [
        # Valid case
        (
            ["test.expt", "--outfile", "test.nxs", "--chunks", "1", "10", "10", "--nproc", "4"],
            {"expt": "test.expt", "outfile": "test.nxs", "chunks": (1, 10, 10), "nproc": 4},
            None,
        ),
        # Missing required argument
        (
            ["--outfile", "test.nxs", "--chunks", "1", "10", "10", "--nproc", "4"],
            None,
            SystemExit,  # argparse throws SystemExit on error
        ),
        # Invalid type for chunks
        (
            ["test.expt", "--outfile", "test.nxs", "--chunks", "a", "b", "c", "--nproc", "4"],
            None,
            SystemExit,
        ),
        # Too few chunks
        (
            ["test.expt", "--outfile", "test.nxs", "--chunks", "1", "10", "--nproc", "4"],
            None,
            SystemExit,
        ),
    ],
)
def test_import_data_parse_arguments(args, expected, raises):
    if raises:
        with pytest.raises(raises):
            import_data_parse_arguments(args=args)
    else:
        params = import_data_parse_arguments(args=args)
        assert params.expt == expected["expt"]
        assert params.outfile == expected["outfile"]
        assert tuple(params.chunks) == expected["chunks"]
        assert params.nproc == expected["nproc"]


@pytest.mark.parametrize(
    "args,expected,raises",
    [
        # Valid case
        (
            ["test.expt", "--sample_spacing", "1", "10", "10", "--outfile", "geometry.nxs"],
            {"expt": "test.expt", "sample_spacing": (1, 10, 10), "outfile": "geometry.nxs"},
            None,
        ),
        # Missing required argument
        (
            ["--sample_spacing", "1", "10", "10", "--outfile", "geometry.nxs"],
            None,
            SystemExit,  # argparse throws SystemExit on error
        ),
        # Invalid type for sample_spacing
        (
            ["test.expt", "--sample_spacing", "a", "b", "c", "--outfile", "geometry.nxs"],
            None,
            SystemExit,
        ),
        # Too few sample_spacing values
        (
            ["test.expt", "--sample_spacing", "1", "10", "--outfile", "geometry.nxs"],
            None,
            SystemExit,
        ),
    ],
)
def test_import_geometry_parse_arguments(args, expected, raises):
    """Test the import data command line argument parsing."""
    if raises:
        with pytest.raises(raises):
            import_geometry_parse_arguments(args=args)
    else:
        params = import_geometry_parse_arguments(args=args)
        assert params.expt == expected["expt"]
        assert tuple(params.sample_spacing) == expected["sample_spacing"]
        assert params.outfile == expected["outfile"]


@pytest.mark.parametrize(
    "args,expected,raises",
    [
        # Valid case
        (
            [
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
            ],
            {
                "geom": "geometry.nxs",
                "data": "data.nxs",
                "count_threshold": 1000.0,
                "sigma_cutoff": 3.0,
                "outfile": "peaks.nxs",
                "nproc": 2,
            },
            None,
        ),
        # Missing required argument
        (
            [
                "geometry.nxs",
                "--count_threshold",
                "1000",
                "--sigma_cutoff",
                "3.0",
                "--outfile",
                "peaks.nxs",
                "--nproc",
                "2",
            ],
            None,
            SystemExit,  # argparse throws SystemExit on error
        ),
        # Invalid type for count_threshold
        (
            [
                "geometry.nxs",
                "data.nxs",
                "--count_threshold",
                "invalid",
                "--sigma_cutoff",
                "3.0",
                "--outfile",
                "peaks.nxs",
                "--nproc",
                "2",
            ],
            None,
            SystemExit,
        ),
        # Invalid type for sigma_cutoff
        (
            [
                "geometry.nxs",
                "data.nxs",
                "--count_threshold",
                "1000",
                "--sigma_cutoff",
                "invalid",
                "--outfile",
                "peaks.nxs",
                "--nproc",
                "2",
            ],
            None,
            SystemExit,
        ),
    ],
)
def test_find_peaks_parse_arguments(args, expected, raises):
    """Test the find peaks command line argument parsing."""
    if raises:
        with pytest.raises(raises):
            find_peaks_parse_arguments(args=args)
    else:
        params = find_peaks_parse_arguments(args=args)
        assert params.geom == expected["geom"]
        assert params.data == expected["data"]
        assert params.count_threshold == expected["count_threshold"]
        assert params.sigma_cutoff == expected["sigma_cutoff"]
        assert params.outfile == expected["outfile"]
        assert params.nproc == expected["nproc"]


@pytest.mark.parametrize(
    "args,expected,raises",
    [
        # Valid case
        (
            [
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
            ],
            {
                "geom": "geometry.nxs",
                "data": "data.nxs",
                "peaks": "peaks.nxs",
                "sigma_cutoff": 3.0,
                "outfile": "mask.nxs",
                "nproc": 4,
                "bragg": True,
            },
            None,
        ),
        # Missing required argument
        (
            [
                "geometry.nxs",
                "data.nxs",
                "--sigma_cutoff",
                "3.0",
                "--outfile",
                "mask.nxs",
                "--nproc",
                "4",
            ],
            None,
            SystemExit,  # argparse throws SystemExit on error
        ),
        # Invalid type for sigma_cutoff
        (
            [
                "geometry.nxs",
                "data.nxs",
                "peaks.nxs",
                "--sigma_cutoff",
                "invalid",
                "--outfile",
                "mask.nxs",
                "--nproc",
                "4",
                "--bragg",
            ],
            None,
            SystemExit,
        ),
    ],
)
def test_mask_peaks_parse_arguments(args, expected, raises):
    """Test the mask peaks command line argument parsing."""
    if raises:
        with pytest.raises(raises):
            mask_peaks_parse_arguments(args=args)
    else:
        params = mask_peaks_parse_arguments(args=args)
        assert params.geom == expected["geom"]
        assert params.data == expected["data"]
        assert params.peaks == expected["peaks"]
        assert params.sigma_cutoff == expected["sigma_cutoff"]
        assert params.outfile == expected["outfile"]
        assert params.nproc == expected["nproc"]
        assert params.bragg == expected["bragg"]


@pytest.mark.parametrize(
    "args,expected,raises",
    [
        # Valid case
        (
            [
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
            ],
            {
                "data": "data.nxs",
                "bins": (2, 50, 30),
                "outfile": "binned_data.nxs",
                "valid_range": (0, 1000),
                "nproc": 4,
            },
            None,
        ),
        # Incorrect number of bins
        (
            ["data.nxs", "2", "50", "--outfile", "binned_data.nxs", "--nproc", "4"],
            None,
            SystemExit,  # argparse throws SystemExit on error
        ),
        # valid_range is not length 2
        (
            ["data.nxs", "2", "50", "30", "--outfile", "binned_data.nxs", "--valid_range", "0", "--nproc", "4"],
            None,
            SystemExit,  # argparse throws SystemExit on error
        ),
    ],
)
def test_bin_image_series_parse_arguments(args, expected, raises):
    """Test the bin image series command line argument parsing."""
    if raises:
        with pytest.raises(raises):
            bin_image_series_parse_arguments(args=args)
    else:
        params = bin_image_series_parse_arguments(args=args)
        assert params.data == expected["data"]
        assert tuple(params.bins) == expected["bins"]
        assert params.outfile == expected["outfile"]
        assert tuple(params.valid_range) == expected["valid_range"]
        assert params.nproc == expected["nproc"]


@pytest.mark.parametrize(
    "args,expected,raises",
    [
        # Valid case
        (
            [
                "hkl1.nxs",
                "hkl2.nxs",
                "--scale",
                "scale1.nxs",
                "scale2.nxs",
                "--outfile",
                "merged.nxs",
                "--outlier",
                "3.0",
                "--split",
                "randomHalf",
            ],
            {
                "hkl": ["hkl1.nxs", "hkl2.nxs"],
                "scale": ["scale1.nxs", "scale2.nxs"],
                "outfile": "merged.nxs",
                "outlier": 3.0,
                "split": "randomHalf",
                "geometry": None,
                "no_scaling": False,
                "no_offset": False,
                "no_absorption": False,
                "no_detector": False,
            },
            None,
        ),
        # Valid case with one set of hkl, scale files
        (
            [
                "hkl.nxs",
                "--scale",
                "scale.nxs",
                "--outfile",
                "merged.nxs",
                "--outlier",
                "3.0",
                "--split",
                "randomHalf",
            ],
            {
                "hkl": ["hkl.nxs"],
                "scale": ["scale.nxs"],
                "outfile": "merged.nxs",
                "outlier": 3.0,
                "split": "randomHalf",
                "geometry": None,
                "no_scaling": False,
                "no_offset": False,
                "no_absorption": False,
                "no_detector": False,
            },
            None,
        ),
        # Invalid case with "--split Friedel" but no geometry file
        (
            [
                "hkl1.nxs",
                "hkl2.nxs",
                "--scale",
                "scale1.nxs",
                "scale2.nxs",
                "--outfile",
                "merged.nxs",
                "--outlier",
                "3.0",
                "--split",
                "Friedel",
            ],
            None,
            SystemExit,  # argparse throws SystemExit on error
        ),
    ],
)
def test_merge_parse_arguments(args, expected, raises):
    """Test the merge command line argument parsing."""
    if raises:
        with pytest.raises(raises):
            merge_parse_arguments(args=args)
    else:
        params = merge_parse_arguments(args=args)
        assert params.hkl == expected["hkl"]
        assert params.scale == expected["scale"]
        assert params.outfile == expected["outfile"]
        assert params.outlier == expected["outlier"]
        assert params.split == expected["split"]
        assert params.geometry == expected["geometry"]
        assert params.no_scaling is expected["no_scaling"]
        assert params.no_offset is expected["no_offset"]
        assert params.no_absorption is expected["no_absorption"]
        assert params.no_detector is expected["no_detector"]


@pytest.mark.parametrize(
    "args,expected,raises",
    [
        # Valid case
        (
            [
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
            ],
            {
                "geom": "geom.nxs",
                "data": "data.nxs",
                "mask": "mask.nxs",
                "subdivide": (2, 2, 2),
                "max_spread": 1.5,
                "outfile": "integrated.nxs",
                "nproc": 4,
            },
            None,
        ),
        # Incorrect number of subdivisions
        (
            [
                "geom.nxs",
                "data.nxs",
                "--mask",
                "mask.nxs",
                "--subdivide",
                "2",
                "--max_spread",
                "1.5",
                "--outfile",
                "integrated.nxs",
                "--nproc",
                "4",
            ],
            None,
            SystemExit,  # argparse throws SystemExit on error
        ),
        # Valid case with no mask
        (
            [
                "geom.nxs",
                "data.nxs",
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
            ],
            {
                "geom": "geom.nxs",
                "data": "data.nxs",
                "mask": None,
                "subdivide": (2, 2, 2),
                "max_spread": 1.5,
                "outfile": "integrated.nxs",
                "nproc": 4,
            },
            None,
        ),
    ],
)
def test_integrate_parse_arguments(args, expected, raises):
    """Test the integrate command line argument parsing."""
    if raises:
        with pytest.raises(raises):
            integrate_parse_arguments(args=args)
    else:
        params = integrate_parse_arguments(args=args)
        assert params.geom == expected["geom"]
        assert params.data == expected["data"]
        assert params.mask == expected["mask"]
        assert tuple(params.subdivide) == expected["subdivide"]
        assert params.max_spread == expected["max_spread"]
        assert params.outfile == expected["outfile"]
        assert params.nproc == expected["nproc"]


@pytest.mark.parametrize(
    "args,expected,raises",
    [
        # Valid case with outfile specified
        (
            [
                "crystal1/integrated.nxs",
                "crystal2/integrated.nxs",
                "--absorption.enable",
                "True",
                "--absorption.nx",
                "10",
                "--absorption.ny",
                "10",
                "--absorption.dphi",
                "15.0",
                "--absorption.niter",
                "5",
                "--absorption.x2tol",
                "0.01",
                "--absorption.outlier",
                "3.0",
                "--outfile",
                "crystal1/scales.nxs",
                "crystal2/scales.nxs",
            ],
            {
                "hkl": ["crystal1/integrated.nxs", "crystal2/integrated.nxs"],
                "absorption_enable": True,
                "absorption_nx": 10,
                "absorption_ny": 10,
                "absorption_dphi": 15.0,
                "absorption_niter": 5,
                "absorption_x2tol": 0.01,
                "absorption_outlier": 3.0,
                "outfile": ["crystal1/scales.nxs", "crystal2/scales.nxs"],
            },
            None,
        ),
        # Valid case with no outfile specified
        (
            [
                "crystal1/integrated.nxs",
                "crystal2/integrated.nxs",
                "--absorption.enable",
                "True",
                "--absorption.nx",
                "10",
                "--absorption.ny",
                "10",
                "--absorption.dphi",
                "15.0",
                "--absorption.niter",
                "5",
                "--absorption.x2tol",
                "0.01",
                "--absorption.outlier",
                "3.0",
            ],
            {
                "hkl": ["crystal1/integrated.nxs", "crystal2/integrated.nxs"],
                "absorption_enable": True,
                "absorption_nx": 10,
                "absorption_ny": 10,
                "absorption_dphi": 15.0,
                "absorption_niter": 5,
                "absorption_x2tol": 0.01,
                "absorption_outlier": 3.0,
                "outfile": ["crystal1/scales.nxs", "crystal2/scales.nxs"],
            },
            None,
        ),
        # Valid case where outfile is not specified and input hkl files follow a pattern
        (
            [
                "integrated_1.nxs",
                "integrated_2.nxs",
                "--absorption.enable",
                "True",
                "--absorption.nx",
                "10",
                "--absorption.ny",
                "10",
                "--absorption.dphi",
                "15.0",
                "--absorption.niter",
                "5",
                "--absorption.x2tol",
                "0.01",
                "--absorption.outlier",
                "3.0",
            ],
            {
                "hkl": ["integrated_1.nxs", "integrated_2.nxs"],
                "absorption_enable": True,
                "absorption_nx": 10,
                "absorption_ny": 10,
                "absorption_dphi": 15.0,
                "absorption_niter": 5,
                "absorption_x2tol": 0.01,
                "absorption_outlier": 3.0,
                "outfile": ["scales_1.nxs", "scales_2.nxs"],
            },
            None,
        ),
    ],
)
def test_scale_parse_arguments(args, expected, raises):
    """Test the scale command line argument parsing."""
    if raises:
        with pytest.raises(raises):
            scale_parse_arguments(args=args)
    else:
        params = scale_parse_arguments(args=args)
        assert params.hkl == expected["hkl"]
        assert params.absorption_enable == expected["absorption_enable"]
        assert params.absorption_nx == expected["absorption_nx"]
        assert params.absorption_ny == expected["absorption_ny"]
        assert params.absorption_dphi == expected["absorption_dphi"]
        assert params.absorption_niter == expected["absorption_niter"]
        assert params.absorption_x2tol == expected["absorption_x2tol"]
        assert params.absorption_outlier == expected["absorption_outlier"]
        assert params.outfile == expected["outfile"]
