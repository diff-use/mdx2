import pytest

from mdx2.command_line.bin_image_series import parse_arguments as bin_image_series_parse_arguments
from mdx2.command_line.correct import parse_arguments as correct_parse_arguments
from mdx2.command_line.find_peaks import parse_arguments as find_peaks_parse_arguments
from mdx2.command_line.import_data import parse_arguments as import_data_parse_arguments
from mdx2.command_line.import_geometry import parse_arguments as import_geometry_parse_arguments
from mdx2.command_line.integrate import parse_arguments as integrate_parse_arguments
from mdx2.command_line.map import parse_arguments as map_parse_arguments
from mdx2.command_line.mask_peaks import parse_arguments as mask_peaks_parse_arguments
from mdx2.command_line.merge import parse_arguments as merge_parse_arguments
from mdx2.command_line.reintegrate import parse_arguments as reintegrate_parse_arguments
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
        # Negative sample_spacing value
        (
            ["test.expt", "--sample_spacing", "1", "-10", "10", "--outfile", "geometry.nxs"],
            None,
            ValueError,
        ),
        # Zero sample_spacing value
        (
            ["test.expt", "--sample_spacing", "0", "10", "10", "--outfile", "geometry.nxs"],
            None,
            ValueError,
        ),
    ],
)
def test_import_geometry_parse_arguments(args, expected, raises):
    """Test the import geometry command line argument parsing."""
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
        # Zero sigma_cutoff
        (
            ["geometry.nxs", "data.nxs", "--count_threshold", "1000", "--sigma_cutoff", "0"],
            None,
            ValueError,  # __post_init__ raises ValueError for sigma_cutoff <= 0
        ),
        # Negative sigma_cutoff
        (
            ["geometry.nxs", "data.nxs", "--count_threshold", "1000", "--sigma_cutoff", "-1.5"],
            None,
            ValueError,  # __post_init__ raises ValueError for sigma_cutoff <= 0
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
        # Zero sigma_cutoff
        (
            ["geometry.nxs", "data.nxs", "peaks.nxs", "--sigma_cutoff", "0"],
            None,
            ValueError,  # __post_init__ raises ValueError for sigma_cutoff <= 0
        ),
        # Negative sigma_cutoff
        (
            ["geometry.nxs", "data.nxs", "peaks.nxs", "--sigma_cutoff", "-2.0"],
            None,
            ValueError,  # __post_init__ raises ValueError for sigma_cutoff <= 0
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
        # Negative bin size
        (
            ["data.nxs", "-1", "50", "30", "--outfile", "binned_data.nxs", "--nproc", "4"],
            None,
            ValueError,  # __post_init__ raises ValueError for negative bins
        ),
        # Zero bin size
        (
            ["data.nxs", "2", "0", "30", "--outfile", "binned_data.nxs", "--nproc", "4"],
            None,
            ValueError,  # __post_init__ raises ValueError for zero bins
        ),
        # Invalid valid_range: min >= max
        (
            ["data.nxs", "2", "50", "30", "--valid_range", "1000", "0"],
            None,
            ValueError,  # __post_init__ raises ValueError for invalid valid_range
        ),
        # Invalid valid_range: min == max
        (
            ["data.nxs", "2", "50", "30", "--valid_range", "100", "100"],
            None,
            ValueError,  # __post_init__ raises ValueError for invalid valid_range
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
                "scaling": True,
                "offset": True,
                "absorption": True,
                "detector": True,
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
                "scaling": True,
                "offset": True,
                "absorption": True,
                "detector": True,
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
            ValueError,  # __post_init__ raises ValueError for validation errors
        ),
        # Zero outlier
        (
            ["hkl.nxs", "--scale", "scale.nxs", "--outlier", "0"],
            None,
            ValueError,  # __post_init__ raises ValueError for outlier <= 0
        ),
        # Negative outlier
        (
            ["hkl.nxs", "--scale", "scale.nxs", "--outlier", "-2.5"],
            None,
            ValueError,  # __post_init__ raises ValueError for outlier <= 0
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
        assert params.scaling is expected["scaling"]
        assert params.offset is expected["offset"]
        assert params.absorption is expected["absorption"]
        assert params.detector is expected["detector"]


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
        # Zero subdivide element
        (
            ["geom.nxs", "data.nxs", "--subdivide", "0", "2", "2", "--max_spread", "1.5"],
            None,
            ValueError,  # __post_init__ raises ValueError for subdivide <= 0
        ),
        # Negative subdivide element
        (
            ["geom.nxs", "data.nxs", "--subdivide", "2", "-1", "2", "--max_spread", "1.5"],
            None,
            ValueError,  # __post_init__ raises ValueError for subdivide <= 0
        ),
        # Zero max_spread
        (
            ["geom.nxs", "data.nxs", "--subdivide", "2", "2", "2", "--max_spread", "0"],
            None,
            ValueError,  # __post_init__ raises ValueError for max_spread <= 0
        ),
        # Negative max_spread
        (
            ["geom.nxs", "data.nxs", "--subdivide", "2", "2", "2", "--max_spread", "-1.5"],
            None,
            ValueError,  # __post_init__ raises ValueError for max_spread <= 0
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
                "absorption.enable": True,
                "absorption.nx": 10,
                "absorption.ny": 10,
                "absorption.dphi": 15.0,
                "absorption.niter": 5,
                "absorption.x2tol": 0.01,
                "absorption.outlier": 3.0,
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
                "absorption.enable": True,
                "absorption.nx": 10,
                "absorption.ny": 10,
                "absorption.dphi": 15.0,
                "absorption.niter": 5,
                "absorption.x2tol": 0.01,
                "absorption.outlier": 3.0,
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
                "absorption.enable": True,
                "absorption.nx": 10,
                "absorption.ny": 10,
                "absorption.dphi": 15.0,
                "absorption.niter": 5,
                "absorption.x2tol": 0.01,
                "absorption.outlier": 3.0,
                "outfile": ["scales_1.nxs", "scales_2.nxs"],
            },
            None,
        ),
        # Invalid case: mismatched number of input and output files
        (
            [
                "integrated_1.nxs",
                "integrated_2.nxs",
                "--outfile",
                "scales.nxs",
            ],
            None,
            ValueError,
        ),
        # Invalid case: duplicate input files
        (
            [
                "integrated.nxs",
                "integrated.nxs",
            ],
            None,
            ValueError,
        ),
        # Invalid case: duplicate input files with explicit outfiles
        (
            [
                "data.nxs",
                "data.nxs",
                "--outfile",
                "scales_1.nxs",
                "scales_2.nxs",
            ],
            None,
            ValueError,
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
        assert params.absorption.enable == expected["absorption.enable"]
        assert params.absorption.nx == expected["absorption.nx"]
        assert params.absorption.ny == expected["absorption.ny"]
        assert params.absorption.dphi == expected["absorption.dphi"]
        assert params.absorption.niter == expected["absorption.niter"]
        assert params.absorption.x2tol == expected["absorption.x2tol"]
        assert params.absorption.outlier == expected["absorption.outlier"]
        assert params.outfile == expected["outfile"]


@pytest.mark.parametrize(
    "args,expected,raises",
    [
        # Valid case with background
        (
            [
                "geom.nxs",
                "integrated.nxs",
                "--background",
                "background.nxs",
                "--outfile",
                "corrected.nxs",
            ],
            {
                "geom": "geom.nxs",
                "hkl": "integrated.nxs",
                "background": "background.nxs",
                "attenuation": True,
                "efficiency": True,
                "polarization": True,
                "lorentz": False,
                "p1": False,
                "outfile": "corrected.nxs",
            },
            None,
        ),
        # Valid case without background
        (
            [
                "geom.nxs",
                "integrated.nxs",
                "--outfile",
                "corrected.nxs",
            ],
            {
                "geom": "geom.nxs",
                "hkl": "integrated.nxs",
                "background": None,
                "attenuation": True,
                "efficiency": True,
                "polarization": True,
                "lorentz": False,
                "p1": False,
                "outfile": "corrected.nxs",
            },
            None,
        ),
        # Valid case with correction flags
        (
            [
                "geom.nxs",
                "integrated.nxs",
                "--background",
                "background.nxs",
                "--lorentz",
                "True",
                "--p1",
                "True",
                "--attenuation",
                "False",
                "--outfile",
                "corrected.nxs",
            ],
            {
                "geom": "geom.nxs",
                "hkl": "integrated.nxs",
                "background": "background.nxs",
                "attenuation": False,
                "efficiency": True,
                "polarization": True,
                "lorentz": True,
                "p1": True,
                "outfile": "corrected.nxs",
            },
            None,
        ),
        # Missing required argument
        (
            ["geom.nxs"],
            None,
            SystemExit,
        ),
    ],
)
def test_correct_parse_arguments(args, expected, raises):
    """Test the correct command line argument parsing."""
    if raises:
        with pytest.raises(raises):
            correct_parse_arguments(args=args)
    else:
        params = correct_parse_arguments(args=args)
        assert params.geom == expected["geom"]
        assert params.hkl == expected["hkl"]
        assert params.background == expected["background"]
        assert params.attenuation == expected["attenuation"]
        assert params.efficiency == expected["efficiency"]
        assert params.polarization == expected["polarization"]
        assert params.lorentz == expected["lorentz"]
        assert params.p1 == expected["p1"]
        assert params.outfile == expected["outfile"]


@pytest.mark.parametrize(
    "args,expected,raises",
    [
        # Valid case with both mask and background
        (
            [
                "geom.nxs",
                "data.nxs",
                "--mask",
                "mask.nxs",
                "--scale",
                "scales.nxs",
                "--background",
                "background.nxs",
                "--subdivide",
                "2",
                "2",
                "2",
                "--output",
                "counts",
                "--outfile",
                "reintegrated.nxs",
                "--nproc",
                "4",
            ],
            {
                "geom": "geom.nxs",
                "data": "data.nxs",
                "mask": "mask.nxs",
                "scale": "scales.nxs",
                "background": "background.nxs",
                "subdivide": (2, 2, 2),
                "output": "counts",
                "outfile": "reintegrated.nxs",
                "nproc": 4,
            },
            None,
        ),
        # Valid case with mask but no background
        (
            [
                "geom.nxs",
                "data.nxs",
                "--mask",
                "mask.nxs",
                "--scale",
                "scales.nxs",
                "--subdivide",
                "2",
                "2",
                "2",
                "--outfile",
                "reintegrated.nxs",
                "--nproc",
                "4",
            ],
            {
                "geom": "geom.nxs",
                "data": "data.nxs",
                "mask": "mask.nxs",
                "scale": "scales.nxs",
                "background": None,
                "subdivide": (2, 2, 2),
                "output": "counts",
                "outfile": "reintegrated.nxs",
                "nproc": 4,
            },
            None,
        ),
        # Valid case with background but no mask
        (
            [
                "geom.nxs",
                "data.nxs",
                "--scale",
                "scales.nxs",
                "--background",
                "background.nxs",
                "--subdivide",
                "2",
                "2",
                "2",
                "--outfile",
                "reintegrated.nxs",
                "--nproc",
                "4",
            ],
            {
                "geom": "geom.nxs",
                "data": "data.nxs",
                "mask": None,
                "scale": "scales.nxs",
                "background": "background.nxs",
                "subdivide": (2, 2, 2),
                "output": "counts",
                "outfile": "reintegrated.nxs",
                "nproc": 4,
            },
            None,
        ),
        # Valid case with neither mask nor background
        (
            [
                "geom.nxs",
                "data.nxs",
                "--subdivide",
                "2",
                "2",
                "2",
                "--outfile",
                "reintegrated.nxs",
                "--nproc",
                "4",
            ],
            {
                "geom": "geom.nxs",
                "data": "data.nxs",
                "mask": None,
                "scale": None,
                "background": None,
                "subdivide": (2, 2, 2),
                "output": "counts",
                "outfile": "reintegrated.nxs",
                "nproc": 4,
            },
            None,
        ),
        # Missing required argument
        (
            ["geom.nxs"],
            None,
            SystemExit,
        ),
        # Zero subdivide element
        (
            ["geom.nxs", "data.nxs", "--subdivide", "0", "2", "2"],
            None,
            ValueError,  # __post_init__ raises ValueError for subdivide <= 0
        ),
        # Negative subdivide element
        (
            ["geom.nxs", "data.nxs", "--subdivide", "2", "-1", "2"],
            None,
            ValueError,  # __post_init__ raises ValueError for subdivide <= 0
        ),
    ],
)
def test_reintegrate_parse_arguments(args, expected, raises):
    """Test the reintegrate command line argument parsing."""
    if raises:
        with pytest.raises(raises):
            reintegrate_parse_arguments(args=args)
    else:
        params = reintegrate_parse_arguments(args=args)
        assert params.geom == expected["geom"]
        assert params.data == expected["data"]
        assert params.mask == expected["mask"]
        assert params.scale == expected["scale"]
        assert params.background == expected["background"]
        assert tuple(params.subdivide) == expected["subdivide"]
        assert params.output == expected["output"]
        assert params.outfile == expected["outfile"]
        assert params.nproc == expected["nproc"]


@pytest.mark.parametrize(
    "args,expected,raises",
    [
        # Valid case
        (
            [
                "geometry.nxs",
                "hkl_table.nxs",
                "--limits",
                "0",
                "10",
                "0",
                "10",
                "0",
                "10",
                "--outfile",
                "map.nxs",
            ],
            {
                "geom": "geometry.nxs",
                "hkl": "hkl_table.nxs",
                "limits": (0.0, 10.0, 0.0, 10.0, 0.0, 10.0),
                "symmetry": True,
                "signal": "intensity",
                "outfile": "map.nxs",
            },
            None,
        ),
        # Valid limits: lmin == lmax
        (
            ["geometry.nxs", "hkl_table.nxs", "--limits", "0", "10", "0", "10", "5", "5"],
            {
                "geom": "geometry.nxs",
                "hkl": "hkl_table.nxs",
                "limits": (0.0, 10.0, 0.0, 10.0, 5.0, 5.0),
                "symmetry": True,
                "signal": "intensity",
                "outfile": "map.nxs",
            },
            None,
        ),
        # Invalid limits: hmin > hmax
        (
            ["geometry.nxs", "hkl_table.nxs", "--limits", "10", "5", "0", "10", "0", "10"],
            None,
            ValueError,  # __post_init__ raises ValueError for hmin > hmax
        ),
        # Invalid limits: kmin > kmax
        (
            ["geometry.nxs", "hkl_table.nxs", "--limits", "0", "10", "11", "10", "0", "10"],
            None,
            ValueError,  # __post_init__ raises ValueError for kmin > kmax
        ),
        # Invalid limits: lmin > lmax
        (
            ["geometry.nxs", "hkl_table.nxs", "--limits", "0", "10", "0", "10", "15", "10"],
            None,
            ValueError,  # __post_init__ raises ValueError for lmin > lmax
        ),
    ],
)
def test_map_parse_arguments(args, expected, raises):
    """Test the map command line argument parsing."""
    if raises:
        with pytest.raises(raises):
            map_parse_arguments(args=args)
    else:
        params = map_parse_arguments(args=args)
        assert params.geom == expected["geom"]
        assert params.hkl == expected["hkl"]
        assert tuple(params.limits) == expected["limits"]
        assert params.symmetry == expected["symmetry"]
        assert params.signal == expected["signal"]
        assert params.outfile == expected["outfile"]
