import logging
import time

import numpy as np
import pytest
from scipy.interpolate import interpn

from mdx2.utils import interp2, interp3, interp_g2g_bilinear, interp_g2g_trilinear

logger = logging.getLogger(__name__)


class TestInterpG2GBilinear:
    """Test suite comparing utils.interp_g2g_bilinear to scipy.interpolate.interpn"""

    @pytest.fixture
    def simple_grid(self):
        """Simple 2D grid for basic testing"""
        axes = [np.arange(10), np.arange(10)]
        data = np.random.rand(10, 10)
        return axes, data

    def test_same_grid(self, simple_grid):
        """Test interpolation onto the same grid returns original data"""
        axes, data = simple_grid

        result = interp_g2g_bilinear(*axes, data, *axes)

        np.testing.assert_allclose(result, data, rtol=1e-10)

    def test_upsampled_grid(self, simple_grid):
        """Test interpolation onto a finer grid"""
        axes, data = simple_grid

        # Create finer grid (2x resolution) - stay slightly inside bounds
        new_axes = [
            np.linspace(axes[0][0] + 0.01, axes[0][-1] - 0.01, 19),
            np.linspace(axes[1][0] + 0.01, axes[1][-1] - 0.01, 19),
        ]

        result = interp_g2g_bilinear(*axes, data, *new_axes)

        # Compare with scipy
        points_x, points_y = np.meshgrid(*new_axes, indexing="ij")
        points = np.column_stack([points_x.ravel(), points_y.ravel()])
        result_scipy = interpn(axes, data, points, method="linear").reshape(result.shape)

        np.testing.assert_allclose(result, result_scipy, rtol=1e-10)

    def test_downsampled_grid(self, simple_grid):
        """Test interpolation onto a coarser grid"""
        axes, data = simple_grid

        # Create coarser grid - stay inside bounds
        new_axes = [
            np.linspace(axes[0][0] + 0.5, axes[0][-1] - 0.5, 5),
            np.linspace(axes[1][0] + 0.5, axes[1][-1] - 0.5, 5),
        ]

        result = interp_g2g_bilinear(*axes, data, *new_axes)

        # Compare with scipy
        points_x, points_y = np.meshgrid(*new_axes, indexing="ij")
        points = np.column_stack([points_x.ravel(), points_y.ravel()])
        result_scipy = interpn(axes, data, points, method="linear").reshape(result.shape)

        np.testing.assert_allclose(result, result_scipy, rtol=1e-10)

    def test_boundary_values(self, simple_grid):
        """Test that boundary values are handled correctly - SPECIFICALLY TESTS BOUNDARIES"""
        axes, data = simple_grid

        # Grid that includes exact boundary points
        new_axes = [np.linspace(axes[0][0], axes[0][-1], 15), np.linspace(axes[1][0], axes[1][-1], 15)]

        result = interp_g2g_bilinear(*axes, data, *new_axes)

        # Compare with scipy
        points_x, points_y = np.meshgrid(*new_axes, indexing="ij")
        points = np.column_stack([points_x.ravel(), points_y.ravel()])
        result_scipy = interpn(axes, data, points, method="linear").reshape(result.shape)

        np.testing.assert_allclose(result, result_scipy, rtol=1e-10)

        # Check corners explicitly
        np.testing.assert_allclose(result[0, 0], data[0, 0], rtol=1e-10)
        np.testing.assert_allclose(result[-1, -1], data[-1, -1], rtol=1e-10)
        np.testing.assert_allclose(result[0, -1], data[0, -1], rtol=1e-10)
        np.testing.assert_allclose(result[-1, 0], data[-1, 0], rtol=1e-10)

    def test_nonuniform_grid(self):
        """Test with non-uniformly spaced source and target grids"""
        axes = [np.array([0, 1, 3, 7, 10]), np.array([0, 2, 5, 8, 10])]
        data = np.random.rand(5, 5)

        # Non-uniform target grid - stay inside bounds
        new_axes = [np.array([0.5, 2, 4, 6, 8.5]), np.array([1, 3, 6, 9])]

        result = interp_g2g_bilinear(*axes, data, *new_axes)

        # Compare with scipy
        points_x, points_y = np.meshgrid(*new_axes, indexing="ij")
        points = np.column_stack([points_x.ravel(), points_y.ravel()])
        result_scipy = interpn(axes, data, points, method="linear").reshape(result.shape)

        np.testing.assert_allclose(result, result_scipy, rtol=1e-10)

    def test_rectangular_grids(self):
        """Test with non-square grids"""
        axes = [np.linspace(0, 10, 20), np.linspace(0, 5, 10)]
        data = np.random.rand(20, 10)

        # Stay inside bounds
        new_axes = [np.linspace(0.1, 9.9, 25), np.linspace(0.1, 4.9, 8)]

        result = interp_g2g_bilinear(*axes, data, *new_axes)

        # Compare with scipy
        points_x, points_y = np.meshgrid(*new_axes, indexing="ij")
        points = np.column_stack([points_x.ravel(), points_y.ravel()])
        result_scipy = interpn(axes, data, points, method="linear").reshape(result.shape)

        np.testing.assert_allclose(result, result_scipy, rtol=1e-10)

    def test_single_point_grids(self):
        """Test with single-point target grid"""
        axes = [np.arange(10), np.arange(10)]
        data = np.random.rand(10, 10)

        # Single point in the middle (away from boundaries)
        new_axes = [np.array([4.5]), np.array([5.2])]

        result = interp_g2g_bilinear(*axes, data, *new_axes)

        # Compare with scipy
        points = np.array([[4.5, 5.2]])
        result_scipy = interpn(axes, data, points, method="linear").reshape(result.shape)

        np.testing.assert_allclose(result, result_scipy, rtol=1e-10)


class TestInterpG2GTrilinear:
    """Test suite comparing utils.interp_g2g_trilinear to scipy.interpolate.interpn"""

    @pytest.fixture
    def simple_grid(self):
        """Simple 3D grid for basic testing"""
        axes = [np.arange(10), np.arange(10), np.arange(10)]
        data = np.random.rand(10, 10, 10)
        return axes, data

    def test_same_grid(self, simple_grid):
        """Test interpolation onto the same grid returns original data"""
        axes, data = simple_grid

        result = interp_g2g_trilinear(*axes, data, *axes)

        np.testing.assert_allclose(result, data, rtol=1e-10)

    def test_upsampled_grid(self, simple_grid):
        """Test interpolation onto a finer grid"""
        axes, data = simple_grid

        # Create finer grid (2x resolution) - stay slightly inside bounds
        new_axes = [
            np.linspace(axes[0][0] + 0.01, axes[0][-1] - 0.01, 19),
            np.linspace(axes[1][0] + 0.01, axes[1][-1] - 0.01, 19),
            np.linspace(axes[2][0] + 0.01, axes[2][-1] - 0.01, 19),
        ]

        result = interp_g2g_trilinear(*axes, data, *new_axes)

        # Compare with scipy
        points_x, points_y, points_z = np.meshgrid(*new_axes, indexing="ij")
        points = np.column_stack([points_x.ravel(), points_y.ravel(), points_z.ravel()])
        result_scipy = interpn(axes, data, points, method="linear").reshape(result.shape)

        np.testing.assert_allclose(result, result_scipy, rtol=1e-10)

    def test_downsampled_grid(self, simple_grid):
        """Test interpolation onto a coarser grid"""
        axes, data = simple_grid

        # Create coarser grid - stay inside bounds
        new_axes = [
            np.linspace(axes[0][0] + 0.5, axes[0][-1] - 0.5, 5),
            np.linspace(axes[1][0] + 0.5, axes[1][-1] - 0.5, 5),
            np.linspace(axes[2][0] + 0.5, axes[2][-1] - 0.5, 5),
        ]

        result = interp_g2g_trilinear(*axes, data, *new_axes)

        # Compare with scipy
        points_x, points_y, points_z = np.meshgrid(*new_axes, indexing="ij")
        points = np.column_stack([points_x.ravel(), points_y.ravel(), points_z.ravel()])
        result_scipy = interpn(axes, data, points, method="linear").reshape(result.shape)

        np.testing.assert_allclose(result, result_scipy, rtol=1e-10)

    def test_boundary_values(self, simple_grid):
        """Test that boundary values are handled correctly - SPECIFICALLY TESTS BOUNDARIES"""
        axes, data = simple_grid

        # Grid that includes exact boundary points
        new_axes = [
            np.linspace(axes[0][0], axes[0][-1], 15),
            np.linspace(axes[1][0], axes[1][-1], 15),
            np.linspace(axes[2][0], axes[2][-1], 15),
        ]

        result = interp_g2g_trilinear(*axes, data, *new_axes)

        # Compare with scipy
        points_x, points_y, points_z = np.meshgrid(*new_axes, indexing="ij")
        points = np.column_stack([points_x.ravel(), points_y.ravel(), points_z.ravel()])
        result_scipy = interpn(axes, data, points, method="linear").reshape(result.shape)

        np.testing.assert_allclose(result, result_scipy, rtol=1e-10)

        # Check corners explicitly
        np.testing.assert_allclose(result[0, 0, 0], data[0, 0, 0], rtol=1e-10)
        np.testing.assert_allclose(result[-1, -1, -1], data[-1, -1, -1], rtol=1e-10)
        np.testing.assert_allclose(result[0, 0, -1], data[0, 0, -1], rtol=1e-10)
        np.testing.assert_allclose(result[-1, -1, 0], data[-1, -1, 0], rtol=1e-10)

    def test_nonuniform_grid(self):
        """Test with non-uniformly spaced source and target grids"""
        axes = [np.array([0, 1, 3, 7, 10]), np.array([0, 2, 5, 8, 10]), np.array([0, 1.5, 4, 7, 10])]
        data = np.random.rand(5, 5, 5)

        # Non-uniform target grid - stay inside bounds
        new_axes = [np.array([0.5, 2, 4, 6, 8.5]), np.array([1, 3, 6, 9]), np.array([0.5, 3, 5.5, 8])]

        result = interp_g2g_trilinear(*axes, data, *new_axes)

        # Compare with scipy
        points_x, points_y, points_z = np.meshgrid(*new_axes, indexing="ij")
        points = np.column_stack([points_x.ravel(), points_y.ravel(), points_z.ravel()])
        result_scipy = interpn(axes, data, points, method="linear").reshape(result.shape)

        np.testing.assert_allclose(result, result_scipy, rtol=1e-10)

    def test_anisotropic_grids(self):
        """Test with grids of very different sizes in each dimension"""
        axes = [np.linspace(0, 10, 50), np.linspace(0, 5, 10), np.linspace(0, 2, 5)]
        data = np.random.rand(50, 10, 5)

        # Stay inside bounds
        new_axes = [np.linspace(0.1, 9.9, 40), np.linspace(0.1, 4.9, 12), np.linspace(0.05, 1.95, 8)]

        # Time the Numba implementation
        start = time.time()
        result = interp_g2g_trilinear(*axes, data, *new_axes)
        numba_time = time.time() - start

        # Time the scipy implementation
        start = time.time()
        points_x, points_y, points_z = np.meshgrid(*new_axes, indexing="ij")
        points = np.column_stack([points_x.ravel(), points_y.ravel(), points_z.ravel()])
        result_scipy = interpn(axes, data, points, method="linear").reshape(result.shape)
        scipy_time = time.time() - start

        logger.info("\ntest_anisotropic_grids timing:")
        logger.info(f"  Numba interp_g2g_trilinear: {numba_time * 1000:.1f}ms")
        logger.info(f"  Scipy interpn: {scipy_time * 1000:.1f}ms")
        logger.info(f"  Scipy is {scipy_time / numba_time:.1f}x slower")
        logger.info(f"  Total test time: {(numba_time + scipy_time) * 1000:.1f}ms")

        np.testing.assert_allclose(result, result_scipy, rtol=1e-10)

    def test_single_point_grids(self):
        """Test with single-point target grid"""
        axes = [np.arange(10), np.arange(10), np.arange(10)]
        data = np.random.rand(10, 10, 10)

        # Single point in the middle (away from boundaries)
        new_axes = [np.array([4.5]), np.array([5.2]), np.array([3.7])]

        result = interp_g2g_trilinear(*axes, data, *new_axes)

        # Compare with scipy
        points = np.array([[4.5, 5.2, 3.7]])
        result_scipy = interpn(axes, data, points, method="linear").reshape(result.shape)

        np.testing.assert_allclose(result, result_scipy, rtol=1e-10)

    def test_4d_data(self):
        """Test trilinear interpolation with 4D data (3D grid + vector components)"""
        axes = [np.arange(8), np.arange(8), np.arange(8)]
        data = np.random.rand(8, 8, 8, 3)  # 3 vector components

        # Stay inside bounds
        new_axes = [np.linspace(0.1, 6.9, 10), np.linspace(0.1, 6.9, 10), np.linspace(0.1, 6.9, 10)]

        result = interp_g2g_trilinear(*axes, data, *new_axes)

        # Compare each component with scipy
        for component in range(3):
            points_x, points_y, points_z = np.meshgrid(*new_axes, indexing="ij")
            points = np.column_stack([points_x.ravel(), points_y.ravel(), points_z.ravel()])
            result_scipy = interpn(axes, data[..., component], points, method="linear").reshape(
                result[..., component].shape
            )

            np.testing.assert_allclose(result[..., component], result_scipy, rtol=1e-10)


class TestInterp2:
    """Test suite comparing utils.interp2 to scipy.interpolate.interpn"""

    @pytest.fixture
    def simple_grid(self):
        """Simple 2D grid for basic testing"""
        axes = [np.arange(10), np.arange(10)]
        data = np.random.rand(10, 10)
        return axes, data

    @pytest.fixture
    def random_points(self):
        """Random interpolation points within bounds"""
        np.random.seed(42)
        x = np.random.uniform(0, 9, size=100)
        y = np.random.uniform(0, 9, size=100)
        return x, y

    def test_random_points_match_scipy(self, simple_grid, random_points):
        """Test that interp2 matches scipy.interpolate.interpn for random points"""
        axes, data = simple_grid
        x, y = random_points

        # Our implementation
        result_ours = interp2(*axes, data, x, y, order=1)

        # Scipy reference
        points = np.column_stack([x, y])
        result_scipy = interpn(axes, data, points, method="linear")

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)

    def test_grid_points_exact(self, simple_grid):
        """Test that interpolation at grid points returns exact values"""
        axes, data = simple_grid

        # Sample at grid points
        x = np.array([0, 5, 9], dtype=float)
        y = np.array([0, 5, 9], dtype=float)

        result_ours = interp2(*axes, data, x, y, order=1)
        result_scipy = interpn(axes, data, np.column_stack([x, y]), method="linear")

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)

        # Check actual values at grid points
        expected = np.array([data[0, 0], data[5, 5], data[9, 9]])
        np.testing.assert_allclose(result_ours, expected, rtol=1e-10)

    def test_midpoint_interpolation(self, simple_grid):
        """Test interpolation at midpoints between grid points"""
        axes, data = simple_grid

        # Midpoints should be average of surrounding values
        x = np.array([0.5, 4.5])
        y = np.array([0.5, 4.5])

        result_ours = interp2(*axes, data, x, y, order=1)
        result_scipy = interpn(axes, data, np.column_stack([x, y]), method="linear")

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)

    def test_edge_behavior(self, simple_grid):
        """Test behavior at and slightly beyond boundaries"""
        axes, data = simple_grid

        # Points at edges and slightly beyond
        x = np.array([0.0, 9.0, -0.1, 9.1])
        y = np.array([0.0, 0.0, 0.0, 0.0])

        result_ours = interp2(*axes, data, x, y, order=1)

        # Scipy with bounds_error=False and fill_value=None uses nearest for extrapolation
        result_scipy = interpn(
            axes, data, np.column_stack([x, y]), method="linear", bounds_error=False, fill_value=None
        )

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)

    def test_nonuniform_grid(self):
        """Test with non-uniformly spaced grid"""
        axes = [np.array([0, 1, 3, 7, 10]), np.array([0, 2, 5, 8, 10])]
        data = np.random.rand(5, 5)

        # Random points within bounds
        np.random.seed(123)
        x = np.random.uniform(0, 10, size=50)
        y = np.random.uniform(0, 10, size=50)

        result_ours = interp2(*axes, data, x, y, order=1)
        result_scipy = interpn(axes, data, np.column_stack([x, y]), method="linear")

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)

    def test_single_point(self, simple_grid):
        """Test interpolation of a single point"""
        axes, data = simple_grid

        x = np.array([4.5])
        y = np.array([5.2])

        result_ours = interp2(*axes, data, x, y, order=1)
        result_scipy = interpn(axes, data, np.column_stack([x, y]), method="linear")

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)
        assert result_ours.shape == (1,)

    def test_rectangular_grid(self):
        """Test with non-square grid"""
        axes = [np.linspace(0, 10, 20), np.linspace(0, 5, 10)]
        data = np.random.rand(20, 10)

        np.random.seed(789)
        x = np.random.uniform(0, 9.9, size=100)
        y = np.random.uniform(0, 4.9, size=100)

        result_ours = interp2(*axes, data, x, y, order=1)
        result_scipy = interpn(axes, data, np.column_stack([x, y]), method="linear")

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)


class TestInterp3:
    """Test suite comparing utils.interp3 to scipy.interpolate.interpn"""

    @pytest.fixture
    def simple_grid(self):
        """Simple 3D grid for basic testing"""
        axes = [np.arange(10), np.arange(10), np.arange(10)]
        data = np.random.rand(10, 10, 10)
        return axes, data

    @pytest.fixture
    def random_points(self):
        """Random interpolation points within bounds"""
        np.random.seed(42)
        x = np.random.uniform(0, 9, size=100)
        y = np.random.uniform(0, 9, size=100)
        z = np.random.uniform(0, 9, size=100)
        return x, y, z

    def test_random_points_match_scipy(self, simple_grid, random_points):
        """Test that interp3 matches scipy.interpolate.interpn for random points"""
        axes, data = simple_grid
        x, y, z = random_points

        # Our implementation
        result_ours = interp3(*axes, data, x, y, z, order=1)

        # Scipy reference
        points = np.column_stack([x, y, z])
        result_scipy = interpn(axes, data, points, method="linear")

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)

    def test_grid_points_exact(self, simple_grid):
        """Test that interpolation at grid points returns exact values"""
        axes, data = simple_grid

        # Sample at grid points
        x = np.array([0, 5, 9], dtype=float)
        y = np.array([0, 5, 9], dtype=float)
        z = np.array([0, 5, 9], dtype=float)

        result_ours = interp3(*axes, data, x, y, z, order=1)
        result_scipy = interpn(axes, data, np.column_stack([x, y, z]), method="linear")

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)

        # Check actual values at grid points
        expected = np.array([data[0, 0, 0], data[5, 5, 5], data[9, 9, 9]])
        np.testing.assert_allclose(result_ours, expected, rtol=1e-10)

    def test_midpoint_interpolation(self, simple_grid):
        """Test interpolation at midpoints between grid points"""
        axes, data = simple_grid

        # Midpoints should be average of surrounding values
        x = np.array([0.5, 4.5])
        y = np.array([0.5, 4.5])
        z = np.array([0.5, 4.5])

        result_ours = interp3(*axes, data, x, y, z, order=1)
        result_scipy = interpn(axes, data, np.column_stack([x, y, z]), method="linear")

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)

    def test_edge_behavior(self, simple_grid):
        """Test behavior at and slightly beyond boundaries"""
        axes, data = simple_grid

        # Points at edges and slightly beyond
        x = np.array([0.0, 9.0, -0.1, 9.1])
        y = np.array([0.0, 0.0, 0.0, 0.0])
        z = np.array([0.0, 0.0, 0.0, 0.0])

        result_ours = interp3(*axes, data, x, y, z, order=1)

        # Scipy with bounds_error=False and fill_value=None uses nearest for extrapolation
        result_scipy = interpn(
            axes, data, np.column_stack([x, y, z]), method="linear", bounds_error=False, fill_value=None
        )

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)

    def test_nonuniform_grid(self):
        """Test with non-uniformly spaced grid"""
        axes = [np.array([0, 1, 3, 7, 10]), np.array([0, 2, 5, 8, 10]), np.array([0, 1.5, 4, 7, 10])]
        data = np.random.rand(5, 5, 5)

        # Random points within bounds
        np.random.seed(123)
        x = np.random.uniform(0, 10, size=50)
        y = np.random.uniform(0, 10, size=50)
        z = np.random.uniform(0, 10, size=50)

        result_ours = interp3(*axes, data, x, y, z, order=1)
        result_scipy = interpn(axes, data, np.column_stack([x, y, z]), method="linear")

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)

    def test_single_point(self, simple_grid):
        """Test interpolation of a single point"""
        axes, data = simple_grid

        x = np.array([4.5])
        y = np.array([5.2])
        z = np.array([3.7])

        result_ours = interp3(*axes, data, x, y, z, order=1)
        result_scipy = interpn(axes, data, np.column_stack([x, y, z]), method="linear")

        np.testing.assert_allclose(result_ours, result_scipy, rtol=1e-10)
        assert result_ours.shape == (1,)
