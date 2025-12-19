import numpy as np
import pytest

from mdx2.processing import _histogram


class TestHistogram:
    """Tests for the custom histogram function against np.histogram"""

    def test_uniform_bins_basic(self):
        """Test with simple uniform integer bins"""
        data = np.array([0, 1, 2, 3, 4, 5, 1, 2, 3])
        bin_edges = np.array([0, 1, 2, 3, 4, 5, 6])

        custom_counts = _histogram(data, bin_edges)
        np_counts, _ = np.histogram(data, bin_edges)

        np.testing.assert_array_equal(custom_counts, np_counts)

    def test_uniform_bins_with_negatives(self):
        """Test that negative values are ignored with uniform bins starting at 0"""
        data = np.array([-2, -1, 0, 1, 2, 3, 4])
        bin_edges = np.array([0, 1, 2, 3, 4, 5])

        custom_counts = _histogram(data, bin_edges)
        np_counts, _ = np.histogram(data, bin_edges)

        np.testing.assert_array_equal(custom_counts, np_counts)

    def test_nonuniform_bins(self):
        """Test with non-uniform bin spacing"""
        data = np.array([0, 5, 10, 15, 25, 50, 100, 200])
        bin_edges = np.array([0, 10, 20, 50, 100, 200, 500])

        custom_counts = _histogram(data, bin_edges)
        np_counts, _ = np.histogram(data, bin_edges)

        np.testing.assert_array_equal(custom_counts, np_counts)

    def test_hdr_bins_mixed(self):
        """Test HDR-style bins with uniform start and non-uniform tail"""
        # Bins: [0,1), [1,2), [2,3), [3,5), [5,10), [10,20)
        bin_edges = np.array([0, 1, 2, 3, 5, 10, 20])
        data = np.array([0, 1, 2, 3, 4, 6, 8, 12, 15, 19])

        custom_counts = _histogram(data, bin_edges)
        np_counts, _ = np.histogram(data, bin_edges)

        np.testing.assert_array_equal(custom_counts, np_counts)

    def test_edge_values(self):
        """Test behavior with values exactly on bin edges"""
        data = np.array([0, 5, 10, 15, 20])
        bin_edges = np.array([0, 5, 10, 15, 20])

        custom_counts = _histogram(data, bin_edges)
        np_counts, _ = np.histogram(data, bin_edges)

        np.testing.assert_array_equal(custom_counts, np_counts)

    def test_max_edge_value(self):
        """Test that values equal to max edge go in last bin"""
        data = np.array([0, 5, 10, 10])
        bin_edges = np.array([0, 5, 10])

        custom_counts = _histogram(data, bin_edges)
        # np.histogram puts max edge value in last bin
        np_counts, _ = np.histogram(data, bin_edges)

        np.testing.assert_array_equal(custom_counts, np_counts)

    def test_out_of_range_values(self):
        """Test handling of values outside bin range"""
        data = np.array([-5, 0, 5, 10, 25, 100])
        bin_edges = np.array([0, 5, 10, 20])

        custom_counts = _histogram(data, bin_edges)
        np_counts, _ = np.histogram(data, bin_edges)

        np.testing.assert_array_equal(custom_counts, np_counts)

    def test_large_uniform_region(self):
        """Test with large uniform region at start"""
        data = np.random.randint(0, 100, size=1000)
        # First 100 bins uniform, then non-uniform
        bin_edges = np.concatenate([np.arange(0, 100), np.array([100, 200, 500, 1000, 2000])])

        custom_counts = _histogram(data, bin_edges)
        np_counts, _ = np.histogram(data, bin_edges)

        np.testing.assert_array_equal(custom_counts, np_counts)

    def test_empty_data(self):
        """Test with empty data array"""
        data = np.array([], dtype=np.int64)
        bin_edges = np.array([0, 1, 2, 3, 4, 5])

        custom_counts = _histogram(data, bin_edges)
        np_counts, _ = np.histogram(data, bin_edges)

        np.testing.assert_array_equal(custom_counts, np_counts)

    def test_single_bin(self):
        """Test with only one bin"""
        data = np.array([0, 1, 2, 5, 5, 5])
        bin_edges = np.array([0, 10])

        custom_counts = _histogram(data, bin_edges)
        np_counts, _ = np.histogram(data, bin_edges)

        np.testing.assert_array_equal(custom_counts, np_counts)

    def test_dtype_preservation(self):
        """Test that output dtype is int64"""
        data = np.array([0, 1, 2, 3], dtype=np.uint16)
        bin_edges = np.array([0, 1, 2, 3, 4], dtype=np.uint16)

        custom_counts = _histogram(data, bin_edges)

        assert custom_counts.dtype == np.int64
        assert len(custom_counts) == len(bin_edges) - 1

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_random_data(self, seed):
        """Test with random data and various bin configurations"""
        np.random.seed(seed)
        data = np.random.randint(0, 1000, size=10000)

        # HDR-style bins
        bin_edges = [0]
        val = 1
        while val < 1000:
            bin_edges.append(val)
            val = int(val * 1.2) + 1
        bin_edges.append(1000)
        bin_edges = np.array(bin_edges)

        custom_counts = _histogram(data, bin_edges)
        np_counts, _ = np.histogram(data, bin_edges)

        np.testing.assert_array_equal(custom_counts, np_counts)

    def test_repeated_values(self):
        """Test with many repeated values"""
        data = np.array([5] * 100 + [10] * 50 + [15] * 25)
        bin_edges = np.array([0, 5, 10, 15, 20])

        custom_counts = _histogram(data, bin_edges)
        np_counts, _ = np.histogram(data, bin_edges)

        np.testing.assert_array_equal(custom_counts, np_counts)
