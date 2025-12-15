"""Algorithms to compute data statistics"""

import numpy as np
from numba import jit
from scipy.stats import poisson

from mdx2.utils import slice_sections


def find_peaks(data, threshold=None, p_value=1e-6, bin_size=None):
    """Find peaks in data using HDR histogram for threshold estimation"""
    # first, handle the simple case where threshold is provided
    if threshold is not None:
        peak_mask = data >= threshold
        return peak_mask
    # otherwise, estimate threshold dynamically
    # if bin_size is not provided, use the entire data as a single bin
    if bin_size is None:
        bin_size = data.shape
    # check that bin_size is a list or array of length equal to data.ndim
    if len(bin_size) != data.ndim:
        raise ValueError("bin_size must be a list or array of length equal to data.ndim")
    # if bin_size has any None values, replace them with the corresponding data.shape values
    bin_size = [data.shape[i] if bin_size[i] is None else bin_size[i] for i in range(data.ndim)]

    # now, compute the number of bins in each dimension
    bin_size = np.array(bin_size)
    nbins = np.ceil(np.array(data.shape) / bin_size).astype(int)

    # initialize the peak mask
    peak_mask = np.zeros(data.shape, dtype=bool)  # True where peaks are found

    # now, generate the slices along each dimension of the array
    slice_axes = [slice_sections(sh, nb) for sh, nb in zip(data.shape, nbins)]
    # iterate over all combinations of slices
    for ind in np.ndindex(*nbins):
        # get the slices for this multi-index
        sl = tuple(slice_axes[dim][ind[dim]] for dim in range(data.ndim))
        data_chunk = data[sl]
        hdr_hist = HDRHistogram(
            data_chunk.ravel()
        )  # note: this works only because masked values are negative, and bin edges start at 0
        threshold = hdr_hist.estimate_peak_threshold(p_value=p_value)
        peak_mask[sl] = data_chunk >= threshold

    return peak_mask


@jit
def _histogram(data, bin_edges):
    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)

    # Find the breakpoint: last bin with uniform integer width
    break_idx = 0
    for i in range(len(bin_edges) - 1):
        if bin_edges[i + 1] - bin_edges[i] == 1:
            break_idx = i + 1
        else:
            break

    # Separate uniform and non-uniform regions
    # uniform_edges = bin_edges[: break_idx + 1]
    nonuniform_edges = bin_edges[break_idx:]

    for value in data:
        # Handle values in uniform bin region (direct indexing)
        if value < break_idx:
            if value >= 0:
                counts[value] += 1
        # Handle values in non-uniform bin region (binary search)
        elif value >= bin_edges[break_idx]:
            if value > bin_edges[-1]:
                continue
            if value == bin_edges[-1]:
                counts[-1] += 1
                continue

            # Binary search in non-uniform region
            left = 0
            right = len(nonuniform_edges) - 1

            while right - left > 1:
                mid = (left + right) // 2
                if nonuniform_edges[mid] <= value:
                    left = mid
                else:
                    right = mid

            # Offset by break_idx to get correct index in counts array
            counts[break_idx + left] += 1

    return counts


class HDRHistogram:
    """High Dynamic Range Histogram for Poisson data"""

    def __init__(self, data, ratio=0.125):
        self.bin_edges = self._calc_hdr_bins(dtype=data.dtype, ratio=ratio)
        # check if data is a masked array, and if so, use only the valid data
        if np.ma.isMaskedArray(data):
            data = data.compressed()
        else:
            data = data.ravel()
        self.counts = _histogram(data, self.bin_edges)

    def _calc_hdr_bins(self, dtype, ratio):
        """Calculate histogram bin edges for a high dynamic range histogram"""
        if not np.issubdtype(dtype, np.integer):
            raise ValueError("dtype must be an integer data type")
        val = np.iinfo(dtype).max
        bin_edges = [val]
        while val:
            step = max(int(val * ratio), 1)
            val = val - step
            bin_edges.append(val)
        bin_edges.reverse()
        return np.array(bin_edges, dtype=dtype)

    def mode(self):
        """Compute the mode of the histogram"""
        bin_widths = np.diff(self.bin_edges)
        left_edges = self.bin_edges[:-1] - 0.5
        bin_centers = left_edges + 0.5 * bin_widths
        mode = bin_centers[np.argmax(self.counts / bin_widths)]
        return mode

    def threshold_mean(self, threshold):
        """Compute the mean of histogram values below a given threshold"""
        bin_widths = np.diff(self.bin_edges)
        left_edges = self.bin_edges[:-1] - 0.5
        bin_centers = left_edges + 0.5 * bin_widths
        ind_threshold = np.searchsorted(bin_centers, threshold)
        counts_below_threshold = self.counts[:ind_threshold]
        bin_centers_below_threshold = bin_centers[:ind_threshold]
        total_counts = np.sum(counts_below_threshold)
        if total_counts == 0:
            return 0.0
        mean = np.sum(bin_centers_below_threshold * counts_below_threshold) / total_counts
        return mean

    def estimate_peak_threshold(self, p_value=1e-6, count_rate=None):
        """Estimate threshold using Poisson distribution"""
        if count_rate is None:
            threshold = poisson.ppf(1 - p_value, self.mode() + 1)
            count_rate = self.threshold_mean(threshold)
        threshold = poisson.ppf(1 - p_value, count_rate)  # why does this return float? shouldn't it be int?
        return int(threshold)
