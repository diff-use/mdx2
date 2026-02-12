"""helper functions for the visualization report"""

import numpy as np
import xarray as xr

from mdx2.data import HKLTable


def unique_slices(symmetry):
    """
    Determine the unique slice directions according to symmetry.

    Returns a tuple of (axis, coord) pairs, where axis is one of 0, 1, 2 (h, k, l), and
    coord is the value of the coordinate for that axis (e.g. 0 for the central slice).
    """
    h, k, l = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")
    h0, k0, l0, _ = symmetry.to_asu(h.ravel(), k.ravel(), l.ravel())
    h0 = h0.reshape((3, 3, 3))
    k0 = k0.reshape((3, 3, 3))
    l0 = l0.reshape((3, 3, 3))

    unique_slices = {}
    for axis in [2, 1, 0]:
        for index, coord in enumerate([0, 1, -1]):
            h0j = h0.take(indices=index, axis=axis).ravel()
            k0j = k0.take(indices=index, axis=axis).ravel()
            l0j = l0.take(indices=index, axis=axis).ravel()
            unique_hkl0j = tuple(sorted(set((x, y, z) for x, y, z in zip(h0j, k0j, l0j))))
            # check if t0j is already in unique_slices, if not, add it:
            if unique_hkl0j not in unique_slices:
                unique_slices[unique_hkl0j] = (axis, coord)

    slice_lookup = {0: [], 1: [], 2: []}  # i.e. h,k,l
    for axis, coord in unique_slices.values():
        slice_lookup[axis].append(coord)
    return slice_lookup


def extract_central_slice(hkl_table, symmetry, crystal, slice_index, slice_coordinate=0.0, signal="intensity"):
    """Get a central slice, symmetry-expanded, as xr.DataArray."""
    if not hasattr(hkl_table, signal):
        raise ValueError(f"HKLTable does not contain a column named '{signal}'")

    UB = crystal.ub_matrix
    s = UB @ np.stack((hkl_table.h, hkl_table.k, hkl_table.l))
    s = np.sqrt(np.sum(s * s, axis=0))
    smax = s.max()

    hlim, klim, llim = _get_slice_limits(smax, UB, slice_index, slice_coordinate)

    # generate a grid of points
    ndiv = hkl_table.ndiv
    Hlim = tuple(np.round(h * ndiv[0]).astype(int) for h in hlim)
    Klim = tuple(np.round(k * ndiv[1]).astype(int) for k in klim)
    Llim = tuple(np.round(l * ndiv[2]).astype(int) for l in llim)
    h_axis = np.arange(Hlim[0], Hlim[1] + 1) / ndiv[0]
    k_axis = np.arange(Klim[0], Klim[1] + 1) / ndiv[1]
    l_axis = np.arange(Llim[0], Llim[1] + 1) / ndiv[2]
    h, k, l = np.meshgrid(h_axis, k_axis, l_axis, indexing="ij")

    hkl_slice = HKLTable(h.ravel(), k.ravel(), l.ravel(), ndiv=ndiv)
    hkl_slice = hkl_slice.to_asu(symmetry)

    slice_data = hkl_table.lookup(hkl_slice.h, hkl_slice.k, hkl_slice.l, signal).reshape(h.shape)

    slice_arr = xr.DataArray(
        data=slice_data,
        dims=["h", "k", "l"],
        coords={"h": h_axis, "k": k_axis, "l": l_axis},
    )

    return slice_arr


def hkl_to_cartesian(slice_arr, crystal, slice_index):
    """Convert a slice from fractional to Cartesian coordinates."""
    UB = _reorient_reciprocal_axes(crystal.ub_matrix, slice_index)
    h, k, l = np.meshgrid(slice_arr.coords["h"], slice_arr.coords["k"], slice_arr.coords["l"], indexing="ij")
    hkl = np.stack((h, k, l))
    sxyz = np.tensordot(UB, hkl, axes=1)
    slice_cart = xr.DataArray(
        data=slice_arr.data,
        dims=["h", "k", "l"],
        coords={
            "sx": (("h", "k", "l"), sxyz[0, ...]),
            "sy": (("h", "k", "l"), sxyz[1, ...]),
            "sz": (("h", "k", "l"), sxyz[2, ...]),
        },
    )
    # convert to 2D array
    slice_cart = slice_cart.isel(**[{"h": 0}, {"k": 0}, {"l": 0}][slice_index])
    return slice_cart


def _reorient_reciprocal_axes(UB, slice_index):
    """rotate the reciprocal cell so the other two axes lie in the x,y plane."""

    t = np.array([[1, 0, 0], [0, 1e-12, 0], [0, 0, 0]])
    # this is the correct "t" for slice_index=2, we need to cyclically permute it for the other slice indices.
    t = np.roll(t, slice_index - 2, axis=0)
    u, _, v = np.linalg.svd(UB @ t)
    R = v @ u.T
    if np.linalg.det(R) < 0:
        R[2, :] *= -1
    return R @ UB


def _get_slice_limits(smax, UB, slice_index, slice_coordinate=0.0):
    """Get the limits of a central slice in reciprocal space."""

    # NOTE: this uses limits from the central slice.
    # If slice_coordinate is not zero, the result is technically incorrect,
    # but works OK for small offsets, which is what we're doing here (e.g. +/- 0.5)

    x_index = (slice_index + 1) % 3
    y_index = (slice_index + 2) % 3

    UBinv = np.linalg.inv(UB)

    limits = [None, None, None]

    plane_normal = UBinv[slice_index, :] / np.linalg.norm(UBinv[slice_index, :])
    x_intersection_vector = np.cross(plane_normal, UB[:, y_index]) / np.linalg.norm(UB[:, y_index])
    y_intersection_vector = np.cross(plane_normal, UB[:, x_index]) / np.linalg.norm(UB[:, x_index])

    x_max = smax * np.abs(np.dot(UBinv[x_index, :], x_intersection_vector))
    y_max = smax * np.abs(np.dot(UBinv[y_index, :], y_intersection_vector))
    # print('At',axes[slice_index], '= 0, maximum value of',axes[x_index], '=',x_max, 'and', axes[y_index], '=',y_max)

    # limits = {axes[slice_index]: (0,0), axes[x_index]: (-x_max, x_max), axes[y_index]: (-y_max, y_max)}
    limits[slice_index] = (slice_coordinate, slice_coordinate)
    limits[x_index] = (-x_max, x_max)
    limits[y_index] = (-y_max, y_max)
    return limits
