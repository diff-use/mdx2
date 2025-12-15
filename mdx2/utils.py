import warnings

import numpy as np
from numba import float64, jit


# DEPRECATED: Import I/O functions for backward compatibility
# These will be removed in a future version
def _deprecated_import(name, new_module="mdx2.io"):
    """Issue a deprecation warning when importing from utils instead of io"""
    warnings.warn(
        f"Importing '{name}' from mdx2.utils is deprecated and will be removed in a future version. "
        f"Please import from {new_module} instead: 'from {new_module} import {name}'",
        DeprecationWarning,
        stacklevel=3,
    )


def __getattr__(name):
    """Lazy import with deprecation warning for I/O functions"""
    io_functions = ["nxload", "nxsave", "loadobj", "saveobj"]
    if name in io_functions:
        _deprecated_import(name)
        from mdx2 import io  # noqa: PLC0415

        return getattr(io, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Explicitly list what's available from this module for backwards compatibility
__all__ = [
    # Numerical functions (current)
    "histogram",
    "interp_g2g_bilinear",
    "interp_g2g_trilinear",
    "interp3",
    "interp2",
    "slice_sections",
    # I/O functions (deprecated, imported from mdx2.io)
    "nxload",  # noqa: F822
    "nxsave",  # noqa: F822
    "loadobj",  # noqa: F822
    "saveobj",  # noqa: F822
]


def slice_sections(Ntotal, Nsections):
    """Slices that divide an array of length Ntotal into Nsections. See np.split_array"""
    Neach_section, extras = divmod(Ntotal, Nsections)
    section_sizes = [0] + extras * [Neach_section + 1] + (Nsections - extras) * [Neach_section]
    div_points = np.array(section_sizes, dtype=int).cumsum()
    slices = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        slices.append(slice(st, end))
    return tuple(slices)


# FUNCTIONS FOR EFFICIENT LINEAR INTERPOLATION


@jit
def histogram(data, bin_edges):
    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    for value in data:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= value < bin_edges[i + 1]:
                counts[i] += 1
                break
        else:
            if value == bin_edges[-1]:
                counts[-1] += 1
    return counts


def interp3(x0, y0, z0, v0, x, y, z, order=1):
    """interpolate from data on a 3D grid to a set of points"""
    if order != 1:
        raise ValueError("Only order=1 (linear interpolation) is supported with numba")
    # Cast all inputs to float64
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    z0 = np.asarray(z0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    return _interp3(x0, y0, z0, v0, x, y, z)


def interp2(x0, y0, v0, x, y, order=1):
    """interpolate from data on a 2D grid to a set of points"""
    if order != 1:
        raise ValueError("Only order=1 (linear interpolation) is supported with numba")
    # Cast all inputs to float64
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return _interp2(x0, y0, v0, x, y)


def interp_g2g_trilinear(x0, y0, z0, v0, x, y, z):
    """
    Trilinear interpolation from one 3D grid to another.

    All inputs are automatically cast to float64 arrays.

    Parameters
    ----------
    x0, y0, z0 : array_like
        Source grid axes
    v0 : array_like
        Source data (3D or 4D)
    x, y, z : array_like
        Target grid axes

    Returns
    -------
    result : ndarray
        Interpolated data on target grid
    """
    # Cast all inputs to float64
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    z0 = np.asarray(z0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if v0.ndim == 3:
        return _trilinear_g2g_3d(x0, y0, z0, v0, x, y, z)
    elif v0.ndim == 4:
        return _trilinear_g2g_4d(x0, y0, z0, v0, x, y, z)
    else:
        raise ValueError(f"v0 must be 3D or 4D, got shape {v0.shape}")


def interp_g2g_bilinear(x0, y0, v0, x, y):
    """
    Bilinear interpolation from one 2D grid to another.

    All inputs are automatically cast to float64 arrays.

    Parameters
    ----------
    x0, y0 : array_like
        Source grid axes
    v0 : array_like
        Source data (2D or 3D)
    x, y : array_like
        Target grid axes

    Returns
    -------
    result : ndarray
        Interpolated data on target grid
    """
    # Cast all inputs to float64
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if v0.ndim == 2:
        return _bilinear_g2g_2d(x0, y0, v0, x, y)
    elif v0.ndim == 3:
        return _bilinear_g2g_3d(x0, y0, v0, x, y)
    else:
        raise ValueError(f"v0 must be 2D or 3D, got shape {v0.shape}")


@jit
def _base_fraction(x, axis=None):
    """
    Map points x onto axis indices using binary search with extrapolation.
    Points outside the axis range will have negative or > (axis.size-1) values.
    """
    n = x.size
    fx = np.empty(n, dtype=np.float64)
    bx = np.empty(n, dtype=np.int32)

    for i in range(n):
        val = x[i]

        # Handle extrapolation below lower bound
        if val <= axis[0]:
            left = 0
            right = 1
        # Handle extrapolation above upper bound
        elif val >= axis[-1]:
            left = axis.size - 2
            right = axis.size - 1
        else:
            # Binary search for the correct interval (within bounds)
            left = 0
            right = axis.size - 1

            while right - left > 1:
                mid = (left + right) // 2
                if axis[mid] <= val:
                    left = mid
                else:
                    right = mid

        # Linear interpolation (or extrapolation) within the interval
        dx = (val - axis[left]) / (axis[right] - axis[left])
        fx[i] = dx
        bx[i] = left

    return bx, fx


@jit
def _fraction(x, axis=None):
    b, f = _base_fraction(x, axis=axis)
    return b + f


@jit
def _map_coordinates_linear_2d(v0, fx, fy):
    """
    Numba-compatible implementation of map_coordinates for 2D arrays with linear interpolation.

    Parameters:
    -----------
    v0 : 2D array
        Input data
    fx, fy : 1D arrays
        Fractional indices for each dimension

    Returns:
    --------
    1D array of interpolated values
    """
    n = fx.size
    result = np.empty(n)

    for i in range(n):
        # Get integer and fractional parts
        ix = int(np.floor(fx[i]))
        iy = int(np.floor(fy[i]))

        # Clip to valid range
        ix = max(0, min(ix, v0.shape[0] - 2))
        iy = max(0, min(iy, v0.shape[1] - 2))

        dx = fx[i] - ix
        dy = fy[i] - iy

        # Bilinear interpolation
        c0 = v0[ix, iy] * (1 - dx) + v0[ix + 1, iy] * dx
        c1 = v0[ix, iy + 1] * (1 - dx) + v0[ix + 1, iy + 1] * dx

        result[i] = c0 * (1 - dy) + c1 * dy

    return result


@jit
def _map_coordinates_linear_3d(v0, fx, fy, fz):
    """
    Numba-compatible implementation of map_coordinates for 3D arrays with linear interpolation.

    Parameters:
    -----------
    v0 : 3D array
        Input data
    fx, fy, fz : 1D arrays
        Fractional indices for each dimension

    Returns:
    --------
    1D array of interpolated values
    """
    n = fx.size
    result = np.empty(n)

    for i in range(n):
        # Get integer and fractional parts
        ix = int(np.floor(fx[i]))
        iy = int(np.floor(fy[i]))
        iz = int(np.floor(fz[i]))

        # Clip to valid range
        ix = max(0, min(ix, v0.shape[0] - 2))
        iy = max(0, min(iy, v0.shape[1] - 2))
        iz = max(0, min(iz, v0.shape[2] - 2))

        dx = fx[i] - ix
        dy = fy[i] - iy
        dz = fz[i] - iz

        # Trilinear interpolation
        c00 = v0[ix, iy, iz] * (1 - dx) + v0[ix + 1, iy, iz] * dx
        c01 = v0[ix, iy, iz + 1] * (1 - dx) + v0[ix + 1, iy, iz + 1] * dx
        c10 = v0[ix, iy + 1, iz] * (1 - dx) + v0[ix + 1, iy + 1, iz] * dx
        c11 = v0[ix, iy + 1, iz + 1] * (1 - dx) + v0[ix + 1, iy + 1, iz + 1] * dx

        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy

        result[i] = c0 * (1 - dz) + c1 * dz

    return result


@jit(
    float64[:](float64[:], float64[:], float64[:], float64[:, :, :], float64[:], float64[:], float64[:]),
    nopython=True,
    cache=True,
)
def _interp3(x0, y0, z0, v0, x, y, z):
    """interpolate from data on a 3D grid to a set of points"""
    fx = _fraction(x, axis=x0)
    fy = _fraction(y, axis=y0)
    fz = _fraction(z, axis=z0)
    return _map_coordinates_linear_3d(v0, fx, fy, fz)


@jit(
    float64[:](float64[:], float64[:], float64[:, :], float64[:], float64[:]),
    nopython=True,
    cache=True,
)
def _interp2(x0, y0, v0, x, y):
    """interpolate from data on a 2D grid to a set of points"""
    fx = _fraction(x, axis=x0)
    fy = _fraction(y, axis=y0)
    return _map_coordinates_linear_2d(v0, fx, fy)


@jit(
    float64[:, :](float64[:], float64[:], float64[:, :], float64[:], float64[:]),
    nopython=True,
    cache=True,
)
def _bilinear_g2g_2d(x0, y0, d0, x, y):
    bx, fx = _base_fraction(x, axis=x0)
    by, fy = _base_fraction(y, axis=y0)

    # Clip the data
    d0 = d0[bx[0] : (bx[-1] + 2), by[0] : (by[-1] + 2)]
    bx = bx - bx[0]
    by = by - by[0]

    dx = fx[:, None]
    dy = fy[None, :]

    c0 = d0[bx, :-1] * (1 - dx) + d0[bx + 1, :-1] * dx
    c1 = d0[bx, 1:] * (1 - dx) + d0[bx + 1, 1:] * dx
    c = c0[:, by] * (1 - dy) + c1[:, by] * dy

    return c


@jit(
    float64[:, :, :](float64[:], float64[:], float64[:, :, :], float64[:], float64[:]),
    nopython=True,
    cache=True,
)
def _bilinear_g2g_3d(x0, y0, d0, x, y):
    bx, fx = _base_fraction(x, axis=x0)
    by, fy = _base_fraction(y, axis=y0)

    # Clip the data
    d0 = d0[bx[0] : (bx[-1] + 2), by[0] : (by[-1] + 2), :]
    bx = bx - bx[0]
    by = by - by[0]

    dx = fx[:, None, None]
    dy = fy[None, :, None]

    c0 = d0[bx, :-1, :] * (1 - dx) + d0[bx + 1, :-1, :] * dx
    c1 = d0[bx, 1:, :] * (1 - dx) + d0[bx + 1, 1:, :] * dx
    c = c0[:, by, :] * (1 - dy) + c1[:, by, :] * dy

    return c


@jit(
    float64[:, :, :](float64[:], float64[:], float64[:], float64[:, :, :], float64[:], float64[:], float64[:]),
    nopython=True,
    cache=True,
)
def _trilinear_g2g_3d(x0, y0, z0, d0, x, y, z):
    bx, fx = _base_fraction(x, axis=x0)
    by, fy = _base_fraction(y, axis=y0)
    bz, fz = _base_fraction(z, axis=z0)

    # Clip the data
    d0 = d0[bx[0] : (bx[-1] + 2), by[0] : (by[-1] + 2), bz[0] : (bz[-1] + 2)]
    bx = bx - bx[0]
    by = by - by[0]
    bz = bz - bz[0]

    dx = fx[:, None, None]
    dy = fy[None, :, None]
    dz = fz[None, None, :]

    c00 = d0[bx, :-1, :-1] * (1 - dx) + d0[bx + 1, :-1, :-1] * dx
    c01 = d0[bx, :-1, 1:] * (1 - dx) + d0[bx + 1, :-1, 1:] * dx
    c10 = d0[bx, 1:, :-1] * (1 - dx) + d0[bx + 1, 1:, :-1] * dx
    c11 = d0[bx, 1:, 1:] * (1 - dx) + d0[bx + 1, 1:, 1:] * dx
    c0 = c00[:, by, :] * (1 - dy) + c10[:, by, :] * dy
    c1 = c01[:, by, :] * (1 - dy) + c11[:, by, :] * dy
    c = c0[:, :, bz] * (1 - dz) + c1[:, :, bz] * dz

    return c


@jit(
    float64[:, :, :, :](float64[:], float64[:], float64[:], float64[:, :, :, :], float64[:], float64[:], float64[:]),
    nopython=True,
    cache=True,
)
def _trilinear_g2g_4d(x0, y0, z0, d0, x, y, z):
    bx, fx = _base_fraction(x, axis=x0)
    by, fy = _base_fraction(y, axis=y0)
    bz, fz = _base_fraction(z, axis=z0)

    # Clip the data
    d0 = d0[bx[0] : (bx[-1] + 2), by[0] : (by[-1] + 2), bz[0] : (bz[-1] + 2), :]
    bx = bx - bx[0]
    by = by - by[0]
    bz = bz - bz[0]

    dx = fx[:, None, None, None]
    dy = fy[None, :, None, None]
    dz = fz[None, None, :, None]

    c00 = d0[bx, :-1, :-1, :] * (1 - dx) + d0[bx + 1, :-1, :-1, :] * dx
    c01 = d0[bx, :-1, 1:, :] * (1 - dx) + d0[bx + 1, :-1, 1:, :] * dx
    c10 = d0[bx, 1:, :-1, :] * (1 - dx) + d0[bx + 1, 1:, :-1, :] * dx
    c11 = d0[bx, 1:, 1:, :] * (1 - dx) + d0[bx + 1, 1:, 1:, :] * dx
    c0 = c00[:, by, :, :] * (1 - dy) + c10[:, by, :, :] * dy
    c1 = c01[:, by, :, :] * (1 - dy) + c11[:, by, :, :] * dy
    c = c0[:, :, bz, :] * (1 - dz) + c1[:, :, bz, :] * dz

    return c
