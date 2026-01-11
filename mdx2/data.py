import abc
import os
import warnings
from copy import deepcopy

import h5py
import hdf5plugin
import numpy as np
import pandas as pd
from nexusformat.nexus import NXdata, NXfield, NXgroup, NXreflections, NXvirtualfield

from mdx2.dxtbx_machinery import Experiment
from mdx2.io import nxload, saveobj


class Peaks:
    """search peaks in image stack"""

    def __init__(self, phi, iy, ix):
        self.phi = phi
        self.iy = iy
        self.ix = ix

    def to_mask(self, phi_axis, iy_axis, ix_axis, mask_in=None):
        shape = (phi_axis.size, iy_axis.size, ix_axis.size)

        ind0 = np.round(np.interp(self.phi, phi_axis, np.arange(shape[0]))).astype(int)
        ind1 = np.round(np.interp(self.iy, iy_axis, np.arange(shape[1]))).astype(int)
        ind2 = np.round(np.interp(self.ix, ix_axis, np.arange(shape[2]))).astype(int)
        if mask_in is not None:
            mask = mask_in
        else:
            mask = np.zeros(shape, dtype="bool")
        mask[ind0, ind1, ind2] = True
        return mask

    @staticmethod
    def where(mask, phi_axis, iy_axis, ix_axis):
        hotpixels = np.argwhere(mask)
        return Peaks(
            phi_axis[hotpixels[:, 0]],
            iy_axis[hotpixels[:, 1]],
            ix_axis[hotpixels[:, 2]],
        )

    @staticmethod
    def stack(peaks):
        phi = np.concatenate([p.phi for p in peaks])
        ix = np.concatenate([p.ix for p in peaks])
        iy = np.concatenate([p.iy for p in peaks])
        return Peaks(phi, iy, ix)

    @property
    def size(self):
        return self.phi.size

    def to_nexus(self):
        return NXreflections(
            name="peaks",
            observed_phi=self.phi,
            observed_px_x=self.ix,
            observed_px_y=self.iy,
        )

    @staticmethod
    def from_nexus(peaks):
        return Peaks(
            peaks.observed_phi.nxvalue,
            peaks.observed_px_y.nxvalue,
            peaks.observed_px_x.nxvalue,
        )


class HKLTable:
    """Container for data in a table with indices h,k,l"""

    def __init__(self, h, k, l, ndiv=(1, 1, 1), **kwargs):
        self.h = h
        self.k = k
        self.l = l
        self.ndiv = tuple(ndiv)
        self.__dict__.update(kwargs)

    @property
    def _data_keys(self):
        return [key for key in self.__dict__ if key not in ["h", "k", "l", "ndiv"]]

    def __len__(self):
        return len(self.h)

    def __getitem__(self, key):
        """Get item by key or slice"""
        if isinstance(key, str):
            return self.__dict__[key]
        elif isinstance(key, slice):
            return HKLTable(
                self.h[key],
                self.k[key],
                self.l[key],
                ndiv=self.ndiv,
                **{k: v[key] for k, v in self.__dict__.items() if k not in ["h", "k", "l", "ndiv"]},
            )
        else:
            raise TypeError("Key must be a string or a slice")
        # TODO: allow fancy indexing if all columns are numpy arrays

    def to_frame(self):
        """convert to pandas dataframe"""
        cols = {k: self.__dict__[k] for k in self.__dict__ if k not in ["ndiv"]}
        return pd.DataFrame(cols)

    def to_asu(self, Symmetry=None):
        """Map to asymmetric unit. If Symmetry is ommitted, P1 space group is assumed (op = 0 for l>=0, op=1 for l<0)"""
        ndiv = self.ndiv
        H = np.round(self.h * ndiv[0]).astype(int)
        K = np.round(self.k * ndiv[1]).astype(int)
        L = np.round(self.l * ndiv[2]).astype(int)

        if Symmetry is None:
            op = (L < 0).astype(int)
            L = np.abs(L)
        else:
            H, K, L, op = Symmetry.to_asu(H, K, L)

        newtable = deepcopy(self)
        newtable.h = H / ndiv[0]
        newtable.k = K / ndiv[1]
        newtable.l = L / ndiv[2]
        newtable.op = op

        return newtable

    # Capital properties H, K, L, and Origin are integer only indexes
    @property
    def H(self):
        return np.round(self.h * self.ndiv[0]).astype(int)

    @property
    def K(self):
        return np.round(self.k * self.ndiv[1]).astype(int)

    @property
    def L(self):
        return np.round(self.l * self.ndiv[2]).astype(int)

    def to_array_index(self, ori=None):
        H, K, L = self.H, self.K, self.L
        if ori is None:
            Ori = (np.min(H), np.min(K), np.min(L))
        else:
            Ori = tuple(np.array(self.ndiv) * np.array(ori))
        return (H - Ori[0], K - Ori[1], L - Ori[2]), Ori

    def from_array_index(self, index, shape, Ori=[0, 0, 0]):
        hklind = np.unravel_index(index, shape)
        hkl = [(ind + o) / n for ind, o, n in zip(hklind, Ori, self.ndiv)]
        return tuple(hkl)

    def unique(self):
        ind, Origin = self.to_array_index()
        sz = [np.max(j) + 1 for j in ind]
        unique_index, index_map, counts = np.unique(
            np.ravel_multi_index(ind, sz),
            return_inverse=True,
            return_counts=True,
        )
        hkl = self.from_array_index(unique_index, sz, Ori=Origin)
        return hkl, index_map, counts

    def bin(self, column_names=None, count_name="count"):
        if column_names is None:
            column_names = self._data_keys

        # catch the case where the HKLTable is empty, and return an empty table with count_name field
        # (used by integrate)
        if len(self) == 0:
            outcols = {k: [] for k in column_names}
            if count_name is not None:
                outcols[count_name] = np.array([], dtype=np.int64)  # np.unique returns counts as int64 dtype
            return HKLTable([], [], [], ndiv=self.ndiv, **outcols)

        # print('finding unique indices')
        (h, k, l), index_map, counts = self.unique()

        outcols = {}
        if count_name is not None:
            # print(f'storing bin counts in column: {count_name}')
            outcols[count_name] = counts
        for key in column_names:
            # print(f'binning data column: {key}')
            outcols[key] = np.bincount(index_map, weights=self.__dict__[key])
        return HKLTable(h, k, l, ndiv=self.ndiv, **outcols)

    @staticmethod
    def concatenate(tabs):
        if not tabs:
            raise ValueError("Cannot concatenate empty list of HKLTable objects")

        ndiv = tabs[0].ndiv  # assume the first one is canonical

        # Validate that all tables have matching ndiv values
        for j in range(1, len(tabs)):
            if tabs[j].ndiv != ndiv:
                raise ValueError(
                    f"Cannot concatenate HKLTable objects with mismatched ndiv values: "
                    f"tabs[0].ndiv={ndiv}, tabs[{j}].ndiv={tabs[j].ndiv}"
                )

        # Find data keys common to all tables
        data_keys = set(tabs[0]._data_keys)
        for j in range(1, len(tabs)):
            data_keys &= set(tabs[j]._data_keys)

        def concat(key):
            return np.concatenate([tab.__dict__[key] for tab in tabs])

        h = concat("h")
        k = concat("k")
        l = concat("l")
        cols = {key: concat(key) for key in data_keys}
        return HKLTable(h, k, l, ndiv=ndiv, **cols)

    @staticmethod
    def from_frame(df):
        """create from pandas dataframe with h,k,l as cols or indices"""
        df = df.reset_index()  # move indices into columns
        h = df.pop("h").to_numpy()
        k = df.pop("k").to_numpy()
        l = df.pop("l").to_numpy()
        data = {key: df[key].to_numpy() for key in df.keys()}
        return HKLTable(h, k, l, **data)

    def to_nexus(self):
        return NXgroup(name="hkl_table", **self.__dict__)

    @staticmethod
    def from_nexus(nxgroup):
        h = nxgroup.h.nxdata
        k = nxgroup.k.nxdata
        l = nxgroup.l.nxdata
        data_keys = [key for key in nxgroup.keys() if key not in ["h", "k", "l"]]
        data = {key: nxgroup[key].nxdata for key in data_keys}
        return HKLTable(h, k, l, **data)

    def lookup(self, h, k, l, column_name, fill_value=np.nan):
        """Lookup values from table using h,k,l indices

        Parameters
        ----------
        h, k, l : array_like
            Miller indices (floats) to lookup
        column_name : str
            Name of column to return values from
        fill_value : scalar, optional
            Value to use for missing indices (default: np.nan)

        Returns
        -------
        values : ndarray
            Array of values with same shape as h,k,l inputs
        """
        # Convert inputs to integer indices
        H_query = np.round(np.asarray(h) * self.ndiv[0]).astype(np.int32)
        K_query = np.round(np.asarray(k) * self.ndiv[1]).astype(np.int32)
        L_query = np.round(np.asarray(l) * self.ndiv[2]).astype(np.int32)

        # Get integer indices from table
        H_table = self.H
        K_table = self.K
        L_table = self.L

        # Create structured arrays for lexicographic sorting
        dt = np.dtype([("h", np.int32), ("k", np.int32), ("l", np.int32)])

        table_structured = np.empty(len(self), dtype=dt)
        table_structured["h"] = H_table
        table_structured["k"] = K_table
        table_structured["l"] = L_table

        query_structured = np.empty(H_query.size, dtype=dt)
        query_structured["h"] = H_query.ravel()
        query_structured["k"] = K_query.ravel()
        query_structured["l"] = L_query.ravel()

        # Sort table
        sort_idx = np.argsort(table_structured)
        table_sorted = table_structured[sort_idx]
        values_sorted = self[column_name][sort_idx]

        # Find matches using searchsorted
        indices = np.searchsorted(table_sorted, query_structured)

        # Handle out of bounds
        indices = np.clip(indices, 0, len(table_sorted) - 1)

        # Check for exact matches
        matches = table_sorted[indices] == query_structured
        result = np.where(matches, values_sorted[indices], fill_value)

        # Reshape to original shape
        return result.reshape(H_query.shape)


class HKLGrid:
    """Container for data on a grid indexed by h,k,l

    Provides a similar interface to mdx2.geometry.GridData, but is specialized for conversion to/from HKLTable
    """

    axes_names = ("h", "k", "l")

    def __init__(
        self, data: dict, ndiv: tuple[int, int, int] = (1, 1, 1), ori: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        if len(ndiv) != 3:
            raise ValueError("ndiv must be a tuple of length 3")
        if len(ori) != 3:
            raise ValueError("ori must be a tuple of length 3")
        if not all(isinstance(o, (int, float)) for o in ori):
            raise ValueError("ori must be a tuple of numbers")
        if not all(isinstance(n, int) and n > 0 for n in ndiv):
            raise ValueError("ndiv must be a tuple of positive integers")
        if not all(isinstance(d, np.ndarray) for d in data.values()):
            raise ValueError("data must be a dictionary of numpy arrays")
        if not all(d.ndim == 3 for d in data.values()):
            raise ValueError("data arrays must be 3-dimensional")
        shapes = [d.shape for d in data.values()]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError("data arrays must have the same shape")
        self.ndiv = ndiv
        self.ori = ori
        self._data = data
        # TODO: there is a potential issue of ori*ndiv not being integer, which could lead to unexpected behavior

    @property
    def axes(self):
        return tuple(np.arange(n) / ndiv + o for n, ndiv, o in zip(self.shape, self.ndiv, self.ori))

    @property
    def ncols(self):
        return len(self._data)

    @property
    def shape(self):
        sh = (0, 0, 0)
        for d in self._data.values():
            sh = tuple(max(s, ds) for s, ds in zip(sh, d.shape))
        return sh

    def __getitem__(self, key):
        """Get item by key or slice

        usage:
        - grid["h"] to get the "h" data
        - grid[:50,:50,:50] to get a sub-grid
        """
        if isinstance(key, str):
            return self._data[key]
        elif isinstance(key, tuple):  # tuple of slices
            new_ori = tuple(ax[sl][0] for ax, sl in zip(self.axes, key))
            return HKLGrid(
                {k: v[key] for k, v in self._data.items()},
                ndiv=self.ndiv,
                ori=new_ori,
            )
        else:
            raise TypeError("Key must be a string or a tuple of slices")

    def _coords_to_indices(self, *hkl):
        """convert hkl coordinates to array indices

        ix, iy, iz = self._coords_to_indices(h, k, l)
        """
        indices = tuple(np.round((np.array(h) - o) * ndiv).astype(int) for h, o, ndiv in zip(hkl, self.ori, self.ndiv))
        return indices

    def _bounds_from_coords(self, *coords):
        bounds = tuple(np.array([np.min(j), np.max(j)]) for j in coords)
        return bounds

    def _padding_from_bounds(self, *bounds):
        """Amount to pad the data in each direction in order to contain a new set of coordinates

        padding = self._padding_from_bounds((hmin, hmax), (kmin, kmax), (lmin, lmax))

        returns None if bounds are contained within current array
        """
        indices = self._coords_to_indices(*bounds)
        padding = tuple((max(-ind[0], 0), max(ind[1] - n + 1, 0)) for n, ind in zip(self.shape, indices))
        if not np.any(padding):
            padding = None
        return padding

    def _is_in_bounds(self, *coords):
        """returns a boolean array that is true for all coordinates within the bounds of the array"""
        indices = self._coords_to_indices(*coords)
        cond = []
        for ind, sh in zip(indices, self.shape):
            cond.append(ind >= 0)
            cond.append(ind < sh)
        isincl = np.all(np.column_stack(cond), axis=1)
        return isincl

    def accumulate(self, *coords, resize=False, **values):
        """Accumulate values on one or more data grids

        Values must be passed as keyword arguments mapping to the keys in data, e.g.:
            accumulate(h, k, l, intensity=arr)
        All values must be arrays of the same shape as the coordinate arrays.

        There are two modes:
        - resize=True, arrays will be resized if necessary to accomodate coordinates outside of current bounds
        - resize=False, values will be silently excluded if they fall outside the current array bounds
        """
        for k in values.keys():
            if k not in self._data:
                raise KeyError(f"key {k} not found in data")
        bounds = self._bounds_from_coords(*coords)
        padding = self._padding_from_bounds(*bounds)
        if padding is not None:
            if resize:
                self.ori = tuple(o - p[0] / ndiv for o, ndiv, p in zip(self.ori, self.ndiv, padding))
                self._data = {k: np.pad(v, pad_width=padding) for k, v in self._data.items()}
            else:
                isincl = self._is_in_bounds(*coords)
                coords = tuple(np.array(c)[isincl] for c in coords)
                values = {k: np.array(v)[isincl] for k, v in values.items()}
        indices = self._coords_to_indices(*coords)
        for k, v in values.items():
            np.add.at(self._data[k], indices, v)
        return self

    def to_table(self, sparse=False):
        """Convert to HKLTable object"""
        hkl = np.meshgrid(*self.axes, indexing="ij")
        hkl = [j.ravel() for j in hkl]
        data_columns = {k: v.ravel() for k, v in self._data.items()}
        if sparse:
            # remove rows where all data values are zero
            is_zero = np.all(np.column_stack([v == 0 for v in data_columns.values()]), axis=1)
            hkl = [j[~is_zero] for j in hkl]
            data_columns = {k: v[~is_zero] for k, v in data_columns.items()}
        return HKLTable(*hkl, **data_columns, ndiv=self.ndiv)

    def accumulate_from_table(self, tab, resize=False):
        # check that ndiv is the same between table and grid:
        if tab.ndiv != self.ndiv:
            raise ValueError("ndiv values do not match")
        self.accumulate(tab.h, tab.k, tab.l, **{k: tab[k] for k in tab._data_keys}, resize=resize)

    @staticmethod
    def from_table(tab):
        # initialize arrays
        data = {k: np.ndarray(shape=[0, 0, 0], dtype=tab[k].dtype) for k in tab._data_keys}
        g = HKLGrid(data, ndiv=tab.ndiv, ori=(tab.h.min(), tab.k.min(), tab.l.min()))
        g.accumulate_from_table(tab, resize=True)
        return g

    def to_nexus(self):
        # TODO: implement this method
        raise NotImplementedError("to_nexus is not implemented")

    @staticmethod
    def from_nexus(nexus_file):
        # TODO: implement this method
        raise NotImplementedError("from_nexus is not implemented")

    def __repr__(self):
        return f"HKLGrid with shape:{self.shape}, ori:{self.ori}, ndiv:{self.ndiv}, keys:{tuple(self._data.keys())}"


class _ImageSeriesBase(abc.ABC):
    """base class for all single-sweep data containers"""

    def __init__(self, phi, iy, ix):
        self.phi = phi
        self.iy = iy
        self.ix = ix

    @property
    def shape(self):
        return (self.phi.size, self.iy.size, self.ix.size)

    @classmethod
    @abc.abstractmethod
    def from_nexus(cls, nxdata):
        pass

    @abc.abstractmethod
    def to_nexus(self):
        pass

    @abc.abstractmethod
    def index(self, miller_index):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, filename, name="image_series", mode="r", strict=False):
        pass


class SparseImageSeries(_ImageSeriesBase):
    """Image data from a single sweep in sparse representation"""

    def __init__(self, phi, iy, ix, indices, values, exposure_times):
        super().__init__(phi, iy, ix)
        self.indices = indices  # should I check that indices are within range? and not repeated?
        self.values = values
        self.exposure_times = exposure_times

    @classmethod
    def from_nexus(cls, nxdata):
        args = [nxdata[k].nxvalue for k in ["phi", "iy", "ix", "indices", "values", "exposure_times"]]
        return cls(*args)

    def to_nexus(self):
        nxobj = NXgroup(name="image_series", **self.__dict__)
        nxobj.attrs["axes"] = ["phi", "iy", "ix"]  # to match ImageSeries
        return nxobj

    def index(self, miller_index):
        phi = self.phi[self.indices[:, 0]]
        iy = self.iy[self.indices[:, 1]]
        ix = self.iy[self.indices[:, 2]]
        exposure_times = self.exposure_times[self.indices[:, 0]]
        h, k, l = miller_index.interpolate(phi, iy, ix)

        return HKLTable(h, k, l, phi=phi, iy=iy, ix=ix, counts=self.values, seconds=exposure_times)

    @classmethod
    def load(cls, filename, name="image_series", mode="r"):
        """Load from nexus file"""
        nxroot = nxload(filename, mode=mode)
        nxs = nxroot["/entry/" + name]
        module_name = nxs.attrs["mdx2_module"]
        class_name = nxs.attrs["mdx2_class"]
        if module_name != "mdx2.data" or class_name != cls.__name__:
            raise ValueError(f"object {name} in file {filename} is not an mdx2.data.{cls.__name__}")
        return cls.from_nexus(nxs)


class DenseImageSeries(_ImageSeriesBase):
    def __init__(self, phi, iy, ix, data, exposure_times, maskval=-1):
        super().__init__(phi, iy, ix)
        self.data = data  # can be NXfield or numpy array, doesn't matter
        self.exposure_times = exposure_times
        self._maskval = maskval

    def index(self, miller_index, mask=None):
        mi = miller_index.regrid(self.phi, self.iy, self.ix)
        phi, iy, ix = np.meshgrid(self.phi, self.iy, self.ix, indexing="ij")
        # HACK to get this the right shape...
        exposure_times = np.tile(self.exposure_times, (self.iy.size, self.ix.size, 1))
        exposure_times = np.moveaxis(exposure_times, 2, 0)
        msk = self.data != self._maskval
        if mask is not None:
            # first, check if mask is an ndarray:
            if not isinstance(mask, np.ndarray) and hasattr(mask, "evaluate"):  # HACK: duck typing for dynamic masks
                mask = mask.evaluate(h=mi.h, k=mi.k, l=mi.l)
            msk = msk & ~mask  # mask is true when pixels are excluded
        return HKLTable(
            mi.h[msk],
            mi.k[msk],
            mi.l[msk],
            phi=phi[msk],
            iy=iy[msk],
            ix=ix[msk],
            counts=self.data[msk],
            seconds=exposure_times[msk],
        )

    @classmethod
    def load(cls, filename, name="image_series", mode="r", strict=False):
        """Load from nexus file

        If strict is True, will raise an error if the mdx2_class is not derived from DenseImageSeries.
        Otherwise, the class must match exactly.
        """
        nxroot = nxload(filename, mode=mode)
        nxs = nxroot["/entry/" + name]
        module_name = nxs.attrs["mdx2_module"]
        class_name = nxs.attrs["mdx2_class"]
        if module_name != "mdx2.data":
            raise ValueError(f"object {name} in file {filename} is not an mdx2.data class")
        if strict:
            if class_name != cls.__name__:
                raise ValueError(f"object {name} in file {filename} is not an mdx2.data.{cls.__name__}")
        else:
            # NOTE: there is some potential for bugs here.
            # - If LazyImageSeries.load is called on a VirtualImageSeries object, the chunks attribute will not work.
            # - [...] < - if more are found, insert here.
            #
            # The best solution (?) would be to implement a load method for each subclass that handles special cases.
            cls_obj = globals().get(class_name, None)
            if cls_obj is None or not issubclass(cls_obj, DenseImageSeries):
                raise ValueError(f"object {name} in file {filename} is not a subclass of DenseImageSeries")
        return cls.from_nexus(nxs)

    @classmethod
    def from_nexus(cls, nxdata, read_data=False):
        phi = nxdata.phi.nxvalue
        iy = nxdata.iy.nxvalue
        ix = nxdata.ix.nxvalue
        exposure_times = nxdata.exposure_times.nxvalue
        data = nxdata[nxdata.attrs["signal"]]  # read the data as a numpy array
        maskval = nxdata.attrs.get("maskval", None)  # earlier versions did not save the maskval
        if read_data:
            data = data.nxvalue
        return cls(phi, iy, ix, data, exposure_times, maskval=maskval)

    def to_nexus(self):
        phi = NXfield(self.phi, name="phi")
        ix = NXfield(self.ix, name="ix")
        iy = NXfield(self.iy, name="iy")
        if isinstance(self.data, NXfield):
            signal = self.data
            signal.name = "data"
        else:
            signal = NXfield(self.data, name="data")
        nxobj = NXdata(signal=signal, axes=[phi, iy, ix], exposure_times=self.exposure_times)
        nxobj.attrs["maskval"] = self._maskval
        return nxobj


class InMemoryImageSeries(DenseImageSeries):
    """Image series that has been loaded into memory (all data are numpy arrays, not nexus NXFields)"""

    def __init__(self, phi, iy, ix, data, exposure_times, maskval=-1):
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array for InMemoryImageSeries")
        super().__init__(phi, iy, ix, data, exposure_times, maskval=maskval)

    def __getitem__(self, sl):
        return InMemoryImageSeries(
            self.phi[sl[0]],
            self.iy[sl[1]],
            self.ix[sl[2]],
            self.data[sl],
            self.exposure_times[sl[0]],
            maskval=self._maskval,
        )

    @property
    def data_masked(self):
        return np.ma.masked_equal(self.data, self._maskval, copy=False)

    @classmethod
    def from_nexus(cls, nxdata):
        super().from_nexus(nxdata, read_data=True)


class LazyImageSeries(DenseImageSeries):
    """Image series backed by a Nexus file with lazy loading and chunked compression"""

    def __init__(self, phi, iy, ix, data, exposure_times, maskval=-1):
        if not isinstance(data, NXfield):
            raise TypeError("data must be a NXfield for LazyImageSeries")
        super().__init__(phi, iy, ix, data, exposure_times, maskval=maskval)

    def __getitem__(self, sl):
        """slicing triggers a read from the NXfield data, returning an InMemoryImageSeries"""
        return InMemoryImageSeries(
            self.phi[sl[0]],
            self.iy[sl[1]],
            self.ix[sl[2]],
            self.data[sl].nxdata,  # force a read and set to numpy array
            self.exposure_times[sl[0]],
            maskval=self._maskval,
        )

    @property
    def chunks(self):
        ch = self.data.chunks
        if ch is not None:
            return ch
        else:
            return (1, self.shape[1], self.shape[2])

    def iter_chunks(self):
        for sl in self.chunk_slice_iterator():
            yield self[sl]

    def chunk_slice_along_axis(self, axis=0):
        c = self.chunks[axis]
        n = self.shape[axis]
        start = range(0, n, c)
        stop = [min(st + c, n) for st in start]
        return [slice(st, sp) for (st, sp) in zip(start, stop)]

    def chunk_slice_iterator(self):
        s1 = self.chunk_slice_along_axis(axis=0)
        s2 = self.chunk_slice_along_axis(axis=1)
        s3 = self.chunk_slice_along_axis(axis=2)
        for a1 in s1:
            for a2 in s2:
                for a3 in s3:
                    yield (a1, a2, a3)

    @staticmethod
    def from_expt(exptfile):
        data_opts = {"dtype": np.int32, "compression": hdf5plugin.LZ4(), "shuffle": True}
        expt = Experiment.from_file(exptfile)
        phi, iy, ix = expt.scan_axes
        shape = (phi.size, iy.size, ix.size)

        # estimate default chunk size
        panel_offset = expt.panel_offset
        dphi = expt.oscillation[1]
        nimgs = _default_stack_from_rotation_angle(dphi)
        nimgs = min(nimgs, shape[0])
        chunks = (nimgs,) + panel_offset

        data = NXfield(shape=shape, name="data", chunks=chunks, **data_opts)
        exposure_times = expt.exposure_times
        return LazyImageSeries(phi, iy, ix, data, exposure_times)

    def save(self, filename, name="image_series", **kwargs):
        """Save to nexus file"""
        nxobj = saveobj(self, filename, name=name, **kwargs)
        return nxobj


class VirtualImageSeries(LazyImageSeries):
    """Image series backed by a virtual datasets in a Nexus file with lazy loading and chunked compression"""

    def __init__(self, phi, iy, ix, data, exposure_times, maskval=-1):
        if not isinstance(data, NXvirtualfield):
            raise TypeError("data must be a NXvirtualfield for VirtualImageSeries")
        super().__init__(phi, iy, ix, data, exposure_times, maskval=maskval)

    @classmethod
    def create(
        cls,
        lazy_image_series,
        filename,
        name="image_series",
        source_template="{prefix}_{index}{ext}",
        source_directory=None,
    ):
        if not isinstance(lazy_image_series, LazyImageSeries):
            raise TypeError("Input lazy_image_series must be a LazyImageSeries object")
        if lazy_image_series.data.nxfilemode is not None:
            raise ValueError("Input lazy_image_series must not be associated with an open file")
        if lazy_image_series.data._value is not None or lazy_image_series.data._memfile is not None:
            raise ValueError("Input lazy_image_series must not have data already loaded in memory")

        slices = [sl for sl in lazy_image_series.chunk_slice_along_axis(0)]
        prefix, ext = os.path.splitext(filename)
        files = [source_template.format(prefix=prefix, name=name, index=index, ext=ext) for index in range(len(slices))]
        if source_directory is not None:
            files = [os.path.join(source_directory, f) for f in files]
            os.makedirs(source_directory, exist_ok=True)
        object_path = f"/entry/{name}"
        data_path = f"{object_path}/data"
        layout = h5py.VirtualLayout(shape=lazy_image_series.data.shape, dtype=lazy_image_series.data.dtype)

        # create the source files and populate the virtual layout
        for sl, fn in zip(slices, files):
            new_phi = lazy_image_series.phi[sl]
            new_exposure_times = lazy_image_series.exposure_times[sl]
            new_data = deepcopy(lazy_image_series.data)
            new_data.shape = (new_phi.size, *new_data.shape[1:])
            if new_data.chunks[0] > new_phi.size:
                new_data.chunks = (new_phi.size, *new_data.chunks[1:])
            new_image_series = LazyImageSeries(
                new_phi,
                lazy_image_series.iy,
                lazy_image_series.ix,
                new_data,
                new_exposure_times,
                maskval=lazy_image_series._maskval,
            )
            layout[sl, :, :] = h5py.VirtualSource(fn, data_path, shape=new_data.shape)
            new_image_series.save(fn, name=name)

        # first, save the lazy dataset
        lazy_image_series.save(filename, name=name)

        # now, do some surgery on the file to replace the dataset with a virtual dataset
        with h5py.File(filename, "r+", libver="latest") as f:
            del f[data_path]
            f.create_virtual_dataset(data_path, layout, fillvalue=lazy_image_series._maskval)
            f[object_path].attrs["mdx2_class"] = cls.__name__

        # now, re-load the object from the file to get the correct VirtualImageSeries object
        vis = cls.load(filename, name=name)
        return vis

    @property
    def virtual_source_files(self):
        """Get list of source files for virtual dataset

        Returns list of file paths and raises RuntimeError if the internal API has changed.

        This property provides safe access to h5py/nexusformat internal attributes.
        """
        try:
            return self.data._vfiles
        except AttributeError:
            # Internal API changed - this is a programming error
            raise RuntimeError(
                "Unable to access virtual dataset source files. The nexusformat/h5py internal API may have changed."
            )

    @property
    def virtual_dataset_path(self):
        """Get the HDF5 path to the virtual dataset.

        Returns HDF5 path, raises RuntimeError if the internal API has changed.

        This property provides safe access to h5py/nexusformat internal attributes.
        """
        try:
            return self.data._vpath
        except AttributeError:
            # Internal API changed - this is a programming error
            raise RuntimeError(
                "Unable to access virtual dataset HDF5 path. The nexusformat/h5py internal API may have changed."
            )

    @property
    def chunks(self):
        source_files = self.virtual_source_files
        dataset_path = self.virtual_dataset_path
        try:
            with h5py.File(source_files[0], "r") as f:
                ch = f[dataset_path].chunks
        except KeyError:
            # Dataset path doesn't exist in source file
            raise RuntimeError(f"Virtual dataset path '{dataset_path}' not found in source file '{source_files[0]}'")
        except (OSError, IOError) as e:
            # File access error - source file missing or inaccessible
            raise FileNotFoundError(f"Cannot access virtual dataset source file '{source_files[0]}': {e}")
        return ch

    def save(self, filename, name="image_series", **kwargs):
        raise NotImplementedError("VirtualImageSeries cannot be saved to a Nexus file directly")

    @staticmethod
    def from_expt(exptfile):
        raise NotImplementedError("VirtualImageSeries cannot be created from an Experiment file directly")


class ImageSeries:
    """Wrapper, acting as a factory for the other objects"""

    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "ImageSeries is deprecated. Please use LazyImageSeries, "
            "VirtualImageSeries, or InMemoryImageSeries directly.",
            DeprecationWarning,
        )
        return super().__new__(cls)

    def __init__(self, phi, iy, ix, data, exposure_times, maskval=-1):
        if isinstance(data, NXfield):
            cls = LazyImageSeries
        elif isinstance(data, NXvirtualfield):
            cls = VirtualImageSeries
        else:
            cls = InMemoryImageSeries
        self._obj = cls(phi, iy, ix, data, exposure_times, maskval=maskval)

    # alternate constructor, to wrap an existing object
    @classmethod
    def from_object(cls, obj):
        image_series = cls.__new__(cls)
        image_series._obj = obj
        return image_series

    @property
    def shape(self):
        return self._obj.shape

    @property
    def phi(self):
        return self._obj.phi

    @property
    def iy(self):
        return self._obj.iy

    @property
    def ix(self):
        return self._obj.ix

    @property
    def data(self):
        return self._obj.data

    @property
    def exposure_times(self):
        return self._obj.exposure_times

    def __getitem__(self, sl):
        return self._obj.__getitem__(sl)

    @property
    def data_masked(self):
        # check if self._obj has data_masked property
        if hasattr(self._obj, "data_masked"):
            return self._obj.data_masked
        elif isinstance(self._obj, LazyImageSeries):
            self._obj = self._obj[:, :, :]  # force load into memory
            return self._obj.data_masked
        else:
            raise AttributeError(f"Underlying object of type {type(self._obj)} has no data_masked property")

    @property
    def virtual_source_files(self):
        if hasattr(self._obj, "virtual_source_files"):
            return self._obj.virtual_source_files
        else:
            raise AttributeError(f"Underlying object of type {type(self._obj)} has no virtual_source_files property")

    @property
    def virtual_dataset_path(self):
        if hasattr(self._obj, "virtual_dataset_path"):
            return self._obj.virtual_dataset_path
        else:
            raise AttributeError(f"Underlying object of type {type(self._obj)} has no virtual_dataset_path property")

    @property
    def chunks(self):
        if hasattr(self._obj, "chunks"):
            return self._obj.chunks
        else:
            raise AttributeError(f"Underlying object of type {type(self._obj)} has no chunks property")

    def iter_chunks(self):
        if hasattr(self._obj, "iter_chunks"):
            return self._obj.iter_chunks()
        else:
            raise AttributeError(f"Underlying object of type {type(self._obj)} has no iter_chunks method")

    def chunk_slice_along_axis(self, *args, **kwargs):
        if hasattr(self._obj, "chunk_slice_along_axis"):
            return self._obj.chunk_slice_along_axis(*args, **kwargs)
        else:
            raise AttributeError(f"Underlying object of type {type(self._obj)} has no chunk_slice_along_axis method")

    def chunk_slice_iterator(self):
        if hasattr(self._obj, "chunk_slice_iterator"):
            return self._obj.chunk_slice_iterator()
        else:
            raise AttributeError(f"Underlying object of type {type(self._obj)} has no chunk_slice_iterator method")

    @staticmethod
    def from_expt(exptfile):
        obj = LazyImageSeries.from_expt(exptfile)
        return ImageSeries.from_object(obj)

    @staticmethod
    def from_nexus(nxdata):
        obj = LazyImageSeries.from_nexus(nxdata)
        return ImageSeries.from_object(obj)

    def to_nexus(self):
        if hasattr(self._obj, "to_nexus"):
            return self._obj.to_nexus()
        else:
            raise AttributeError(f"Underlying object of type {type(self._obj)} has no to_nexus method")

    def save(
        self,
        filename,
        name="image_series",
        virtual=False,
        source_template="{prefix}_{index}{ext}",
        source_directory=None,
        **kwargs,
    ):
        if not virtual:
            if hasattr(self._obj, "save"):
                return self._obj.save(filename, name=name, **kwargs)
            else:
                raise AttributeError(f"Underlying object of type {type(self._obj)} has no save method")
        else:
            if isinstance(self._obj, LazyImageSeries):
                vis = VirtualImageSeries.create(
                    self._obj,
                    filename,
                    name=name,
                    source_template=source_template,
                    source_directory=source_directory,
                )
                self._obj = vis  # update internal object to the VirtualImageSeries
                return vis
            else:
                raise TypeError("Only LazyImageSeries objects can be saved as VirtualImageSeries")

    @staticmethod
    def load(filename, name="image_series", mode="r"):
        """Load from nexus file"""
        obj = LazyImageSeries.load(filename, name=name, mode=mode)
        return ImageSeries.from_object(obj)

    def index(self, *args, **kwargs):
        if hasattr(self._obj, "index"):
            return self._obj.index(*args, **kwargs)
        else:
            raise AttributeError(f"Underlying object of type {type(self._obj)} has no index method")


# SOME UTILITY FUNCTIONS USED ABOVE


def _default_stack_from_rotation_angle(ang):
    """Compute a reasonable image stack size (1--50 images) given the rotation angle per frame (degrees)

    The heuristics are designed to work with typical choices of angular range per image, such as
    0.1, 0.025, 1/3, 0.04, etc. The algorithm chooses either 40, 45, or 50 image stack size if the
    total oscillation is less than 6 degrees. Otherwise, it chooses a stack to give either 5 or 6 degrees
    of total oscillation. If the heuristics fail, it will make a sane choice: 10 frames for large rotation
    per frame, and 40 frames if the rotation is small, and 50 frames if there is no rotation.
    """
    if ang == 0:
        numer, denom = 0, 1
    else:
        numer = 180
        denom = np.round(numer / ang).astype(int)
    if denom < numer * 7:
        if 5 * denom % numer == 0:
            nimgs = 5 * denom // numer
        elif 6 * denom % numer == 0:
            nimgs = 6 * denom // numer
        else:
            nimgs = 10
    elif 2 * 50 * numer % denom == 0:
        nimgs = 50
    elif 2 * 40 * numer % denom == 0:
        nimgs = 40
    elif 2 * 45 * numer % denom == 0:
        nimgs = 45
    elif 4 * 50 * numer % denom == 0:
        nimgs = 50
    elif 4 * 45 * numer % denom == 0:
        nimgs = 45
    else:
        nimgs = 40
    return nimgs
