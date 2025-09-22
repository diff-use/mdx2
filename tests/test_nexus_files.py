import nexusformat.nexus as nxs
import numpy as np

from mdx2.data import ImageSeries
from mdx2.utils import nxsave


def test_writing_data_after_nxsave(tmp_path):
    """check that data can be written after saving a Nexus object"""

    # write a nexus file with an empty NXfield
    filename = tmp_path / "test_data_after_nxsave.nxs"
    nxfield = nxs.NXfield(shape=(10,), dtype="int32")
    nxdata = nxs.NXdata(nxfield)
    nxsave(nxdata, filename)
    # now, write some data to the NXfield
    nxfield[0] = 42

    # reload the file and check that the data was written correctly
    nxroot = nxs.nxload(filename)
    assert nxroot.entry.data.signal.nxdata[0] == 42


def test_ImageSeries_virtual_dataset_creation(tmp_path):
    """check that ImageSeries can be saved with a virtual dataset"""
    data_opts = {"dtype": np.int32, "compression": "gzip", "compression_opts": 1, "shuffle": True}

    phi = np.arange(20)
    iy = np.arange(10)
    ix = np.arange(10)
    data = nxs.NXfield(shape=(20, 10, 10), name="data", **data_opts, chunks=(8, 5, 5))
    exposure_times = 0.1 * np.ones_like(phi)

    image_series = ImageSeries(phi, iy, ix, data, exposure_times)
    filename = tmp_path / "test_image_series_virtual.nxs"
    nxobj = image_series.save(filename, virtual=True)
    assert isinstance(nxobj.data, nxs.NXvirtualfield)
    assert isinstance(image_series.data, nxs.NXvirtualfield)
    assert image_series.chunks == (8, 5, 5)
    assert image_series.data.shape == (20, 10, 10)
    assert len(nxobj.data._vfiles) == 3  # source files are present

    # write some data to the second source file, check that it appears in the virtual dataset
    source_series = ImageSeries.load(nxobj.data._vfiles[1], mode="r+")
    source_series.data[0, 0, 0] = 666
    assert image_series.data[8, 0, 0] == 666
