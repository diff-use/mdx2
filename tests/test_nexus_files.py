import nexusformat.nexus as nxs

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
