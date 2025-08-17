import re

import h5py as h5
import numpy as np

from mdx2 import __version__ as mdx2_version
from mdx2.geometry import Crystal
from mdx2.utils import nxsave


def test_mdx2_version_number():
    """check that mdx2.__version__ matches semver semantics"""
    semver_regex = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"  # noqa: E501
    match = re.match(semver_regex, mdx2_version)
    assert match
    major, minor, patch, prerelease, buildmetadata = match.groups()
    assert major is not None
    assert minor is not None
    assert patch is not None
    # prerelease and buildmetadata can be None


def test_mdx2_utils_nxsave(tmp_path):
    """check that mdx2.utils.nxsave adds a version number"""

    crystal = Crystal(
        space_group="P 1",
        unit_cell=np.array([100.0, 100.0, 100.0, 90.0, 90.0, 90.0]),
        orientation_matrix=np.identity(3),
        ub_matrix=np.identity(3),
    )

    filename = tmp_path / "test_crystal.nxs"
    nxsave(crystal.to_nexus(), filename)

    with h5.File(filename, "r") as f:
        assert "mdx2_version" in f.attrs
        assert f.attrs["mdx2_version"] == mdx2_version
