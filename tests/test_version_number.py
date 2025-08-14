import re

from mdx2 import __version__ as mdx2_version


def test_mdx2_version_number():
    """check that mdx2.__version__ matches semver semantics"""
    semver_regex = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"  # noqa: E501
    match = re.match(semver_regex, mdx2_version)
    assert match
    major, minor, patch, prerelease, buildmetadata = match.groups()
