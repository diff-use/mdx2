import re

from mdx2 import getVersionNumber


def test_version_number():
    """version string returned by getVersionNumber should match symver semantics"""
    version_string = getVersionNumber()
    semver_regex = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"  # noqa: E501
    match = re.match(semver_regex, version_string)
    assert match
    major, minor, patch, prerelease, buildmetadata = match.groups()
