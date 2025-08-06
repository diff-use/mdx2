import pkg_resources


def getVersionNumber():
    version = pkg_resources.require("mdx2")[0].version
    return version


__version__ = getVersionNumber()
