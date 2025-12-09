import importlib

from loguru import logger
from nexusformat.nexus import NXentry, NXgroup, NXroot, NXvirtualfield
from nexusformat.nexus import nxload as nexus_nxload

import mdx2

# Configure default logging for this module when used outside command-line tools
# Remove default handler and add a custom one with WARNING level and simplified format
logger.remove()  # Remove default handler
logger.add(
    lambda msg: print(msg, end=""),  # Print to stderr (default behavior)
    level="WARNING",
    format="<level>{level}</level>: {message}\n",
    colorize=True,
)

# FUNCTIONS FOR LOADING AND SAVING MDX2 CLASSES TO NEXUS FILES


def _patch_virtualfields(g):
    """Recursively patch NXvirtualfields in a Nexus group to set their shape."""
    if isinstance(g, NXgroup):
        for entry in g.entries.values():
            _patch_virtualfields(entry)
    elif isinstance(g, NXvirtualfield):
        logger.debug(f"patching virtual field: {g.nxpath}")
        with g.nxfile as f:
            g._shape = f.get(g.nxpath).shape


def nxload(filename, mode="r", **kwargs):
    """Wrapper around nexusformat.nexus.nxload to check mdx2 version."""
    nxroot = nexus_nxload(filename, mode=mode, **kwargs)
    mdx2_version_file = nxroot.attrs.get("mdx2_version")
    if mdx2_version_file != mdx2.__version__:
        logger.warning(f"mdx2 version mismatch: file version {mdx2_version_file}, installed version {mdx2.__version__}")
    _patch_virtualfields(nxroot)
    return nxroot


def nxsave(nxsobj, filename, mode="w", **kwargs):
    """Wrapper around nexusformat.nexus.nxsave to add mdx2 version."""
    nxroot = NXroot(NXentry(nxsobj))
    nxroot.attrs["mdx2_version"] = mdx2.__version__
    nxroot.save(filename, mode=mode, **kwargs)
    return nxroot


def loadobj(filename, objectname):
    # simple wrapper to load mdx2 objects from nxs files
    # handles import using the mdx2_module and mdx2_class attributes
    # does a from_nexus() call to instantiate the class
    nxroot = nxload(filename, "r")
    mdx2_version_file = nxroot.attrs.get("mdx2_version")
    if mdx2_version_file != mdx2.__version__:
        pass  # TODO: handle version mismatch
    nxs = nxroot["/entry/" + objectname]
    mod = nxs.attrs["mdx2_module"]
    cls = nxs.attrs["mdx2_class"]
    _tmp = importlib.__import__(mod, fromlist=[cls])
    Class = getattr(_tmp, cls)
    logger.info(f"Loading {objectname} from {filename} as {mod}.{cls}")
    return Class.from_nexus(nxs)


def saveobj(obj, filename, name=None, append=False, mode="w"):
    # simple wrapper to save mdx2 objects as nxs files
    #
    nxsobj = obj.to_nexus()
    if name is not None:
        nxsobj.rename(name)
    logger.info(f"Saving {nxsobj.nxname} ({type(obj)}) to {filename}")
    nxsobj.attrs["mdx2_module"] = type(obj).__module__
    nxsobj.attrs["mdx2_class"] = type(obj).__name__
    if append:
        root = nxload(filename, "r+")
        root["entry/" + nxsobj.nxname] = nxsobj
    else:
        nxsave(nxsobj, filename, mode=mode)
    return nxsobj
