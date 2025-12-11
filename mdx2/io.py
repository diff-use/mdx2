import importlib
import warnings

from nexusformat.nexus import NXentry, NXgroup, NXroot, NXvirtualfield
from nexusformat.nexus import nxload as nexus_nxload

import mdx2

# FUNCTIONS FOR LOADING AND SAVING MDX2 CLASSES TO NEXUS FILES


def _patch_virtualfields(g):
    """Recursively patch NXvirtualfields in a Nexus group to set their shape."""
    if isinstance(g, NXgroup):
        for entry in g.entries.values():
            _patch_virtualfields(entry)
    elif isinstance(g, NXvirtualfield):
        # Patch virtual field shape - this has been thoroughly tested but may need
        # debugging if nexusformat package changes its internal handling of virtual datasets
        with g.nxfile as f:
            g._shape = f.get(g.nxpath).shape


def nxload(filename, mode="r", **kwargs):
    """Wrapper around nexusformat.nexus.nxload to check mdx2 version."""
    nxroot = nexus_nxload(filename, mode=mode, **kwargs)
    mdx2_version_file = nxroot.attrs.get("mdx2_version")
    if mdx2_version_file != mdx2.__version__:
        warnings.warn(
            f"mdx2 version mismatch: file version {mdx2_version_file}, installed version {mdx2.__version__}",
            UserWarning,
            stacklevel=2,
        )
    _patch_virtualfields(nxroot)
    return nxroot


def nxsave(nxsobj, filename, mode="w", **kwargs):
    """Wrapper around nexusformat.nexus.nxsave to add mdx2 version."""
    nxroot = NXroot(NXentry(nxsobj))
    nxroot.attrs["mdx2_version"] = mdx2.__version__
    nxroot.save(filename, mode=mode, **kwargs)
    return nxroot


def loadobj(filename, objectname):
    """Load mdx2 objects from nexus files with security validation.

    Parameters
    ----------
    filename : str
        Path to the nexus file
    objectname : str
        Name of the object within the nexus file entry

    Returns
    -------
    object
        The deserialized mdx2 object

    Raises
    ------
    ValueError
        If the module is not from the mdx2 package
    TypeError
        If the class does not have a callable from_nexus method
    """
    nxroot = nxload(filename, "r")
    nxs = nxroot["/entry/" + objectname]
    mod = nxs.attrs["mdx2_module"]
    cls = nxs.attrs["mdx2_class"]

    # Security: Only allow mdx2 modules to prevent arbitrary code execution
    if not mod.startswith("mdx2."):
        raise ValueError(f"Untrusted module '{mod}': only mdx2.* modules are allowed")

    # Import and get the class
    _tmp = importlib.__import__(mod, fromlist=[cls])
    Class = getattr(_tmp, cls)

    # Validate the class has the expected interface
    if not hasattr(Class, "from_nexus") or not callable(getattr(Class, "from_nexus")):
        raise TypeError(f"{mod}.{cls} does not have a callable from_nexus method")

    return Class.from_nexus(nxs)


def saveobj(obj, filename, name=None, append=False, mode="w"):
    # simple wrapper to save mdx2 objects as nxs files
    if not hasattr(obj, "to_nexus"):
        raise TypeError(f"{type(obj).__name__} does not support Nexus serialization (missing to_nexus method)")
    nxsobj = obj.to_nexus()
    if name is not None:
        nxsobj.rename(name)
    nxsobj.attrs["mdx2_module"] = type(obj).__module__
    nxsobj.attrs["mdx2_class"] = type(obj).__name__
    if append:
        root = nxload(filename, "r+")
        root["entry/" + nxsobj.nxname] = nxsobj
    else:
        nxsave(nxsobj, filename, mode=mode)
    return nxsobj
