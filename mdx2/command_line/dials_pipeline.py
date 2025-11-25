from __future__ import annotations

import logging
import math
import os
import platform
import sys

import cctbx
import xia2.Driver.timing
import xia2.Handlers.Streams
import xia2.XIA2Version
from xia2.Handlers.Environment import df
from xia2.Handlers.Flags import Flags
from xia2.XIA2Version import Version

logger = logging.getLogger("xia2.cli.xia2_main")

# monkey patch report generation to skip ccp4 dependent steps

from xia2.cli.xia2_html import generate_xia2_html  # noqa: E402


def _generate_xia2_html_no_ccp4(xinfo, filename="xia2.html", params=None):
    """Generate xia2 HTML report, skipping CCP4 dependent analyses."""
    logger.info("Generating xia2 HTML report (CCP4 dependent analyses skipped).")

    # Proceed with generating the report, but skip CCP4 dependent sections
    if params:
        params.xtriage_analysis = False
        params.include_radiation_damage = False

    # Call the original function with modified parameters
    generate_xia2_html(xinfo, filename=filename, params=params)


generate_xia2_html = _generate_xia2_html_no_ccp4  # noqa: F811


def _check_environment_noccp4():
    """Check the environment we are running in..."""

    if sys.hexversion < 0x02070000:
        raise RuntimeError("Python versions older than 2.7 are not supported")

    executable = sys.executable
    cctbx_dir = os.sep.join(cctbx.__file__.split(os.sep)[:-3])

    # to help wrapper code - print process id...

    logger.debug("Process ID: %d", os.getpid())

    logger.info("Environment configuration...")
    logger.info("Python => %s", executable)
    logger.info("CCTBX => %s", cctbx_dir)

    ccp4_keys = ["CCP4", "CCP4_SCR"]
    for k in ccp4_keys:
        v = os.getenv(k)
        if not v:
            logger.warning("%s not defined - some features will be unavailable", k)

    logger.info("Starting directory: %s", Flags.get_starting_directory())
    logger.info("Working directory: %s", os.getcwd())
    logger.info("Free space:        %.2f GB", df() / math.pow(2, 30))

    hostname = platform.node().split(".")[0]
    logger.info("Host: %s", hostname)

    logger.info("Contact: xia2.support@gmail.com")

    logger.info(Version)


import xia2.Applications.xia2_main  # noqa: E402

xia2.Applications.xia2_main.check_environment = _check_environment_noccp4

from xia2.Modules.Scaler.CommonScaler import CommonScaler  # noqa: E402


def _skip_scale_finish_chunk_6_add_free_r(self):
    """Skip adding free R flags if CCP4 is not available."""
    logger.info("Skipping addition of free R flags (CCP4 not available).")
    hklout = self._scalr_scaled_reflection_files["mtz_merged"]
    self._scalr_scaled_reflection_files["mtz"] = hklout
    del self._scalr_scaled_reflection_files["mtz_merged"]


def _skip_scale_finish_chunk_8_raddam(self):
    """Skip Raddam analysis if CCP4 is not available."""
    logger.info("Skipping Raddam analysis (CCP4 not available).")


CommonScaler._scale_finish_chunk_6_add_free_r = _skip_scale_finish_chunk_6_add_free_r
CommonScaler._scale_finish_chunk_8_raddam = _skip_scale_finish_chunk_8_raddam

from xia2.cli.xia2_main import run  # noqa: E402

if __name__ == "__main__":
    run()
