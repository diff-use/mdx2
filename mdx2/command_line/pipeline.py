"""
Prefect pipeline for the single-crystal example (insulin-tutorial).

Runs the full single-crystal workflow matching the reference script:
DIALS: import (data) → find_spots → index → refine → import (background);
mdx2: import_data refined.expt (--datastore) → import_geometry → find_peaks → mask_peaks →
import_data background (--datastore_bg) → bin_image_series → integrate → correct →
scale [--mca2020] → merge → map slice → map offslice.

If mdx2_single_crystal.json or deployment.json exists in the current working
directory (or in working_dir when set), its keys are used to fill in the flow
parameters. Use raw_data_dir to read inputs from another directory (e.g. raw_data/insulin)
and write all outputs to working_dir (e.g. processed_data/insulin).

Run from the repo root or from examples/insulin-tutorial:
  python -m mdx2.command_line.pipeline
  python -m mdx2.command_line.pipeline --working_dir /path/to/insulin-tutorial
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from prefect import flow
from prefect.logging import get_run_logger

from mdx2.command_line.prefect_flows import run_conda_command, run_mdx2_cli_command


# Config file names to look for in the working directory (first found wins)
CONFIG_FILENAMES = ("mdx2_single_crystal.json", "deployment.json")

# Keys allowed in the config file (must match flow parameters)
CONFIG_KEYS = frozenset({
    "working_dir",
    "raw_data_dir",
    "run_dials",
    "crystal_images",
    "background_images",
    "space_group",
    "refined_expt",
    "background_expt",
    "integrate_subdivide",
    "count_threshold",
    "sigma_cutoff",
    "nproc",
    "datastore",
    "datastore_bg",
    "mca2020",
})


def load_single_crystal_config(search_dir: Path) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Load single-crystal pipeline parameters from search_dir.
    Looks for mdx2_single_crystal.json then deployment.json.
    Returns (params_dict, path_to_file) or ({}, None) if none found.
    """
    for name in CONFIG_FILENAMES:
        path = search_dir / name
        if not path.is_file():
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        return {k: v for k, v in data.items() if k in CONFIG_KEYS}, path
    return {}, None


def _cmd(command: str, args: list, working_dir: Optional[str]):
    """Run one mdx2 CLI command as a Prefect task."""
    return run_mdx2_cli_command(command, args, working_dir=working_dir)


@flow(name="single-crystal-example", log_prints=True)
def single_crystal_workflow(
    working_dir: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
    run_dials: bool = True,
    crystal_images: str = "images/insulin_2_1",
    background_images: str = "images/insulin_2_bkg",
    space_group: int = 199,
    refined_expt: str = "refined.expt",
    background_expt: str = "background.expt",
    integrate_subdivide: str = "4 4 4",
    count_threshold: int = 20,
    sigma_cutoff: int = 3,
    nproc: int = 1,
    datastore: str = "datastore",
    datastore_bg: str = "datastore_bg",
    mca2020: bool = False,
) -> list:
    """
    Run the single-crystal (insulin-tutorial) pipeline: DIALS then mdx2.

    When run_dials is True (default), runs DIALS in raw_data_dir (or working_dir):
    dials.import → find_spots → index → refine, then background import. Produces
    refined.expt and background.expt. Then runs the mdx2 chain: import_geometry →
    import_data → find_peaks → mask_peaks → (background bin) → integrate → correct →
    scale → merge → map. Set run_dials=false if refined.expt and background.expt
    already exist. Config file (mdx2_single_crystal.json or deployment.json) can
    override all parameters.
    """
    logger = get_run_logger()

    # Search for config file in working_dir or cwd
    search_dir = Path(working_dir) if working_dir else Path.cwd()
    config, config_path = load_single_crystal_config(search_dir)
    if config:
        logger.info("Loaded parameters from %s", config_path)
        working_dir = config.get("working_dir", working_dir)
        raw_data_dir = config.get("raw_data_dir", raw_data_dir)
        run_dials = config.get("run_dials", run_dials)
        crystal_images = config.get("crystal_images", crystal_images)
        background_images = config.get("background_images", background_images)
        space_group = config.get("space_group", space_group)
        refined_expt = config.get("refined_expt", refined_expt)
        background_expt = config.get("background_expt", background_expt)
        integrate_subdivide = config.get("integrate_subdivide", integrate_subdivide)
        count_threshold = config.get("count_threshold", count_threshold)
        sigma_cutoff = config.get("sigma_cutoff", sigma_cutoff)
        nproc = config.get("nproc", nproc)
        datastore = config.get("datastore", datastore)
        datastore_bg = config.get("datastore_bg", datastore_bg)
        mca2020 = config.get("mca2020", mca2020)
        # Normalize integrate_subdivide (JSON may give list or string)
        if isinstance(integrate_subdivide, list):
            integrate_subdivide = " ".join(str(x) for x in integrate_subdivide)
        # If working_dir was relative in config, resolve against search_dir
        if working_dir and not Path(working_dir).is_absolute():
            working_dir = str((search_dir / working_dir).resolve())
        # If raw_data_dir was relative in config, resolve against search_dir
        if raw_data_dir and not Path(raw_data_dir).is_absolute():
            raw_data_dir = str((search_dir / raw_data_dir).resolve())

    wd = working_dir
    base = Path(working_dir) if working_dir else Path.cwd()
    raw_base = Path(raw_data_dir).resolve() if raw_data_dir else base

    if not run_dials:
        if not (raw_base / refined_expt).exists():
            raise FileNotFoundError(
                f"Expected {refined_expt} in {raw_base}. "
                "Set raw_data_dir, set run_dials=true to run DIALS, or place files in working_dir."
            )
        if not (raw_base / background_expt).exists():
            raise FileNotFoundError(
                f"Expected {background_expt} in {raw_base}. "
                "Set run_dials=true to run DIALS or place files (see examples/insulin-tutorial/README.md)."
            )

    # Paths for inputs: from raw_data_dir when set, else working_dir
    refined_path = str(raw_base / refined_expt)
    background_path = str(raw_base / background_expt)

    results = []
    dials_wd = str(raw_base)

    # DIALS steps (run in raw_base so image paths resolve)
    if run_dials:
        logger.info("Running DIALS: import → find_spots → index → refine, then background import")
        # 1. dials.import crystal images → imported.expt
        results.append(run_conda_command(["dials.import", crystal_images], working_dir=dials_wd))
        # 2. dials.find_spots → strong.refl
        results.append(run_conda_command(["dials.find_spots", "imported.expt"], working_dir=dials_wd))
        # 3. dials.index → indexed.expt, indexed.refl
        results.append(
            run_conda_command(
                ["dials.index", "imported.expt", "strong.refl", f"space_group={space_group}"],
                working_dir=dials_wd,
            )
        )
        # 4. dials.refine → refined.expt, refined.refl
        results.append(
            run_conda_command(["dials.refine", "indexed.expt", "indexed.refl"], working_dir=dials_wd)
        )
        # 5. dials.import background images → background.expt
        results.append(
            run_conda_command(
                ["dials.import", background_images, "output.experiments=background.expt"],
                working_dir=dials_wd,
            )
        )

    # mdx2 steps (order matches reference: import_data → import_geometry → find_peaks → mask_peaks → bkg → integrate → correct → scale → merge → map slice → map offslice)
    # 1. mdx2.import_data refined.expt --nproc N --datastore datastore → data.nxs
    results.append(
        _cmd(
            "import_data",
            [refined_path, "--nproc", str(nproc), "--datastore", datastore],
            wd,
        )
    )
    # 2. mdx2.import_geometry refined.expt → geometry.nxs
    results.append(_cmd("import_geometry", [refined_path], wd))
    # 3. mdx2.find_peaks geometry.nxs data.nxs --nproc N --count_threshold 20 → peaks.nxs
    results.append(
        _cmd(
            "find_peaks",
            [
                "geometry.nxs",
                "data.nxs",
                "--nproc",
                str(nproc),
                "--count_threshold",
                str(count_threshold),
            ],
            wd,
        )
    )
    # 4. mdx2.mask_peaks geometry.nxs data.nxs peaks.nxs --nproc N --sigma_cutoff 3 → mask.nxs
    results.append(
        _cmd(
            "mask_peaks",
            [
                "geometry.nxs",
                "data.nxs",
                "peaks.nxs",
                "--nproc",
                str(nproc),
                "--sigma_cutoff",
                str(sigma_cutoff),
            ],
            wd,
        )
    )
    # 5. mdx2.import_data background.expt --outfile bkg_data.nxs --nproc N --datastore datastore_bg
    results.append(
        _cmd(
            "import_data",
            [
                background_path,
                "--outfile",
                "bkg_data.nxs",
                "--nproc",
                str(nproc),
                "--datastore",
                datastore_bg,
            ],
            wd,
        )
    )
    # 6. mdx2.bin_image_series bkg_data.nxs 10 20 20 --valid_range 0 200 --outfile bkg_data_binned.nxs --nproc N
    results.append(
        _cmd(
            "bin_image_series",
            [
                "bkg_data.nxs",
                "10",
                "20",
                "20",
                "--valid_range",
                "0",
                "200",
                "--outfile",
                "bkg_data_binned.nxs",
                "--nproc",
                str(nproc),
            ],
            wd,
        )
    )
    # 7. mdx2.integrate geometry.nxs data.nxs --mask mask.nxs --subdivide ... --nproc N
    results.append(
        _cmd(
            "integrate",
            [
                "geometry.nxs",
                "data.nxs",
                "--mask",
                "mask.nxs",
                "--subdivide",
                *integrate_subdivide.split(),
                "--nproc",
                str(nproc),
            ],
            wd,
        )
    )
    # 8. mdx2.correct geometry.nxs integrated.nxs --background bkg_data_binned.nxs → corrected.nxs
    results.append(
        _cmd(
            "correct",
            [
                "geometry.nxs",
                "integrated.nxs",
                "--background",
                "bkg_data_binned.nxs",
            ],
            wd,
        )
    )
    # 9. mdx2.scale corrected.nxs [--mca2020] → scales.nxs
    scale_args = ["corrected.nxs"]
    if mca2020:
        scale_args.append("--mca2020")
    results.append(_cmd("scale", scale_args, wd))
    # 10. mdx2.merge corrected.nxs --scale scales.nxs → merged.nxs
    results.append(_cmd("merge", ["corrected.nxs", "--scale", "scales.nxs"], wd))
    # 11. mdx2.map slice: -50 50 -50 50 0 0 → slice.nxs
    results.append(
        _cmd(
            "map",
            [
                "geometry.nxs",
                "merged.nxs",
                "--limits",
                "-50",
                "50",
                "-50",
                "50",
                "0",
                "0",
                "--outfile",
                "slice.nxs",
            ],
            wd,
        )
    )
    # 12. mdx2.map offslice: -50 50 -50 50 0.5 0.5 → offslice.nxs
    results.append(
        _cmd(
            "map",
            [
                "geometry.nxs",
                "merged.nxs",
                "--limits",
                "-50",
                "50",
                "-50",
                "50",
                "0.5",
                "0.5",
                "--outfile",
                "offslice.nxs",
            ],
            wd,
        )
    )

    logger.success("Single-crystal example pipeline finished.")
    return results


if __name__ == "__main__":
    import os

    # Default working_dir to insulin-tutorial if we're in the mdx2 repo
    cwd = Path.cwd()
    insulin_tutorial = cwd / "examples" / "insulin-tutorial"
    if insulin_tutorial.is_dir():
        default_working_dir = str(insulin_tutorial)
    else:
        default_working_dir = str(cwd)

    api_url = os.getenv("PREFECT_API_URL", "http://prefect-server:4200/api")
    print(f"Connecting to Prefect API at: {api_url}")
    print(f"Running single-crystal example (working_dir={default_working_dir})")

    # Run the flow (blocking)
    single_crystal_workflow(working_dir=default_working_dir)
