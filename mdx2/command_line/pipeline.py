"""
Prefect pipeline for the single-crystal example (insulin-tutorial).

Runs the full single-crystal workflow matching the reference script:
DIALS: import (data) → find_spots → index → refine → import (background);
mdx2: import_data refined.expt (--datastore) → import_geometry → find_peaks → mask_peaks →
import_data background (--datastore_bg) → bin_image_series → integrate → correct →
scale [--mca2020] → merge → map slice → map offslice.

If deployment.json exists in the current working directory (or in working_dir
when set), its keys are used to fill in the flow parameters. Use --file to
specify a config file by name. Use raw_data_dir to read
inputs from another directory (e.g. raw_data/insulin) and write all outputs to
working_dir (e.g. processed_data/insulin).

Modes:
  prefect (default when Prefect is installed): run as Prefect flow/tasks.
  local: run the same pipeline locally without Prefect (e.g. if Prefect is not installed).
  Use --mode local to force local execution; use --mode prefect to require Prefect.

Run from the mdx2-dev env (e.g. in Docker or with micromamba activate mdx2-dev):
  mdx2.pipeline --file single_crystal_workflow.json
  mdx2.pipeline --working_dir /path/to/processed_data
  mdx2.pipeline --mode local --working_dir /path/to/processed_data

Or as a module:
  python -m mdx2.command_line.pipeline --file single_crystal_workflow.json
  python -m mdx2.command_line.pipeline --working_dir /path/to/insulin-tutorial
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from pathlib import Path as _Path
from typing import Any, Dict, List, Optional, Tuple

# Optional Prefect: allow running in "local" mode when Prefect is not installed
PREFECT_AVAILABLE = False
flow = None
task = None
get_run_logger = None
run_conda_command = None
run_mdx2_cli_command = None

try:
    from prefect import flow as _flow, task as _task
    from prefect.logging import get_run_logger as _get_run_logger

    _prefect_dir = _Path(__file__).resolve().parent.parent.parent / "prefect"
    if _prefect_dir.is_dir() and str(_prefect_dir) not in sys.path:
        sys.path.insert(0, str(_prefect_dir))
    from prefect_flows import run_conda_command as _run_conda, run_mdx2_cli_command as _run_mdx2

    PREFECT_AVAILABLE = True
    flow = _flow
    task = _task
    get_run_logger = _get_run_logger
    run_conda_command = _run_conda
    run_mdx2_cli_command = _run_mdx2
except Exception:
    pass

_log = logging.getLogger(__name__)


def _run_conda_command_local(
    argv: List[str],
    working_dir: Optional[str] = None,
    conda_env: str = "mdx2-dev",
) -> subprocess.CompletedProcess:
    """Run a conda command locally (no Prefect). Same behavior as prefect_flows.run_conda_command."""
    cmd = ["micromamba", "run", "-n", conda_env] + argv
    _log.info("Running: %s (cwd=%s)", " ".join(cmd), working_dir)
    result = subprocess.run(
        cmd,
        cwd=working_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        _log.info("STDOUT:\n%s", result.stdout)
    if result.stderr:
        _log.warning("STDERR:\n%s", result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result


def _run_mdx2_cli_command_local(
    command: str,
    args: List[str],
    working_dir: Optional[str] = None,
    conda_env: str = "mdx2-dev",
) -> subprocess.CompletedProcess:
    """Run an mdx2 CLI command locally (no Prefect). Same behavior as prefect_flows.run_mdx2_cli_command."""
    cmd = [
        "micromamba", "run", "-n", conda_env,
        "python", "-m", f"mdx2.command_line.{command}",
    ] + args
    _log.info("Running command: %s", " ".join(cmd))
    if working_dir:
        _log.info("Working directory: %s", working_dir)
    result = subprocess.run(
        cmd,
        cwd=working_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        _log.info("STDOUT:\n%s", result.stdout)
    if result.stderr:
        _log.warning("STDERR:\n%s", result.stderr)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result


# Config file names to look for in the working directory (first found wins)
CONFIG_FILENAMES = ("deployment.json",)

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
    Looks for deployment.json.
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


def load_single_crystal_config_from_file(path: Path) -> Tuple[Dict[str, Any], Path]:
    """
    Load single-crystal pipeline parameters from an explicit config file path.
    Returns (params_dict, path_to_file). Raises FileNotFoundError if path does not exist.
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k in CONFIG_KEYS}, path


# Pipeline step tasks (each step is a task with persist_result=True for result storage)
# Only defined when Prefect is available
if PREFECT_AVAILABLE:

    @task(persist_result=True, name="dials-import")
    def task_dials_import(argv: list, working_dir: Optional[str]):
        return run_conda_command(argv, working_dir=working_dir)

    @task(persist_result=True, name="dials-find-spots")
    def task_dials_find_spots(argv: list, working_dir: Optional[str]):
        return run_conda_command(argv, working_dir=working_dir)

    @task(persist_result=True, name="dials-index")
    def task_dials_index(argv: list, working_dir: Optional[str]):
        return run_conda_command(argv, working_dir=working_dir)

    @task(persist_result=True, name="dials-refine")
    def task_dials_refine(argv: list, working_dir: Optional[str]):
        return run_conda_command(argv, working_dir=working_dir)

    @task(persist_result=True, name="dials-import-background")
    def task_dials_import_background(argv: list, working_dir: Optional[str]):
        return run_conda_command(argv, working_dir=working_dir)

    @task(persist_result=True, name="mdx2-import-data")
    def task_mdx2_import_data(command: str, args: list, working_dir: Optional[str]):
        return run_mdx2_cli_command(command, args, working_dir=working_dir)

    @task(persist_result=True, name="mdx2-import-geometry")
    def task_mdx2_import_geometry(command: str, args: list, working_dir: Optional[str]):
        return run_mdx2_cli_command(command, args, working_dir=working_dir)

    @task(persist_result=True, name="mdx2-find-peaks")
    def task_mdx2_find_peaks(command: str, args: list, working_dir: Optional[str]):
        return run_mdx2_cli_command(command, args, working_dir=working_dir)

    @task(persist_result=True, name="mdx2-mask-peaks")
    def task_mdx2_mask_peaks(command: str, args: list, working_dir: Optional[str]):
        return run_mdx2_cli_command(command, args, working_dir=working_dir)

    @task(persist_result=True, name="mdx2-bin-image-series")
    def task_mdx2_bin_image_series(command: str, args: list, working_dir: Optional[str]):
        return run_mdx2_cli_command(command, args, working_dir=working_dir)

    @task(persist_result=True, name="mdx2-integrate")
    def task_mdx2_integrate(command: str, args: list, working_dir: Optional[str]):
        return run_mdx2_cli_command(command, args, working_dir=working_dir)

    @task(persist_result=True, name="mdx2-correct")
    def task_mdx2_correct(command: str, args: list, working_dir: Optional[str]):
        return run_mdx2_cli_command(command, args, working_dir=working_dir)

    @task(persist_result=True, name="mdx2-scale")
    def task_mdx2_scale(command: str, args: list, working_dir: Optional[str]):
        return run_mdx2_cli_command(command, args, working_dir=working_dir)

    @task(persist_result=True, name="mdx2-merge")
    def task_mdx2_merge(command: str, args: list, working_dir: Optional[str]):
        return run_mdx2_cli_command(command, args, working_dir=working_dir)

    @task(persist_result=True, name="mdx2-map")
    def task_mdx2_map(command: str, args: list, working_dir: Optional[str]):
        return run_mdx2_cli_command(command, args, working_dir=working_dir)

    @flow(name="single-crystal-example", log_prints=True)
    def single_crystal_workflow(
        working_dir: Optional[str] = None,
        raw_data_dir: Optional[str] = None,
        config_file: Optional[str] = None,
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
        already exist. Config file (deployment.json) can override all parameters.
        """
        logger = get_run_logger()

        # Load config: from --file path if given, else search working_dir or cwd for default names
        if config_file:
            config, config_path = load_single_crystal_config_from_file(Path(config_file))
            search_dir = config_path.parent
        else:
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
            results.append(task_dials_import(["dials.import", crystal_images], dials_wd))
            # 2. dials.find_spots → strong.refl
            results.append(task_dials_find_spots(["dials.find_spots", "imported.expt"], dials_wd))
            # 3. dials.index → indexed.expt, indexed.refl
            results.append(
                task_dials_index(
                    ["dials.index", "imported.expt", "strong.refl", f"space_group={space_group}"],
                    dials_wd,
                )
            )
            # 4. dials.refine → refined.expt, refined.refl
            results.append(
                task_dials_refine(["dials.refine", "indexed.expt", "indexed.refl"], dials_wd)
            )
            # 5. dials.import background images → background.expt
            results.append(
                task_dials_import_background(
                    ["dials.import", background_images, "output.experiments=background.expt"],
                    dials_wd,
                )
            )

        # mdx2 steps (order matches reference: import_data → import_geometry → find_peaks → mask_peaks → bkg → integrate → correct → scale → merge → map slice → map offslice)
        # 1. mdx2.import_data refined.expt --nproc N --datastore datastore → data.nxs
        results.append(
            task_mdx2_import_data(
                "import_data",
                [refined_path, "--nproc", str(nproc), "--datastore", datastore],
                wd,
            )
        )
        # 2. mdx2.import_geometry refined.expt → geometry.nxs
        results.append(task_mdx2_import_geometry("import_geometry", [refined_path], wd))
        # 3. mdx2.find_peaks geometry.nxs data.nxs --nproc N --count_threshold 20 → peaks.nxs
        results.append(
            task_mdx2_find_peaks(
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
            task_mdx2_mask_peaks(
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
            task_mdx2_import_data(
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
            task_mdx2_bin_image_series(
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
            task_mdx2_integrate(
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
            task_mdx2_correct(
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
        results.append(task_mdx2_scale("scale", scale_args, wd))
        # 10. mdx2.merge corrected.nxs --scale scales.nxs → merged.nxs
        results.append(task_mdx2_merge("merge", ["corrected.nxs", "--scale", "scales.nxs"], wd))
        # 11. mdx2.map slice: -50 50 -50 50 0 0 → slice.nxs
        results.append(
            task_mdx2_map(
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
            task_mdx2_map(
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


def single_crystal_workflow_local(
    working_dir: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
    config_file: Optional[str] = None,
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
    Run the single-crystal (insulin-tutorial) pipeline locally without Prefect.
    Same steps and parameters as single_crystal_workflow; use when Prefect is not installed.
    """
    # Load config (same logic as Prefect flow)
    if config_file:
        config, config_path = load_single_crystal_config_from_file(Path(config_file))
        search_dir = config_path.parent
    else:
        search_dir = Path(working_dir) if working_dir else Path.cwd()
        config, config_path = load_single_crystal_config(search_dir)
    if config:
        _log.info("Loaded parameters from %s", config_path)
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
        if isinstance(integrate_subdivide, list):
            integrate_subdivide = " ".join(str(x) for x in integrate_subdivide)
        if working_dir and not Path(working_dir).is_absolute():
            working_dir = str((search_dir / working_dir).resolve())
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

    refined_path = str(raw_base / refined_expt)
    background_path = str(raw_base / background_expt)
    results: list = []
    dials_wd = str(raw_base)

    def run_conda(argv: list, cwd: Optional[str]):
        return _run_conda_command_local(argv, working_dir=cwd)

    def run_mdx2(cmd: str, args: list, cwd: Optional[str]):
        return _run_mdx2_cli_command_local(cmd, args, working_dir=cwd)

    if run_dials:
        _log.info("Running DIALS: import → find_spots → index → refine, then background import")
        results.append(run_conda(["dials.import", crystal_images], dials_wd))
        results.append(run_conda(["dials.find_spots", "imported.expt"], dials_wd))
        results.append(
            run_conda(
                ["dials.index", "imported.expt", "strong.refl", f"space_group={space_group}"],
                dials_wd,
            )
        )
        results.append(run_conda(["dials.refine", "indexed.expt", "indexed.refl"], dials_wd))
        results.append(
            run_conda(
                ["dials.import", background_images, "output.experiments=background.expt"],
                dials_wd,
            )
        )

    results.append(run_mdx2("import_data", [refined_path, "--nproc", str(nproc), "--datastore", datastore], wd))
    results.append(run_mdx2("import_geometry", [refined_path], wd))
    results.append(
        run_mdx2(
            "find_peaks",
            [
                "geometry.nxs", "data.nxs", "--nproc", str(nproc),
                "--count_threshold", str(count_threshold),
            ],
            wd,
        )
    )
    results.append(
        run_mdx2(
            "mask_peaks",
            [
                "geometry.nxs", "data.nxs", "peaks.nxs",
                "--nproc", str(nproc), "--sigma_cutoff", str(sigma_cutoff),
            ],
            wd,
        )
    )
    results.append(
        run_mdx2(
            "import_data",
            [background_path, "--outfile", "bkg_data.nxs", "--nproc", str(nproc), "--datastore", datastore_bg],
            wd,
        )
    )
    results.append(
        run_mdx2(
            "bin_image_series",
            [
                "bkg_data.nxs", "10", "20", "20",
                "--valid_range", "0", "200", "--outfile", "bkg_data_binned.nxs", "--nproc", str(nproc),
            ],
            wd,
        )
    )
    results.append(
        run_mdx2(
            "integrate",
            [
                "geometry.nxs", "data.nxs", "--mask", "mask.nxs",
                "--subdivide", *integrate_subdivide.split(), "--nproc", str(nproc),
            ],
            wd,
        )
    )
    results.append(
        run_mdx2(
            "correct",
            ["geometry.nxs", "integrated.nxs", "--background", "bkg_data_binned.nxs"],
            wd,
        )
    )
    scale_args = ["corrected.nxs"]
    if mca2020:
        scale_args.append("--mca2020")
    results.append(run_mdx2("scale", scale_args, wd))
    results.append(run_mdx2("merge", ["corrected.nxs", "--scale", "scales.nxs"], wd))
    results.append(
        run_mdx2(
            "map",
            ["geometry.nxs", "merged.nxs", "--limits", "-50", "50", "-50", "50", "0", "0", "--outfile", "slice.nxs"],
            wd,
        )
    )
    results.append(
        run_mdx2(
            "map",
            ["geometry.nxs", "merged.nxs", "--limits", "-50", "50", "-50", "50", "0.5", "0.5", "--outfile", "offslice.nxs"],
            wd,
        )
    )

    _log.info("Single-crystal example pipeline finished (local mode).")
    return results


# When Prefect is not available, expose local workflow so "single_crystal_workflow" is always callable
if not PREFECT_AVAILABLE:
    single_crystal_workflow = single_crystal_workflow_local  # type: ignore[misc, assignment]


def main() -> None:
    """CLI entry point: parse --file, --working_dir, --mode; run Prefect flow or local pipeline."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Run the single-crystal (insulin-tutorial) pipeline. Use --mode prefect for Prefect flow, --mode local to run without Prefect.",
    )
    parser.add_argument(
        "--file",
        "-f",
        metavar="JSON",
        default=None,
        help="If not set, looks for deployment.json in working dir.",
    )
    parser.add_argument(
        "--working_dir",
        "-w",
        metavar="DIR",
        default=None,
        help="Working directory for outputs. Defaults to insulin-tutorial if in mdx2 repo, else cwd.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=("prefect", "local", "auto"),
        default="auto",
        help="prefect: run as Prefect flow (requires Prefect). local: run locally without Prefect. auto: use Prefect if available, else local (default).",
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    if args.working_dir is not None:
        working_dir = args.working_dir
    elif (cwd / "examples" / "insulin-tutorial").is_dir():
        working_dir = str(cwd / "examples" / "insulin-tutorial")
    else:
        working_dir = str(cwd)

    use_prefect = args.mode == "prefect" or (args.mode == "auto" and PREFECT_AVAILABLE)

    if args.mode == "prefect" and not PREFECT_AVAILABLE:
        print("Prefect is not installed or could not be imported. Use --mode local to run the pipeline without Prefect.")
        sys.exit(1)

    if use_prefect:
        api_url = os.getenv("PREFECT_API_URL", "http://prefect-server:4200/api")
        print(f"Connecting to Prefect API at: {api_url}")
        print(f"Running single-crystal pipeline (working_dir={working_dir})")
        if args.file:
            print(f"Config file: {args.file}")
        single_crystal_workflow(working_dir=working_dir, config_file=args.file)
    else:
        if args.mode == "auto":
            print("Running in local mode (Prefect not available).")
        print(f"Running single-crystal pipeline locally (working_dir={working_dir})")
        if args.file:
            print(f"Config file: {args.file}")
        single_crystal_workflow_local(working_dir=working_dir, config_file=args.file)


if __name__ == "__main__":
    main()
