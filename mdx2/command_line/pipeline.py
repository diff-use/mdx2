"""
Prefect pipeline for the single-crystal workflow.

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
import os
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
    from prefect_flows import (
        directory_setup as _directory_setup,
        populate_deployment_images as _populate_task,
        run_conda_command as _run_conda,
        run_mdx2_cli_command as _run_mdx2,
    )

    PREFECT_AVAILABLE = True
    flow = _flow
    task = _task
    get_run_logger = _get_run_logger
    run_conda_command = _run_conda
    run_mdx2_cli_command = _run_mdx2
except Exception:
    pass

_log = logging.getLogger(__name__)


def _tee_output_to_file_local(result: subprocess.CompletedProcess, log_file: str) -> None:
    """Write stdout and stderr to log_file (tee-like behavior)."""
    with open(log_file, "w") as f:
        if result.stdout:
            f.write(result.stdout)
        if result.stderr:
            if result.stdout and not result.stdout.endswith("\n"):
                f.write("\n")
            f.write(result.stderr)


def _run_conda_command_local(
    argv: List[str],
    working_dir: Optional[str] = None,
    conda_env: str = "mdx2-dev",
    log_file: Optional[str] = None,
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
    if log_file:
        _tee_output_to_file_local(result, log_file)
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
    log_file: Optional[str] = None,
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
    if log_file:
        _tee_output_to_file_local(result, log_file)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result


def _default_deployment_json(crystal_name: str, raw_dir: str, processed_dir: str) -> dict:
    """Return default deployment.json content for a crystal. raw_data_dir is relative to processed_data/<crystal>/.

    crystal_files/background_files are intentionally omitted here; they are populated later by
    populate_deployment_images based on discovered HDF5 masters.
    """
    return {
        "working_dir": ".",
        "raw_data_dir": f"../../{raw_dir}/{crystal_name}",
        "run_dials": True,
        "space_group": 199,
        "refined_expt": "refined.expt",
        "background_expt": "background.expt",
        "integrate_subdivide": "4 4 4",
        "count_threshold": 20,
        "sigma_cutoff": 3,
        "nproc": 1,
        "datastore": "datastore",
        "datastore_bg": "datastore_bg",
        "mca2020": False,
    }


def _create_processed_dirs(raw_path: Path, processed_path: Path) -> List[Path]:
    """Create processed_path/<name>/ for each symlink or subdir in raw_path; create deployment.json if missing."""
    created: List[Path] = []
    if not raw_path.is_dir():
        return created
    raw_dir_name = raw_path.name
    processed_dir_name = processed_path.name
    for p in sorted(raw_path.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_symlink():
            target = p.resolve()
            if not target.is_dir():
                continue
        elif not p.is_dir():
            continue
        out_path = processed_path / p.name
        out_path.mkdir(parents=True, exist_ok=True)
        deployment_file = out_path / "deployment.json"
        if not deployment_file.exists():
            config = _default_deployment_json(p.name, raw_dir_name, processed_dir_name)
            with open(deployment_file, "w") as f:
                json.dump(config, f, indent=2)
        created.append(out_path)
    return created


def _populate_deployment_images(deployment_file: str) -> dict:
    """
    Populate deployment.json with paths discovered from raw HDF5 masters:
    - background_files: list of *_bg_*master.h5 relative to raw_data_dir
    - crystal_files: list of non-bg *_master.h5 relative to raw_data_dir

    Returns the updated config dict. Call before validation so empty values can be filled.
    """
    dep_path = Path(deployment_file).resolve()
    if not dep_path.is_file():
        raise FileNotFoundError(f"deployment.json not found: {dep_path}")

    with open(dep_path) as f:
        config = json.load(f)

    raw_data_dir = config.get("raw_data_dir")
    if not raw_data_dir:
        raise ValueError(f"raw_data_dir is missing in {dep_path}")

    raw_base = (dep_path.parent / raw_data_dir).resolve()
    if not raw_base.is_dir():
        _log.warning("raw_data_dir does not exist: %s", raw_base)
        return config

    bg_abs = sorted(set(raw_base.glob("*/*_bg_*master.h5")))
    all_master_abs = sorted(set(raw_base.glob("*/*_master.h5")))
    data_abs = [p for p in all_master_abs if "_bg_" not in p.name]

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(raw_base))
        except ValueError:
            return str(p)

    bg_files = [rel(p) for p in bg_abs]
    crystal_files = [rel(p) for p in data_abs]

    config["background_files"] = bg_files
    config["crystal_files"] = crystal_files

    with open(dep_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    _log.info(
        "Updated %s (crystal_files=%s, background_files=%s)",
        dep_path,
        len(crystal_files),
        len(bg_files),
    )
    return config


# Config file names to look for in the working directory (first found wins)
CONFIG_FILENAMES = ("deployment.json",)

# Keys allowed in the config file (must match flow parameters)
CONFIG_KEYS = frozenset({
    "working_dir",
    "raw_data_dir",
    "raw_dir",
    "processed_dir",
    "run_dials",
    "crystal_files",
    "background_files",
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
    def task_dials_import(argv: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_conda_command(argv, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="dials-find-spots")
    def task_dials_find_spots(argv: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_conda_command(argv, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="dials-index")
    def task_dials_index(argv: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_conda_command(argv, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="dials-refine")
    def task_dials_refine(argv: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_conda_command(argv, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="dials-import-background")
    def task_dials_import_background(argv: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_conda_command(argv, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="mdx2-import-data")
    def task_mdx2_import_data(command: str, args: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_mdx2_cli_command(command, args, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="mdx2-import-geometry")
    def task_mdx2_import_geometry(command: str, args: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_mdx2_cli_command(command, args, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="mdx2-find-peaks")
    def task_mdx2_find_peaks(command: str, args: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_mdx2_cli_command(command, args, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="mdx2-mask-peaks")
    def task_mdx2_mask_peaks(command: str, args: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_mdx2_cli_command(command, args, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="mdx2-bin-image-series")
    def task_mdx2_bin_image_series(command: str, args: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_mdx2_cli_command(command, args, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="mdx2-integrate")
    def task_mdx2_integrate(command: str, args: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_mdx2_cli_command(command, args, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="mdx2-correct")
    def task_mdx2_correct(command: str, args: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_mdx2_cli_command(command, args, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="mdx2-scale")
    def task_mdx2_scale(command: str, args: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_mdx2_cli_command(command, args, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="mdx2-merge")
    def task_mdx2_merge(command: str, args: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_mdx2_cli_command(command, args, working_dir=working_dir, log_file=log_file)

    @task(persist_result=True, name="mdx2-map")
    def task_mdx2_map(command: str, args: list, working_dir: Optional[str], log_file: Optional[str] = None):
        return run_mdx2_cli_command(command, args, working_dir=working_dir, log_file=log_file)

    @flow(name="single-crystal-workflow", log_prints=True)
    def single_crystal_workflow(
        working_dir: Optional[str] = None,
        raw_data_dir: Optional[str] = None,
        config_file: Optional[str] = None,
        raw_dir: str = "raw_data",
        processed_dir: str = "processed_data",
        run_dials: bool = True,
        crystal_files: Optional[List[str]] = None,
        background_files: Optional[List[str]] = None,
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
        Run the single-crystal workflow: DIALS then mdx2.

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
            cfg_path = Path(config_file)
            if not cfg_path.is_absolute():
                base = (Path(working_dir) if working_dir else Path.cwd()).resolve()
                cfg_path = (base / config_file).resolve()
            config, config_path = load_single_crystal_config_from_file(cfg_path)
            search_dir = config_path.parent
        else:
            search_dir = Path(working_dir) if working_dir else Path.cwd()
            config, config_path = load_single_crystal_config(search_dir)
        if config:
            logger.info("Loaded parameters from %s", config_path)
            working_dir = config.get("working_dir", working_dir)
            raw_data_dir = config.get("raw_data_dir", raw_data_dir)
            raw_dir = config.get("raw_dir", raw_dir)
            processed_dir = config.get("processed_dir", processed_dir)
            run_dials = config.get("run_dials", run_dials)
            crystal_files = config.get("crystal_files", crystal_files)
            background_files = config.get("background_files", background_files)
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
        # Populate crystal_files/background_files from discovered HDF5 masters if empty
        if run_dials and raw_data_dir and config_path and (not crystal_files or not background_files):
            try:
                populated = _populate_task(str(config_path))
                crystal_files = populated.get("crystal_files") or crystal_files
                background_files = populated.get("background_files") or background_files
            except Exception as e:
                logger.warning("populate_deployment_images failed: %s", e)
        # Normalize to list (config may have string for single path)
        def _ensure_list(x): return [x] if isinstance(x, str) else (x or [])
        crystal_files = _ensure_list(crystal_files)
        background_files = _ensure_list(background_files)
        if run_dials:
            if not crystal_files:
                raise ValueError(
                    "crystal_files must be set when run_dials is true. "
                    "Either add crystal_files to deployment.json (e.g. [\"images/insulin_2_1\"]), "
                    "or ensure raw_data_dir exists and contains *_master.h5 for populate_deployment_images to discover."
                )
            if not background_files:
                raise ValueError(
                    "background_files must be set when run_dials is true. "
                    "Either add background_files to deployment.json (e.g. [\"images/insulin_2_bkg\"]), "
                    "or ensure raw_data_dir exists and contains *_bg_*master.h5 for populate_deployment_images to discover."
                )

        wd = working_dir
        base = Path(working_dir) if working_dir else Path.cwd()
        raw_base = Path(raw_data_dir).resolve() if raw_data_dir else base

        # Run directory_setup first when raw_data_dir points to a raw_data-style folder.
        # Use project root (search_dir.parent.parent for processed_data/<crystal>/ layout) so we create
        # raw_data/ and processed_data/ in the writable project tree, not in resolved symlink targets (e.g. /mnt/chess).
        if raw_data_dir:
            setup_base = search_dir.parent.parent
            _directory_setup(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                working_dir=str(setup_base),
            )

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

        # Paths for inputs: when run_dials, DIALS writes to wd; else expect refined.expt/background.expt in raw_base.
        # Use absolute paths; commands run from writable wd_path.
        wd_path = Path(wd).resolve()
        if run_dials:
            refined_path = str(wd_path / refined_expt)
            background_path = str(wd_path / background_expt)
        else:
            refined_path = str((raw_base / refined_expt).resolve())
            background_path = str((raw_base / background_expt).resolve())

        results = []
        # Run DIALS and mdx2 from writable wd. Use absolute paths for file inputs.
        dials_wd = str(wd_path)

        if run_dials:
            logger.info("Running DIALS: import → find_spots → index → refine, then background import")
            crystal_path = str((raw_base / crystal_files[0]).resolve())
            background_path_in = str((raw_base / background_files[0]).resolve())
            # 1. dials.import crystal images → imported.expt
            results.append(task_dials_import(
                ["dials.import", crystal_path], dials_wd,
                log_file=str(wd_path / "01_dials_import.log"),
            ))
            # 2. dials.find_spots → strong.refl
            results.append(task_dials_find_spots(
                ["dials.find_spots", "imported.expt"], dials_wd,
                log_file=str(wd_path / "02_dials_find_spots.log"),
            ))
            # 3. dials.index → indexed.expt, indexed.refl
            results.append(
                task_dials_index(
                    ["dials.index", "imported.expt", "strong.refl", f"space_group={space_group}"],
                    dials_wd,
                    log_file=str(wd_path / "03_dials_index.log"),
                )
            )
            # 4. dials.refine → refined.expt, refined.refl
            results.append(
                task_dials_refine(
                    ["dials.refine", "indexed.expt", "indexed.refl"], dials_wd,
                    log_file=str(wd_path / "04_dials_refine.log"),
                )
            )
            # 5. dials.import background images → background.expt
            results.append(
                task_dials_import_background(
                    ["dials.import", background_path_in, "output.experiments=background.expt"],
                    dials_wd,
                    log_file=str(wd_path / "05_dials_import_background.log"),
                )
            )

        # mdx2 steps: run from wd with absolute paths for file inputs
        wd_str = str(wd_path)
        geom_abs = str(wd_path / "geometry.nxs")
        data_abs = str(wd_path / "data.nxs")
        peaks_abs = str(wd_path / "peaks.nxs")
        mask_abs = str(wd_path / "mask.nxs")
        bkg_data_abs = str(wd_path / "bkg_data.nxs")
        bkg_binned_abs = str(wd_path / "bkg_data_binned.nxs")
        integrated_abs = str(wd_path / "integrated.nxs")
        corrected_abs = str(wd_path / "corrected.nxs")
        scales_abs = str(wd_path / "scales.nxs")
        merged_abs = str(wd_path / "merged.nxs")
        # 1. mdx2.import_data refined.expt --nproc N --datastore datastore → data.nxs
        results.append(
            task_mdx2_import_data(
                "import_data",
                [refined_path, "--nproc", str(nproc), "--datastore", datastore],
                wd_str,
                log_file=str(wd_path / "06_mdx2_import_data.log"),
            )
        )
        # 2. mdx2.import_geometry refined.expt → geometry.nxs
        results.append(task_mdx2_import_geometry(
            "import_geometry", [refined_path], wd_str,
            log_file=str(wd_path / "07_mdx2_import_geometry.log"),
        ))
        # 3. mdx2.find_peaks geometry.nxs data.nxs --nproc N --count_threshold 20 → peaks.nxs
        results.append(
            task_mdx2_find_peaks(
                "find_peaks",
                [geom_abs, data_abs, "--nproc", str(nproc), "--count_threshold", str(count_threshold)],
                wd_str,
                log_file=str(wd_path / "08_mdx2_find_peaks.log"),
            )
        )
        # 4. mdx2.mask_peaks geometry.nxs data.nxs peaks.nxs --nproc N --sigma_cutoff 3 → mask.nxs
        results.append(
            task_mdx2_mask_peaks(
                "mask_peaks",
                [geom_abs, data_abs, peaks_abs, "--nproc", str(nproc), "--sigma_cutoff", str(sigma_cutoff)],
                wd_str,
                log_file=str(wd_path / "09_mdx2_mask_peaks.log"),
            )
        )
        # 5. mdx2.import_data background.expt --outfile bkg_data.nxs --nproc N --datastore datastore_bg
        results.append(
            task_mdx2_import_data(
                "import_data",
                [background_path, "--outfile", "bkg_data.nxs", "--nproc", str(nproc), "--datastore", datastore_bg],
                wd_str,
                log_file=str(wd_path / "10_mdx2_import_data_bg.log"),
            )
        )
        # 6. mdx2.bin_image_series bkg_data.nxs 10 20 20 --valid_range 0 200 --outfile bkg_data_binned.nxs --nproc N
        results.append(
            task_mdx2_bin_image_series(
                "bin_image_series",
                [bkg_data_abs, "10", "20", "20", "--valid_range", "0", "200", "--outfile", "bkg_data_binned.nxs", "--nproc", str(nproc)],
                wd_str,
                log_file=str(wd_path / "11_mdx2_bin_image_series.log"),
            )
        )
        # 7. mdx2.integrate geometry.nxs data.nxs --mask mask.nxs --subdivide ... --nproc N
        results.append(
            task_mdx2_integrate(
                "integrate",
                [geom_abs, data_abs, "--mask", mask_abs, "--subdivide", *integrate_subdivide.split(), "--nproc", str(nproc)],
                wd_str,
                log_file=str(wd_path / "12_mdx2_integrate.log"),
            )
        )
        # 8. mdx2.correct geometry.nxs integrated.nxs --background bkg_data_binned.nxs → corrected.nxs
        results.append(
            task_mdx2_correct(
                "correct",
                [geom_abs, integrated_abs, "--background", bkg_binned_abs],
                wd_str,
                log_file=str(wd_path / "13_mdx2_correct.log"),
            )
        )
        # 9. mdx2.scale corrected.nxs [--mca2020] → scales.nxs
        scale_args = [corrected_abs]
        if mca2020:
            scale_args.append("--mca2020")
        results.append(task_mdx2_scale("scale", scale_args, wd_str, log_file=str(wd_path / "14_mdx2_scale.log")))
        # 10. mdx2.merge corrected.nxs --scale scales.nxs → merged.nxs
        results.append(task_mdx2_merge(
            "merge", [corrected_abs, "--scale", scales_abs], wd_str,
            log_file=str(wd_path / "15_mdx2_merge.log"),
        ))
        # 11. mdx2.map slice: -50 50 -50 50 0 0 → slice.nxs
        results.append(
            task_mdx2_map(
                "map",
                [geom_abs, merged_abs, "--limits", "-50", "50", "-50", "50", "0", "0", "--outfile", "slice.nxs"],
                wd_str,
                log_file=str(wd_path / "16_mdx2_map_slice.log"),
            )
        )
        # 12. mdx2.map offslice: -50 50 -50 50 0.5 0.5 → offslice.nxs
        results.append(
            task_mdx2_map(
                "map",
                [geom_abs, merged_abs, "--limits", "-50", "50", "-50", "50", "0.5", "0.5", "--outfile", "offslice.nxs"],
                wd_str,
                log_file=str(wd_path / "17_mdx2_map_offslice.log"),
            )
        )

        logger.info("Single-crystal workflow finished.")
        return results


def single_crystal_workflow_local(
    working_dir: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
    config_file: Optional[str] = None,
    raw_dir: str = "raw_data",
    processed_dir: str = "processed_data",
    run_dials: bool = True,
    crystal_files: Optional[List[str]] = None,
    background_files: Optional[List[str]] = None,
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
    Run the single-crystal workflow locally without Prefect.
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
        raw_dir = config.get("raw_dir", raw_dir)
        processed_dir = config.get("processed_dir", processed_dir)
        run_dials = config.get("run_dials", run_dials)
        crystal_files = config.get("crystal_files", crystal_files)
        background_files = config.get("background_files", background_files)
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
    # Populate crystal_files/background_files from discovered HDF5 masters if empty
    if run_dials and raw_data_dir and config_path and (not crystal_files or not background_files):
        try:
            populated = _populate_deployment_images(str(config_path))
            crystal_files = populated.get("crystal_files") or crystal_files
            background_files = populated.get("background_files") or background_files
        except Exception as e:
            _log.warning("populate_deployment_images failed: %s", e)
    def _ensure_list(x): return [x] if isinstance(x, str) else (x or [])
    crystal_files = _ensure_list(crystal_files)
    background_files = _ensure_list(background_files)
    if run_dials:
        if not crystal_files:
            raise ValueError(
                "crystal_files must be set when run_dials is true. "
                "Either add crystal_files to deployment.json (e.g. [\"images/insulin_2_1\"]), "
                "or ensure raw_data_dir exists and contains *_master.h5 for populate_deployment_images to discover."
            )
        if not background_files:
            raise ValueError(
                "background_files must be set when run_dials is true. "
                "Either add background_files to deployment.json (e.g. [\"images/insulin_2_bkg\"]), "
                "or ensure raw_data_dir exists and contains *_bg_*master.h5 for populate_deployment_images to discover."
            )

    wd = working_dir
    base = Path(working_dir) if working_dir else Path.cwd()
    raw_base = Path(raw_data_dir).resolve() if raw_data_dir else base

    # Run directory setup first when raw_data_dir points to a raw_data-style folder.
    # Use project root (search_dir.parent.parent) so we use writable raw_data/processed_data, not resolved symlink targets.
    if raw_data_dir:
        setup_base = search_dir.parent.parent
        raw_path = setup_base / raw_dir
        processed_path = setup_base / processed_dir
        _create_processed_dirs(raw_path, processed_path)

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

    wd_path = Path(wd).resolve()
    if run_dials:
        refined_path = str(wd_path / refined_expt)
        background_path = str(wd_path / background_expt)
    else:
        refined_path = str((raw_base / refined_expt).resolve())
        background_path = str((raw_base / background_expt).resolve())

    results: list = []
    dials_wd = str(wd_path)  # Run DIALS and mdx2 from writable wd (raw_base may be read-only)

    def run_conda(argv: list, cwd: Optional[str], log_file: Optional[str] = None):
        return _run_conda_command_local(argv, working_dir=cwd, log_file=log_file)

    def run_mdx2(cmd: str, args: list, cwd: Optional[str], log_file: Optional[str] = None):
        return _run_mdx2_cli_command_local(cmd, args, working_dir=cwd, log_file=log_file)

    if run_dials:
        crystal_path = str((raw_base / crystal_files[0]).resolve())
        background_path_in = str((raw_base / background_files[0]).resolve())
        _log.info("Running DIALS: import → find_spots → index → refine, then background import")
        results.append(run_conda(["dials.import", crystal_path], dials_wd, str(wd_path / "01_dials_import.log")))
        results.append(run_conda(["dials.find_spots", "imported.expt"], dials_wd, str(wd_path / "02_dials_find_spots.log")))
        results.append(
            run_conda(
                ["dials.index", "imported.expt", "strong.refl", f"space_group={space_group}"],
                dials_wd,
                str(wd_path / "03_dials_index.log"),
            )
        )
        results.append(run_conda(
            ["dials.refine", "indexed.expt", "indexed.refl"], dials_wd,
            str(wd_path / "04_dials_refine.log"),
        ))
        results.append(
            run_conda(
                ["dials.import", background_path_in, "output.experiments=background.expt"],
                dials_wd,
                str(wd_path / "05_dials_import_background.log"),
            )
        )

    wd_str = str(wd_path)
    geom_abs = str(wd_path / "geometry.nxs")
    data_abs = str(wd_path / "data.nxs")
    peaks_abs = str(wd_path / "peaks.nxs")
    mask_abs = str(wd_path / "mask.nxs")
    bkg_data_abs = str(wd_path / "bkg_data.nxs")
    bkg_binned_abs = str(wd_path / "bkg_data_binned.nxs")
    integrated_abs = str(wd_path / "integrated.nxs")
    corrected_abs = str(wd_path / "corrected.nxs")
    scales_abs = str(wd_path / "scales.nxs")
    merged_abs = str(wd_path / "merged.nxs")
    results.append(run_mdx2("import_data", [refined_path, "--nproc", str(nproc), "--datastore", datastore], wd_str, str(wd_path / "06_mdx2_import_data.log")))
    results.append(run_mdx2("import_geometry", [refined_path], wd_str, str(wd_path / "07_mdx2_import_geometry.log")))
    results.append(run_mdx2("find_peaks", [geom_abs, data_abs, "--nproc", str(nproc), "--count_threshold", str(count_threshold)], wd_str, str(wd_path / "08_mdx2_find_peaks.log")))
    results.append(run_mdx2("mask_peaks", [geom_abs, data_abs, peaks_abs, "--nproc", str(nproc), "--sigma_cutoff", str(sigma_cutoff)], wd_str, str(wd_path / "09_mdx2_mask_peaks.log")))
    results.append(run_mdx2("import_data", [background_path, "--outfile", "bkg_data.nxs", "--nproc", str(nproc), "--datastore", datastore_bg], wd_str, str(wd_path / "10_mdx2_import_data_bg.log")))
    results.append(run_mdx2("bin_image_series", [bkg_data_abs, "10", "20", "20", "--valid_range", "0", "200", "--outfile", "bkg_data_binned.nxs", "--nproc", str(nproc)], wd_str, str(wd_path / "11_mdx2_bin_image_series.log")))
    results.append(run_mdx2("integrate", [geom_abs, data_abs, "--mask", mask_abs, "--subdivide", *integrate_subdivide.split(), "--nproc", str(nproc)], wd_str, str(wd_path / "12_mdx2_integrate.log")))
    results.append(run_mdx2("correct", [geom_abs, integrated_abs, "--background", bkg_binned_abs], wd_str, str(wd_path / "13_mdx2_correct.log")))
    scale_args = [corrected_abs]
    if mca2020:
        scale_args.append("--mca2020")
    results.append(run_mdx2("scale", scale_args, wd_str, str(wd_path / "14_mdx2_scale.log")))
    results.append(run_mdx2("merge", [corrected_abs, "--scale", scales_abs], wd_str, str(wd_path / "15_mdx2_merge.log")))
    results.append(run_mdx2("map", [geom_abs, merged_abs, "--limits", "-50", "50", "-50", "50", "0", "0", "--outfile", "slice.nxs"], wd_str, str(wd_path / "16_mdx2_map_slice.log")))
    results.append(run_mdx2("map", [geom_abs, merged_abs, "--limits", "-50", "50", "-50", "50", "0.5", "0.5", "--outfile", "offslice.nxs"], wd_str, str(wd_path / "17_mdx2_map_offslice.log")))

    _log.info("Single-crystal workflow finished (local mode).")
    return results


# When Prefect is not available, expose local workflow so "single_crystal_workflow" is always callable
if not PREFECT_AVAILABLE:
    single_crystal_workflow = single_crystal_workflow_local  # type: ignore[misc, assignment]


def main() -> None:
    """CLI entry point: parse --file, --working_dir, --mode; run Prefect flow or local pipeline."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Run the single-crystal workflow. Use --mode prefect for Prefect flow, --mode local to run without Prefect.",
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

    # Resolve relative config file path against working_dir so it works when cwd differs (e.g. in Docker/Prefect)
    config_file = args.file
    if config_file and not Path(config_file).is_absolute():
        base = Path(working_dir).resolve()
        config_file = str((base / config_file).resolve())

    use_prefect = args.mode == "prefect" or (args.mode == "auto" and PREFECT_AVAILABLE)

    if args.mode == "prefect" and not PREFECT_AVAILABLE:
        print("Prefect is not installed or could not be imported. Use --mode local to run the pipeline without Prefect.")
        sys.exit(1)

    if use_prefect:
        api_url = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")
        os.environ["PREFECT_API_URL"] = api_url
        print(f"Connecting to Prefect API at: {api_url}")
        print(f"Running single-crystal workflow (working_dir={working_dir})")
        if config_file:
            print(f"Config file: {config_file}")
        single_crystal_workflow(working_dir=working_dir, config_file=config_file)
    else:
        if args.mode == "auto":
            print("Running in local mode (Prefect not available).")
        print(f"Running single-crystal workflow locally (working_dir={working_dir})")
        if config_file:
            print(f"Config file: {config_file}")
        single_crystal_workflow_local(working_dir=working_dir, config_file=config_file)


if __name__ == "__main__":
    main()
