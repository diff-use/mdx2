"""
Prefect flows for running mdx2 CLI commands as orchestrated tasks.

This module provides Prefect flows and tasks that wrap mdx2 command-line tools,
making it easy to run pipeline commands with workflow orchestration.
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from prefect import flow, task
from prefect.logging import get_run_logger


def _resolve_raw_entries(raw_dir: Path) -> List[Tuple[str, Path]]:
    """Return list of (name, resolved_path) for each symlink or subdir in raw_dir."""
    entries: List[Tuple[str, Path]] = []
    if not raw_dir.is_dir():
        return entries
    for p in sorted(raw_dir.iterdir()):
        if p.name.startswith("."):
            continue
        if p.is_symlink():
            target = p.resolve()
            if target.is_dir():
                entries.append((p.name, target))
        elif p.is_dir():
            entries.append((p.name, p.resolve()))
    return entries


@task(name="run-mdx2-command", log_prints=True)
def run_mdx2_cli_command(
    command: str,
    args: List[str],
    working_dir: Optional[str] = None,
    conda_env: str = "mdx2-dev",
) -> subprocess.CompletedProcess:
    """
    Run an mdx2 CLI command as a Prefect task.
    
    Args:
        command: The mdx2 command to run (e.g., 'map', 'integrate', 'scale')
        args: List of command-line arguments to pass to the command
        working_dir: Working directory for the command (default: current directory)
        conda_env: Conda environment name to activate (default: 'mdx2-dev')
    
    Returns:
        CompletedProcess object with returncode, stdout, stderr
    
    Example:
        >>> result = run_mdx2_cli_command(
        ...     command="map",
        ...     args=["geom.nxs", "hkl.nxs", "--outfile", "map.nxs"]
        ... )
    """
    logger = get_run_logger()
    
    # Build the command: activate conda env and run the module
    cmd = [
        "micromamba", "run", "-n", conda_env,
        "python", "-m", f"mdx2.command_line.{command}"
    ] + args
    
    logger.info(f"Running command: {' '.join(cmd)}")
    if working_dir:
        logger.info(f"Working directory: {working_dir}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=False,  # Don't raise on non-zero exit, we'll handle it
        )
        
        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"STDERR:\n{result.stderr}")
        
        if result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        
        logger.success(f"Command '{command}' completed successfully")
        return result
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command execution failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error running command: {e}")
        raise


@task(name="run-conda-command", log_prints=True)
def run_conda_command(
    argv: List[str],
    working_dir: Optional[str] = None,
    conda_env: str = "mdx2-dev",
) -> subprocess.CompletedProcess:
    """
    Run an arbitrary command in the conda environment (e.g. DIALS commands).
    argv: e.g. ["dials.import", "images/insulin_2_1"]
    """
    logger = get_run_logger()
    cmd = ["micromamba", "run", "-n", conda_env] + argv
    logger.info("Running: %s (cwd=%s)", " ".join(cmd), working_dir)
    try:
        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            logger.info("STDOUT:\n%s", result.stdout)
        if result.stderr:
            logger.warning("STDERR:\n%s", result.stderr)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr
            )
        return result
    except subprocess.CalledProcessError as e:
        logger.error("Command failed: %s", e)
        raise


@flow(name="mdx2-pipeline", log_prints=True)
def custom_workflow(
    commands: List[dict],
    working_dir: Optional[str] = None,
    conda_env: str = "mdx2-dev",
) -> List[subprocess.CompletedProcess]:
    """
    Run a sequence of mdx2 CLI commands as a Prefect flow.
    
    Args:
        commands: List of command dictionaries, each with:
            - 'command': CLI command name (e.g., 'integrate', 'scale', 'map')
            - 'args': List of arguments for the command
        working_dir: Working directory for all commands
        conda_env: Conda environment name
    
    Returns:
        List of CompletedProcess objects, one per command
    
    Example:
        >>> results = mdx2_pipeline_flow([
        ...     {
        ...         "command": "integrate",
        ...         "args": ["geom.nxs", "data.nxs", "--outfile", "integrated.nxs"]
        ...     },
        ...     {
        ...         "command": "scale",
        ...         "args": ["integrated.nxs", "--outfile", "scaled.nxs"]
        ...     },
        ...     {
        ...         "command": "map",
        ...         "args": ["geom.nxs", "scaled.nxs", "--outfile", "map.nxs"]
        ...     }
        ... ])
    """
    logger = get_run_logger()
    logger.info(f"Starting mdx2 pipeline with {len(commands)} commands")
    
    results = []
    for i, cmd_config in enumerate(commands, 1):
        command = cmd_config["command"]
        args = cmd_config.get("args", [])
        
        logger.info(f"Step {i}/{len(commands)}: Running '{command}'")
        
        result = run_mdx2_cli_command(
            command=command,
            args=args,
            working_dir=working_dir,
            conda_env=conda_env,
        )
        results.append(result)
    
    logger.success(f"Pipeline completed successfully with {len(results)} commands")
    return results


# Public alias for the pipeline flow
mdx2_pipeline_flow = custom_workflow


# Convenience flows for common single-command operations
@flow(name="mdx2-map", log_prints=True)
def map_flow(
    geom: str,
    hkl: str,
    outfile: str = "map.nxs",
    symmetry: bool = True,
    limits: tuple = (0, 10, 0, 10, 0, 10),
    signal: str = "intensity",
    working_dir: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Prefect flow for running mdx2 map command."""
    args = [geom, hkl, "--outfile", outfile]
    if not symmetry:
        args.append("--no-symmetry")
    if limits != (0, 10, 0, 10, 0, 10):
        args.extend(["--limits"] + [str(x) for x in limits])
    if signal != "intensity":
        args.extend(["--signal", signal])
    
    return run_mdx2_cli_command("map", args, working_dir=working_dir)


@flow(name="mdx2-integrate", log_prints=True)
def integrate_flow(
    geom: str,
    data: str,
    outfile: str = "integrated.nxs",
    mask: Optional[str] = None,
    subdivide: tuple = (1, 1, 1),
    max_spread: float = 1.0,
    nproc: int = 1,
    working_dir: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Prefect flow for running mdx2 integrate command."""
    args = [geom, data, "--outfile", outfile]
    if mask:
        args.extend(["--mask", mask])
    if subdivide != (1, 1, 1):
        args.extend(["--subdivide"] + [str(x) for x in subdivide])
    if max_spread != 1.0:
        args.extend(["--max-spread", str(max_spread)])
    if nproc != 1:
        args.extend(["--nproc", str(nproc)])
    
    return run_mdx2_cli_command("integrate", args, working_dir=working_dir)


@flow(name="mdx2-scale", log_prints=True)
def scale_flow(
    hkl: str,
    outfile: str = "scaled.nxs",
    working_dir: Optional[str] = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Prefect flow for running mdx2 scale command."""
    args = [hkl, "--outfile", outfile]
    # Add any additional kwargs as command-line arguments
    for key, value in kwargs.items():
        if value is not None:
            args.extend([f"--{key.replace('_', '-')}", str(value)])
    
    return run_mdx2_cli_command("scale", args, working_dir=working_dir)


@flow(name="process-raw-data", log_prints=True)
def process_raw_data_flow(
    raw_dir: str = "raw_data",
    processed_dir: str = "processed_data",
    geom_file: str = "geometry.nxs",
    data_file: str = "data.nxs",
    mask_file: Optional[str] = None,
    working_dir: Optional[str] = None,
    nproc: int = 1,
    fail_fast: bool = False,
) -> List[subprocess.CompletedProcess]:
    """
    Run the raw_data -> processed_data pipeline for all symlinks in raw_data.
    For each entry: creates processed_data/<name>/ and runs integrate -> scale -> map.
    """
    logger = get_run_logger()
    base = Path(working_dir) if working_dir else Path.cwd()
    raw_path = (base / raw_dir).resolve()
    processed_path = (base / processed_dir).resolve()

    if not raw_path.is_dir():
        raise FileNotFoundError(f"raw_data directory not found: {raw_path}")

    entries = _resolve_raw_entries(raw_path)
    if not entries:
        logger.warning("No symlinks or subdirectories found in %s", raw_path)
        return []

    logger.info("Found %s entries in raw_data: %s", len(entries), [e[0] for e in entries])
    all_results: List[subprocess.CompletedProcess] = []

    for name, resolved_path in entries:
        out_path = processed_path / name
        try:
            out_path.mkdir(parents=True, exist_ok=True)
            geom = str(resolved_path / geom_file)
            data = str(resolved_path / data_file)
            integrated = str(out_path / "integrated.nxs")
            scaled = str(out_path / "scaled.nxs")
            map_out = str(out_path / "map.nxs")

            if not (Path(geom).exists() and Path(data).exists()):
                raise FileNotFoundError(
                    f"Expected {geom_file} and {data_file} in {resolved_path}"
                )

            integrate_args = [geom, data, "--outfile", integrated, "--nproc", str(nproc)]
            if mask_file and (resolved_path / mask_file).exists():
                integrate_args.extend(["--mask", str(resolved_path / mask_file)])

            all_results.append(
                run_mdx2_cli_command("integrate", integrate_args, working_dir=working_dir)
            )
            all_results.append(
                run_mdx2_cli_command(
                    "scale", [integrated, "--outfile", scaled], working_dir=working_dir
                )
            )
            all_results.append(
                run_mdx2_cli_command(
                    "map", [geom, scaled, "--outfile", map_out], working_dir=working_dir
                )
            )
            logger.success("Pipeline finished for %s -> %s", name, out_path)
        except Exception as e:
            if fail_fast:
                raise
            logger.exception("Failed to process %s: %s", name, e)

    return all_results


if __name__ == "__main__":
    import os
    from prefect import serve

    from mdx2.command_line.pipeline import single_crystal_workflow

    api_url = os.getenv("PREFECT_API_URL", "http://prefect-server:4200/api")
    print(f"Connecting to Prefect API at: {api_url}")

    serve(
        custom_workflow.to_deployment(name="custom-workflow-deployment"),
        process_raw_data_flow.to_deployment(name="process-raw-data-deployment"),
        single_crystal_workflow.to_deployment(name="single-crystal-example-deployment"),
    )
