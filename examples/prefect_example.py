"""
Example script showing how to use Prefect flows to run mdx2 CLI commands.

This demonstrates:
1. Running a single CLI command as a Prefect task
2. Running a pipeline of multiple commands
3. Using convenience flows for common operations

Run from mdx2 repo root, or ensure prefect/ is on PYTHONPATH.
"""

import sys
from pathlib import Path

_prefect_dir = Path(__file__).resolve().parent.parent / "prefect"
if _prefect_dir.is_dir() and str(_prefect_dir) not in sys.path:
    sys.path.insert(0, str(_prefect_dir))

from prefect_flows import (
    integrate_flow,
    map_flow,
    mdx2_pipeline_flow,
    run_mdx2_cli_command,
    scale_flow,
)


def example_single_command():
    """Example: Run a single mdx2 command"""
    print("Example 1: Running a single map command")
    result = run_mdx2_cli_command(
        command="map",
        args=["geom.nxs", "hkl.nxs", "--outfile", "map.nxs", "--signal", "intensity"],
    )
    print(f"Result: {result.returncode}")


def example_convenience_flow():
    """Example: Use convenience flow for map command"""
    print("Example 2: Using convenience map_flow")
    result = map_flow(
        geom="geom.nxs",
        hkl="hkl.nxs",
        outfile="map.nxs",
        symmetry=True,
        signal="intensity",
    )
    print(f"Result: {result.returncode}")


def example_pipeline():
    """Example: Run a pipeline of multiple commands"""
    print("Example 3: Running a pipeline of commands")
    
    commands = [
        {
            "command": "integrate",
            "args": [
                "geom.nxs",
                "data.nxs",
                "--outfile",
                "integrated.nxs",
                "--nproc",
                "4",
            ],
        },
        {
            "command": "scale",
            "args": ["integrated.nxs", "--outfile", "scaled.nxs"],
        },
        {
            "command": "map",
            "args": [
                "geom.nxs",
                "scaled.nxs",
                "--outfile",
                "map.nxs",
                "--signal",
                "intensity",
            ],
        },
    ]
    
    results = mdx2_pipeline_flow(commands)
    print(f"Pipeline completed with {len(results)} commands")


def example_individual_flows():
    """Example: Run individual flows in sequence"""
    print("Example 4: Running individual flows")
    
    # Integrate
    integrate_result = integrate_flow(
        geom="geom.nxs",
        data="data.nxs",
        outfile="integrated.nxs",
        nproc=4,
    )
    
    # Scale
    scale_result = scale_flow(
        hkl="integrated.nxs",
        outfile="scaled.nxs",
    )
    
    # Map
    map_result = map_flow(
        geom="geom.nxs",
        hkl="scaled.nxs",
        outfile="map.nxs",
    )
    
    print("All flows completed")


if __name__ == "__main__":
    # Uncomment the example you want to run:
    # example_single_command()
    # example_convenience_flow()
    # example_pipeline()
    # example_individual_flows()
    
    print("See the function definitions above for examples")
    print("To run, uncomment one of the example functions above")
