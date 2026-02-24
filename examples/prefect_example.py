"""
Example script showing how to use Prefect flows to run mdx2 CLI commands.

This demonstrates:
1. Running a single CLI command as a Prefect task
2. Running a pipeline of multiple commands
3. Using convenience flows for common operations
"""

from mdx2.command_line.prefect_flows import (
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
