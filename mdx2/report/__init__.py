"""report generation utilities"""

import os
import platform
import subprocess
from datetime import datetime
from importlib import resources

import papermill as pm

import mdx2  # for version info


def _get_default_metadata():
    """assign defaults to notebook metadata fields, return as a dict"""
    # First, try to get author using various methods
    # 1. if on linux, use the "getent" command to get the full name of the user
    # 2. if on mac, use the "id -F" command to get the full name of the user
    # 3. if the above fail, use the USER environment variable
    author = None
    try:
        if platform.system() == "Linux":
            author = subprocess.check_output(["getent", "passwd", os.getlogin()]).decode().split(":")[4].split(",")[0]
        elif platform.system() == "Darwin":
            author = subprocess.check_output(["id", "-F"]).decode().strip()
    except Exception:
        pass
    if not author:
        author = os.environ.get("USER", None)

    # Next, get the current date and time using python's datetime module,
    # formatted in a human-readable way (e.g. "2024-06-01 12:00:00")
    date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # get the mdx2 version using mdx2.__version__
    version = mdx2.__version__

    # now, get the environment (i,e. hostname)
    environment = platform.node()

    # get the current working directory
    working_directory = os.getcwd()

    return {
        "author": author,
        "date_created": date_created,
        "mdx2_version": version,
        "environment": environment,
        "working_directory": working_directory,
    }


def execute_notebook(template_name, output_path=None, parameters={}, **kwargs):
    """execute a notebook template and save the result to output_path

    Parameters
    ----------
    template_name : str
        name of the notebook template to execute. This should be the name of a .ipynb file in mdx2.report.templates
    output_path : str, optional
        path to save the executed notebook to. If None, the template name will be used (with .ipynb extension)
    parameters : dict, optional
        dictionary of parameters to pass to the notebook template.
    **kwargs
        additional keyword arguments to pass to papermill.execute_notebook.

    See https://papermill.readthedocs.io/en/latest/api.html#papermill.execute_notebook for more details

    Returns
    -------
    None
    """

    if output_path is None:
        output_path = f"{template_name}.ipynb"

    default_metadata = _get_default_metadata()

    # allow user parameters to override default metadata
    parameters = {**default_metadata, **parameters, "notebook_template": template_name}

    # get the path to the notebook template
    template_resource = resources.files("mdx2.report").joinpath("templates", f"{template_name}.ipynb")
    with resources.as_file(template_resource) as template_path:
        pm.execute_notebook(input_path=template_path, output_path=output_path, parameters=parameters, **kwargs)
