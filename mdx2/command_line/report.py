"""
Generate jupyter notebook reports.
"""

from dataclasses import asdict, dataclass
from typing import ClassVar, Optional, Union

from simple_parsing import field
from simple_parsing.helpers.fields import subparsers

from mdx2.command_line import make_argument_parser, with_parsing
from mdx2.report import execute_notebook

# first, I need to define dataclass parameters for each template, for use with simple-parsing


@dataclass
class Metadata:
    """metadata fields that can be overriden on the CLI (see mdx2.report._get_default_metadata)

    All fields are optional, and if not provided, will be populated by mdx2.report with default values.
    """

    author: Optional[str] = None
    date_created: Optional[str] = None
    mdx2_version: Optional[str] = None
    environment: Optional[str] = None
    working_directory: Optional[str] = None


@dataclass
class ExecutableNotebook:
    """base class for executable notebooks."""

    _template_name: ClassVar[str] = field(init=False)  # name of the notebook template to execute, set by subclasses

    def execute(self, **kwargs):
        """execute the report generation using the provided parameters"""
        execute_notebook(
            template_name=self._template_name,
            parameters=asdict(self),
            **kwargs,
        )


@dataclass
class TemplateParameters(ExecutableNotebook):
    """dummy template for testing purposes"""

    _template_name = "_template"

    input_files: list[str]  # list of input file paths, required
    model_names: Optional[list[str]] = (
        None  # optional list of model names, overriding defaults defined in _template.ipynb
    )
    pi: Optional[float] = None  # optional numerical value of pi, overriding default defined in _template.ipynb


@dataclass
class VisualizationParameters(ExecutableNotebook):
    """parameters for the visualization.ipynb template"""

    _template_name = "visualization"

    geom: str  # NeXus file containing symmetry and crystal
    hkl: str  # NeXus file containing hkl_table


@dataclass
class ScalingModelParameters(ExecutableNotebook):
    """parameters for the visualization.ipynb template"""

    _template_name = "scaling_model"

    input_files: list[str]  # list of input file paths, required
    model_names: Optional[list[str]] = (
        None  # optional list of model names, overriding defaults defined in scaling_model.ipynb
    )
    shared_detector_model: Optional[bool] = None  # set to False to show separate detector models for each input file.


@dataclass
class Parameters:
    """parameters for the report generation"""

    report: Union[TemplateParameters, ScalingModelParameters] = subparsers(
        {p._template_name: p for p in [TemplateParameters, ScalingModelParameters]},
    )
    metadata: Metadata = Metadata()  # metadata fields that can be overridden on the CLI


def run_report(params):
    """main function to run the report generation"""
    params.report.execute(metadata=asdict(params.metadata))


# NOTE: parse_arguments is imported by the testing framework
parse_arguments = make_argument_parser(Parameters, __doc__)

# NOTE: run is the main entry point for the command line script
run = with_parsing(parse_arguments)(run_report)


if __name__ == "__main__":
    run()
