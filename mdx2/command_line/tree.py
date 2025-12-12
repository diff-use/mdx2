"""
View the NeXus file tree
"""

from dataclasses import dataclass

from simple_parsing import field

from mdx2.command_line import make_argument_parser, with_parsing
from mdx2.io import nxload


@dataclass
class Parameters:
    filename: str = field(positional=True, help="NeXus file name")


parse_arguments = make_argument_parser(Parameters, __doc__)


@with_parsing(parse_arguments)
def run(params):
    nxs = nxload(params.filename, "r")
    print(f"{params.filename}:", nxs.tree)


if __name__ == "__main__":
    run()
