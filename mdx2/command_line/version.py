"""
Print the version number
"""

import os
import sys
from dataclasses import dataclass

import mdx2
from mdx2.command_line import make_argument_parser, with_parsing


@dataclass
class Parameters:
    pass


parse_arguments = make_argument_parser(Parameters, __doc__)


@with_parsing(parse_arguments)
def run(params):
    print("mdx2:", mdx2.__version__)
    print("Python {0.major}.{0.minor}.{0.micro}".format(sys.version_info))
    print(f"Installed in: {os.path.split(mdx2.__file__)[0]}")


if __name__ == "__main__":
    run()
