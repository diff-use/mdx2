"""Common utilities for command-line tools"""

import sys
from functools import wraps

from loguru import logger


def with_logging(log_filename=None, log_level="INFO"):
    """
    Decorator to set up logging for command-line tools.

    Args:
        log_filename: Name of log file. If None, uses module name.
        log_level: Logging level (default: INFO)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(args=None):
            # Remove default handler
            logger.remove()

            # Determine log filename
            if log_filename is None:
                module_name = func.__module__.split(".")[-1]
                logfile = f"mdx2.{module_name}.log"
            else:
                logfile = log_filename

            # Add file and stderr handlers
            logger.add(logfile, level=log_level)
            logger.add(sys.stderr, level=log_level)

            try:
                result = func(args)
                logger.success("done")
                return result
            except Exception as e:
                logger.error(f"Error: {e}")
                raise

        return wrapper

    return decorator
