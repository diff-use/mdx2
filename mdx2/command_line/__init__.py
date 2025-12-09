"""Common utilities for command-line tools"""

import sys
import time
from datetime import datetime
from functools import wraps

from loguru import logger


def log_parallel_backend(parallel):
    """
    Log information about the active joblib Parallel backend.

    Attempts to access backend information via private API. If this fails
    (e.g., due to API changes), logs a warning but does not raise an error.

    Parameters
    ----------
    parallel : joblib.Parallel
        The Parallel instance to inspect
    """
    try:
        if hasattr(parallel, "_backend") and parallel._backend:
            backend_name = parallel._backend.__class__.__name__
        else:
            backend_name = "Unknown"
        logger.info("Using backend: {}, n_jobs: {}", backend_name, parallel.n_jobs)
    except Exception as e:
        logger.warning("Could not determine joblib backend details: {}", e)


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

            # Determine log filename and module name
            if log_filename is None:
                module_name = func.__module__.split(".")[-1]
                logfile = f"mdx2.{module_name}.log"
            else:
                logfile = log_filename
                module_name = func.__module__.split(".")[-1]

            # File format: detailed with full timestamp, NO color tags for plain text
            file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

            # Stderr format: streamlined with fixed-width time and level, WITH colors
            stderr_format = "<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> | <level>{message}</level>"

            # Add handlers with different formats
            logger.add(logfile, level=log_level, format=file_format, colorize=False)
            logger.add(sys.stderr, level=log_level, format=stderr_format, colorize=True)

            # Log start with full timestamp and module name
            start_time = time.time()
            start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Starting mdx2.{module_name} at {start_datetime}")

            try:
                result = func(args)
                elapsed = time.time() - start_time
                logger.success(f"mdx2.{module_name} completed in {elapsed:.2f} seconds")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"mdx2.{module_name} failed after {elapsed:.2f} seconds: {e}")
                raise

        return wrapper

    return decorator
