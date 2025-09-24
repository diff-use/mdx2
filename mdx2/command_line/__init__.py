import logging


def configure_logging(filename=None, level=logging.INFO):
    """Configure logging to console and optionally to a file."""

    handlers = []

    if filename is not None:
        # configure the logger
        file_handler = logging.FileHandler(filename)
        file_formatter = logging.Formatter("%(asctime)s %(module)s %(levelname)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        handlers.append(file_handler)

    # Console handler: simple format (message only)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(level=level, handlers=handlers)
