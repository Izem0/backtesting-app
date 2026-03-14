import logging


def setup_logger(
    name: str = "root",
    log_level=logging.INFO,
    format_str=None,
    date_fmt=None,
    handlers=None,
) -> logging.Logger:
    """
    Creates a logger with specified configuration.

    Args:
        name (str): The name of the logger.
        log_level (int, optional): The logging level. Defaults to logging.INFO.
        format_str (str, optional): The format string for the logger.
        handlers (list, optional): A list of handlers to add to the logger.

    Returns:
        logging.Logger: The created logger.
    """
    if handlers is None:
        handlers = []

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not format_str:
        format_str = (
            "%(name)s | "
            "[%(asctime)s] "
            "{%(filename)s:%(lineno)d} "
            "%(levelname)s - "
            "%(message)s"
        )
    if not date_fmt:
        date_fmt = "%Y-%m-%dT%H:%M:%S%z"

    formatter = logging.Formatter(format_str, datefmt=date_fmt)

    if not handlers:
        # Create a default console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
