"""Configure the global logger through ``luxonis_ml``."""

import lightning.pytorch as pl
import numpy as np
import pydantic
import torch
from luxonis_ml.typing import PathType
from luxonis_ml.utils import setup_logging as ml_setup_logging


def setup_logging(
    *, file: PathType | None = None, use_rich: bool = True
) -> None:
    """Set up the global logger for the package.

    The function calls ``luxonis_ml.utils.setup_logging`` with ``file``
    and ``use_rich``. That function does these steps:

    - It reads the log level from the ``LOG_LEVEL`` environment
      variable. It raises a ``ValueError`` for an invalid level.
    - It removes the existing ``loguru`` handlers.
    - It adds a console handler. This handler skips a log record with
      a ``file_only`` extra, for example a record from
      ``logger.bind(file_only=True)``.
    - It adds a file handler when ``file`` is not ``None``. This
      handler also writes the records with a ``file_only`` extra.
    - It sends Python warnings to the logger.
    - It installs an exception hook that prints a summary of an
      uncaught ``pydantic`` validation error. The environment variable
      ``LUXONISML_DISABLE_PRETTY_VALIDATION_ERRORS`` turns the hook off.

    When ``use_rich`` is ``True``, the rich tracebacks of logged
    exceptions show no source code for the frames of
    ``lightning.pytorch``, ``torch``, ``pydantic``, and ``numpy``. They
    still show the file and the line of these frames.

    The package calls this function when it loads its modules. The
    config calls it again when it validates ``rich_logging``.
    `LuxonisModel` calls it with the log file of the run.

    Args:
        file (``PathType | None``): The path of the log file. The logger
            appends to the file. ``None`` writes no log file.
        use_rich (bool): When ``True``, write the console output with
            a rich handler to ``stdout``. When ``False``, write plain log
            lines to ``stderr``.

    """
    ml_setup_logging(
        file=file,
        use_rich=use_rich,
        tracebacks_suppress=[pl, torch, pydantic, np],
    )
