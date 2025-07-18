import lightning.pytorch as pl
import numpy as np
import pydantic
import torch
from luxonis_ml.typing import PathType
from luxonis_ml.utils import setup_logging as ml_setup_logging


def setup_logging(
    *, file: PathType | None = None, use_rich: bool = True
) -> None:
    ml_setup_logging(
        file=file,
        use_rich=use_rich,
        tracebacks_suppress=[pl, torch, pydantic, np],
    )
