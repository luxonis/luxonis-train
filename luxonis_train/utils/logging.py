import lightning.pytorch as pl
import numpy as np
import pydantic
import torch
from luxonis_ml.typing import PathType
from luxonis_ml.utils import setup_logging as ml_setup_logging


def setup_logging(*, file: PathType | None = None) -> None:
    ml_setup_logging(file=file, tracebacks_suppress=[pl, torch, pydantic, np])
