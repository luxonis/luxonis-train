from lightning.pytorch.loggers.logger import Logger
from luxonis_ml.tracker import LuxonisTracker


class LuxonisTrackerPL(LuxonisTracker, Logger):
    """Implementation of LuxonisTracker that is compatible with PytorchLightning."""

    ...
