from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)

from luxonis_train.utils.registry import CALLBACKS

from .archive_on_train_end import ArchiveOnTrainEnd
from .export_on_train_end import ExportOnTrainEnd
from .gpu_stats_monitor import GPUStatsMonitor
from .luxonis_progress_bar import LuxonisProgressBar
from .metadata_logger import MetadataLogger
from .module_freezer import ModuleFreezer
from .test_on_train_end import TestOnTrainEnd
from .upload_checkpoint import UploadCheckpoint

CALLBACKS.register_module(module=EarlyStopping)
CALLBACKS.register_module(module=LearningRateMonitor)
CALLBACKS.register_module(module=ModelCheckpoint)
CALLBACKS.register_module(module=RichModelSummary)
CALLBACKS.register_module(module=DeviceStatsMonitor)


__all__ = [
    "ArchiveOnTrainEnd",
    "ExportOnTrainEnd",
    "LuxonisProgressBar",
    "MetadataLogger",
    "ModuleFreezer",
    "TestOnTrainEnd",
    "UploadCheckpoint",
    "GPUStatsMonitor",
]
