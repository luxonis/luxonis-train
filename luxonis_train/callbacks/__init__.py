from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelPruning,
    RichModelSummary,
    StochasticWeightAveraging,
    Timer,
)

from luxonis_train.utils.registry import CALLBACKS

from .archive_on_train_end import ArchiveOnTrainEnd
from .export_on_train_end import ExportOnTrainEnd
from .gpu_stats_monitor import GPUStatsMonitor
from .luxonis_progress_bar import (
    BaseLuxonisProgressBar,
    LuxonisRichProgressBar,
    LuxonisTQDMProgressBar,
)
from .metadata_logger import MetadataLogger
from .module_freezer import ModuleFreezer
from .test_on_train_end import TestOnTrainEnd
from .upload_checkpoint import UploadCheckpoint
from .gradcam_visializer import GradCamCallback

CALLBACKS.register_module(module=EarlyStopping)
CALLBACKS.register_module(module=LearningRateMonitor)
CALLBACKS.register_module(module=ModelCheckpoint)
CALLBACKS.register_module(module=RichModelSummary)
CALLBACKS.register_module(module=DeviceStatsMonitor)
CALLBACKS.register_module(module=GradientAccumulationScheduler)
CALLBACKS.register_module(module=StochasticWeightAveraging)
CALLBACKS.register_module(module=Timer)
CALLBACKS.register_module(module=ModelPruning)
CALLBACKS.register_module(module=GradCamCallback)

__all__ = [
    "ArchiveOnTrainEnd",
    "ExportOnTrainEnd",
    "LuxonisTQDMProgressBar",
    "LuxonisRichProgressBar",
    "BaseLuxonisProgressBar",
    "MetadataLogger",
    "ModuleFreezer",
    "TestOnTrainEnd",
    "UploadCheckpoint",
    "GPUStatsMonitor",
    "GradCamCallback",
]
