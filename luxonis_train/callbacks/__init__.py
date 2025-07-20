from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    GradientAccumulationScheduler,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelPruning,
    StochasticWeightAveraging,
    Timer,
)

from luxonis_train.registry import CALLBACKS

from .archive_on_train_end import ArchiveOnTrainEnd
from .ema import EMACallback
from .export_on_train_end import ExportOnTrainEnd
from .gpu_stats_monitor import GPUStatsMonitor
from .gradcam_visualizer import GradCamCallback
from .luxonis_model_summary import LuxonisModelSummary
from .luxonis_progress_bar import (
    BaseLuxonisProgressBar,
    LuxonisRichProgressBar,
    LuxonisTQDMProgressBar,
)
from .metadata_logger import MetadataLogger
from .test_on_train_end import TestOnTrainEnd
from .training_manager import TrainingManager
from .upload_checkpoint import UploadCheckpoint

CALLBACKS.register(module=EarlyStopping)
CALLBACKS.register(module=LearningRateMonitor)
CALLBACKS.register(module=ModelCheckpoint)
CALLBACKS.register(module=LuxonisModelSummary)
CALLBACKS.register(module=DeviceStatsMonitor)
CALLBACKS.register(module=GradientAccumulationScheduler)
CALLBACKS.register(module=StochasticWeightAveraging)
CALLBACKS.register(module=Timer)
CALLBACKS.register(module=ModelPruning)
CALLBACKS.register(module=GradCamCallback)
CALLBACKS.register(module=EMACallback)
CALLBACKS.register(module=TrainingManager)


__all__ = [
    "ArchiveOnTrainEnd",
    "BaseLuxonisProgressBar",
    "EMACallback",
    "ExportOnTrainEnd",
    "GPUStatsMonitor",
    "GradCamCallback",
    "LuxonisModelSummary",
    "LuxonisRichProgressBar",
    "LuxonisTQDMProgressBar",
    "MetadataLogger",
    "TestOnTrainEnd",
    "TrainingManager",
    "UploadCheckpoint",
]
