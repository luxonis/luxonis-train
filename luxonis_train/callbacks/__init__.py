"""Callbacks that run at points in the training loop.

An entry of ``trainer.callbacks`` in the config names a callback of the
`CALLBACKS` registry. Its ``params`` go to the constructor. The run
does not build an entry whose ``active`` is false.

A run gets these callbacks without an entry in the config:

- `GracefulInterruptCallback` and `FailOnNoTrainBatches`.
- `LuxonisRichProgressBar` when ``rich_logging`` is true, and
  `LuxonisTQDMProgressBar` when it is false.
- `TrainingManager` and `LuxonisModelSummary`.
- A ``ModelCheckpoint`` on the lowest validation loss, and a second one
  on the main metric when the config has one.
- `AIMETCallback` when ``exporter.aimet.active`` is true.
- A ``GradientAccumulationScheduler`` when
  ``trainer.accumulate_grad_batches`` is set and no such callback is in
  the list.

With ``trainer.smart_cfg_auto_populate``, the config also adds the
checkpoint upload, testing, and conversion callbacks when they are
missing. `ConvertOnTrainEnd` replaces separate export and archive
callbacks when it is configured explicitly.

"""

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

from .aimet_callback import AIMETCallback
from .archive_on_train_end import ArchiveOnTrainEnd
from .convert_on_train_end import ConvertOnTrainEnd
from .ema import EMACallback
from .export_on_train_end import ExportOnTrainEnd
from .fail_on_no_train_batches import FailOnNoTrainBatches
from .gpu_stats_monitor import GPUStatsMonitor
from .graceful_interrupt import GracefulInterruptCallback
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
from .training_progress_callback import TrainingProgressCallback
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
CALLBACKS.register(module=GracefulInterruptCallback)
CALLBACKS.register(module=TrainingProgressCallback)
CALLBACKS.register(module=ConvertOnTrainEnd)
CALLBACKS.register(module=FailOnNoTrainBatches)


__all__ = [
    "AIMETCallback",
    "ArchiveOnTrainEnd",
    "BaseLuxonisProgressBar",
    "ConvertOnTrainEnd",
    "EMACallback",
    "ExportOnTrainEnd",
    "FailOnNoTrainBatches",
    "GPUStatsMonitor",
    "GracefulInterruptCallback",
    "GradCamCallback",
    "LuxonisModelSummary",
    "LuxonisRichProgressBar",
    "LuxonisTQDMProgressBar",
    "MetadataLogger",
    "TestOnTrainEnd",
    "TrainingManager",
    "TrainingProgressCallback",
    "UploadCheckpoint",
]
