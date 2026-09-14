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

When ``trainer.smart_cfg_auto_populate`` is true, the config also adds
`UploadCheckpoint`, `TestOnTrainEnd`, and `ConvertOnTrainEnd` to
``trainer.callbacks`` when they are missing.

The registry holds every callback above. It also holds
`ArchiveOnTrainEnd`, `EMACallback`, `ExportOnTrainEnd`,
`GPUStatsMonitor`, `GradCamCallback`, `MetadataLogger`, and
`TrainingProgressCallback`. It also holds these ``lightning.pytorch``
callbacks: ``DeviceStatsMonitor``, ``EarlyStopping``,
``LearningRateMonitor``, ``ModelPruning``,
``StochasticWeightAveraging``, and ``Timer``.

Lightning calls the callbacks in this order:

1. `GracefulInterruptCallback`, `FailOnNoTrainBatches`, and the progress
   bar.
2. `TrainingManager`, `LuxonisModelSummary`, and `AIMETCallback`.
3. The entries of ``trainer.callbacks``, in the order of the config.
   The config moves `EMACallback` to the front.
4. The ``GradientAccumulationScheduler``.
5. Each ``ModelCheckpoint``.

`ConvertOnTrainEnd` exports, archives, and converts in one step. Prefer
it over a separate `ExportOnTrainEnd` and `ArchiveOnTrainEnd`. When
``trainer.callbacks`` lists an active `ConvertOnTrainEnd`, the config
deactivates the other two. The check runs before
``trainer.smart_cfg_auto_populate`` adds `ConvertOnTrainEnd`, so an
added one leaves them active.

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
