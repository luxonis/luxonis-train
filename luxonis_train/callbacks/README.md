# Callbacks

List of all supported callbacks.

**Note**: The order of callbacks being called is determined by the order in which they are defined in the configuration file.

## Table Of Contents

- [`PytorchLightning` Callbacks](#pytorchlightning-callbacks)
- [`GracefulInterruptCallback`](#gracefulinterruptcallback)
- [`FailOnNoTrainBatches`](#failonnotrainbatches)
- [`LuxonisModelSummary`](#luxonismodelsummary)
- [`LuxonisRichProgressBar`](#luxonisrichprogressbar)
- [`LuxonisTQDMProgressBar`](#luxonistqdmprogressbar)
- [`TrainingManager`](#trainingmanager)
- [`ExportOnTrainEnd`](#exportontrainend)
- [`ArchiveOnTrainEnd`](#archiveontrainend)
- [`ConvertOnTrainEnd`](#convertontrainend)
- [`MetadataLogger`](#metadatalogger)
- [`TestOnTrainEnd`](#testontrainend)
- [`UploadCheckpoint`](#uploadcheckpoint)
- [`GradCamCallback`](#gradcamcallback)
- [`EMACallback`](#emacallback)
- [`TrainingProgressCallback`](#trainingprogresscallback)

## `PytorchLightning` Callbacks

List of supported callbacks from `lightning.pytorch`.

- [`GPUStatsMonitor`](https://pytorch-lightning.readthedocs.io/en/1.5.10/api/pytorch_lightning.callbacks.gpu_stats_monitor.html): Monitors and logs GPU stats during training.
- [`DeviceStatsMonitor`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.DeviceStatsMonitor.html#lightning.pytorch.callbacks.DeviceStatsMonitor): Monitors and logs device stats (CPU/GPU) during training.
- [`EarlyStopping`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping): Stops training when a monitored metric has stopped improving.
- [`LearningRateMonitor`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html#lightning.pytorch.callbacks.LearningRateMonitor): Logs learning rate during training.
- [`ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint): Saves model checkpoints during training.
- [`RichModelSummary`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichModelSummary.html#lightning.pytorch.callbacks.RichModelSummary): Provides a detailed summary of the model.
- [`GradientAccumulationScheduler`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.GradientAccumulationScheduler.html#lightning.pytorch.callbacks.GradientAccumulationScheduler): Adjusts gradient accumulation dynamically during training.
- [`StochasticWeightAveraging`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.StochasticWeightAveraging.html#lightning.pytorch.callbacks.StochasticWeightAveraging): Averages model weights to improve generalization.
- [`Timer`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Timer.html#lightning.pytorch.callbacks.Timer): Times training, validation, and test loops.
- [`ModelPruning`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelPruning.html#lightning.pytorch.callbacks.ModelPruning): Prunes model weights during training.

**Default**: `ModelCheckpoint` is added automatically (minimum validation loss, and best main metric if configured). `GradientAccumulationScheduler` is added automatically when `trainer.accumulate_grad_batches` is set and no scheduler is configured.

## `GracefulInterruptCallback`

Handles SIGINT/SIGTERM gracefully. On the first interrupt it saves a `resume.ckpt`, stops training, and skips remaining train-end callbacks. On the second interrupt it exits immediately.

**Default**: Added automatically.

**Parameters:**

| Key        | Type                       | Default value | Description                                               |
| ---------- | -------------------------- | ------------- | --------------------------------------------------------- |
| `save_dir` | `Path`                     | -             | Directory where `resume.ckpt` will be stored.             |
| `tracker`  | `LuxonisTrackerPL \| None` | `None`        | Optional tracker used to upload the interrupt checkpoint. |

## `FailOnNoTrainBatches`

Raises an error if Lightning computes zero training batches after data setup. The error message includes the current dataset size and the minimum required size based on `batch_size`, `world_size`, `drop_last`, and `limit_train_batches`.

**Default**: Added automatically.

## `LuxonisModelSummary`

Custom model summary based on `RichModelSummary`. It prints a rich table to the console when `rich_logging=True`, and a plain table otherwise. A copy is also logged to the log file.

**Default**: Added automatically with `max_depth=2`, and `rich` set from `cfg.rich_logging`.

**Parameters:**

| Key    | Type   | Default value | Description                                                    |
| ------ | ------ | ------------- | -------------------------------------------------------------- |
| `rich` | `bool` | `True`        | Enables rich rendering. Falls back to plain text when `False`. |

## `LuxonisRichProgressBar`

Rich progress bar used when `rich_logging=True`. It prints metrics in rich tables and mirrors output to the log file.

**Default**: Added automatically when `rich_logging=True`.

## `LuxonisTQDMProgressBar`

TQDM progress bar used when `rich_logging=False`. It prints metrics as tables in logs.

**Default**: Added automatically when `rich_logging=False`.

## `TrainingManager`

Manages frozen nodes and training strategies. It freezes configured nodes before training, unfreezes them at the specified epoch, and updates strategy parameters after backprop.

**Default**: Added automatically.

## `ExportOnTrainEnd`

Performs export on train end with best weights.
This callback only exports to ONNX; it does not run HubAI or blobconverter conversions.

**Parameters:**

| Key                    | Type                        | Default value | Description                                                                                                                                                     |
| ---------------------- | --------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `preferred_checkpoint` | `Literal["metric", "loss"]` | `"metric"`    | Which checkpoint should the callback use. If the preferred checkpoint is not available, the other option is used. If none is available, the callback is skipped |

## `ArchiveOnTrainEnd`

Callback to create an `NN Archive` at the end of the training.

**Parameters:**

| Key                    | Type                        | Default value | Description                                                                                                                                                     |
| ---------------------- | --------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `preferred_checkpoint` | `Literal["metric", "loss"]` | `"metric"`    | Which checkpoint should the callback use. If the preferred checkpoint is not available, the other option is used. If none is available, the callback is skipped |

## `ConvertOnTrainEnd`

Unified callback that exports, archives, and converts the archive to the target platform at the end of training. This is the recommended callback for model conversion as it combines the functionality of `ExportOnTrainEnd` and `ArchiveOnTrainEnd`, and also runs platform-specific conversions (blobconverter or HubAI SDK) if configured.

**Default**: Added automatically when `smart_cfg_auto_populate=True` (default).

**Steps:**

<ol>
  <li>Exports the model to ONNX</li>
  <li>Creates an NN Archive from the ONNX</li>
  <li>Runs blobconverter if `exporter.blobconverter.active` is `true`</li>
  <li>Runs HubAI SDK conversion if `exporter.hubai.active` is `true`</li>
</ol>

**Parameters:**

| Key                    | Type                        | Default value | Description                                                                                                                                                     |
| ---------------------- | --------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `preferred_checkpoint` | `Literal["metric", "loss"]` | `"metric"`    | Which checkpoint should the callback use. If the preferred checkpoint is not available, the other option is used. If none is available, the callback is skipped |

## `MetadataLogger`

Callback that logs training metadata.

Metadata include all defined hyperparameters together with git hashes of `luxonis-ml` and `luxonis-train` packages. Also stores this information locally.

**Parameters:**

| Key           | Type        | Default value | Description                                                                                                                |
| ------------- | ----------- | ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `hyperparams` | `list[str]` | `[]`          | List of hyperparameters to log. The hyperparameters are provided as config keys in dot notation. _E.g._ `"trainer.epochs"` |

## `TestOnTrainEnd`

Callback to perform a test run at the end of the training.

**Default**: Added automatically when `smart_cfg_auto_populate=True` (default).

**Parameters:**

| Key    | Type                              | Default value | Description                     |
| ------ | --------------------------------- | ------------- | ------------------------------- |
| `view` | `Literal["train", "val", "test"]` | `"test"`      | Which view to use for the test. |

## `UploadCheckpoint`

Callback that uploads currently the best checkpoint (based on validation loss) to the tracker location - where all other logs are stored.

**Default**: Added automatically when `smart_cfg_auto_populate=True` (default).

## `GradCamCallback`

Callback to visualize gradients using Grad-CAM. Works only during validation.

**Parameters:**

| Key             | Type  | Default value      | Description                                          |
| --------------- | ----- | ------------------ | ---------------------------------------------------- |
| `target_layer`  | `int` | -                  | Layer to visualize gradients.                        |
| `class_idx`     | `int` | 0                  | Index of the class for visualization. Defaults to 0. |
| `log_n_batches` | `int` | 1                  | Number of batches to log. Defaults to 1.             |
| `task`          | `str` | `"classification"` | The type of task. Defaults to "classification".      |

## `EMACallback`

A callback that maintains an exponential moving average (EMA) of the model's parameters. The EMA weights are employed during validation, testing, visualizations, and for exporting the model, ensuring the deployed model benefits from smoother, more stable parameters.

**Parameters:**

| Key                 | Type    | Default value | Description                                                                        |
| ------------------- | ------- | ------------- | ---------------------------------------------------------------------------------- |
| `decay`             | `float` | `0.5`         | The decay factor for updating the EMA. Higher values yield slower updates.         |
| `use_dynamic_decay` | `bool`  | `True`        | If enabled, adjusts the decay factor dynamically based on the training iteration.  |
| `decay_tau`         | `float` | `2000`        | The time constant (tau) for dynamic decay, influencing how quickly the EMA adapts. |

## `TrainingProgressCallback`

Callback that publishes training progress and timing metrics.

**Parameters:**

| Key                   | Type  | Default value | Description                                                                                                             |
| --------------------- | ----- | ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `log_every_n_batches` | `int` | `1`           | How often to log progress metrics (every N batches). 1 for real-time updates, higher values to reduce logging overhead. |

**Published Metrics:**

| Metric Key                     | Description                                   |
| ------------------------------ | --------------------------------------------- |
| `train/epoch_progress_percent` | Percentage (0-100) of current epoch completed |
| `train/epoch_duration_sec`     | Time elapsed so far in current epoch          |
| `train/epoch_completion_sec`   | Total duration of completed epoch in seconds  |
