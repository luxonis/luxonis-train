# Callbacks

List of all supported callbacks.

## Table Of Contents

- [`PytorchLightning` Callbacks](#pytorchlightning-callbacks)
- [`ExportOnTrainEnd`](#exportontrainend)
- [`ArchiveOnTrainEnd`](#archiveontrainend)
- [`MetadataLogger`](#metadatalogger)
- [`TestOnTrainEnd`](#testontrainend)
- [`UploadCheckpoint`](#uploadcheckpoint)

## `PytorchLightning` Callbacks

List of supported callbacks from `lightning.pytorch`.

- [`GPUStatsMonitor`](https://pytorch-lightning.readthedocs.io/en/1.5.10/api/pytorch_lightning.callbacks.gpu_stats_monitor.html)
- [`DeviceStatsMonitor`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.DeviceStatsMonitor.html#lightning.pytorch.callbacks.DeviceStatsMonitor)
- [`EarlyStopping`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping)
- [`LearningRateMonitor`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html#lightning.pytorch.callbacks.LearningRateMonitor)
- [`ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint)
- [`RichModelSummary`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichModelSummary.html#lightning.pytorch.callbacks.RichModelSummary)
- [`GradientAccumulationScheduler`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.GradientAccumulationScheduler.html#lightning.pytorch.callbacks.GradientAccumulationScheduler)
- [`StochasticWeightAveraging`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.StochasticWeightAveraging.html#lightning.pytorch.callbacks.StochasticWeightAveraging)
- [`Timer`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Timer.html#lightning.pytorch.callbacks.Timer)
- [`ModelPruning`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelPruning.html#lightning.pytorch.callbacks.ModelPruning)

## `ExportOnTrainEnd`

Performs export on train end with best weights.

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

## `MetadataLogger`

Callback that logs training metadata.

Metadata include all defined hyperparameters together with git hashes of `luxonis-ml` and `luxonis-train` packages. Also stores this information locally.

**Parameters:**

| Key           | Type        | Default value | Description                                                                                                                |
| ------------- | ----------- | ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `hyperparams` | `list[str]` | `[]`          | List of hyperparameters to log. The hyperparameters are provided as config keys in dot notation. _E.g._ `"trainer.epochs"` |

## `TestOnTrainEnd`

Callback to perform a test run at the end of the training.

## `UploadCheckpoint`

Callback that uploads currently the best checkpoint (based on validation loss) to the tracker location - where all other logs are stored.

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

Callback that updates the stored parameters using a moving average.

**Parameters:**

| Key                 | Type    | Default value | Description                                                                                     |
| ------------------- | ------- | ------------- | ----------------------------------------------------------------------------------------------- |
| `decay`             | `float` | `0.5`         | Decay factor for the moving average.                                                            |
| `use_dynamic_decay` | `bool`  | `True`        | Whether to use dynamic decay.                                                                   |
| `decay_tau`         | `float` | `2000`        | Decay tau for dynamic decay.                                                                    |
| `device`            | `str`   | `None`        | Device to use for the moving average. If `None`, the device is inferred from the model's device |
