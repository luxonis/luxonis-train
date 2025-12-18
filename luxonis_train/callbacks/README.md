# Callbacks

List of all supported callbacks.

**Note**: The order of callbacks being called is determined by the order in which they are defined in the configuration file.

## Table Of Contents

- [`PytorchLightning` Callbacks](#pytorchlightning-callbacks)
- [`LuxonisBatchSizeFinder`](#batchsizefinder)
- [`ExportOnTrainEnd`](#exportontrainend)
- [`ArchiveOnTrainEnd`](#archiveontrainend)
- [`MetadataLogger`](#metadatalogger)
- [`TestOnTrainEnd`](#testontrainend)
- [`UploadCheckpoint`](#uploadcheckpoint)
- [`GradCamCallback`](#gradcamcallback)
- [`EMACallback`](#emacallback)

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

## `LuxonisBatchSizeFinder`

Automatically finds the largest batch size that fits in memory without causing out-of-memory (OOM) errors. This callback extends [PyTorch Lightning's LuxonisBatchSizeFinder](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BatchSizeFinder.html) to also update the training config with the optimal batch size found.

When the optimal batch size is found:
- If the found batch size is **higher** than the configured batch size, the batch size is automatically updated to account for the hardware being able to handle larger batches.
- If the found batch size is **lower** than the configured batch size, the batch size is automatically updated to prevent OOM errors.
- The updated batch size is persisted to the training config file.

**Note**: This is an [experimental](https://lightning.ai/docs/pytorch/stable/versioning.html#experimental-api) feature from PyTorch Lightning.

**Parameters:**

| Key               | Type                           | Default value  | Description                                                                                                        |
| ----------------- | ------------------------------ | -------------- | ------------------------------------------------------------------------------------------------------------------ |
| `mode`            | `Literal["power", "binsearch"]`| `"power"`      | Search strategy. `"power"` increases batch size by powers of 2. `"binsearch"` performs binary search after OOM.    |
| `steps_per_trial` | `int`                          | `3`            | Number of steps to run for each batch size trial.                                                                  |
| `init_val`        | `int`                          | `2`            | Initial batch size to start the search from.                                                                       |
| `max_trials`      | `int`                          | `25`           | Maximum number of trials to run before terminating.                                                                |
| `batch_arg_name`  | `str`                          | `"batch_size"` | Name of the batch size attribute in the data loader.                                                               |

**Example Configuration:**

```yaml
trainer:
  batch_size: 2  # Start with a small batch size
  callbacks:
    - name: LuxonisBatchSizeFinder
      params:
        mode: power
        steps_per_trial: 3
        init_val: 2
        max_trials: 25
```

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

**Parameters:**

| Key    | Type                              | Default value | Description                     |
| ------ | --------------------------------- | ------------- | ------------------------------- |
| `view` | `Literal["train", "val", "test"]` | \`"test"      | Which view to use for the test. |

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

A callback that maintains an exponential moving average (EMA) of the model's parameters. The EMA weights are employed during validation, testing, visualizations, and for exporting the model, ensuring the deployed model benefits from smoother, more stable parameters.

**Parameters:**

| Key                 | Type    | Default value | Description                                                                        |
| ------------------- | ------- | ------------- | ---------------------------------------------------------------------------------- |
| `decay`             | `float` | `0.5`         | The decay factor for updating the EMA. Higher values yield slower updates.         |
| `use_dynamic_decay` | `bool`  | `True`        | If enabled, adjusts the decay factor dynamically based on the training iteration.  |
| `decay_tau`         | `float` | `2000`        | The time constant (tau) for dynamic decay, influencing how quickly the EMA adapts. |
