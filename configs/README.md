# Configuration

The configuration is defined in a `YAML` file, which you must provide.
The configuration file consists of a few major blocks that are described below.
You can create your own config or use/edit one of the examples.

## Table Of Contents

- [Top-level Options](#top-level-options)
- [Model](#model)
  - [Nodes](#nodes)
    - [Losses](#losses)
    - [Metrics](#metrics)
    - [Visualizers](#visualizers)
- [Tracker](#tracker)
- [Loader](#loader)
  - [`LuxonisLoaderTorch`](#luxonisloadertorch)
- [Trainer](#trainer)
  - [Preprocessing](#preprocessing)
    - [Augmentations](#augmentations)
  - [Callbacks](#callbacks)
  - [Optimizer](#optimizer)
  - [Scheduler](#scheduler)
- [Exporter](#exporter)
  - [`ONNX`](#onnx)
  - [Blob](#blob)
- [Tuner](#tuner)
  - [Storage](#storage)
- [ENVIRON](#environ)

## Top-level Options

| Key        | Type                    | Description      |
| ---------- | ----------------------- | ---------------- |
| `model`    | [`model`](#model)       | Model section    |
| `loader`   | [`loader`](#loader)     | Loader section   |
| `train`    | [`train`](#train)       | Train section    |
| `tracker`  | [`tracker`](#tracker)   | Tracker section  |
| `trainer`  | [`trainer`](#trainer)   | Trainer section  |
| `exporter` | [`exporter`](#exporter) | Exporter section |
| `tuner`    | [`tuner`](#tuner)       | Tuner section    |

## Model

This is the most important block, that **must be always defined by the user**. There are two different ways you can create the model.

| Key                | Type   | Default value | Description                                                |
| ------------------ | ------ | ------------- | ---------------------------------------------------------- |
| `name`             | `str`  | `"model"`     | Name of the model                                          |
| `weights`          | `path` | `None`        | Path to weights to load                                    |
| `predefined_model` | `str`  | `None`        | Name of a predefined model to use                          |
| `params`           | `dict` | `{}`          | Parameters for the predefined model                        |
| `nodes`            | `list` | `[]`          | List of nodes (see [nodes](#nodes))                        |
| `losses`           | `list` | `[]`          | List of losses (see [losses](#losses))                     |
| `metrics`          | `list` | `[]`          | List of metrics (see [metrics](#metrics))                  |
| `visualziers`      | `list` | `[]`          | List of visualizers (see [visualizers](#visualizers))      |
| `outputs`          | `list` | `[]`          | List of outputs nodes, inferred from nodes if not provided |

### Nodes

For list of all nodes, see [nodes](../luxonis_train/nodes/README.md).

| Key                       | Type                   | Default value | Description                                                                                                                                 |
| ------------------------- | ---------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                    | `str`                  | -             | Name of the node                                                                                                                            |
| `alias`                   | `str`                  | `None`        | Custom name for the node                                                                                                                    |
| `params`                  | `dict`                 | `{}`          | Parameters for the node                                                                                                                     |
| `inputs`                  | `list`                 | `[]`          | List of input nodes for this node, if empty, the node is understood to be an input node of the model                                        |
| `freezing.active`         | `bool`                 | `False`       | whether to freeze the modules so the weights are not updated                                                                                |
| `freezing.unfreeze_after` | `int \| float \| None` | `None`        | After how many epochs should the modules be unfrozen, can be `int` for a specific number of epochs or `float` for a portion of the training |
| `remove_on_export`        | `bool`                 | `False`       | Whether the node should be removed when exporting                                                                                           |
| `losses`                  | `list`                 | `[]`          | List of losses attached to this node                                                                                                        |
| `metrics`                 | `list`                 | `[]`          | List of metrics attached to this node                                                                                                       |
| `visualizers`             | `list`                 | `[]`          | List of visualizers attached to this node                                                                                                   |

#### Losses

At least one node must have a loss attached to it.
You can see the list of all currently supported loss functions and their parameters [here](../luxonis_train/attached_modules/losses/README.md).

| Key      | Type    | Default value | Description                              |
| -------- | ------- | ------------- | ---------------------------------------- |
| `weight` | `float` | `1.0`         | Weight of the loss used in the final sum |
| `alias`  | `str`   | `None`        | Custom name for the loss                 |
| `params` | `dict`  | `{}`          | Additional parameters for the loss       |

#### Metrics

In this section, you configure which metrics should be used for which node.
You can see the list of all currently supported metrics and their parameters [here](../luxonis_train/attached_modules/metrics/README.md).

| Key              | Type   | Default value | Description                                                                            |
| ---------------- | ------ | ------------- | -------------------------------------------------------------------------------------- |
| `is_main_metric` | `bool` | `False`       | Marks this specific metric as the main one. Main metric is used for saving checkpoints |
| `alias`          | `str`  | `None`        | Custom name for the metric                                                             |
| `params`         | `dict` | `{}`          | Additional parameters for the metric                                                   |

#### Visualizers

In this section, you configure which visualizers should be used for which node. Visualizers are responsible for creating images during training.
You can see the list of all currently supported visualizers and their parameters [here](../luxonis_train/attached_modules/visualizers/README.md).

| Key      | Type   | Default value | Description                              |
| -------- | ------ | ------------- | ---------------------------------------- |
| `alias`  | `str`  | `None`        | Custom name for the visualizer           |
| `params` | `dict` | `{}`          | Additional parameters for the visualizer |

**Example:**

```yaml
name: "SegmentationHead"
inputs:
  - "RepPANNeck"
losses:
  - name: "BCEWithLogitsLoss"
metrics:
  - name: "F1Score"
    params:
      task: "binary"
  - name: "JaccardIndex"
    params:
      task: "binary"
visualizers:
  - name: "SegmentationVisualizer"
    params:
      colors: "#FF5055"
```

## Tracker

This library uses [`LuxonisTrackerPL`](https://github.com/luxonis/luxonis-ml/blob/b2399335efa914ef142b1b1a5db52ad90985c539/src/luxonis_ml/ops/tracker.py#L152).
You can configure it like this:

| Key              | Type          | Default value | Description                                                |
| ---------------- | ------------- | ------------- | ---------------------------------------------------------- |
| `project_name`   | `str \| None` | `None`        | Name of the project used for logging                       |
| `project_id`     | `str \| None` | `None`        | ID of the project used for logging (relevant for `MLFlow`) |
| `run_name`       | `str \| None` | `None`        | Name of the run. If empty, then it will be auto-generated  |
| `run_id`         | `str \| None` | `None`        | ID of an already created run (relevant for `MLFLow`)       |
| `save_directory` | `str`         | `"output"`    | Path to the save directory                                 |
| `is_tensorboard` | `bool`        | `True`        | Whether to use `Tensorboard`                               |
| `is_wandb`       | `bool`        | `False`       | Whether to use `WandB`                                     |
| `wandb_entity`   | `str \| None` | `None`        | Name of `WandB` entity                                     |
| `is_mlflow`      | `bool`        | `False`       | Whether to use `MLFlow`                                    |

**Example:**

```yaml
tracker:
  project_name: "project_name"
  save_directory: "output"
  is_tensorboard: true
  is_wandb: false
  is_mlflow: false
```

## Loader

This section controls the data loading process and parameters regarding the dataset.

To store and load the data we use `LuxonisDataset` and `LuxonisLoader.` For specific config parameters refer to [`LuxonisML`](https://github.com/luxonis/luxonis-ml).

| Key            | Type               | Default value          | Description                          |
| -------------- | ------------------ | ---------------------- | ------------------------------------ |
| `name`         | `str`              | `"LuxonisLoaderTorch"` | Name of the Loader                   |
| `image_source` | `str`              | `"image"`              | Name of the input image group        |
| `train_view`   | `str \| list[str]` | `"train"`              | splits to use for training           |
| `val_view`     | `str \| list[str]` | `"val"`                | splits to use for validation         |
| `test_view`    | `str \| list[str]` | `"test"`               | splits to use for testing            |
| `params`       | `dict[str, Any]`   | `{}`                   | Additional parameters for the loader |

### `LuxonisLoaderTorch`

By default, `LuxonisLoaderTorch` can either use an existing `LuxonisDataset` or create a new one if it can be parsed automatically by `LuxonisParser` (check [`LuxonisML`](https://github.com/luxonis/luxonis-ml) `data` sub-package for more info).

In most cases you want to set one of the parameters below. You can check all the parameters in the `LuxonisLoaderTorch` class itself.

| Key            | Type  | Default value | Description                                                          |
| -------------- | ----- | ------------- | -------------------------------------------------------------------- |
| `dataset_name` | `str` | `None`        | Name of an existing `LuxonisDataset`                                 |
| `dataset_dir`  | `str` | `None`        | Location of the data from which new `LuxonisDataset` will be created |

**Example:**

```yaml
loader:
  # using default loader with an existing dataset
  params:
    dataset_name: "dataset_name"
```

```yaml
loader:
  # using default loader with a directory
  params:
    dataset_name: "dataset_name"
    dataset_dir: "path/to/dataset"
```

## Trainer

Here you can change everything related to actual training of the model.

| Key                       | Type                                           | Default value | Description                                                                                                                                      |
| ------------------------- | ---------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `seed`                    | `int`                                          | `None`        | Seed for reproducibility                                                                                                                         |
| `deterministic`           | `bool \| "warn" \| None`                       | `None`        | Whether PyTorch should use deterministic backend                                                                                                 |
| `batch_size`              | `int`                                          | `32`          | Batch size used for training                                                                                                                     |
| `accumulate_grad_batches` | `int`                                          | `1`           | Number of batches for gradient accumulation                                                                                                      |
| `use_weighted_sampler`    | `bool`                                         | `False`       | Whether to use `WeightedRandomSampler` for training, only works with classification tasks                                                        |
| `epochs`                  | `int`                                          | `100`         | Number of training epochs                                                                                                                        |
| `n_workers`               | `int`                                          | `4`           | Number of workers for data loading                                                                                                               |
| `validation_interval`     | `int`                                          | `5`           | Frequency of computing metrics on validation data                                                                                                |
| `n_log_images`            | `int`                                          | `4`           | Maximum number of images to visualize and log                                                                                                    |
| `skip_last_batch`         | `bool`                                         | `True`        | Whether to skip last batch while training                                                                                                        |
| `accelerator`             | `Literal["auto", "cpu", "gpu"]`                | `"auto"`      | What accelerator to use for training                                                                                                             |
| `devices`                 | `int \| list[int] \| str`                      | `"auto"`      | Either specify how many devices to use (int), list specific devices, or use "auto" for automatic configuration based on the selected accelerator |
| `matmul_precision`        | `Literal["medium", "high", "highest"] \| None` | `None`        | Sets the internal precision of float32 matrix multiplications                                                                                    |
| `strategy`                | `Literal["auto", "ddp"]`                       | `"auto"`      | What strategy to use for training                                                                                                                |
| `n_sanity_val_steps`      | `int`                                          | `2`           | Number of sanity validation steps performed before training                                                                                      |
| `profiler`                | `Literal["simple", "advanced"] \| None`        | `None`        | PL profiler for GPU/CPU/RAM utilization analysis                                                                                                 |
| `verbose`                 | `bool`                                         | `True`        | Print all intermediate results to console                                                                                                        |
| `pin_memory`              | `bool`                                         | `True`        | Whether to pin memory in the `DataLoader`                                                                                                        |
| `save_top_k`              | `-1 \| NonNegativeInt`                         | `3`           | Save top K checkpoints based on validation loss when training                                                                                    |

**Example:**

```yaml

trainer:
  accelerator: "auto"
  devices: "auto"
  strategy: "auto"

  n_sanity_val_steps: 1
  profiler: null
  verbose: true
  batch_size: 8
  accumulate_grad_batches: 1
  epochs: 200
  n_workers: 8
  validation_interval: 10
  n_log_images: 8
  skip_last_batch: true
  log_sub_losses: true
  save_top_k: 3
```

### Preprocessing

We use [`Albumentations`](https://albumentations.ai/docs/) library for `augmentations`. [Here](https://albumentations.ai/docs/api_reference/full_reference/#pixel-level-transforms) you can see a list of all pixel level augmentations supported, and [here](https://albumentations.ai/docs/api_reference/full_reference/#spatial-level-transforms) you see all spatial level transformations. In the configuration you can specify any augmentation from these lists and their parameters.

Additionally, we support `Mosaic4` and `MixUp` batch augmentations and letterbox resizing if `keep_aspect_ratio: true`.

| Key                 | Type         | Default value | Description                                                                                                                                                             |
| ------------------- | ------------ | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `train_image_size`  | `list[int]`  | `[256, 256]`  | Image size used for training as `[height, width]`                                                                                                                       |
| `keep_aspect_ratio` | `bool`       | `True`        | Whether to keep the aspect ratio while resizing                                                                                                                         |
| `train_rgb`         | `bool`       | `True`        | Whether to train on RGB or BGR images                                                                                                                                   |
| `normalize.active`  | `bool`       | `True`        | Whether to use normalization                                                                                                                                            |
| `normalize.params`  | `dict`       | `{}`          | Parameters for normalization, see [Normalize](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize) |
| `augmentations`     | `list[dict]` | `[]`          | List of `Albumentations` augmentations                                                                                                                                  |

#### Augmentations

| Key      | Type   | Default value | Description                        |
| -------- | ------ | ------------- | ---------------------------------- |
| `name`   | `str`  | -             | Name of the augmentation           |
| `active` | `bool` | `True`        | Whether the augmentation is active |
| `params` | `dict` | `{}`          | Parameters of the augmentation     |

**Example:**

```yaml

trainer:
  preprocessing:
    # using YAML capture to reuse the image size
    train_image_size: [&height 384, &width 384]
    keep_aspect_ratio: true
    train_rgb: true
    normalize:
      active: true
    augmentations:
      - name: "Defocus"
        params:
          p: 0.1
      - name: "Sharpen"
        params:
          p: 0.1
      - name: "Flip"
      - name: "RandomRotate90"
      - name: "Mosaic4"
        params:
          out_width: *width
          out_height: *height

```

### Callbacks

Callbacks sections contain a list of callbacks.
More information on callbacks and a list of available ones can be found [here](../luxonis_train/callbacks/README.md)
Each callback is a dictionary with the following fields:

| Key      | Type   | Default value | Description                |
| -------- | ------ | ------------- | -------------------------- |
| `name`   | `str`  | -             | Name of the callback       |
| `active` | `bool` | `True`        | Whether callback is active |
| `params` | `dict` | `{}`          | Parameters of the callback |

**Example:**

```yaml

trainer:
  callbacks:
    - name: "LearningRateMonitor"
      params:
        logging_interval: "step"
    - name: MetadataLogger
      params:
        hyperparams: ["trainer.epochs", "trainer.batch_size"]
    - name: "EarlyStopping"
      params:
        patience: 3
        monitor: "val/loss"
        mode: "min"
        verbose: true
    - name: "ExportOnTrainEnd"
    - name: "TestOnTrainEnd"
```

### Optimizer

What optimizer to use for training.
List of all optimizers can be found [here](https://pytorch.org/docs/stable/optim.html).

| Key      | Type   | Default value | Description                 |
| -------- | ------ | ------------- | --------------------------- |
| `name`   | `str`  | `"Adam"`      | Name of the optimizer       |
| `params` | `dict` | `{}`          | Parameters of the optimizer |

**Example:**

```yaml
optimizer:
  name: "SGD"
  params:
    lr: 0.02
    momentum: 0.937
    nesterov: true
    weight_decay: 0.0005
```

### Scheduler

What scheduler to use for training.
List of all optimizers can be found [here](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).

| Key      | Type   | Default value  | Description                 |
| -------- | ------ | -------------- | --------------------------- |
| `name`   | `str`  | `"ConstantLR"` | Name of the scheduler       |
| `params` | `dict` | `{}`           | Parameters of the scheduler |

**Example:**

```yaml
trainer:
  scheduler:
    name: "CosineAnnealingLR"
    params:
      T_max: *epochs
      eta_min: 0
```

## Exporter

Here you can define configuration for exporting.

| Key                      | Type                              | Default value | Description                                                                                    |
| ------------------------ | --------------------------------- | ------------- | ---------------------------------------------------------------------------------------------- |
| `name`                   | `str \| None`                     | `None`        | Name of the exported model                                                                     |
| `input_shape`            | `list\[int\] \| None`             | `None`        | Input shape of the model. If not provided, inferred from the dataset                           |
| `data_type`              | `Literal["INT8", "FP16", "FP32"]` | `"FP16"`      | Data type of the exported model. Only used for conversion to BLOB                              |
| `reverse_input_channels` | `bool`                            | `True`        | Whether to reverse the image channels in the exported model. Relevant for `BLOB` export        |
| `scale_values`           | `list[float] \| None`             | `None`        | What scale values to use for input normalization. If not provided, inferred from augmentations |
| `mean_values`            | `list[float] \| None`             | `None`        | What mean values to use for input normalization. If not provided, inferred from augmentations  |
| `upload_to_run`          | `bool`                            | `True`        | Whether to upload the exported files to tracked run as artifact                                |
| `upload_url`             | `str \| None`                     | `None`        | Exported model will be uploaded to this URL if specified                                       |
| `output_names`           | `list[str] \| None`               | `None`        | Optional list of output names to override the default ones (deprecated)                        |

### `ONNX`

Option specific for `ONNX` export.

| Key             | Type                     | Default value | Description                       |
| --------------- | ------------------------ | ------------- | --------------------------------- |
| `opset_version` | `int`                    | `12`          | Which `ONNX` opset version to use |
| `dynamic_axes`  | `dict[str, Any] \| None` | `None`        | Whether to specify dynamic axes   |

### Blob

| Key       | Type                                                             | Default value | Description                              |
| --------- | ---------------------------------------------------------------- | ------------- | ---------------------------------------- |
| `active`  | `bool`                                                           | `False`       | Whether to export to `BLOB` format       |
| `shaves`  | `int`                                                            | `6`           | How many shaves                          |
| `version` | `Literal["2021.2", "2021.3", "2021.4", "2022.1", "2022.3_RVC3"]` | `"2022.1"`    | `OpenVINO` version to use for conversion |

**Example:**

```yaml
exporter:
  onnx:
    opset_version: 11
  blobconverter:
    active: true
    shaves: 8
```

## Tuner

Here you can specify options for tuning.

| Key                      | Type              | Default value  | Description                                                                                                                                                                                                                                                                                                                 |
| ------------------------ | ----------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `study_name`             | `str`             | `"test-study"` | Name of the study                                                                                                                                                                                                                                                                                                           |
| `continue_exising_study` | `bool`            | `True`         | Whether to continue an existing study with this name                                                                                                                                                                                                                                                                        |
| `use_pruner`             | `bool`            | `True`         | Whether to use the `MedianPruner`                                                                                                                                                                                                                                                                                           |
| `n_trials`               | `int \| None`     | `15`           | Number of trials for each process. `None` represents no limit in terms of number of trials                                                                                                                                                                                                                                  |
| `timeout`                | `int \| None`     | `None`         | Stop study after the given number of seconds                                                                                                                                                                                                                                                                                |
| `params`                 | `dict[str, list]` | `{}`           | Which parameters to tune. The keys should be in the format `key1.key2.key3_<type>`. Type can be one of `[categorical, float, int, longuniform, uniform, subset]`. For more information about the types, visit [`Optuna` documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html) |

> \[!NOTE\]
> `"subset"` sampling is currently only supported for augmentations.
> You can specify a set of augmentations defined in `trainer` to choose from.
> Every run, only a subset of random $N$ augmentations will be active (`is_active` parameter will be `True` for chosen ones and `False` for the rest in the set).

### Storage

| Key            | Type                         | Default value | Description                                         |
| -------------- | ---------------------------- | ------------- | --------------------------------------------------- |
| `active`       | `bool`                       | `True`        | Whether to use storage to make the study persistent |
| `storage_type` | `Literal["local", "remote"]` | `"local"`     | Type of the storage                                 |

**Example:**

```yaml
t uner:
  study_name: "seg_study"
  n_trials: 10
  storage:
    storage_type: "local"
  params:
    trainer.optimizer.name_categorical: ["Adam", "SGD"]
    trainer.optimizer.params.lr_float: [0.0001, 0.001]
    trainer.batch_size_int: [4, 16, 4]
    # each run will have 2 of the following augmentations active
    trainer.preprocessing.augmentations_subset: [["Defocus", "Sharpen", "Flip"], 2]
```

## ENVIRON

A special section of the config file where you can specify environment variables.
For more info on the variables, see [Credentials](../README.md#credentials).

> \[!WARNING\]
> This is not a recommended way due to possible leakage of secrets!
> This section is intended for testing purposes only!
> Use environment variables or `.env` files instead.

| Key                        | Type                                                       | Default value    |
| -------------------------- | ---------------------------------------------------------- | ---------------- |
| `AWS_ACCESS_KEY_ID`        | `str \| None`                                              | `None`           |
| `AWS_SECRET_ACCESS_KEY`    | `str \| None`                                              | `None`           |
| `AWS_S3_ENDPOINT_URL`      | `str \| None`                                              | `None`           |
| `MLFLOW_CLOUDFLARE_ID`     | `str \| None`                                              | `None`           |
| `MLFLOW_CLOUDFLARE_SECRET` | `str \| None`                                              | `None`           |
| `MLFLOW_S3_BUCKET`         | `str \| None`                                              | `None`           |
| `MLFLOW_S3_ENDPOINT_URL`   | `str \| None`                                              | `None`           |
| `MLFLOW_TRACKING_URI`      | `str \| None`                                              | `None`           |
| `POSTGRES_USER`            | `str \| None`                                              | `None`           |
| `POSTGRES_PASSWORD`        | `str \| None`                                              | `None`           |
| `POSTGRES_HOST`            | `str \| None`                                              | `None`           |
| `POSTGRES_PORT`            | `str \| None`                                              | `None`           |
| `POSTGRES_DB`              | `str \| None`                                              | `None`           |
| `LUXONISML_BUCKET`         | `str \| None`                                              | `None`           |
| `LUXONISML_BASE_PATH`      | `str`                                                      | `"~/luxonis_ml"` |
| `LUXONISML_TEAM_ID`        | `str`                                                      | `"offline"`      |
| `LOG_LEVEL`                | `Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]` | `"INFO"`         |
