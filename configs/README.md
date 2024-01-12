# Configuration

The configuration is defined in a yaml file, which you must provide.
The configuration file consists of a few major blocks that are described below.
You can create your own config or use/edit one of the examples.

## Table Of Contents

- [Top-level Options](#top-level-options)
- [Model](#model)
  - [Nodes](#nodes)
  - [Attached Modules](#attached-modules)
    - [Losses](#losses)
    - [Metrics](#metrics)
    - [Visualizers](#visualizers)
- [Tracker](#tracker)
- [Dataset](#dataset)
- [Trainer](#train)
  - [Preprocessing](#preprocessing)
  - [Optimizer](#optimizer)
  - [Scheduler](#scheduler)
  - [Callbacks](#callbacks)
- [Exporter](#exporter)
  - [ONNX](#onnx)
  - [Blob](#blob)
- [Tuner](#tuner)
  - [Storage](#storage)
- [ENVIRON](#environ)

## Top-level Options

| Key           | Type                  | Default value | Description                                   |
| ------------- | --------------------- | ------------- | --------------------------------------------- |
| use_rich_text | bool                  | True          | whether to use rich text for console printing |
| model         | [Model](#model)       |               | model section                                 |
| dataset       | [dataset](#dataset)   |               | dataset section                               |
| train         | [train](#train)       |               | train section                                 |
| tracker       | [tracker](#tracker)   |               | tracker section                               |
| trainer       | [trainer](#trainer)   |               | trainer section                               |
| exporter      | [exporter](#exporter) |               | exporter section                              |
| tuner         | [tuner](#tuner)       |               | tuner section                                 |

## Model

This is the most important block, that **must be always defined by the user**. There are two different ways you can create the model.

| Key              | Type | Default value | Description                                                |
| ---------------- | ---- | ------------- | ---------------------------------------------------------- |
| name             | str  |               | name of the model                                          |
| weights          | path | None          | path to weights to load                                    |
| predefined_model | str  | None          | name of a predefined model to use                          |
| params           | dict | {}            | parameters for the predefined model                        |
| nodes            | list | \[\]          | list of nodes (see [nodes](#nodes)                         |
| losses           | list | \[\]          | list of losses (see [losses](#losses)                      |
| metrics          | list | \[\]          | list of metrics (see [metrics](#metrics)                   |
| visualziers      | list | \[\]          | list of visualizers (see [visualizers](#visualizers)       |
| outputs          | list | \[\]          | list of outputs nodes, inferred from nodes if not provided |

### Nodes

For list of all nodes, see [nodes](src/luxonis_train/nodes/README.md).

| Key           | Type | Default value | Description                                                                                          |
| ------------- | ---- | ------------- | ---------------------------------------------------------------------------------------------------- |
| name          | str  |               | name of the node                                                                                     |
| override_name | str  | None          | custom name for the node                                                                             |
| params        | dict | {}            | parameters for the node                                                                              |
| inputs        | list | \[\]          | list of input nodes for this node, if empty, the node is understood to be an input node of the model |
| frozen        | bool | False         | whether should the node be trained                                                                   |

### Attached Modules

Modules that are attached to a node. This include losses, metrics and visualziers.

| Key           | Type | Default value | Description                                 |
| ------------- | ---- | ------------- | ------------------------------------------- |
| name          | str  |               | name of the module                          |
| attached_to   | str  |               | Name of the node the module is attached to. |
| override_name | str  | None          | custom name for the module                  |
| params        | dict | {}            | parameters of the module                    |

#### Losses

At least one node must have a loss attached to it.
You can see the list of all currently supported loss functions and their parameters [here](./src/luxonis_train/attached_modules/losses/README.md).

| Key    | Type  | Default value | Description                              |
| ------ | ----- | ------------- | ---------------------------------------- |
| weight | float | 1.0           | weight of the loss used in the final sum |

#### Metrics

In this section, you configure which metrics should be used for which node.
You can see the list of all currently supported metrics and their parameters [here](./src/luxonis_train/attached_modules/metrics/README.md).

| Key            | Type | Default value | Description                                                                             |
| -------------- | ---- | ------------- | --------------------------------------------------------------------------------------- |
| is_main_metric | bool | False         | Marks this specific metric as the main one. Main metric is used for saving checkpoints. |

#### Visualizers

In this section, you configure which visualizers should be used for which node. Visualizers are responsible for creating images during training.
You can see the list of all currently supported visualizers and their parameters [here](./src/luxonis_train/attached_modules/visualizers/README.md).

Visualizers have no specific configuration.

## Tracker

This library uses [LuxonisTrackerPL](https://github.com/luxonis/luxonis-ml/blob/b2399335efa914ef142b1b1a5db52ad90985c539/src/luxonis_ml/ops/tracker.py#L152).
You can configure it like this:

| Key            | Type        | Default value | Description                                                |
| -------------- | ----------- | ------------- | ---------------------------------------------------------- |
| project_name   | str \| None | None          | Name of the project used for logging.                      |
| project_id     | str \| None | None          | Id of the project used for logging (relevant for MLFlow).  |
| run_name       | str \| None | None          | Name of the run. If empty, then it will be auto-generated. |
| run_id         | str \| None | None          | Id of an already created run (relevant for MLFLow.)        |
| save_directory | str         | "output"      | Path to the save directory.                                |
| is_tensorboard | bool        | True          | Whether to use tensorboard.                                |
| is_wandb       | bool        | False         | Whether to use WandB.                                      |
| wandb_entity   | str \| None | None          | Name of WandB entity.                                      |
| is_mlflow      | bool        | False         | Whether to use MLFlow.                                     |

## Dataset

To store and load the data we use LuxonisDataset and LuxonisLoader. For specific config parameters refer to [LuxonisML](https://github.com/luxonis/luxonis-ml).

| Key            | Type                                     | Default value       | Description                                    |
| -------------- | ---------------------------------------- | ------------------- | ---------------------------------------------- |
| dataset_name   | str \| None                              | None                | name of the dataset                            |
| team_id        | str \| None                              | None                | team under which you can find all datasets     |
| dataset_id     | str \| None                              | None                | id of the dataset                              |
| bucket_type    | Literal\["intenal", "external"\]         | internal            | type of underlying storage                     |
| bucket_storage | Literal\["local", "s3", "gcc", "azure"\] | BucketStorage.LOCAL | underlying object storage for a bucket         |
| train_view     | str                                      | train               | view to use for training                       |
| val_view       | str                                      | val                 | view to use for validation                     |
| test_view      | str                                      | test                | view to use for testing                        |
| json_mode      | bool                                     | False               | load using JSON annotations instead of MongoDB |

## Trainer

Here you can change everything related to actual training of the model.

| Key                     | Type                                    | Default value | Description                                                                                                                                      |
| ----------------------- | --------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| batch_size              | int                                     | 32            | batch size used for training                                                                                                                     |
| accumulate_grad_batches | int                                     | 1             | number of batches for gradient accumulation                                                                                                      |
| use_weighted_sampler    | bool                                    | False         | bool if use WeightedRandomSampler for training, only works with classification tasks                                                             |
| epochs                  | int                                     | 100           | number of training epochs                                                                                                                        |
| num_workers             | int                                     | 2             | number of workers for data loading                                                                                                               |
| train_metrics_interval  | int                                     | -1            | frequency of computing metrics on train data, -1 if don't perform                                                                                |
| validation_interval     | int                                     | 1             | frequency of computing metrics on validation data                                                                                                |
| num_log_images          | int                                     | 4             | maximum number of images to visualize and log                                                                                                    |
| skip_last_batch         | bool                                    | True          | whether to skip last batch while training                                                                                                        |
| accelerator             | Literal\["auto", "cpu", "gpu"\]         | "auto"        | What accelerator to use for training.                                                                                                            |
| devices                 | int \| list\[int\] \| str               | "auto"        | Either specify how many devices to use (int), list specific devices, or use "auto" for automatic configuration based on the selected accelerator |
| strategy                | Literal\["auto", "ddp"\]                | "auto"        | What strategy to use for training.                                                                                                               |
| num_sanity_val_steps    | int                                     | 2             | Number of sanity validation steps performed before training.                                                                                     |
| profiler                | Literal\["simple", "advanced"\] \| None | None          | PL profiler for GPU/CPU/RAM utilization analysis                                                                                                 |
| verbose                 | bool                                    | True          | Print all intermediate results to console.                                                                                                       |

### Preprocessing

We use [Albumentations](https://albumentations.ai/docs/) library for `augmentations`. [Here](https://albumentations.ai/docs/api_reference/full_reference/#pixel-level-transforms) you can see a list of all pixel level augmentations supported, and [here](https://albumentations.ai/docs/api_reference/full_reference/#spatial-level-transforms) you see all spatial level transformations. In config you can specify any augmentation from this lists and their params. Additionaly we support `Mosaic4` batch augmentation and letterbox resizing if `keep_aspect_ratio: True`.

| Key               | Type                                                                                 | Default value | Description                                                                                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------ | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| train_image_size  | list\[int\]                                                                          | \[256, 256\]  | image size used for training \[height, width\]                                                                                                                          |
| keep_aspect_ratio | bool                                                                                 | True          | bool if keep aspect ration while resizing                                                                                                                               |
| train_rgb         | bool                                                                                 | True          | bool if train on rgb or bgr                                                                                                                                             |
| normalize.active  | bool                                                                                 | True          | bool if use normalization                                                                                                                                               |
| normalize.params  | dict                                                                                 | {}            | params for normalization, see [documentation](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.Normalize) |
| augmentations     | list\[{"name": Name of the augmentation, "params": Parameters of the augmentation}\] | \[\]          | list of Albumentations augmentations                                                                                                                                    |

### Optimizer

What optimizer to use for training.
List of all optimizers can be found [here](https://pytorch.org/docs/stable/optim.html).

| Key    | Type | Default value | Description                  |
| ------ | ---- | ------------- | ---------------------------- |
| name   | str  |               | Name of the optimizer.       |
| params | dict | {}            | Parameters of the optimizer. |

### Scheduler

What scheduler to use for training.
List of all optimizers can be found [here](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).

| Key    | Type | Default value | Description                  |
| ------ | ---- | ------------- | ---------------------------- |
| name   | str  |               | Name of the scheduler.       |
| params | dict | {}            | Parameters of the scheduler. |

### Callbacks

Callbacks sections contains a list of callbacks.
More information on callbacks and a list of available ones can be found [here](src/luxonis_train/callbacks/README.md)
Each callback is a dictionary with the following fields:

| Key    | Type | Default value | Description                 |
| ------ | ---- | ------------- | --------------------------- |
| name   | str  |               | Name of the callback.       |
| params | dict | {}            | Parameters of the callback. |

## Exporter

Here you can define configuration for exporting.

| Key                    | Type                              | Default value   | Description                                                                                     |
| ---------------------- | --------------------------------- | --------------- | ----------------------------------------------------------------------------------------------- |
| export_save_directory  | str                               | "output_export" | Where to save the exported files.                                                               |
| input_shape            | list\[int\] \| None               | None            | Input shape of the model. If not provided, inferred from the dataset.                           |
| export_model_name      | str                               | "model"         | Name of the exported model.                                                                     |
| data_type              | Literal\["INT8", "FP16", "FP32"\] | "FP16"          | Data type of the exported model.                                                                |
| reverse_input_channels | bool                              | True            | Whether to reverse the image channels in the exported model. Relevant for `.blob` export        |
| scale_values           | list\[float\] \| None             | None            | What scale values to use for input normalization. If not provided, inferred from augmentations. |
| mean_values            | list\[float\] \| None             | None            | What mean values to use for input normalizations. If not provided, inferred from augmentations. |
| upload_directory       | str \| None                       | None            | Where to upload the exported models.                                                            |

### ONNX

Option specific for ONNX export.

| Key           | Type                     | Default value | Description                      |
| ------------- | ------------------------ | ------------- | -------------------------------- |
| opset_version | int                      | 12            | Which opset version to use.      |
| dynamic_axes  | dict\[str, Any\] \| None | None          | Whether to specify dinamic axes. |

### Blob

| Key    | Type | Default value | Description                          |
| ------ | ---- | ------------- | ------------------------------------ |
| active | bool | False         | Whether to export to `.blob` format. |
| shaves | int  | 6             | How many shaves.                     |

## Tuner

Here you can specify options for tuning.

| Key        | Type              | Default value | Description                                                                                                                                                                                                                                                                                                        |
| ---------- | ----------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| study_name | str               | "test-study"  | Name of the study.                                                                                                                                                                                                                                                                                                 |
| use_pruner | bool              | True          | Whether to use the MedianPruner.                                                                                                                                                                                                                                                                                   |
| n_trials   | int \| None       | 15            | Number of trials for each process. `None` represents no limit in terms of numbner of trials.                                                                                                                                                                                                                       |
| timeout    | int \| None       | None          | Stop study after the given number of seconds.                                                                                                                                                                                                                                                                      |
| params     | dict\[str, list\] | {}            | Which parameters to tune. The keys should be in the format `key1.key2.key3_<type>`. Type can be one of `[categorical, float, int, longuniform, uniform]`. For more information about the types, visit [Optuna documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html). |

Example of params for tuner block:

```yaml
tuner:
  params:
    trainer.optimizer.name_categorical: ["Adam", "SGD"]
    trainer.optimizer.params.lr_float: [0.0001, 0.001]
    trainer.batch_size_int: [4, 16, 4]
```

### Storage

| Key          | Type                         | Default value | Description                                          |
| ------------ | ---------------------------- | ------------- | ---------------------------------------------------- |
| active       | bool                         | True          | Whether to use storage to make the study persistent. |
| storage_type | Literal\["local", "remote"\] | "local"       | Type of the storage.                                 |

## ENVIRON

A special section of the config file where you can specify environment variables.
For more info on the variables, see [Credentials](../README.md#credentials).

**NOTE**

This is not a recommended way due to possible leakage of secrets. This section is intended for testing purposes only.

| Key                      | Type                                                       | Default value  | Description |
| ------------------------ | ---------------------------------------------------------- | -------------- | ----------- |
| AWS_ACCESS_KEY_ID        | str \| None                                                | None           |             |
| AWS_SECRET_ACCESS_KEY    | str \| None                                                | None           |             |
| AWS_S3_ENDPOINT_URL      | str \| None                                                | None           |             |
| MLFLOW_CLOUDFLARE_ID     | str \| None                                                | None           |             |
| MLFLOW_CLOUDFLARE_SECRET | str \| None                                                | None           |             |
| MLFLOW_S3_BUCKET         | str \| None                                                | None           |             |
| MLFLOW_S3_ENDPOINT_URL   | str \| None                                                | None           |             |
| MLFLOW_TRACKING_URI      | str \| None                                                | None           |             |
| POSTGRES_USER            | str \| None                                                | None           |             |
| POSTGRES_PASSWORD        | str \| None                                                | None           |             |
| POSTGRES_HOST            | str \| None                                                | None           |             |
| POSTGRES_PORT            | str \| None                                                | None           |             |
| POSTGRES_DB              | str \| None                                                | None           |             |
| LUXONISML_BUCKET         | str \| None                                                | None           |             |
| LUXONISML_BASE_PATH      | str                                                        | "~/luxonis_ml" |             |
| LUXONISML_TEAM_ID        | str                                                        | "offline"      |             |
| LUXONISML_TEAM_NAME      | str                                                        | "offline"      |             |
| LOG_LEVEL                | Literal\["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"\] | "INFO"         |             |
