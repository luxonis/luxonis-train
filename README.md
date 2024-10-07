# Luxonis Training Framework

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![MacOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=apple&logoColor=white)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyBadge](https://img.shields.io/pypi/pyversions/luxonis-train?logo=data:image/svg+xml%3Bbase64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHlZZWxsb3ciIGdyYWRpZW50VHJhbnNmb3JtPSJyb3RhdGUoNDUpIj4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iI2ZlNSIgb2Zmc2V0PSIwLjYiLz4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iI2RhMSIgb2Zmc2V0PSIxIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJweUJsdWUiIGdyYWRpZW50VHJhbnNmb3JtPSJyb3RhdGUoNDUpIj4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iIzY5ZiIgb2Zmc2V0PSIwLjQiLz4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iIzQ2OCIgb2Zmc2V0PSIxIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogIDwvZGVmcz4KCiAgPHBhdGggZD0iTTI3LDE2YzAtNyw5LTEzLDI0LTEzYzE1LDAsMjMsNiwyMywxM2wwLDIyYzAsNy01LDEyLTExLDEybC0yNCwwYy04LDAtMTQsNi0xNCwxNWwwLDEwbC05LDBjLTgsMC0xMy05LTEzLTI0YzAtMTQsNS0yMywxMy0yM2wzNSwwbDAtM2wtMjQsMGwwLTlsMCwweiBNODgsNTB2MSIgZmlsbD0idXJsKCNweUJsdWUpIi8+CiAgPHBhdGggZD0iTTc0LDg3YzAsNy04LDEzLTIzLDEzYy0xNSwwLTI0LTYtMjQtMTNsMC0yMmMwLTcsNi0xMiwxMi0xMmwyNCwwYzgsMCwxNC03LDE0LTE1bDAtMTBsOSwwYzcsMCwxMyw5LDEzLDIzYzAsMTUtNiwyNC0xMywyNGwtMzUsMGwwLDNsMjMsMGwwLDlsMCwweiBNMTQwLDUwdjEiIGZpbGw9InVybCgjcHlZZWxsb3cpIi8+CgogIDxjaXJjbGUgcj0iNCIgY3g9IjY0IiBjeT0iODgiIGZpbGw9IiNGRkYiLz4KICA8Y2lyY2xlIHI9IjQiIGN4PSIzNyIgY3k9IjE1IiBmaWxsPSIjRkZGIi8+Cjwvc3ZnPgo=)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![CI](https://github.com/luxonis/luxonis-train/actions/workflows/ci.yaml/badge.svg)
![Docs](https://github.com/luxonis/luxonis-train/actions/workflows/docs.yaml/badge.svg)
[![codecov](https://codecov.io/gh/luxonis/luxonis-train/graph/badge.svg?token=647MTHBYD5)](https://codecov.io/gh/luxonis/luxonis-train)

<a name="overview"></a>

## 🌟 Overview

Luxonis Training Framework (`LuxonisTrain`) is a flexible and easy-to-use tool for training deep learning models. It is built on top of `PyTorch Lightning` and provides a simple interface for training, testing, and exporting models.

- **No Code Approach:** No coding skills are required to use `LuxonisTrain`. All you need to do is to define a simple configuration file in a user-friendly `YAML` format.
- **Simplicity:** You can jump right in using our set of predefined configuration files for most common computer vision tasks.
- **Extensibility:** Define your custom components using an easy-to-use Python API that does the heavy lifting for you.
- **Built for the Edge:** `LuxonisTrain` was built with edge devices in mind, focusing on optimal architectures to run with limited compute.

> \[!WARNING\]
> **The project is in a beta state and might be unstable or contain bugs - please report any feedback.**

## 📜 Table Of Contents

- [🌟 Overview](#overview)
- [🛠️ Installation](#installation)
- [📝 Usage](#usage)
  - [💻 CLI](#cli)
- [⚙️ Configuration](#configuration)
- [💾 Data](#data-loading)
  - [📂 Data Directory](#data-directory)
  - [🚀 `LuxonisDataset`](#luxonis-dataset)
- [🏋️‍♂️Training](#training)
- [✍ Testing](#testing)
- [🧠 Inference](#inference)
- [🤖 Exporting](#exporting)
- [🗂️ NN Archive](#nn-archive)
- [🔬 Tuning](#tuning)
- [🎨 Customizations](#customizations)
- [📚 Tutorials and Examples](#tutorials)
- [🔑 Credentials](#credentials)
- [🤝 Contributing](#contributing)
- [📄 License](#license)

<a name="installation"></a>

## 🛠️ Installation

`luxonis-train` is hosted on PyPI and can be installed with `pip` as:

```bash
pip install luxonis-train
```

This command will also create a `luxonis_train` executable in your `PATH`. For more information on how to use the CLI, see [CLI Usage](#cli).

<a name="usage"></a>

## 📝 Usage

You can use `LuxonisTrain` either from the command line or from a Python script.

<a name="cli"></a>

### 💻 CLI

The CLI is the most straightforward way how to use `LuxonisTrain`. The CLI provides several commands for training, testing, tuning, exporting and more.

**Available commands:**

- `train` - Start the training process
- `test` - Test the model on a specific dataset view
- `infer` - Run inference on a dataset, image directory, or a video file.
- `export` - Export the model to either `ONNX` or `BLOB` format that can be run on edge devices
- `archive` - Create an `NN Archive` file that can be used with our `DepthAI` API (coming soon)
- `tune` - Tune the hyperparameters of the model for better performance
- `inspect` - Inspect the dataset you are using and visualize the annotations

To learn more information about any of these commands, run

```bash
luxonis_train <command> --help
```

Specific usage examples can be found in the respective sections below.

<a name="configuration"></a>

## ⚙️ Configuration

The entire configuration is specified in a `YAML` file. This includes the model topology, loss functions, metrics,
optimizers, and all the other components. For extensive list of all options, specific instructions and example
configuration files, see [Configuration](https://github.com/luxonis/luxonis-train/blob/main/configs/README.md).

We provide a set of predefined configuration files for most common computer vision tasks.
You can find them in the `configs` directory.

In the following examples, we will be using the `configs/detection_light_model.yaml` configuration file.

<a name="data-loading"></a>

## 🚀 Data

`LuxonisTrain` supports several ways of loading data:

- using a data directory in one of the supported formats
- using an already existing dataset in our custom `LuxonisDataset` format
- using a custom loader
  - to learn how to implement and use custom loaders, see [Customizations](#customizations)

<a name="data-directory"></a>

### 📂 Data Directory

The easiest way to load data is to use a directory with the dataset in one of the supported formats.

The supported formats are:

- `COCO` - We support COCO JSON format in two variants:
  - [`RoboFlow`](https://roboflow.com/formats/coco-json)
  - [`FiftyOne`](https://docs.voxel51.com/user_guide/export_datasets.html#cocodetectiondataset-export)
- [`Pascal VOC XML`](https://roboflow.com/formats/pascal-voc-xml)
- [`YOLO Darknet TXT`](https://roboflow.com/formats/yolo-darknet-txt)
- [`YOLOv4 PyTorch TXT`](https://roboflow.com/formats/yolov4-pytorch-txt)
- [`MT YOLOv6`](https://roboflow.com/formats/mt-yolov6)
- [`CreateML JSON`](https://roboflow.com/formats/createml-json)
- [`TensorFlow Object Detection CSV`](https://roboflow.com/formats/tensorflow-object-detection-csv)
- `Classification Directory` - A directory with subdirectories for each class
  ```plaintext
  dataset_dir/
  ├── train/
  │   ├── class1/
  │   │   ├── img1.jpg
  │   │   ├── img2.jpg
  │   │   └── ...
  │   ├── class2/
  │   └── ...
  ├── valid/
  └── test/
  ```
- `Segmentation Mask Directory` - A directory with images and corresponding masks.
  ```plaintext
  dataset_dir/
  ├── train/
  │   ├── img1.jpg
  │   ├── img1_mask.png
  │   ├── ...
  │   └── _classes.csv
  ├── valid/
  └── test/
  ```
  The masks are stored as grayscale `PNG` images where each pixel value corresponds to a class.
  The mapping from pixel values to classes is defined in the `_classes.csv` file.
  ```csv
  Pixel Value, Class
  0, background
  1, class1
  2, class2
  3, class3
  ```

To use a directory loader, you need to specify the `dataset_dir` parameter in the config file.

`dataset_dir` can be one of the following:

- local path to the dataset directory
- URL to a remote dataset
  - the dataset will be downloaded to a `"data"` directory in the current working directory
  - supported URL protocols:
    - `s3://bucket/path/to/directory` for AWS S3
    - `gs://buclet/path/to/directory` for Google Cloud Storage
    - `roboflow://workspace/project/version/format` for `RoboFlow` datasets
      - `workspace` - name of the workspace the dataset belongs to
      - `project` - name of the project the dataset belongs to
      - `version` - version of the dataset
      - `format` - one of `coco`, `darknet`, `voc`, `yolov4pytorch`, `mt-yolov6`, `createml`, `tensorflow`, `folder`, `png-mask-semantic`
      - **example:** `roboflow://team-roboflow/coco-128/2/coco`

```yaml
loader:
  params:
    # Optional, the dataset will be created under this name.
    # If not specified, the name of the dataset will be
    # the same as the name of the dataset directory.
    dataset_name: "coco_test"

    # Path to the dataset directory. It can be be either a local path
    # or an URL to a remote dataset.
    dataset_dir: "roboflow://team-roboflow/coco-128/2/coco"

    # One of coco, voc, darknet, yolov4, yolov6, createml, tfcsv, clsdir, or segmask.
    # Notice the values of `dataset_type` here are a bit different
    # from the dataset formats in the RoboFlow URL.
    dataset_type: "coco"
```

<a name="luxonis-dataset"></a>

### 💾 `LuxonisDataset`

`LuxonisDataset` is our custom dataset format designed for easy and efficient dataset management.
To learn more about how to create a dataset in this format from scratch, see the [Luxonis ML](https://github.com/luxonis/luxonis-ml) repository.

To use the `LuxonisDataset` as a source of the data, specify the following in the config file:

```yaml
loader:
  params:
    # name of the dataset
    dataset_name: "dataset_name"

    # one of local (default), s3, gcs
    bucket_storage: "local"
```

> \[!TIP\]
> To inspect the loader output, use the `luxonis_train inspect` command:
>
> ```bash
> luxonis_train inspect --config <config.yaml> --view <train/val/test>
> ```
>
> **The `inspect` command is currently only available in the CLI**

<a name="training"></a>

## 🏋️‍♂️ Training

Once you've created your `config.yaml` file you can start the training process by running:

**CLI:**

```bash
luxonis_train train --config configs/detection_light_model.yaml
```

If you wish to change some config parameters without modifying the config file,
you can do this by providing key-value pairs as arguments. Example of this is:

```bash
luxonis_train train                           \
  --config configs/detection_light_model.yaml \
  loader.params.dataset_dir "roboflow://team-roboflow/coco-128/2/coco"
```

Where keys and values are space separated and sub-keys are dot (`.`) separated. If the configuration field is a list, then key/sub-key should be a number (_e.g._ `trainer.preprocessing.augmentations.0.params.p 1`).

**Python:**

```python
from luxonis_train import LuxonisModel

model = LuxonisModel(
  "configs/detection_light_model.yaml",
  {"loader.params.dataset_dir": "roboflow://team-roboflow/coco-128/2/coco"}
)
model.train()
```

If not explicitly disabled, the training process will be monitored by `TensorBoard`. To start the `TensorBoard` server, run:

```bash
tensorboard --logdir output/tensorboard_logs
```

This command will start the server and print the URL where you can access the `TensorBoard` dashboard.

By default, the files produced during the training run will be saved in the `output` directory.
Individual runs will be saved under a randomly generated run name.

Assuming all optional callbacks are enabled, the output directory will be similar to the following:

```plaintext
output/
├── tensorboard_logs/
└── 0-red-puma/
    ├── config.yaml
    ├── luxonis_train.log
    ├── metadata.yaml
    ├── best_val_metrics/
    │   └── model_metric_name=metric_value_loss=loss_value.ckpt
    ├── min_val_loss/
    │   └── model_loss=loss_value.ckpt
    ├── export/
    │   ├── model.onnx
    │   └── model.blob
    └── archive/
        └── model.onnx.tar.gz
```

<a name="testing"></a>

## ✍ Testing

To test the model on a specific dataset view (`train`, `test`, or `val`), use the following command:

**CLI:**

```bash
luxonis_train test --config configs/detection_light_model.yaml \
                   --view val                                  \
                   --weights path/to/checkpoint.ckpt
```

**Python:**

```python
from luxonis_train import LuxonisModel

model = LuxonisModel("configs/detection_light_model.yaml")
model.test(weights="path/to/checkpoint.ckpt")
```

The testing process can be started automatically at the end of the training by using the `TestOnTrainEnd` callback.
To learn more about callbacks, see [Callbacks](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/callbacks/README.md).

<a name="inference"></a>

## 🧠 Inference

You can use the `infer` command to run inference on a dataset, image directory, or a video file.

**CLI:**

To run the inference on a dataset view and show the results on screen:

```bash
luxonis_train infer --config configs/detection_light_model.yaml \
                    --view val                                  \
                    --weights path/to/checkpoint.ckpt
```

To run the inference on a video file and show the results on screen:

```bash
luxonis_train infer --config configs/detection_light_model.yaml \
                    --weights path/to/checkpoint.ckpt           \
                    --source-path path/to/video.mp4
```

To run the inference on an image directory and save the results to another directory:

```bash
luxonis_train infer --config configs/detection_light_model.yaml \
                    --weights path/to/checkpoint.ckpt           \
                    --source-path path/to/images                \
                    --save-dir path/to/save_directory
```

**Python:**

```python
from luxonis_train import LuxonisModel

model = LuxonisModel("configs/detection_light_model.yaml")

# infer on a dataset view
model.infer(weights="path/to/checkpoint.ckpt", view="val")

# infer on a video file
model.infer(weights="path/to/checkpoint.ckpt", source_path="path/to/video.mp4")

# infer on an image directory and save the results
model.infer(
    weights="path/to/checkpoint.ckpt",
    source_path="path/to/images",
    save_dir="path/to/save_directory",
)
```

<a name="exporting"></a>

## 🤖 Exporting

We support export to `ONNX`, and `BLOB` formats, latter of which can run on OAK-D cameras.

To configure the exporter, you can specify the [exporter](https://github.com/luxonis/luxonis-train/blob/main/configs/README.md#exporter) section in the config file.

By default, the exporter will export the model only to the `ONNX` format.

You can see an example export configuration [here](https://github.com/luxonis/luxonis-train/blob/main/configs/example_export.yaml).

**CLI:**

```bash
luxonis_train export --config configs/example_export.yaml --weights path/to/weights.ckpt
```

**Python:**

```python
from luxonis_train import LuxonisModel

model = LuxonisModel("configs/example_export.yaml")
model.export(weights="path/to/weights.ckpt")
```

The export process can be run automatically at the end of the training by using the `ExportOnTrainEnd` callback.

<a name="nn-archive"></a>

## 🗂️ NN Archive

The models can also be exported to our custom `NN Archive` format. `NN Archive` is a `.tar.gz` file that can be easily used with the [`DepthAI`](https://github.com/luxonis/depthai) API.

The archive contains the exported model together with all the metadata needed for running the model with no additional configuration from the user.

**CLI:**

```bash
luxonis_train archive                         \
  --config configs/detection_light_model.yaml \
  --weights path/to/checkpoint.ckpt
```

Or you can specify the path to the exported model if you already have it:

```bash
luxonis_train archive                         \
  --config configs/detection_light_model.yaml \
  --executable path/to/exported_model.onnx
```

**Python:**

```python
from luxonis_train import LuxonisModel

model = LuxonisModel("configs/detection_light_model.yaml")
model.archive("path/to/exported_model.onnx")
# model.archive(weights="path/to/checkpoint.ckpt")
```

The archive can be created automatically at the end of the training by using the `ArchiveOnTrainEnd` callback.

<a name="tuning"></a>

## 🔬 Tuning

The `tune` command can be used to search for the optimal hyperparameters of the model in order to boost its performance.
The tuning is powered by [`Optuna`](https://optuna.org/).
To use tuning, you have to specify the [tuner](https://github.com/luxonis/luxonis-train/blob/main/configs/README.md#tuner) section in the config file.

You can see an example tuning configuration [here](https://github.com/luxonis/luxonis-train/blob/main/configs/example_tuning.yaml).

**CLI:**

```bash
luxonis_train tune --config configs/example_tuning.yaml
```

**Python:**

```python
from luxonis_train import LuxonisModel

model = LuxonisModel("configs/example_tuning.yaml")
model.tune()
```

<a name="customizations"></a>

## 🎨 Customizations

We provide a registry interface through which you can create new
[loaders](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/loaders/README.md),
[nodes](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/nodes/README.md),
[losses](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/attached_modules/losses/README.md),
[metrics](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/attached_modules/metrics/README.md),
[visualizers](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/attached_modules/visualizers/README.md),
[callbacks](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/callbacks/README.md),
[optimizers](https://github.com/luxonis/luxonis-train/blob/main/configs/README.md#optimizer),
and [schedulers](https://github.com/luxonis/luxonis-train/blob/main/configs/README.md#scheduler).

Registered components can be then referenced in the config file. Custom components need to inherit from their respective base classes:

- Loader - [`BaseLoader`](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/loaders/base_loader.py)
- Node - [`BaseNode`](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/models/nodes/base_node.py)
- Loss - [`BaseLoss`](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/attached_modules/losses/base_loss.py)
- Metric - [`BaseMetric`](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/attached_modules/metrics/base_metric.py)
- Visualizer - [`BaseVisualizer`](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/attached_modules/visualizers/base_visualizer.py)
- Callback - [`lightning.pytorch.callbacks.Callback`](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html), requires manual registration to the `CALLBACKS` registry
- Optimizer - [`torch.optim.Optimizer`](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer), requires manual registration to the `OPTIMIZERS` registry
- Scheduler - [`torch.optim.lr_scheduler.LRScheduler`](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate), requires manual registration to the `SCHEDULERS` registry

Here is an example of how to create custom loss and optimizer:

```python
from torch.optim import Optimizer
from luxonis_train.utils.registry import OPTIMIZERS
from luxonis_train.attached_modules.losses import BaseLoss

@OPTIMIZERS.register_module()
class CustomOptimizer(Optimizer):
    ...

# Subclasses of BaseNode, LuxonisLoss, LuxonisMetric
# and BaseVisualizer are registered automatically.
class CustomLoss(BaseLoss):
    # This class is automatically registered under `CustomLoss` name.
    def __init__(self, k_steps: int, **kwargs):
        super().__init__(**kwargs)
        ...
```

In the configuration file you can reference the `CustomOptimizer` and `CustomLoss` by their names:

```yaml
losses:
  - name: CustomLoss
    params:  # additional parameters
      k_steps: 12
```

The files containing the custom components must be sourced before the training script is run. To do that in CLI, you can use the `--source` argument:

```bash
luxonis_train --source custom_components.py train --config config.yaml
```

You can also run the training from a python script:

```python
from custom_components import *
from luxonis_train import LuxonisModel

model = LuxonisModel("config.yaml")
model.train()
```

For more information on how to define custom components, consult the respective in-source documentation.

<a name="tutorials"></a>

## 📚 Tutorials and Examples

We are actively working on providing examples and tutorials for different parts of the library which will help you to start more easily. The tutorials can be found [here](https://github.com/luxonis/depthai-ml-training/tree/master) and will be updated regularly.

<a name="credentials"></a>

## 🔑 Credentials

Local use is supported by default. In addition, we also integrate several cloud services which can be primarily used for logging the training progress and storing data. To use these services, you usually need to load specific environment variables to set up the correct credentials.

You have these options how to set up the environment variables:

- Using standard environment variables
- Specifying the variables in a `.env` file. If a variable is both in the environment and present in `.env` file, the exported variable takes precedence.
- Specifying the variables in the [ENVIRON](https://github.com/luxonis/luxonis-train/blob/main/configs/README.md#environ) section of the config file. Variables defined in the config file will take precedence over environment and `.env` variables. Note that this is not a recommended way due to security reasons.

The following storage services are supported:

- `AWS S3`, requires the following environment variables:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_S3_ENDPOINT_URL`
- `Google Cloud Storage`, requires the following environment variables:
  - `GOOGLE_APPLICATION_CREDENTIALS`
- `RoboFlow`, requires the following environment variables:
  - `ROBOFLOW_API_KEY`

For logging and tracking, we support:

- `MLFlow`, requires the following environment variables:
  - `MLFLOW_S3_BUCKET`
  - `MLFLOW_S3_ENDPOINT_URL`
  - `MLFLOW_TRACKING_URI`
- `WandB`, requires the following environment variables:
  - `WANDB_API_KEY`

There is an option for remote `POSTGRESS` database storage for [Tuning](#tuning).
You need to specify the following env variables in order to connect to the database:

- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`

<a name="contributing"></a>

## 🤝 Contributing

If you want to contribute to the development, consult the [Contribution guide](https://github.com/luxonis/luxonis-train/blob/main/CONTRIBUTING.md) for further instructions.

<a name="license"></a>

## 📄 License

This project is licensed under the [Apache License, Version 2.0](https://opensource.org/license/apache-2-0/) - see the [LICENSE](LICENSE) file for details.
