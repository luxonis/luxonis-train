# Luxonis Training Framework

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![MacOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=apple&logoColor=white)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyBadge](https://github.com/luxonis/luxonis-train/blob/main/media/pybadge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![CI](https://github.com/luxonis/luxonis-train/actions/workflows/ci.yaml/badge.svg)
![Docs](https://github.com/luxonis/luxonis-train/actions/workflows/docs.yaml/badge.svg)
[![codecov](https://codecov.io/gh/luxonis/luxonis-train/graph/badge.svg?token=647MTHBYD5)](https://codecov.io/gh/luxonis/luxonis-train)

Luxonis training framework (`luxonis-train`) is intended for training deep learning models that can run fast on OAK products.

**The project is in a beta state and might be unstable or contain bugs - please report any feedback.**

## Table Of Contents

- [Installation](#installation)
- [Training](#training)
- [Customizations](#customizations)
- [Tuning](#tuning)
- [Exporting](#exporting)
- [Credentials](#credentials)
- [Contributing](#contributing)

## Installation

`luxonis-train` is hosted on PyPi and can be installed with `pip` as:

```bash
pip install luxonis-train
```

This command will also create a `luxonis_train` executable in your `PATH`.
See `luxonis_train --help` for more information.

## Usage

The entire configuration is specified in a `yaml` file. This includes the model
structure, used losses, metrics, optimizers etc. For specific instructions and example
configuration files, see [Configuration](https://github.com/luxonis/luxonis-train/blob/main/configs/README.md).

## Data Loading

LuxonisTrain supports several ways of loading data:

- from an existing dataset in the Luxonis Dataset Format
- from a directory in one of the supported formats (_e.g._ COCO, VOC, _etc._)
- using a custom loader

### Luxonis Dataset Format

The default loader used with `LuxonisTrain` is `LuxonisLoaderTorch`. It can either load data from an already created dataset in the `LuxonisDataFormat` or create a new dataset automatically from a set of supported formats.

For instructions on how to create a dataset in the LDF, follow the
[examples](https://github.com/luxonis/luxonis-ml/tree/main/examples) in
the [luxonis-ml](https://github.com/luxonis/luxonis-ml) repository.

To use the Luxonis Dataset Loader, specify the following in the config file:

```yaml

loader:
  params:
    # name of the created dataset
    dataset_name: dataset_name
    # one of local (default), s3, gcs
    bucket_storage: local
```

### Parsing from Directory

The supported formats are:

- COCO - We support COCO JSON format in two variants:
  - [RoboFlow](https://roboflow.com/formats/coco-json)
  - [FiveOne](https://docs.voxel51.com/user_guide/export_datasets.html#cocodetectiondataset-export)
- [Pascal VOC XML](https://roboflow.com/formats/pascal-voc-xml)
- [YOLO Darknet TXT](https://roboflow.com/formats/yolo-darknet-txt)
- [YOLOv4 PyTorch TXT](https://roboflow.com/formats/yolov4-pytorch-txt)
- [MT YOLOv6](https://roboflow.com/formats/mt-yolov6)
- [CreateML JSON](https://roboflow.com/formats/createml-json)
- [TensorFlow Object Detection CSV](https://roboflow.com/formats/tensorflow-object-detection-csv)
- Classification Directory - A directory with subdirectories for each class
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
- Segmentation Mask Directory - A directory with images and corresponding masks.
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
  The masks are stored as grayscale PNG images where each pixel value corresponds to a class.
  The mapping from pixel values to class is defined in the `_classes.csv` file.
  ```csv
  Pixel Value, Class
  0, background
  1, class1
  2, class2
  3, class3
  ```

To use a directory loader, specify the following in the config file:

```yaml

loader:
  params:
    dataset_dir: path/to/dataset
    # one of voc, darknet, yolov4, yolov6, createml, tfcsv, clsdir, segmask
    dataset_type: coco
```

### Custom Loader

To learn how to implement and use a custom loader, see [customization](#customizations).

The loader can be referenced in the configuration file using its class name:

```yaml
loader:
  name: CustomLoader
  params:
    # additional parameters to be passed to the loade constructor
```

To inspect the loader output, use the `luxonis_train inspect` command:

```bash
luxonis_train inspect --config <config.yaml> --view <train/val/test>
```

## Training

Once you've created your `config.yaml` file you can train the model using this command:

```bash
luxonis_train train --config config.yaml
```

If you wish to manually override some config parameters you can do this by providing the key-value pairs. Example of this is:

```bash
luxonis_train train --config config.yaml trainer.batch_size 8 trainer.epochs 10
```

where key and value are space separated and sub-keys are dot (`.`) separated. If the configuration field is a list, then key/sub-key should be a number (e.g. `trainer.preprocessing.augmentations.0.name RotateCustom`).

## Testing

To test the model on a specific dataset view (`train`, `test`, or `val`), use the following command:

```bash
luxonis_train test --config <config.yaml> --view <train/test/val>
```

## Tuning

To improve training performance you can use `Tuner` for hyperparameter optimization.
To use tuning, you have to specify [tuner](https://github.com/luxonis/luxonis-train/blob/main/configs/README.md#tuner) section in the config file.

To start the tuning, run

```bash
luxonis_train tune --config config.yaml
```

You can see an example tuning configuration [here](https://github.com/luxonis/luxonis-train/blob/main/configs/example_tuning.yaml).

## Exporting

We support export to `ONNX`, and `DepthAI .blob format` which is used for OAK cameras. By default, we export to `ONNX` format.

To configure the exporter, you can specify the [exporter](https://github.com/luxonis/luxonis-train/blob/main/configs/README.md#exporter) section in the config file.

You can see an example export configuration [here](https://github.com/luxonis/luxonis-train/blob/main/configs/example_export.yaml).

To export the model, run

```bash
luxonis_train export --config config.yaml model.weights path/to/weights.ckpt
```

The export process can be run automatically at the end of the training by using the `ExportOnTrainEnd` callback.

To learn about callbacks, see [callbacks](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/callbacks/README.md).

## NN Archive Support

The models can also be exported to our custom NN Archive format.

```bash
luxonis_train archive --executable path/to/exported_model.onnx --config config.yaml
```

This will create a `.tar.gz` file which can be used with the [DepthAI](https://github.com/luxonis/depthai) API.

The archive can be created automatically at the end of the training by using the `ArchiveOnTrainEnd` callback.

## Customizations

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

- Loader - [BaseLoader](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/loaders/base_loader.py)
- Node - [BaseNode](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/models/nodes/base_node.py)
- Loss - [BaseLoss](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/attached_modules/losses/base_loss.py)
- Metric - [BaseMetric](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/attached_modules/metrics/base_metric.py)
- Visualizer - [BaseVisualizer](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/attached_modules/visualizers/base_visualizer.py)
- Callback - [Callback from lightning.pytorch.callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html), requires manual registration to the `CALLBACKS` registry
- Optimizer - [Optimizer from torch.optim](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer), requires manual registration to the `OPTIMIZERS` registry
- Scheduler - [LRScheduler fro torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate), requires manual registration to the `SCHEDULERS` registry

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

In the configuration file you reference the `CustomOptimizer` and `CustomLoss` by their names:

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

For `LuxonisTrain` to recognize the custom components, they need to be imported before the main training script is run.

If you're using the CLI, you can import the custom components by specifying the `--source` flag:

```bash
luxonis_train --source custom_components.py train --config config.yaml
```

Otherwise, you can import the custom components in your custom main script:

```python
from custom_components import *
from luxonis_train import LuxonisModel

model = LuxonisModel("config.yaml")
model.train()
```

For more information on how to define custom components, consult the respective in-source documentation.

## Credentials

Local use is supported by default. In addition, we also integrate some cloud services which can be primarily used for logging and storing. When these are used, you need to load environment variables to set up the correct credentials.

You have these options how to set up the environment variables:

- Using standard environment variables
- Specifying the variables in a `.env` file. If a variable is both in the environment and present in `.env` file, the exported variable takes precedence.
- Specifying the variables in the [ENVIRON](https://github.com/luxonis/luxonis-train/blob/main/configs/README.md#environ) section of the config file. Note that this is not a recommended way. Variables defined in config take precedence over environment and `.env` variables.

The following storage services are supported:

- AWS S3, requires the following environment variables:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_S3_ENDPOINT_URL`
- Google Cloud Storage, requires the following environment variables:
  - `GOOGLE_APPLICATION_CREDENTIALS`

For logging and tracking, we support:

- MLFlow, requires the following environment variables:
  - `MLFLOW_S3_BUCKET`
  - `MLFLOW_S3_ENDPOINT_URL`
  - `MLFLOW_TRACKING_URI`
- WandB, requires the following environment variables:
  - `WANDB_API_KEY`

There is an option for remote `POSTGRESS` storage for [Tuning](#tuning). To connect to the database you need to specify the following env variables:

- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`

## Contributing

If you want to contribute to the development, install the dev version of the package:

```bash
pip install luxonis-train[dev]
```

Consult the [Contribution guide](https://github.com/luxonis/luxonis-train/blob/main/CONTRIBUTING.md) for further instructions.
