# Luxonis Training Framework

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)
![MacOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=apple&logoColor=white)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyBadge](media/pybadge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![UnitTests](https://github.com/luxonis/models/actions/workflows/tests.yaml/badge.svg)
![Docs](https://github.com/luxonis/luxonis-ml/actions/workflows/docs.yaml/badge.svg)
[![Coverage](media/coverage_badge.svg)](https://github.com/luxonis/models/actions)

Luxonis training framework (`luxonis-train`) is intended for training deep learning models that can run fast on OAK products.

The project is in alpha state - please report any feedback.

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
configuration files, see [Configuration](./configs/README.md).

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

## Tuning

To improve training performance you can use `Tuner` for hyperparameter optimization.
To use tuning, you have to specify [tuner](configs/README.md#tuner) section in the config file.

To start the tuning, run

```bash
luxonis_train tune --config config.yaml
```

You can see an example tuning configuration [here](configs/example_tuning.yaml).

## Exporting

We support export to `ONNX`, and `DepthAI .blob format` which is used for OAK cameras. By default, we export to `ONNX` format.

To use the exporter, you have to specify the [exporter](configs/README.md#exporter) section in the config file.

Once you have the config file ready you can export the model using

```bash
luxonis_train export --config config.yaml
```

You can see an example export configuration [here](configs/example_export.yaml).

## Customizations

We provide a registry interface through which you can create new [nodes](src/luxonis_train/nodes/README.md), [losses](src/luxonis_train/attached_modules/losses/README.md), [metrics](src/luxonis_train/attached_modules/metrics/README.md), [visualizers](src/luxonis_train/attached_modules/visualizers/README.md), [callbacks](src/luxonis_train/callbacks/README.md), [optimizers](configs/README.md#optimizer), and [schedulers](configs/README.md#scheduler).

Registered components can be then referenced in the config file. Custom components need to inherit from their respective base classes:

- Node - [BaseNode](src/luxonis_train/models/nodes/base_node.py)
- Loss - [BaseLoss](src/luxonis_train/attached_modules/losses/base_loss.py)
- Metric - [BaseMetric](src/luxonis_train/attached_modules/metrics/base_metric.py)
- Visualizer - [BaseVisualizer](src/luxonis_train/attached_modules/visualizers/base_visualizer.py)
- Callback - [Callback from lightning.pytorch.callbacks](lightning.pytorch.callbacks)
- Optimizer - [Optimizer from torch.optim](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)
- Scheduler - [LRScheduler from torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

Here is an example of how to create a custom components:

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

And then in the config you reference this `CustomOptimizer` and `CustomLoss` by their names.

```yaml
losses:
  - name: CustomLoss
    params:  # additional parameters
      k_steps: 12

```

For more information on how to define custom components, consult the respective in-source documentation.

## Credentials

Local use is supported by default. In addition, we also integrate some cloud services which can be primarily used for logging and storing. When these are used, you need to load environment variables to set up the correct credentials.

You have these options how to set up the environment variables:

- Using standard environment variables
- Specifying the variables in a `.env` file. If a variable is both in the environment and present in `.env` file, the exported variable takes precedense.
- Specifying the variables in the [ENVIRON](configs/README.md#environ) section of the config file. Note that this is not a recommended way. Variables defined in config take precedense over environment and `.env` variables.

### S3

If you are working with LuxonisDataset that is hosted on S3, you need to specify these env variables:

```bash
AWS_ACCESS_KEY_ID=**********
AWS_SECRET_ACCESS_KEY=**********
AWS_S3_ENDPOINT_URL=**********
```

### MLFlow

If you want to use MLFlow for logging and storing artifacts you also need to specify MLFlow-related env variables like this:

```bash
MLFLOW_S3_BUCKET=**********
MLFLOW_S3_ENDPOINT_URL=**********
MLFLOW_TRACKING_URI=**********
```

### WanDB

If you are using WanDB for logging, you have to sign in first in your environment.

### POSTGRESS

There is an option for remote storage for [Tuning](#tuning). We use POSTGRES and to connect to the database you need to specify the folowing env variables:

```bash
POSTGRES_USER=**********
POSTGRES_PASSWORD=**********
POSTGRES_HOST=**********
POSTGRES_PORT=**********
POSTGRES_DB=**********
```

## Contributing

If you want to contribute to the development, install the dev version of the package:

```bash
pip install luxonis-train[dev]
```

Consult the [Contribution guide](CONTRIBUTING.md) before making a pull request.
