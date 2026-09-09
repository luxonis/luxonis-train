"""The configuration schema and the predefined models.

`Config` parses and validates the YAML file that defines a run.
`luxonis_train.config.config` holds the section models, and
`luxonis_train.config.predefined_models` holds the models that build a
node graph from a few parameters.

"""

from .config import (
    AttachedModuleConfig,
    Config,
    ExportConfig,
    LossModuleConfig,
    MetricModuleConfig,
    NodeConfig,
    TrainerConfig,
)

# So predefined models get registered
from .predefined_models import *
from .predefined_models.base_predefined_model import BasePredefinedModel

__all__ = [
    "AttachedModuleConfig",
    "BasePredefinedModel",
    "Config",
    "ExportConfig",
    "LossModuleConfig",
    "MetricModuleConfig",
    "NodeConfig",
    "TrainerConfig",
]
