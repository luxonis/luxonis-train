import pytest
from luxonis_ml.typing import Params
from torch import nn
from torch._prims_common import Tensor

from luxonis_train import BaseNode, LuxonisModel, Tasks


class Backbone(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class Neck(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(64, 128, 3)
        self.conv2 = nn.Conv2d(128, 256, 3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        return self.conv2(x)


class Head(BaseNode):
    task = Tasks.CLASSIFICATION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.branch1 = nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3),
            nn.MaxPool2d(2),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 3),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 4 * 12, 10)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = x1 + x2
        return self.fc(self.flatten(x))


@pytest.fixture
def config() -> Params:
    return {
        "model": {
            "name": "test_finetuning",
            "nodes": [
                {
                    "name": "Backbone",
                    "finetuning": {
                        "parameters": [
                            {"name": "conv1"},
                            {"name": "conv2"},
                        ],
                        "optimizer": {
                            "params": {"lr": 0.001},
                        },
                    },
                },
                {
                    "name": "Neck",
                    "finetuning": {
                        "optimizer": {"name": "AdamW"},
                    },
                },
                {
                    "name": "Head",
                    "losses": [{"name": "CrossEntropyLoss"}],
                    "metrics": [{"name": "Accuracy"}],
                    "finetuning": [
                        {
                            "parameters": [{"name": "branch1"}],
                            "optimizer": {
                                "name": "SGD",
                                "params": {"lr": 0.01},
                            },
                            "scheduler": {
                                "name": "CosineAnnealingLR",
                            },
                        },
                        {
                            "parameters": [{"module_type": "Linear"}],
                            "optimizer": {
                                "params": {"weight_decay": 0.01},
                            },
                        },
                        {
                            "parameters": [{"module_type": "Conv2d"}],
                            "optimizer": {
                                "params": {"weight_decay": 0.02},
                            },
                            "scheduler": {
                                "name": "StepLR",
                                "params": {"step_size": 10},
                            },
                        },
                    ],
                },
            ],
        }
    }


def test_finetuning(config: Params, opts: Params):
    model = LuxonisModel(
        config, opts | {"loader.params.n_classes": 10}, debug_mode=True
    )
    model.train()
    optimizers = model.lightning_module.optimizers()
    schedulers = model.lightning_module.lr_schedulers()
    assert isinstance(optimizers, list)
    assert isinstance(schedulers, list)
    assert len(optimizers) == len(schedulers) == 4
