from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import pytest
from luxonis_ml.typing import Params
from torch import Tensor, nn
from torch.optim import Optimizer

from luxonis_train import BaseNode, LuxonisModel, Tasks


class Backbone(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.conv2 = nn.Conv2d(4, 6, 3)
        self.conv3 = nn.Conv2d(6, 8, 3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class Neck(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2d(8, 8, 3)
        self.conv2 = nn.Conv2d(8, 8, 3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        return self.conv2(x)


class Head(BaseNode):
    task = Tasks.CLASSIFICATION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.branch1 = nn.Sequential(
            nn.Conv2d(8, 8, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 3),
            nn.MaxPool2d(2),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(8, 8, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 3),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 4 * 12, 10)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = x1 + x2
        return self.fc(self.flatten(x))


class TinyHead(BaseNode):
    task = Tasks.CLASSIFICATION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 3),
            nn.MaxPool2d(2),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 3),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 6 * 14, 10)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = x1 + x2
        return self.fc(self.flatten(x))


@dataclass
class OptimizerSnapshot:
    model: LuxonisModel
    optimizers: list[Optimizer]
    schedulers: list[Any]
    names_by_id: dict[int, str]


def head_node(finetuning: Any | None = None) -> Params:
    node: Params = {
        "name": "Head",
        "losses": [{"name": "CrossEntropyLoss"}],
        "metrics": [{"name": "Accuracy"}],
    }
    if finetuning is not None:
        node["finetuning"] = finetuning
    return node


def tiny_head_node(finetuning: Any | None = None) -> Params:
    node = head_node(finetuning)
    node["name"] = "TinyHead"
    node["alias"] = "Head"
    return node


def node(name: str, finetuning: Any | None = None) -> Params:
    if name == "Head":
        return head_node(finetuning)
    node_cfg: Params = {"name": name}
    if finetuning is not None:
        node_cfg["finetuning"] = finetuning
    return node_cfg


def config(nodes: list[Params], trainer: Params | None = None) -> Params:
    trainer_cfg: dict[str, Any] = {
        "scheduler": {"name": "ConstantLR", "params": {"factor": 1.0}}
    }
    if trainer is not None:
        trainer_cfg |= trainer
    return cast(
        Params,
        {
            "model": {"name": "test_finetuning", "nodes": nodes},
            "trainer": trainer_cfg,
        },
    )


def build_snapshot(config: Params, opts: Params) -> OptimizerSnapshot:
    model = LuxonisModel(
        deepcopy(config),
        opts | {"loader.params.n_classes": 10},
        allow_empty_dataset=True,
    )
    optimizers, schedulers = model.lightning_module.configure_optimizers()
    return OptimizerSnapshot(
        model=model,
        optimizers=list(optimizers),
        schedulers=list(schedulers),
        names_by_id=parameter_names_by_id(model),
    )


def parameter_names_by_id(model: LuxonisModel) -> dict[int, str]:
    names: dict[int, str] = {}
    for node_name, luxonis_node in model.lightning_module.nodes.items():
        for module_name, module in luxonis_node.module.named_modules():
            if list(module.parameters()) and not list(module.children()):
                for param_name, param in module.named_parameters():
                    name = (
                        f"{node_name}.{module.__class__.__name__}."
                        f"{module_name}.{param_name}"
                    )
                    names[id(param)] = name
    return names


def scheduler(scheduler: Any) -> Any:
    if isinstance(scheduler, dict):
        return scheduler["scheduler"]
    return scheduler


def group_names(
    snapshot: OptimizerSnapshot, optimizer: Optimizer, group_idx: int
) -> set[str]:
    return {
        snapshot.names_by_id[id(param)]
        for param in optimizer.param_groups[group_idx]["params"]
    }


def optimizer_group_names(
    snapshot: OptimizerSnapshot, optimizer: Optimizer
) -> list[set[str]]:
    return [
        group_names(snapshot, optimizer, group_idx)
        for group_idx, _ in enumerate(optimizer.param_groups)
    ]


def optimizer_names(
    snapshot: OptimizerSnapshot, optimizer: Optimizer
) -> set[str]:
    return {
        snapshot.names_by_id[id(param)]
        for group in optimizer.param_groups
        for param in group["params"]
    }


def names_for_ids(
    snapshot: OptimizerSnapshot, parameter_ids: set[int]
) -> set[str]:
    return {
        snapshot.names_by_id[param_id]
        for param_id in parameter_ids
        if param_id in snapshot.names_by_id
    }


def ids_for_names(snapshot: OptimizerSnapshot, names: set[str]) -> set[int]:
    ids_by_name = {
        name: param_id for param_id, name in snapshot.names_by_id.items()
    }
    return {ids_by_name[name] for name in names}


def optimizer_parameter_names(snapshot: OptimizerSnapshot) -> set[str]:
    return {
        snapshot.names_by_id[id(param)]
        for optimizer in snapshot.optimizers
        for group in optimizer.param_groups
        for param in group["params"]
    }


def optimizer_parameter_ids(snapshot: OptimizerSnapshot) -> set[int]:
    return {
        id(param)
        for optimizer in snapshot.optimizers
        for group in optimizer.param_groups
        for param in group["params"]
    }


def trainable_parameter_names(snapshot: OptimizerSnapshot) -> set[str]:
    return {
        snapshot.names_by_id[id(param)]
        for param in snapshot.model.lightning_module.parameters()
        if param.requires_grad
    }


def trainable_parameter_ids(snapshot: OptimizerSnapshot) -> set[int]:
    return {
        id(param)
        for param in snapshot.model.lightning_module.parameters()
        if param.requires_grad
    }


def matching_names(snapshot: OptimizerSnapshot, *parts: str) -> set[str]:
    return {
        name
        for name in snapshot.names_by_id.values()
        if all(part in name for part in parts)
    }


def find_group(
    snapshot: OptimizerSnapshot, expected_names: set[str]
) -> tuple[int, Optimizer, dict[str, Any]]:
    for optimizer_idx, optimizer in enumerate(snapshot.optimizers):
        for group in optimizer.param_groups:
            names = {
                snapshot.names_by_id[id(param)] for param in group["params"]
            }
            if names == expected_names:
                return optimizer_idx, optimizer, group
    groups = [
        group_names(snapshot, optimizer, group_idx)
        for optimizer in snapshot.optimizers
        for group_idx, _ in enumerate(optimizer.param_groups)
    ]
    raise AssertionError(
        f"Could not find parameter group {expected_names}. Groups: {groups}"
    )


def assert_no_duplicate_parameters(snapshot: OptimizerSnapshot) -> None:
    param_ids = [
        id(param)
        for optimizer in snapshot.optimizers
        for group in optimizer.param_groups
        for param in group["params"]
    ]
    assert len(param_ids) == len(set(param_ids))


def assert_all_trainable_parameters_assigned(
    snapshot: OptimizerSnapshot,
) -> None:
    assert optimizer_parameter_ids(snapshot) == trainable_parameter_ids(
        snapshot
    )


def assert_group_options(
    group: dict[str, Any], expected: dict[str, float | int]
) -> None:
    for key, value in expected.items():
        assert group[key] == pytest.approx(value)
