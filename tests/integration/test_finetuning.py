from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import pytest
from luxonis_ml.typing import Params
from torch import nn
from torch._prims_common import Tensor
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import ConstantLR, LambdaLR, StepLR

from luxonis_train import BaseNode, LuxonisModel, Tasks
from luxonis_train.config.config import ParameterPattern


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


def _head_node(finetuning: Any | None = None) -> Params:
    node: Params = {
        "name": "Head",
        "losses": [{"name": "CrossEntropyLoss"}],
        "metrics": [{"name": "Accuracy"}],
    }
    if finetuning is not None:
        node["finetuning"] = finetuning
    return node


def _tiny_head_node(finetuning: Any | None = None) -> Params:
    node = _head_node(finetuning)
    node["name"] = "TinyHead"
    node["alias"] = "Head"
    return node


def _node(name: str, finetuning: Any | None = None) -> Params:
    if name == "Head":
        return _head_node(finetuning)
    node: Params = {"name": name}
    if finetuning is not None:
        node["finetuning"] = finetuning
    return node


def _config(nodes: list[Params], trainer: Params | None = None) -> Params:
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


def _build_snapshot(config: Params, opts: Params) -> OptimizerSnapshot:
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
        names_by_id=_parameter_names_by_id(model),
    )


def _parameter_names_by_id(model: LuxonisModel) -> dict[int, str]:
    names: dict[int, str] = {}
    for node_name, node in model.lightning_module.nodes.items():
        for module_name, module in node.module.named_modules():
            if list(module.parameters()) and not list(module.children()):
                for param_name, param in module.named_parameters():
                    name = (
                        f"{node_name}.{module.__class__.__name__}."
                        f"{module_name}.{param_name}"
                    )
                    names[id(param)] = name
    return names


def _scheduler(scheduler: Any) -> Any:
    if isinstance(scheduler, dict):
        return scheduler["scheduler"]
    return scheduler


def _group_names(
    snapshot: OptimizerSnapshot, optimizer: Optimizer, group_idx: int
) -> set[str]:
    return {
        snapshot.names_by_id[id(param)]
        for param in optimizer.param_groups[group_idx]["params"]
    }


def _all_optimizer_names(snapshot: OptimizerSnapshot) -> set[str]:
    return {
        snapshot.names_by_id[id(param)]
        for optimizer in snapshot.optimizers
        for group in optimizer.param_groups
        for param in group["params"]
    }


def _matching_names(snapshot: OptimizerSnapshot, *parts: str) -> set[str]:
    return {
        name
        for name in snapshot.names_by_id.values()
        if all(part in name for part in parts)
    }


def _find_group(
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
        _group_names(snapshot, optimizer, group_idx)
        for optimizer in snapshot.optimizers
        for group_idx, _ in enumerate(optimizer.param_groups)
    ]
    raise AssertionError(
        f"Could not find parameter group {expected_names}. Groups: {groups}"
    )


def _assert_no_duplicate_parameters(snapshot: OptimizerSnapshot) -> None:
    param_ids = [
        id(param)
        for optimizer in snapshot.optimizers
        for group in optimizer.param_groups
        for param in group["params"]
    ]
    assert len(param_ids) == len(set(param_ids))


@pytest.fixture
def config() -> Params:
    return _config(
        [
            _node(
                "Backbone",
                {
                    "parameters": [
                        {"name": "conv1"},
                        {"name": "conv2"},
                    ],
                    "optimizer": {
                        "params": {"lr": 0.001},
                    },
                },
            ),
            _node(
                "Neck",
                {
                    "optimizer": {"name": "AdamW"},
                },
            ),
            _head_node(
                [
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
                ]
            ),
        ]
    )


def test_finetuning(config: Params, opts: Params):
    snapshot = _build_snapshot(config, opts)
    optimizers = snapshot.optimizers
    schedulers = snapshot.schedulers
    assert len(optimizers) == len(schedulers) == 4


@pytest.mark.parametrize(
    ("parameters", "expected_parts"),
    [
        (None, ("Head.",)),
        ("fc", ("Head.Linear.fc",)),
        ([{"name": "branch[12]\\.0"}], ("branch1.0", "branch2.0")),
        ([{"module_type": "Linear"}], ("Head.Linear.fc",)),
        ([{"name": "fc", "module_type": "Linear"}], ("Head.Linear.fc",)),
        ([{"name": "branch1\\.0"}, {"name": "fc"}], ("branch1.0", "fc")),
    ],
)
def test_valid_parameter_selectors(
    parameters: Any, expected_parts: tuple[str, ...], opts: Params
):
    snapshot = _build_snapshot(
        _config(
            [
                _tiny_head_node(
                    {
                        "parameters": parameters,
                        "optimizer": {
                            "name": "SGD",
                            "params": {"lr": 0.123},
                        },
                    }
                )
            ]
        ),
        opts,
    )

    expected_names = set().union(
        *(_matching_names(snapshot, part) for part in expected_parts)
    )
    _, optimizer, group = _find_group(snapshot, expected_names)
    assert isinstance(optimizer, SGD)
    assert group["lr"] == pytest.approx(0.123)
    _assert_no_duplicate_parameters(snapshot)


def test_combined_selector_requires_name_and_module_type():
    assert ParameterPattern(name="fc", module_type="Linear").matches(
        "Linear", "fc.weight"
    )
    assert not ParameterPattern(name="fc", module_type="Conv2d").matches(
        "Linear", "fc.weight"
    )


def test_overlapping_rules_claim_parameters_once(opts: Params):
    snapshot = _build_snapshot(
        _config(
            [
                _tiny_head_node(
                    [
                        {
                            "parameters": [{"name": "branch1"}],
                            "optimizer": {"params": {"lr": 0.001}},
                        },
                        {
                            "parameters": [{"module_type": "Conv2d"}],
                            "optimizer": {"params": {"lr": 0.002}},
                        },
                    ]
                )
            ]
        ),
        opts,
    )

    branch1 = _matching_names(snapshot, "Head.Conv2d.branch1")
    remaining_convs = _matching_names(snapshot, "Head.Conv2d") - branch1
    _, _, branch1_group = _find_group(snapshot, branch1)
    _, _, remaining_group = _find_group(snapshot, remaining_convs)

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 1
    assert branch1_group["lr"] == pytest.approx(0.001)
    assert remaining_group["lr"] == pytest.approx(0.002)
    assert branch1.isdisjoint(remaining_convs)
    _assert_no_duplicate_parameters(snapshot)


def test_same_optimizer_scheduler_minimizes_to_one_optimizer(opts: Params):
    snapshot = _build_snapshot(
        _config(
            [
                _node(
                    "Backbone",
                    [
                        {"parameters": [{"name": "conv1"}]},
                        {"parameters": [{"name": "conv2"}]},
                    ],
                ),
                _node("Neck"),
                _head_node(),
            ],
            trainer={
                "optimizer": {"name": "Adam", "params": {"lr": 1e-4}},
                "scheduler": {
                    "name": "ConstantLR",
                    "params": {"factor": 1.0},
                },
            },
        ),
        opts,
    )

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 1
    assert isinstance(snapshot.optimizers[0], Adam)
    assert isinstance(_scheduler(snapshot.schedulers[0]), ConstantLR)
    #!CHECK
    assert len(snapshot.optimizers[0].param_groups) >= 3
    _find_group(snapshot, _matching_names(snapshot, "Backbone.Conv2d.conv1"))
    _find_group(snapshot, _matching_names(snapshot, "Backbone.Conv2d.conv2"))
    _find_group(snapshot, _matching_names(snapshot, "Backbone.Conv2d.conv3"))
    assert all(
        group["lr"] == pytest.approx(1e-4)
        for group in snapshot.optimizers[0].param_groups
    )


def test_same_optimizer_scheduler_keeps_distinct_hyperparameter_groups(
    opts: Params,
):
    snapshot = _build_snapshot(
        _config(
            [
                _tiny_head_node(
                    [
                        {
                            "parameters": [{"module_type": "Conv2d"}],
                            "optimizer": {"params": {"lr": 1e-3}},
                        },
                        {
                            "parameters": [{"module_type": "Linear"}],
                            "optimizer": {"params": {"lr": 1e-2}},
                        },
                    ]
                )
            ]
        ),
        opts,
    )

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 1
    assert isinstance(snapshot.optimizers[0], Adam)
    assert isinstance(_scheduler(snapshot.schedulers[0]), ConstantLR)
    assert len(snapshot.optimizers[0].param_groups) == 2
    _, _, conv_group = _find_group(
        snapshot, _matching_names(snapshot, "Head.Conv2d")
    )
    _, _, linear_group = _find_group(
        snapshot, _matching_names(snapshot, "Head.Linear.fc")
    )
    assert conv_group["lr"] == pytest.approx(1e-3)
    assert linear_group["lr"] == pytest.approx(1e-2)


def test_different_optimizer_names_create_separate_optimizers_greedy(
    opts: Params,
):
    snapshot = _build_snapshot(
        _config(
            [
                _tiny_head_node(
                    [
                        {
                            "parameters": [{"module_type": "Conv2d"}],
                            "optimizer": {"name": "SGD"},
                        },
                        {
                            "parameters": [{"module_type": "Linear"}],
                            "optimizer": {"name": "AdamW"},
                        },
                    ]
                )
            ]
        ),
        opts,
    )

    # 2 instead of 3 because no parameters are left for the default optimizer
    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 2
    assert {type(optimizer) for optimizer in snapshot.optimizers} == {
        SGD,
        AdamW,
    }
    _assert_no_duplicate_parameters(snapshot)


def test_different_optimizer_names_create_separate_optimizers(opts: Params):
    snapshot = _build_snapshot(
        _config(
            [
                _tiny_head_node(
                    [
                        {
                            "parameters": [{"name": "conv1"}],
                            "optimizer": {"name": "SGD"},
                        },
                        {
                            "parameters": [{"module_type": "Linear"}],
                            "optimizer": {"name": "AdamW"},
                        },
                    ]
                )
            ]
        ),
        opts,
    )

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 3
    assert {type(optimizer) for optimizer in snapshot.optimizers} == {
        SGD,
        AdamW,
        Adam,
    }
    _assert_no_duplicate_parameters(snapshot)


def test_different_scheduler_names_create_separate_optimizers(opts: Params):
    snapshot = _build_snapshot(
        _config(
            [
                _tiny_head_node(
                    [
                        {
                            "parameters": [{"module_type": "Conv2d"}],
                            "scheduler": {
                                "name": "StepLR",
                                "params": {"step_size": 2},
                            },
                        },
                        {"parameters": [{"module_type": "Linear"}]},
                    ]
                )
            ]
        ),
        opts,
    )

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 2
    assert all(
        isinstance(optimizer, Adam) for optimizer in snapshot.optimizers
    )
    assert {
        type(_scheduler(scheduler)) for scheduler in snapshot.schedulers
    } == {
        ConstantLR,
        StepLR,
    }
    _assert_no_duplicate_parameters(snapshot)


# CHECK
def test_inherits_omitted_optimizer_and_scheduler(opts: Params):
    snapshot = _build_snapshot(
        _config(
            [_tiny_head_node({"parameters": [{"module_type": "Linear"}]})],
            trainer={
                "optimizer": {
                    "name": "Adam",
                    "params": {"lr": 0.001, "weight_decay": 0.1},
                },
                "scheduler": {
                    "name": "StepLR",
                    "params": {"step_size": 5, "gamma": 0.5},
                },
            },
        ),
        opts,
    )

    idx, optimizer, group = _find_group(
        snapshot, _matching_names(snapshot, "Head.Linear.fc")
    )
    assert isinstance(optimizer, Adam)
    assert isinstance(_scheduler(snapshot.schedulers[idx]), StepLR)
    assert group["lr"] == pytest.approx(0.001)
    assert group["weight_decay"] == pytest.approx(0.1)
    assert _scheduler(snapshot.schedulers[idx]).step_size == 5
    assert _scheduler(snapshot.schedulers[idx]).gamma == pytest.approx(0.5)


# CHECK
def test_inherits_name_and_merges_params_when_name_is_missing(opts: Params):
    snapshot = _build_snapshot(
        _config(
            [
                _tiny_head_node(
                    {
                        "parameters": [{"module_type": "Linear"}],
                        "optimizer": {"params": {"lr": 0.002}},
                        "scheduler": {"params": {"gamma": 0.1}},
                    }
                )
            ],
            trainer={
                "optimizer": {
                    "name": "Adam",
                    "params": {"lr": 0.001, "weight_decay": 0.1},
                },
                "scheduler": {
                    "name": "StepLR",
                    "params": {"step_size": 5, "gamma": 0.5},
                },
            },
        ),
        opts,
    )

    idx, optimizer, group = _find_group(
        snapshot, _matching_names(snapshot, "Head.Linear.fc")
    )
    scheduler = _scheduler(snapshot.schedulers[idx])
    assert isinstance(optimizer, Adam)
    assert isinstance(scheduler, StepLR)
    assert group["lr"] == pytest.approx(0.002)
    assert group["weight_decay"] == pytest.approx(0.1)
    assert scheduler.step_size == 5
    assert scheduler.gamma == pytest.approx(0.1)


def test_switching_name_replaces_inherited_params(opts: Params):
    snapshot = _build_snapshot(
        _config(
            [
                _tiny_head_node(
                    {
                        "parameters": [{"module_type": "Linear"}],
                        "optimizer": {
                            "name": "SGD",
                            "params": {"lr": 0.03},
                        },
                        "scheduler": {
                            "name": "ConstantLR",
                            "params": {"factor": 1.0, "total_iters": 2},
                        },
                    }
                )
            ],
            trainer={
                "optimizer": {
                    "name": "Adam",
                    "params": {"lr": 0.001, "weight_decay": 0.1},
                },
                "scheduler": {
                    "name": "StepLR",
                    "params": {"step_size": 5, "gamma": 0.5},
                },
            },
        ),
        opts,
    )

    idx, optimizer, group = _find_group(
        snapshot, _matching_names(snapshot, "Head.Linear.fc")
    )
    scheduler = _scheduler(snapshot.schedulers[idx])
    assert isinstance(optimizer, SGD)
    assert isinstance(scheduler, ConstantLR)
    assert group["lr"] == pytest.approx(0.03)
    assert group["weight_decay"] == pytest.approx(0)
    assert scheduler.total_iters == 2


@pytest.mark.parametrize(
    ("finetuning", "match"),
    [
        ({"parameters": []}, "at least one parameter pattern"),
        ({"parameters": ""}, "cannot be empty"),
        ({"parameters": [1]}, "Parameter patterns must be"),
        (
            {"parameters": [{"name": "missing"}]},
            "did not match any available trainable parameters",
        ),
    ],
)
def test_invalid_parameter_selectors(
    finetuning: Params, match: str, opts: Params
):
    with pytest.raises(ValueError, match=match):
        _build_snapshot(_config([_tiny_head_node(finetuning)]), opts)


@pytest.mark.parametrize(
    "finetuning",
    [
        {
            "parameters": [{"module_type": "Linear"}],
            "optimizer": {"name": "MissingOptimizer"},
        },
        {
            "parameters": [{"module_type": "Linear"}],
            "scheduler": {"name": "MissingScheduler"},
        },
    ],
)
def test_unknown_optimizer_or_scheduler_name_raises(
    finetuning: Params, opts: Params
):
    with pytest.raises(KeyError):
        _build_snapshot(_config([_tiny_head_node(finetuning)]), opts)


@pytest.mark.parametrize(
    "finetuning",
    [
        {
            "parameters": [{"module_type": "Linear"}],
            "optimizer": {"params": {"not_a_param": True}},
        },
        {
            "parameters": [{"module_type": "Linear"}],
            "scheduler": {"params": {"not_a_param": True}},
        },
    ],
)
def test_invalid_optimizer_or_scheduler_params_raise(
    finetuning: Params, opts: Params
):
    with pytest.raises(TypeError):
        _build_snapshot(_config([_tiny_head_node(finetuning)]), opts)


def test_training_strategy_excludes_finetuned_parameters(opts: Params):
    snapshot = _build_snapshot(
        _config(
            [
                _tiny_head_node(
                    {
                        "parameters": [{"name": "fc"}],
                        "optimizer": {"name": "Adam"},
                    }
                )
            ],
            trainer={
                "training_strategy": {
                    "name": "TripleLRSGDStrategy",
                    "params": {"lr": 0.02},
                }
            },
        ),
        opts,
    )

    fc_names = _matching_names(snapshot, "Head.Linear.fc")
    idx, finetuning_optimizer, _ = _find_group(snapshot, fc_names)
    strategy_optimizers = [
        optimizer
        for optimizer_idx, optimizer in enumerate(snapshot.optimizers)
        if optimizer_idx != idx
    ]
    assert isinstance(finetuning_optimizer, Adam)
    assert len(strategy_optimizers) == 1
    assert isinstance(strategy_optimizers[0], SGD)
    assert all(
        isinstance(_scheduler(scheduler), LambdaLR)
        for scheduler in snapshot.schedulers
    )
    strategy_names = {
        snapshot.names_by_id[id(param)]
        for group in strategy_optimizers[0].param_groups
        for param in group["params"]
    }
    assert strategy_names == _all_optimizer_names(snapshot) - fc_names
    assert strategy_names.isdisjoint(fc_names)
    _assert_no_duplicate_parameters(snapshot)
