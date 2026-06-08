from typing import Any

import pytest
from luxonis_ml.typing import Params
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ConstantLR, StepLR

from ._helpers import (
    assert_group_options,
    build_snapshot,
    config,
    find_group,
    matching_names,
    scheduler,
    tiny_head_node,
)


@pytest.mark.parametrize(
    (
        "override",
        "expected_optimizer_type",
        "expected_group_options",
    ),
    [
        (None, Adam, {"lr": 0.001, "weight_decay": 0.1}),
        (
            {"params": {"lr": 0.002}},
            Adam,
            {"lr": 0.002, "weight_decay": 0.1},
        ),
        (
            {"name": "Adam", "params": {"lr": 0.003}},
            Adam,
            {"lr": 0.003, "weight_decay": 0.1},
        ),
        ({"name": "SGD", "params": {"lr": 0.03}}, SGD, {"lr": 0.03}),
        (
            {"name": "AdamW", "params": {"lr": 0.04}},
            AdamW,
            {"lr": 0.04, "weight_decay": 0.01},
        ),
    ],
)
def test_optimizer_inheritance_and_override(
    override: Params | None,
    expected_optimizer_type: type,
    expected_group_options: dict[str, float],
    opts: Params,
):
    finetuning: dict[str, Any] = {"parameters": [{"module_type": "Linear"}]}
    if override is not None:
        finetuning["optimizer"] = override

    snapshot = build_snapshot(
        config(
            [tiny_head_node(finetuning)],
            trainer={
                "optimizer": {
                    "name": "Adam",
                    "params": {"lr": 0.001, "weight_decay": 0.1},
                },
            },
        ),
        opts,
    )

    _, optimizer, group = find_group(
        snapshot, matching_names(snapshot, "Head.Linear.fc")
    )
    assert isinstance(optimizer, expected_optimizer_type)
    assert_group_options(group, expected_group_options)
    if expected_optimizer_type is SGD:
        assert group["weight_decay"] == pytest.approx(0)


@pytest.mark.parametrize(
    ("override", "expected_scheduler_type", "expected_attrs"),
    [
        (None, StepLR, {"step_size": 5, "gamma": 0.5}),
        ({"params": {"gamma": 0.1}}, StepLR, {"step_size": 5, "gamma": 0.1}),
        (
            {"name": "StepLR", "params": {"gamma": 0.2}},
            StepLR,
            {"step_size": 5, "gamma": 0.2},
        ),
        (
            {
                "name": "ConstantLR",
                "params": {"factor": 1.0, "total_iters": 2},
            },
            ConstantLR,
            {"factor": 1.0, "total_iters": 2},
        ),
    ],
)
def test_scheduler_inheritance_and_override(
    override: Params | None,
    expected_scheduler_type: type,
    expected_attrs: dict[str, Any],
    opts: Params,
):
    finetuning: dict[str, Any] = {"parameters": [{"module_type": "Linear"}]}
    if override is not None:
        finetuning["scheduler"] = override

    snapshot = build_snapshot(
        config(
            [tiny_head_node(finetuning)],
            trainer={
                "scheduler": {
                    "name": "StepLR",
                    "params": {"step_size": 5, "gamma": 0.5},
                },
            },
        ),
        opts,
    )

    idx, _, _ = find_group(
        snapshot, matching_names(snapshot, "Head.Linear.fc")
    )
    scheduler_cfg = scheduler(snapshot.schedulers[idx])
    assert isinstance(scheduler_cfg, expected_scheduler_type)
    for attr, value in expected_attrs.items():
        assert getattr(scheduler_cfg, attr) == pytest.approx(value)


@pytest.mark.parametrize(
    ("finetuning", "expected_optimizer_type", "expected_scheduler_type"),
    [
        ({}, Adam, StepLR),
        (
            {
                "optimizer": {"params": {"lr": 0.002}},
                "scheduler": {"params": {"gamma": 0.1}},
            },
            Adam,
            StepLR,
        ),
        (
            {
                "optimizer": {"name": "SGD", "params": {"lr": 0.03}},
                "scheduler": {
                    "name": "ConstantLR",
                    "params": {"factor": 1.0, "total_iters": 2},
                },
            },
            SGD,
            ConstantLR,
        ),
    ],
)
def test_optimizer_and_scheduler_inheritance_together(
    finetuning: Params,
    expected_optimizer_type: type,
    expected_scheduler_type: type,
    opts: Params,
):
    finetuning = {
        "parameters": [{"module_type": "Linear"}],
        **finetuning,
    }
    snapshot = build_snapshot(
        config(
            [tiny_head_node(finetuning)],
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

    idx, optimizer, group = find_group(
        snapshot, matching_names(snapshot, "Head.Linear.fc")
    )
    scheduler_cfg = scheduler(snapshot.schedulers[idx])
    assert isinstance(optimizer, expected_optimizer_type)
    assert isinstance(scheduler_cfg, expected_scheduler_type)
    if expected_optimizer_type is SGD:
        assert_group_options(group, {"lr": 0.03})
        assert group["weight_decay"] == pytest.approx(0)
        assert scheduler_cfg.total_iters == 2
    else:
        assert group["weight_decay"] == pytest.approx(0.1)
        assert scheduler_cfg.step_size == 5
