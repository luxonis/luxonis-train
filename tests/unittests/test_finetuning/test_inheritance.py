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
    """Trainer base optimizer is Adam(lr=0.001, weight_decay=0.1).

    The single Head rule targets Linear modules and varies its optimizer
    block. The merge semantics come from
    ``merge_config_items``: when the override omits ``name`` or matches
    the base name, params are merged (override wins per-key). When
    ``name`` differs, the override *replaces* both name and params —
    the new optimizer's own class defaults fill any remaining slots.

    Cases (in the order listed above):
        1. ``None`` — no override at all → inherits Adam(lr=0.001,
           wd=0.1) unchanged.
        2. Params-only override (``lr=0.002``) → same Adam, lr
           replaced, wd inherited from base.
        3. Same name + params override → Adam(lr=0.003, wd=0.1);
           behaves identically to case 2 because ``name`` matches.
        4. Different name (SGD) → SGD(lr=0.03); base's ``wd=0.1`` is
           discarded and SGD has no weight_decay default → the group
           reports ``weight_decay=0`` (the extra assertion at the
           bottom).
        5. Different name (AdamW) → AdamW(lr=0.04); base's wd is
           discarded but AdamW's class default of 0.01 fills in.
    """
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
    """Same merge rules as optimizers, applied to the scheduler.

    Trainer base scheduler is ``StepLR(step_size=5, gamma=0.5)``.

    Cases (in the order listed above):
        1. ``None`` — inherits StepLR(5, 0.5) unchanged.
        2. Params-only (``gamma=0.1``) → StepLR(5, 0.1); step_size
           inherited from base.
        3. Same name + params override (``gamma=0.2``) → StepLR(5,
           0.2); merges per-key.
        4. Different name (ConstantLR) → the base StepLR params are
           discarded and only the override's params
           (``factor=1.0, total_iters=2``) apply.
    """
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
    """Sanity-check that optimizer and scheduler inheritance stay
    independent when both are involved in a single rule. Base is
    Adam(lr=0.001, wd=0.1) + StepLR(step_size=5, gamma=0.5).

    Cases (in the order listed above):
        1. Empty override — everything inherited: Adam(lr=0.001,
           wd=0.1) + StepLR(5, 0.5).
        2. Params-only overrides on both → Adam(lr=0.002, wd=0.1) +
           StepLR(5, gamma=0.1). Merging happens independently on each
           side.
        3. Different name on both → SGD(lr=0.03) (base's wd dropped,
           SGD default weight_decay=0) + ConstantLR(1.0, total_iters=2)
           (base's StepLR params dropped entirely).
    """
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
