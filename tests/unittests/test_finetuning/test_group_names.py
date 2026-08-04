import re
from pathlib import Path
from typing import Any, cast

import pytest
from luxonis_ml.typing import Params

import luxonis_train as lxt
from luxonis_train.lightning.training_plan import (
    _unique_name,
    unwrap_optimizers,
)

from ._helpers import build_snapshot, config, node, tiny_head_node

FINETUNING = [
    {
        "parameters": [{"module_type": "Linear"}],
        "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
    }
]

PG_SUFFIX = re.compile(r"pg\d+")


def test_every_parameter_group_carries_its_plan_name(opts: Params):
    """Every group Lightning sees must expose the plan's name for it, so
    that `LearningRateMonitor` reports a label instead of a position.
    """
    snapshot = build_snapshot(
        config([tiny_head_node(FINETUNING)]),
        opts,
    )
    plan = snapshot.model.lightning_module.training_plan
    assert plan is not None

    planned = [
        group.name for inner in plan.plan.inners for group in inner.groups
    ]
    actual = [
        group["name"]
        for optimizer in plan.inner_optimizers
        for group in optimizer.param_groups
    ]
    assert actual == planned
    # The rule's own group and the node's tail are told apart by name.
    assert set(actual) == {"Head/0", "default/Head"}


def test_group_names_are_unique_across_inner_optimizers(opts: Params):
    """Group names must be unique globally, not merely per inner.

    `LearningRateMonitor` rejects an optimizer holding two groups of the
    same name, and the composite presents every inner's groups as one
    list.
    """
    convolutions = [
        {
            "parameters": [{"module_type": "Conv2d"}],
            "optimizer": {"name": "SGD", "params": {"lr": 0.1}},
        }
    ]
    snapshot = build_snapshot(
        config(
            [
                node("Backbone", convolutions),
                node("Neck", convolutions),
                node("Head", FINETUNING),
            ]
        ),
        opts,
    )
    plan = snapshot.model.lightning_module.training_plan
    assert plan is not None
    assert len(plan.inner_optimizers) > 1, "expected the composite path"

    names = [
        group["name"]
        for optimizer in plan.inner_optimizers
        for group in optimizer.param_groups
    ]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["a", "b"], ["a", "b"]),
        (["a", "a"], ["a", "a-2"]),
        (["a", "a", "a"], ["a", "a-2", "a-3"]),
        (["a", "a-2", "a"], ["a", "a-2", "a-3"]),
    ],
)
def test_colliding_names_are_disambiguated(
    names: list[str], expected: list[str]
):
    used: set[str] = set()
    assert [_unique_name(name, used) for name in names] == expected


def test_unknown_group_options_are_still_rejected(opts: Params):
    """Allowing `name` through the group-option check must not turn it
    into a hole for genuinely invalid options.
    """
    with pytest.raises(TypeError, match="nonsense"):
        build_snapshot(
            config(
                [
                    tiny_head_node(
                        [
                            {
                                "parameters": [{"module_type": "Linear"}],
                                "optimizer": {
                                    "name": "SGD",
                                    "params": {"lr": 0.1, "nonsense": 1},
                                },
                            }
                        ]
                    )
                ]
            ),
            opts,
        )


def test_learning_rate_monitor_logs_one_named_series_per_group(
    opts: Params, tmp_path: Path
):
    """The end-to-end complaint: with several parameter groups, the
    tracker used to receive positional `lr-SGD/pg1`, `lr-SGD/pg2` keys
    that could not be traced back to a group.
    """
    model = lxt.LuxonisModel(
        config([tiny_head_node(FINETUNING)]),
        opts
        | {
            "loader.params.n_classes": 10,
            "trainer.epochs": 1,
            "trainer.accelerator": "cpu",
            "trainer.n_sanity_val_steps": 0,
            "tracker.save_directory": str(tmp_path),
            "trainer.callbacks": [
                *cast(list[Params], opts["trainer.callbacks"]),
                {"name": "LearningRateMonitor"},
            ],
        },
        allow_empty_dataset=True,
    )

    logged: set[str] = set()
    tracker = model.tracker
    original = tracker.log_metrics

    def spy(metrics: dict[str, float], step: int) -> Any:
        logged.update(metrics)
        return original(metrics, step)

    tracker.log_metrics = spy
    model.train()

    # `LearningRateMonitor` prefixes each series with the scheduler
    # config name; only the suffix identifies the group.
    learning_rates = {
        key.split("/", 1)[1] for key in logged if key.startswith("lr")
    }
    assert learning_rates, "`LearningRateMonitor` logged no learning rate"

    optimizers = unwrap_optimizers(model.lightning_module.trainer.optimizers)
    assert learning_rates == {
        group["name"]
        for optimizer in optimizers
        for group in optimizer.param_groups
    }
    # The complaint verbatim: positional `pg1`/`pg2` series are gone.
    assert not any(
        PG_SUFFIX.fullmatch(key.rsplit("/", 1)[-1]) for key in learning_rates
    )
