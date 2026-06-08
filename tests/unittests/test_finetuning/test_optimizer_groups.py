import pytest
from luxonis_ml.typing import Params
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)

from ._helpers import (
    assert_all_trainable_parameters_assigned,
    assert_group_options,
    assert_no_duplicate_parameters,
    build_snapshot,
    config,
    find_group,
    head_node,
    matching_names,
    node,
    optimizer_group_names,
    optimizer_parameter_names,
    scheduler,
    tiny_head_node,
    trainable_parameter_names,
)


@pytest.fixture
def representative_config() -> Params:
    return config(
        [
            node(
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
            node(
                "Neck",
                {
                    "optimizer": {"name": "AdamW"},
                },
            ),
            head_node(
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


def test_representative_finetuning_builds_expected_optimizer_groups(
    representative_config: Params, opts: Params
):
    snapshot = build_snapshot(representative_config, opts)

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 4
    assert [type(optimizer) for optimizer in snapshot.optimizers] == [
        Adam,
        AdamW,
        SGD,
        Adam,
    ]
    assert [
        type(scheduler_cfg)
        for scheduler_cfg in map(scheduler, snapshot.schedulers)
    ] == [ConstantLR, ConstantLR, CosineAnnealingLR, StepLR]
    assert optimizer_group_names(snapshot, snapshot.optimizers[0]) == [
        matching_names(snapshot, "Backbone.Conv2d.conv1")
        | matching_names(snapshot, "Backbone.Conv2d.conv2"),
        matching_names(snapshot, "Backbone.Conv2d.conv3"),
        matching_names(snapshot, "Head.Linear.fc"),
    ]

    _, _, backbone_group = find_group(
        snapshot,
        matching_names(snapshot, "Backbone.Conv2d.conv1")
        | matching_names(snapshot, "Backbone.Conv2d.conv2"),
    )
    _, neck_optimizer, _ = find_group(
        snapshot, matching_names(snapshot, "Neck.")
    )
    _, head_sgd_optimizer, head_branch_group = find_group(
        snapshot, matching_names(snapshot, "Head.Conv2d.branch1")
    )
    _, _, head_linear_group = find_group(
        snapshot, matching_names(snapshot, "Head.Linear.fc")
    )
    _, _, head_conv_group = find_group(
        snapshot,
        matching_names(snapshot, "Head.Conv2d")
        - matching_names(snapshot, "branch1"),
    )

    assert isinstance(neck_optimizer, AdamW)
    assert isinstance(head_sgd_optimizer, SGD)
    assert_group_options(backbone_group, {"lr": 0.001})
    assert_group_options(head_branch_group, {"lr": 0.01})
    assert_group_options(head_linear_group, {"weight_decay": 0.01})
    assert_group_options(head_conv_group, {"weight_decay": 0.02})
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


def test_no_finetuning_uses_single_default_optimizer_for_all_trainable_params(
    opts: Params,
):
    snapshot = build_snapshot(
        config(
            [node("Backbone"), node("Neck"), head_node()],
            trainer={
                "optimizer": {"name": "AdamW", "params": {"lr": 0.004}},
                "scheduler": {
                    "name": "StepLR",
                    "params": {"step_size": 3, "gamma": 0.7},
                },
            },
        ),
        opts,
    )

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 1
    assert isinstance(snapshot.optimizers[0], AdamW)
    assert len(snapshot.optimizers[0].param_groups) == 1
    assert isinstance(scheduler(snapshot.schedulers[0]), StepLR)
    assert scheduler(snapshot.schedulers[0]).step_size == 3
    assert scheduler(snapshot.schedulers[0]).gamma == pytest.approx(0.7)
    assert optimizer_parameter_names(snapshot) == trainable_parameter_names(
        snapshot
    )
    assert_group_options(snapshot.optimizers[0].param_groups[0], {"lr": 0.004})
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


@pytest.mark.parametrize(
    (
        "finetuning",
        "expected_optimizer_count",
        "expected_scheduler_count",
        "expected_optimizer_types",
        "expected_scheduler_types",
        "expected_group_count",
    ),
    [
        (
            [
                {"parameters": [{"name": "conv1"}]},
                {"parameters": [{"name": "conv2"}]},
            ],
            1,
            1,
            {Adam},
            {ConstantLR},
            5,
        ),
        (
            [
                {
                    "parameters": [{"name": "conv1"}],
                    "optimizer": {"name": "SGD"},
                },
                {
                    "parameters": [{"name": "conv2"}],
                    "optimizer": {"name": "AdamW"},
                },
            ],
            3,
            3,
            {Adam, AdamW, SGD},
            {ConstantLR},
            5,
        ),
        (
            [
                {
                    "parameters": [{"name": "conv1"}],
                    "scheduler": {
                        "name": "StepLR",
                        "params": {"step_size": 2},
                    },
                },
                {"parameters": [{"name": "conv2"}]},
            ],
            2,
            2,
            {Adam},
            {ConstantLR, StepLR},
            5,
        ),
        (
            [
                {
                    "parameters": [{"name": "conv1"}],
                    "optimizer": {"params": {"lr": 0.001}},
                },
                {
                    "parameters": [{"name": "conv2"}],
                    "optimizer": {"params": {"lr": 0.002}},
                },
            ],
            1,
            1,
            {Adam},
            {ConstantLR},
            5,
        ),
    ],
)
def test_grouping_matrix(
    finetuning: list[Params],
    expected_optimizer_count: int,
    expected_scheduler_count: int,
    expected_optimizer_types: set[type],
    expected_scheduler_types: set[type],
    expected_group_count: int,
    opts: Params,
):
    snapshot = build_snapshot(
        config(
            [
                node("Backbone", finetuning),
                node("Neck"),
                head_node(),
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

    assert len(snapshot.optimizers) == expected_optimizer_count
    assert len(snapshot.schedulers) == expected_scheduler_count
    assert {type(optimizer) for optimizer in snapshot.optimizers} == (
        expected_optimizer_types
    )
    assert {
        type(scheduler_cfg)
        for scheduler_cfg in map(scheduler, snapshot.schedulers)
    } == expected_scheduler_types
    assert sum(
        len(optimizer.param_groups) for optimizer in snapshot.optimizers
    ) == (expected_group_count)
    find_group(snapshot, matching_names(snapshot, "Backbone.Conv2d.conv1"))
    find_group(snapshot, matching_names(snapshot, "Backbone.Conv2d.conv2"))
    find_group(snapshot, matching_names(snapshot, "Backbone.Conv2d.conv3"))
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


def test_same_optimizer_scheduler_keeps_distinct_hyperparameter_groups(
    opts: Params,
):
    snapshot = build_snapshot(
        config(
            [
                tiny_head_node(
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
    assert isinstance(scheduler(snapshot.schedulers[0]), ConstantLR)
    assert len(snapshot.optimizers[0].param_groups) == 2
    _, _, conv_group = find_group(
        snapshot, matching_names(snapshot, "Head.Conv2d")
    )
    _, _, linear_group = find_group(
        snapshot, matching_names(snapshot, "Head.Linear.fc")
    )
    assert_group_options(conv_group, {"lr": 1e-3})
    assert_group_options(linear_group, {"lr": 1e-2})
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


def test_same_scheduler_name_with_different_params_uses_distinct_optimizers(
    opts: Params,
):
    snapshot = build_snapshot(
        config(
            [
                tiny_head_node(
                    [
                        {
                            "parameters": [{"module_type": "Conv2d"}],
                            "optimizer": {"params": {"lr": 1e-3}},
                            "scheduler": {
                                "name": "StepLR",
                                "params": {"step_size": 2, "gamma": 0.5},
                            },
                        },
                        {
                            "parameters": [{"module_type": "Linear"}],
                            "optimizer": {"params": {"lr": 1e-2}},
                            "scheduler": {
                                "name": "StepLR",
                                "params": {"step_size": 4, "gamma": 0.8},
                            },
                        },
                    ]
                )
            ]
        ),
        opts,
    )

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 2
    conv_idx, _, conv_group = find_group(
        snapshot, matching_names(snapshot, "Head.Conv2d")
    )
    linear_idx, _, linear_group = find_group(
        snapshot, matching_names(snapshot, "Head.Linear.fc")
    )
    conv_scheduler = scheduler(snapshot.schedulers[conv_idx])
    linear_scheduler = scheduler(snapshot.schedulers[linear_idx])

    assert isinstance(conv_scheduler, StepLR)
    assert isinstance(linear_scheduler, StepLR)
    assert conv_scheduler.step_size == 2
    assert conv_scheduler.gamma == pytest.approx(0.5)
    assert linear_scheduler.step_size == 4
    assert linear_scheduler.gamma == pytest.approx(0.8)
    assert_group_options(conv_group, {"lr": 1e-3})
    assert_group_options(linear_group, {"lr": 1e-2})
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


def test_reduce_on_plateau_monitor_uses_formatted_main_metric_name(
    opts: Params,
):
    node_cfg = tiny_head_node(
        {
            "scheduler": {
                "name": "ReduceLROnPlateau",
                "params": {"mode": "max"},
            }
        }
    )
    node_cfg["task_name"] = "classification"
    node_cfg["metrics"] = [{"name": "Accuracy", "is_main_metric": True}]

    snapshot = build_snapshot(config([node_cfg]), opts)
    scheduler_cfg = snapshot.schedulers[0]

    assert isinstance(scheduler_cfg, dict)
    assert isinstance(scheduler_cfg["scheduler"], ReduceLROnPlateau)
    assert (
        scheduler_cfg["monitor"] == "val/metric/classification-Head/Accuracy"
    )


def test_overlapping_rules_claim_parameters_once(opts: Params):
    snapshot = build_snapshot(
        config(
            [
                tiny_head_node(
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

    branch1 = matching_names(snapshot, "Head.Conv2d.branch1")
    remaining_convs = matching_names(snapshot, "Head.Conv2d") - branch1
    _, _, branch1_group = find_group(snapshot, branch1)
    _, _, remaining_group = find_group(snapshot, remaining_convs)

    assert len(snapshot.optimizers) == len(snapshot.schedulers) == 1
    assert_group_options(branch1_group, {"lr": 0.001})
    assert_group_options(remaining_group, {"lr": 0.002})
    assert branch1.isdisjoint(remaining_convs)
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


def test_default_optimizer_receives_unclaimed_trainable_parameters(
    opts: Params,
):
    snapshot = build_snapshot(
        config(
            [
                tiny_head_node(
                    [
                        {
                            "parameters": [{"name": "branch1"}],
                            "optimizer": {"name": "SGD"},
                        },
                    ]
                )
            ],
            trainer={
                "optimizer": {"name": "AdamW", "params": {"lr": 0.004}},
            },
        ),
        opts,
    )

    branch1 = matching_names(snapshot, "Head.Conv2d.branch1")
    default_names = optimizer_parameter_names(snapshot) - branch1
    _, finetuning_optimizer, _ = find_group(snapshot, branch1)
    _, default_optimizer, default_group = find_group(snapshot, default_names)

    assert isinstance(finetuning_optimizer, SGD)
    assert isinstance(default_optimizer, AdamW)
    assert_group_options(default_group, {"lr": 0.004})
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)
