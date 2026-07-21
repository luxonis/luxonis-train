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
    """End-to-end sanity check covering every merging axis at once.

    Setup:
        - No trainer optimizer/scheduler is set explicitly, so the base
          defaults to Adam + ConstantLR(factor=1.0).
        - Backbone: one rule targeting ``conv1`` and ``conv2`` with
          ``lr=0.001``, inheriting Adam and the base ConstantLR.
        - Neck: one rule with no parameter selector (matches everything
          under Neck) overriding the optimizer to AdamW.
        - Head: three rules — ``branch1`` with SGD+CosineAnnealingLR,
          ``Linear`` with Adam+``wd=0.01``+ConstantLR, and remaining
          ``Conv2d`` with Adam+``wd=0.02``+StepLR.

    Expected result (4 optimizers, 4 schedulers):
        1. Adam / ConstantLR — three groups merged by identical
           (optimizer, scheduler, scheduler_params) key: Backbone
           conv1+conv2, Backbone conv3 (from the implicit per-node
           default catch-all), and Head Linear.fc.
        2. AdamW / ConstantLR — Neck (different optimizer name splits
           it off from the Adam bucket).
        3. SGD / CosineAnnealingLR — Head branch1 (different optimizer
           name).
        4. Adam / StepLR — remaining Head Conv2d (same optimizer as #1
           but a different scheduler splits it into its own optimizer).
    """
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
    """Baseline case: no finetuning rules anywhere.

    Setup:
        - No node has ``finetuning`` configured.
        - Trainer overrides the defaults with AdamW(lr=0.004) and
          StepLR(step_size=3, gamma=0.7).

    Expected result:
        A single AdamW optimizer with a single parameter group
        containing every trainable parameter, and a single StepLR
        scheduler with the configured hyperparameters. The finetuning
        code path is bypassed entirely — the fast default in
        ``_extract_optimizer_params`` short-circuits and collects all
        params in one pass.
    """
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
    """Sweep every combination of "does the rule change the optimizer /
    the scheduler / both / neither" on Backbone.conv1 vs conv2, with
    Neck and Head left as defaults. Base trainer is Adam+ConstantLR.

    Each rule contributes 1 param group; every node also contributes a
    default catch-all rule for any unclaimed params (Backbone.conv3,
    Neck.*, Head.*). Rules that end up with the same
    ``(optimizer_name, scheduler_name, scheduler_params)`` key are
    merged into one optimizer with multiple param groups.

    Cases (in the order listed above):
        1. Two rules, both fully inheriting Adam+ConstantLR. All 5
           resulting groups (conv1, conv2, plus 3 defaults) share the
           same key → 1 Adam optimizer with 5 groups.
        2. Rule 1 overrides optimizer to SGD, rule 2 to AdamW. The
           three default groups still land on Adam+ConstantLR, giving
           3 optimizers (SGD, AdamW, Adam) and 5 total groups.
        3. Rule 1 attaches StepLR (different scheduler name), rule 2
           inherits. rule 1 → its own optimizer; rule 2 + 3 defaults
           merge into a shared Adam+ConstantLR optimizer → 2
           optimizers, 5 groups.
        4. Both rules override only optimizer *params* (different lrs).
           Optimizer name and scheduler are unchanged, so all 5 groups
           merge into a single Adam optimizer with distinct
           per-group hyperparameters.
    """
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
    """Two rules that agree on optimizer *and* scheduler but differ in
    their per-group hyperparameters must merge into a single optimizer
    with two param groups.

    Setup:
        - Head has two rules, both defaulting to Adam+ConstantLR.
        - Rule 1 targets Conv2d modules with ``lr=1e-3``.
        - Rule 2 targets Linear modules with ``lr=1e-2``.

    Expected result:
        A single Adam optimizer with two param groups (one per rule),
        each carrying its own learning rate. Only one ConstantLR
        scheduler is created — schedulers are per-optimizer, not
        per-group.
    """
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
    """Scheduler *parameters* participate in the group key alongside the
    scheduler name — two StepLR schedulers with different
    ``step_size``/``gamma`` cannot share an optimizer.

    Setup:
        - Head has two rules, both Adam.
        - Rule 1 (Conv2d): ``StepLR(step_size=2, gamma=0.5)``,
          ``lr=1e-3``.
        - Rule 2 (Linear): ``StepLR(step_size=4, gamma=0.8)``,
          ``lr=1e-2``.

    Expected result:
        Two Adam optimizers, each with its own StepLR (parameters
        preserved verbatim). This is what makes distinct schedules per
        parameter subset actually possible.
    """
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


def test_cosine_annealing_lr_t_max_is_supplied_without_mutating_config(
    opts: Params,
):
    """``CosineAnnealingLR`` requires a ``T_max`` argument that isn't
    generally known at config-authoring time — the builder auto-fills it
    from ``trainer.epochs`` when the user leaves it out.

    Setup:
        - Trainer runs 7 epochs with ``CosineAnnealingLR`` and no
          explicit ``T_max``.

    Expected result:
        A single ``CosineAnnealingLR`` with ``T_max == 7``. The
        original ``scheduler.params`` on the config must remain empty
        — the auto-fill goes into the constructed scheduler, not back
        into the config object (mutating it would corrupt later
        rebuilds and any serialization round-trip).
    """
    snapshot = build_snapshot(
        config(
            [tiny_head_node({"parameters": [{"module_type": "Linear"}]})],
            trainer={
                "epochs": 7,
                "scheduler": {"name": "CosineAnnealingLR"},
            },
        ),
        opts,
    )

    scheduler_cfg = scheduler(snapshot.schedulers[0])

    assert isinstance(scheduler_cfg, CosineAnnealingLR)
    assert (
        scheduler_cfg.T_max
        == snapshot.model.lightning_module.cfg.trainer.epochs
    )
    assert snapshot.model.lightning_module.cfg.trainer.scheduler.params == {}
    assert_no_duplicate_parameters(snapshot)
    assert_all_trainable_parameters_assigned(snapshot)


def test_reduce_on_plateau_monitor_uses_formatted_main_metric_name(
    opts: Params,
):
    """``ReduceLROnPlateau`` needs a ``monitor`` string pointing at a
    logged Lightning metric — the builder derives it from the designated
    main metric.

    Setup:
        - A single Head configured with ``task_name='classification'``
          and Accuracy marked as the main metric.
        - The only rule attaches ``ReduceLROnPlateau(mode='max')`` with
          no explicit ``monitor``.

    Expected result:
        The scheduler is returned as a dict (Lightning's config-style
        payload, not a bare scheduler) with
        ``monitor='val/metric/classification-Head/Accuracy'`` — the
        exact key Lightning logs during validation.
    """
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
    """When two rules would match the same parameter, the earlier rule
    wins — no parameter appears in more than one optimizer group.

    Setup:
        - Head has two rules using the default Adam+ConstantLR.
        - Rule 1: ``name='branch1'`` with ``lr=0.001``.
        - Rule 2: ``module_type='Conv2d'`` with ``lr=0.002``. This
          would *also* match the ``branch1`` conv layers if they were
          still available.

    Expected result:
        Rule 1 claims all ``branch1`` params first; rule 2 sees them as
        already-claimed and only picks up the remaining Conv2d layers.
        Both rules end up in a single Adam optimizer (same
        optimizer+scheduler key), each with the correct lr on its own
        group.
    """
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
    """The implicit per-node default catch-all rule sweeps up whatever
    the explicit finetuning rules didn't claim, using the trainer's base
    optimizer settings.

    Setup:
        - Head has one explicit rule targeting ``branch1`` with SGD.
        - Trainer base optimizer is AdamW(lr=0.004); base scheduler
          uses the config helper's ConstantLR(factor=1.0).

    Expected result:
        Two optimizers — the SGD one holding ``branch1`` (from the
        explicit rule) and an AdamW one holding *every other
        trainable parameter* (from the implicit default rule),
        carrying the trainer's ``lr=0.004``. This is what lets users
        override only a subset without having to enumerate the rest.
    """
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
