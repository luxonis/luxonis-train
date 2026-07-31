from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
from luxonis_ml.typing import Params
from torch.optim import SGD, Adam, Optimizer

from luxonis_train import LuxonisModel
from luxonis_train.callbacks import TrainingManager

from ._helpers import (
    OptimizerSnapshot,
    assert_all_trainable_parameters_assigned,
    assert_group_options,
    config,
    find_group,
    matching_names,
    optimizer_names,
    parameter_names_by_id,
    tiny_head_node,
)


def test_unfreezing_restores_parameters_to_configured_optimizers(
    opts: Params,
):
    node = tiny_head_node(
        {
            "parameters": [{"module_type": "Linear"}],
            "optimizer": {
                "name": "SGD",
                "params": {"lr": 0.02},
            },
        }
    )
    node["freezing"] = {"active": True, "unfreeze_after": 1}
    model = LuxonisModel(
        deepcopy(
            config(
                [node],
                trainer={
                    "optimizer": {
                        "name": "Adam",
                        "params": {"lr": 0.003},
                    }
                },
            )
        ),
        opts | {"loader.params.n_classes": 10},
        allow_empty_dataset=True,
    )
    module = model.lightning_module
    manager = TrainingManager()
    manager.freeze_before_training(module)
    optimizers, schedulers = module.configure_optimizers()
    snapshot = OptimizerSnapshot(
        model=model,
        optimizers=list(optimizers),
        schedulers=list(schedulers),
        names_by_id=parameter_names_by_id(model),
    )

    assert [type(optimizer) for optimizer in optimizers] == [SGD, Adam]
    assert all(
        not optimizer_names(snapshot, optimizer) for optimizer in optimizers
    )

    manager.finetune_function(module, 1, optimizers[0])

    linear_names = matching_names(snapshot, "Head.Linear.fc")
    convolution_names = matching_names(snapshot, "Head.Conv2d")
    _, _, linear_group = find_group(snapshot, linear_names)
    _, _, convolution_group = find_group(snapshot, convolution_names)

    assert optimizer_names(snapshot, optimizers[0]) == linear_names
    assert optimizer_names(snapshot, optimizers[1]) == convolution_names
    assert_group_options(linear_group, {"lr": 0.02})
    assert_group_options(convolution_group, {"lr": 0.003})
    assert_all_trainable_parameters_assigned(snapshot)


def test_unfreezing_default_group_does_not_change_active_group_lr(
    opts: Params,
):
    active_node = tiny_head_node()
    active_node["alias"] = "ActiveHead"
    active_node["input_sources"] = ["image"]
    frozen_node = tiny_head_node()
    frozen_node["alias"] = "FrozenHead"
    frozen_node["input_sources"] = ["image"]
    frozen_node["freezing"] = {
        "active": True,
        "unfreeze_after": 1,
        "lr_after_unfreeze": 0.02,
    }
    model = LuxonisModel(
        deepcopy(
            config(
                [active_node, frozen_node],
                trainer={
                    "optimizer": {
                        "name": "Adam",
                        "params": {"lr": 0.003},
                    }
                },
            )
        ),
        opts | {"loader.params.n_classes": 10},
        allow_empty_dataset=True,
    )
    module = model.lightning_module
    manager = TrainingManager()
    manager.freeze_before_training(module)
    optimizers, schedulers = module.configure_optimizers()
    snapshot = OptimizerSnapshot(
        model=model,
        optimizers=list(optimizers),
        schedulers=list(schedulers),
        names_by_id=parameter_names_by_id(model),
    )
    optimizer = optimizers[0]
    active_names = matching_names(snapshot, "ActiveHead")
    frozen_names = matching_names(snapshot, "FrozenHead")

    assert optimizer_names(snapshot, optimizer) == active_names

    manager.finetune_function(module, 1, optimizer)

    _, _, active_group = find_group(snapshot, active_names)
    _, _, frozen_group = find_group(snapshot, frozen_names)
    assert_group_options(active_group, {"lr": 0.003})
    assert_group_options(frozen_group, {"lr": 0.02})
    assert_all_trainable_parameters_assigned(snapshot)


def _frozen_two_optimizer_model(
    opts: Params, finetune_frozen_node: bool
) -> tuple[Any, TrainingManager, list[Optimizer], Any]:
    """Two heads sharing the loader image, one of them frozen until
    epoch 1.

    `finetune_frozen_node` selects which head carries the extra `SGD`
    rule, and therefore whether the unfrozen parameters are restored by
    extending an existing group in place (the frozen head owns the rule)
    or by `add_param_group` on the optimizer at index 1 (the active head
    owns it). Both routes must end up in the checkpointed state.
    """
    rule: Params = {
        "parameters": [{"module_type": "Linear"}],
        "optimizer": {"name": "SGD", "params": {"lr": 0.02}},
    }
    active = tiny_head_node(None if finetune_frozen_node else rule)
    active["alias"] = "ActiveHead"
    active["input_sources"] = ["image"]
    frozen = tiny_head_node(rule if finetune_frozen_node else None)
    frozen["alias"] = "FrozenHead"
    frozen["input_sources"] = ["image"]
    frozen["freezing"] = {"active": True, "unfreeze_after": 1}

    model = LuxonisModel(
        deepcopy(
            config(
                [active, frozen],
                trainer={
                    "optimizer": {"name": "Adam", "params": {"lr": 0.003}}
                },
            )
        ),
        opts | {"loader.params.n_classes": 10},
        allow_empty_dataset=True,
    )
    module = model.lightning_module
    manager = TrainingManager()
    manager.freeze_before_training(module)
    optimizers = list(module.configure_optimizers()[0])
    # `BaseFinetuning` only ever reads these two attributes off the
    # trainer, so a stub avoids running a real fit.
    trainer: Any = SimpleNamespace(optimizers=optimizers, current_epoch=0)
    return module, manager, optimizers, trainer


def _group_sizes(optimizers: list[Optimizer]) -> list[list[int]]:
    return [
        [len(group["params"]) for group in optimizer.param_groups]
        for optimizer in optimizers
    ]


@pytest.mark.parametrize("finetune_frozen_node", [True, False])
def test_unfreezing_is_recorded_in_finetuning_state(
    opts: Params, finetune_frozen_node: bool
):
    """`BaseFinetuning` checkpoints `_internal_optimizer_metadata` and
    rebuilds `optimizer.param_groups` from it when a run is resumed.

    It records new groups only for the optimizer it is iterating over,
    and only when the group *count* changes. `finetune_function` breaks
    both assumptions: it mutates every optimizer during the first
    per-optimizer call, and it extends an existing group in place. If
    the metadata is not re-snapshotted, resuming after an unfreeze
    either drops the unfrozen parameters or makes
    `Optimizer.load_state_dict` reject the checkpointed state.
    """
    module, manager, optimizers, trainer = _frozen_two_optimizer_model(
        opts, finetune_frozen_node
    )

    manager.on_train_epoch_start(trainer, module)
    trainer.current_epoch = 1
    manager.on_train_epoch_start(trainer, module)

    metadata = manager.state_dict()["internal_optimizer_metadata"]
    stored = [
        [len(group["params"]) for group in metadata[index]]
        for index in range(len(optimizers))
    ]
    assert stored == _group_sizes(optimizers)


@pytest.mark.parametrize("finetune_frozen_node", [True, False])
def test_unfreezing_after_resume_reaches_the_optimizer(
    opts: Params, finetune_frozen_node: bool
):
    """`BaseFinetuning.on_fit_start` restores a resumed run by
    *replacing* `optimizer.param_groups` with freshly built
    dictionaries.

    An unfreeze that happens after such a restore must still land in the
    live groups, so the parameter group has to be looked up on the
    optimizer rather than cached as a dictionary reference.
    """
    module, manager, optimizers, trainer = _frozen_two_optimizer_model(
        opts, finetune_frozen_node
    )
    manager.on_train_epoch_start(trainer, module)  # epoch 0, still frozen

    resumed = TrainingManager()
    resumed.load_state_dict(manager.state_dict())
    resumed.on_fit_start(trainer, module)
    trainer.current_epoch = 1
    resumed.on_train_epoch_start(trainer, module)

    assigned = {
        id(parameter)
        for optimizer in optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable = {
        id(parameter)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    assert trainable - assigned == set()
