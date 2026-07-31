from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

from luxonis_ml.typing import Params
from torch.optim import SGD, Adam

from luxonis_train import LuxonisModel
from luxonis_train.callbacks import TrainingManager

from ._helpers import (
    OptimizerSnapshot,
    assert_all_trainable_parameters_assigned,
    assert_group_options,
    assert_total_partition,
    config,
    find_group,
    matching_names,
    parameter_names_by_id,
    tiny_head_node,
)


def _snapshot_after_configure(model: LuxonisModel) -> OptimizerSnapshot:
    module = model.lightning_module
    module.configure_optimizers()
    runtime = module.training_plan
    assert runtime is not None
    return OptimizerSnapshot(
        model=model,
        optimizers=list(runtime.inner_optimizers),
        schedulers=list(runtime.members),
        names_by_id=parameter_names_by_id(model),
    )


def test_unfreezing_restores_parameters_to_configured_optimizers(
    opts: Params,
):
    """Frozen parameters sit inside their configured groups from the
    start; unfreezing is a pure `requires_grad` flip that leaves the
    partition untouched.
    """
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
    manager.setup(cast(Any, None), module, "fit")

    assert not any(
        parameter.requires_grad for parameter in module.parameters()
    )

    snapshot = _snapshot_after_configure(model)

    assert [type(optimizer) for optimizer in snapshot.optimizers] == [
        SGD,
        Adam,
    ]
    assert_total_partition(snapshot)

    manager.on_train_epoch_start(
        cast(Any, SimpleNamespace(current_epoch=1)), module
    )

    linear_names = matching_names(snapshot, "Head.Linear.fc")
    convolution_names = matching_names(snapshot, "Head.Conv2d")
    _, _, linear_group = find_group(snapshot, linear_names)
    _, _, convolution_group = find_group(snapshot, convolution_names)

    assert all(parameter.requires_grad for parameter in module.parameters())
    assert_group_options(linear_group, {"lr": 0.02})
    assert_group_options(convolution_group, {"lr": 0.003})
    assert_all_trainable_parameters_assigned(snapshot)


def test_unfreezing_default_group_does_not_change_active_group_lr(
    opts: Params,
):
    """`lr_after_unfreeze` rebases only the frozen node's groups; the
    groups of other nodes keep their configured learning rate.
    """
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
    manager.setup(cast(Any, None), module, "fit")
    snapshot = _snapshot_after_configure(model)

    (optimizer,) = snapshot.optimizers
    active_names = matching_names(snapshot, "ActiveHead")
    frozen_names = matching_names(snapshot, "FrozenHead")

    assert not any(
        parameter.requires_grad
        for parameter in module.nodes["FrozenHead"].module.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in module.nodes["ActiveHead"].module.parameters()
    )
    assert len(optimizer.param_groups) == 2

    manager.on_train_epoch_start(
        cast(Any, SimpleNamespace(current_epoch=1)), module
    )

    _, _, active_group = find_group(snapshot, active_names)
    _, _, frozen_group = find_group(snapshot, frozen_names)
    assert_group_options(active_group, {"lr": 0.003})
    assert_group_options(frozen_group, {"lr": 0.02, "initial_lr": 0.02})
    assert all(parameter.requires_grad for parameter in module.parameters())
    assert_all_trainable_parameters_assigned(snapshot)
    assert_total_partition(snapshot)
