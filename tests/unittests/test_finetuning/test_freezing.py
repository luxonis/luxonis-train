from copy import deepcopy

from luxonis_ml.typing import Params
from torch.optim import SGD, Adam

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
