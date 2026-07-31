"""Contract tests for `CompositeOptimizer`, the composite scheduler
wrappers, and the training-plan resolution pipeline.
"""

import pytest
import torch
from lightning.fabric.utilities.optimizer import _optimizer_to_device
from lightning.fabric.utilities.types import Optimizable
from lightning.pytorch.core.optimizer import LightningOptimizer
from luxonis_ml.typing import Params
from torch import nn
from torch.optim import LBFGS, SGD, Adam
from torch.optim.lr_scheduler import (
    ConstantLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)

from luxonis_train import LuxonisModel
from luxonis_train.lightning.freezing import FreezeSchedule, NodeFreezePlan
from luxonis_train.lightning.training_plan import (
    build_training_plan,
    resolve_training_plan,
    unwrap_optimizers,
)
from luxonis_train.optimizers.composite_optimizer import CompositeOptimizer
from luxonis_train.schedulers.composite_scheduler import (
    CompositeLRScheduler,
    CompositeReduceLROnPlateau,
    rebase_scheduler_lr,
)

from ._helpers import config, tiny_head_node


def _two_inners() -> tuple[nn.Linear, nn.Linear, SGD, Adam]:
    backbone = nn.Linear(4, 8)
    head = nn.Linear(8, 2)
    sgd = SGD(backbone.parameters(), lr=0.1)
    adam = Adam(head.parameters(), lr=0.01)
    return backbone, head, sgd, adam


def test_composite_optimizer_contract():
    """The composite must satisfy the surface Lightning relies on: the
    `Optimizable` protocol, `LightningOptimizer` dynamic subclassing,
    live `param_groups` concatenation, a writable `state` view, and a
    fixed partition.
    """
    _, _, sgd, adam = _two_inners()
    composite = CompositeOptimizer([sgd, adam])

    assert isinstance(CompositeOptimizer([sgd, adam]), Optimizable)
    assert not hasattr(composite, "optimizer")

    # the same dictionary objects, not copies
    assert composite.param_groups[0] is sgd.param_groups[0]
    assert composite.param_groups[1] is adam.param_groups[0]

    assert "lr" in composite.defaults
    # Adam has betas, SGD does not - the intersection must drop them,
    # otherwise `LearningRateMonitor` would index them on every group.
    assert "betas" not in composite.defaults

    with pytest.raises(TypeError, match="cannot be replaced"):
        composite.param_groups = []
    with pytest.raises(TypeError, match="cannot be replaced"):
        composite.state = {}
    with pytest.raises(RuntimeError, match="fixed partition"):
        composite.add_param_group({"params": []})

    wrapped = LightningOptimizer(composite)
    assert wrapped.optimizer is composite
    assert isinstance(wrapped, CompositeOptimizer)


def test_composite_optimizer_steps_every_inner():
    backbone, head, sgd, adam = _two_inners()
    composite = CompositeOptimizer([sgd, adam])

    def closure() -> torch.Tensor:
        composite.zero_grad()
        loss = backbone(torch.ones(1, 4)).sum() + head(torch.ones(1, 8)).sum()
        loss.backward()
        return loss

    before = [
        parameter.detach().clone()
        for parameter in [*backbone.parameters(), *head.parameters()]
    ]
    loss = composite.step(closure)
    after = [*backbone.parameters(), *head.parameters()]

    assert isinstance(loss, torch.Tensor)
    assert all(
        not torch.equal(a.detach(), b)
        for a, b in zip(after, before, strict=True)
    )


def test_composite_optimizer_state_view_routes_writes():
    """Lightning's `_optimizer_to_device` reassigns into
    `optimizer.state`; the view must route those writes to the owning
    inner optimizer.
    """
    backbone, head, sgd, adam = _two_inners()
    composite = CompositeOptimizer([sgd, adam])
    composite.step(
        lambda: (
            composite.zero_grad(),
            (
                backbone(torch.ones(1, 4)).sum() + head(torch.ones(1, 8)).sum()
            ).backward(),
        )
    )

    assert len(composite.state) == len(adam.state)
    _optimizer_to_device(composite, torch.device("cpu"))

    parameter = next(iter(adam.state))
    composite.state[parameter] = {"marker": 1}
    assert adam.state[parameter] == {"marker": 1}


def test_composite_optimizer_state_dict_round_trip():
    _, _, sgd, adam = _two_inners()
    composite = CompositeOptimizer([sgd, adam])
    state = composite.state_dict()

    assert state["format"] == "luxonis_composite"
    assert state["optimizers"] == ["SGD", "Adam"]
    composite.load_state_dict(state)

    with pytest.raises(ValueError, match="different optimizer"):
        composite.load_state_dict(sgd.state_dict())

    reversed_composite = CompositeOptimizer([adam, sgd])
    with pytest.raises(ValueError, match="does not match"):
        reversed_composite.load_state_dict(state)


def test_composite_optimizer_survives_inner_group_replacement():
    """`Optimizer.load_state_dict` replaces the inner group
    dictionaries; the composite's `param_groups` must reflect the live
    dictionaries afterwards.
    """
    _, _, sgd, adam = _two_inners()
    composite = CompositeOptimizer([sgd, adam])
    sgd.load_state_dict(sgd.state_dict())

    assert composite.param_groups[0] is sgd.param_groups[0]


def test_composite_optimizer_rejects_combined_lbfgs():
    backbone, head, _, _ = _two_inners()
    lbfgs = LBFGS(backbone.parameters())
    adam = Adam(head.parameters())
    with pytest.raises(ValueError, match="LBFGS"):
        CompositeOptimizer([lbfgs, adam])
    CompositeOptimizer([lbfgs])  # a sole LBFGS is fine


def test_unwrap_optimizers():
    _, _, sgd, adam = _two_inners()
    composite = CompositeOptimizer([sgd, adam])
    assert unwrap_optimizers([composite]) == [sgd, adam]
    assert unwrap_optimizers([sgd, adam]) == [sgd, adam]


def test_composite_scheduler_fans_out():
    _, _, sgd, adam = _two_inners()
    composite = CompositeOptimizer([sgd, adam])
    members = [
        StepLR(sgd, step_size=1, gamma=0.1),
        StepLR(adam, step_size=1, gamma=0.5),
    ]
    scheduler = CompositeLRScheduler(composite, members)
    assert scheduler.optimizer is composite

    scheduler.step()

    assert sgd.param_groups[0]["lr"] == pytest.approx(0.01)
    assert adam.param_groups[0]["lr"] == pytest.approx(0.005)
    assert scheduler.get_last_lr() == [
        pytest.approx(0.01),
        pytest.approx(0.005),
    ]

    state = scheduler.state_dict()
    assert state["last_epoch"] == 1
    scheduler.load_state_dict(state)


def test_composite_plateau_scheduler_fans_out():
    _, _, sgd, adam = _two_inners()
    composite = CompositeOptimizer([sgd, adam])
    members = [
        ReduceLROnPlateau(sgd, mode="min", patience=0, factor=0.1),
        ReduceLROnPlateau(adam, mode="min", patience=0, factor=0.5),
    ]
    scheduler = CompositeReduceLROnPlateau(composite, members)
    assert isinstance(scheduler, ReduceLROnPlateau)

    scheduler.step(1.0)
    scheduler.step(2.0)  # worse -> both members reduce

    assert sgd.param_groups[0]["lr"] == pytest.approx(0.01)
    assert adam.param_groups[0]["lr"] == pytest.approx(0.005)
    scheduler.load_state_dict(scheduler.state_dict())


def test_rebase_scheduler_lr_recurses_into_sequential():
    module = nn.Linear(2, 2)
    optimizer = SGD(module.parameters(), lr=0.1)
    sequential = SequentialLR(
        optimizer,
        schedulers=[
            ConstantLR(optimizer, factor=1.0),
            ConstantLR(optimizer, factor=1.0),
        ],
        milestones=[1],
    )
    rebase_scheduler_lr(sequential, 0, 0.5)
    # `SequentialLR` itself carries no `base_lrs`; the rebase must
    # reach every child scheduler.
    assert all(child.base_lrs[0] == 0.5 for child in sequential._schedulers)


def test_freeze_schedule_apply_is_idempotent():
    """Double application must not re-log, design-frozen parameters and
    BatchNorm layers constructed without statistics tracking must keep
    their configuration after unfreezing, and a smaller epoch must re-
    freeze.
    """

    class Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.frozen = nn.Linear(2, 2)
            self.frozen.requires_grad_(False)
            self.bn = nn.BatchNorm2d(2)
            self.static_bn = nn.BatchNorm2d(2, track_running_stats=False)

    module = Module()
    plan = NodeFreezePlan.from_module(
        "node", module, unfreeze_epoch=2, lr_after_unfreeze=None
    )
    schedule = FreezeSchedule([plan])

    schedule.apply(0)
    schedule.apply(0)
    assert not any(
        parameter.requires_grad for parameter in module.parameters()
    )
    assert module.bn.track_running_stats is False
    assert module.static_bn.track_running_stats is False

    schedule.apply(2)
    assert all(
        parameter.requires_grad for parameter in module.linear.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in module.frozen.parameters()
    )
    assert module.bn.track_running_stats is True
    assert module.static_bn.track_running_stats is False

    schedule.apply(1)
    assert not any(
        parameter.requires_grad for parameter in module.parameters()
    )

    assert schedule.is_frozen("node", 1)
    assert not schedule.is_frozen("node", 2)
    assert plan.unfreezes_at(2)
    assert not plan.unfreezes_at(3)


def test_resolution_produces_total_partition(opts: Params):
    node = tiny_head_node(
        {
            "parameters": [{"module_type": "Linear"}],
            "optimizer": {"name": "SGD", "params": {"lr": 0.02}},
        }
    )
    node["freezing"] = {"active": True, "unfreeze_after": 1}
    model = LuxonisModel(
        config(
            [node],
            trainer={"optimizer": {"name": "Adam", "params": {"lr": 0.003}}},
        ),
        opts | {"loader.params.n_classes": 10},
        allow_empty_dataset=True,
    )
    nodes = model.lightning_module.nodes
    plan = resolve_training_plan(model.cfg, nodes)

    claimed = {
        id(parameter)
        for inner in plan.inners
        for group in inner.groups
        for parameter in group.parameters
    }
    every = {
        id(parameter)
        for node_wrapper in nodes.values()
        for parameter in node_wrapper.module.parameters()
    }
    assert claimed == every

    # node-purity: every group of the frozen node holds only its params
    for handle in plan.handles_by_node["Head"]:
        group = plan.inners[handle.inner_index].groups[handle.group_index]
        assert group.node_names == ("Head",)


def test_lr_after_unfreeze_survives_scheduler_steps(opts: Params):
    """`set_group_base_lr` must update the scheduler's `base_lrs`, so
    the new base survives schedulers that compute from the base rather
    than from the current value (previously base-driven schedulers such
    as `LambdaLR` or `CosineAnnealingLR` clobbered `lr_after_unfreeze`
    on the next step).
    """
    import math

    node = tiny_head_node(
        {
            "parameters": [{"module_type": "Linear"}],
            "optimizer": {"name": "SGD", "params": {"lr": 0.02}},
            "scheduler": {
                "name": "CosineAnnealingLR",
                "params": {"T_max": 4},
            },
        }
    )
    model = LuxonisModel(
        config(
            [node],
            trainer={"optimizer": {"name": "Adam", "params": {"lr": 0.003}}},
        ),
        opts | {"loader.params.n_classes": 10, "trainer.epochs": 4},
        allow_empty_dataset=True,
    )
    plan = resolve_training_plan(model.cfg, model.lightning_module.nodes)
    runtime = build_training_plan(plan, model.cfg, None)

    (handle,) = [
        handle
        for handle in runtime.plan.handles_by_node["Head"]
        if runtime.plan.inners[handle.inner_index].optimizer_name == "SGD"
    ]
    runtime.set_group_base_lr(handle, 0.5)
    runtime.members[handle.inner_index].step()

    # the cosine curve continues from the NEW base, not the old 0.02
    expected = 0.5 * (1 + math.cos(math.pi * 1 / 4)) / 2
    assert runtime.group(handle)["lr"] == pytest.approx(expected)
