from types import SimpleNamespace
from typing import Any, cast

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from luxonis_train.lightning.training_plan import GroupHandle
from luxonis_train.strategies.triple_lr_sgd import TripleLRSGDStrategy


class _Core:
    loaders = {"train": range(10)}


class _Cfg:
    class Trainer:
        batch_size = 1
        epochs = 50

    trainer = Trainer()


def _partition(
    strategy: TripleLRSGDStrategy, module: nn.Module
) -> dict[str, list[nn.Parameter]]:
    """Apply the strategy's rules with first-match-wins claiming, the
    way the training plan does.
    """
    rules = strategy.rules()
    groups: dict[str, list[nn.Parameter]] = {rule.tag: [] for rule in rules}
    claimed: set[int] = set()
    for module_name, submodule in module.named_modules():
        for parameter_name, parameter in submodule.named_parameters(
            recurse=False
        ):
            if id(parameter) in claimed:
                continue
            for rule in rules:
                if rule.selector(
                    submodule, module_name, parameter, parameter_name
                ):
                    groups[rule.tag].append(parameter)
                    claimed.add(id(parameter))
                    break
    return groups


def test_triple_lr_sgd():
    """Golden numeric table for the TripleLRSGD warmup and schedule.

    The strategy is attached to a hand-built SGD with the conventional
    three-group layout (batch-norm weights, weights, biases) - exactly
    what the training plan builds from its rules - and driven through a
    real Lightning fit.
    """

    class DummyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.core = _Core()
            self.cfg = _Cfg()
            self.linear = torch.nn.Linear(2, 1)
            self.lr_list = [[], [], []]

        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

        def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
            x = batch
            y = self.forward(x)
            return torch.nn.functional.mse_loss(y, torch.zeros_like(y))

        def configure_optimizers(
            self,
        ) -> tuple[list[Optimizer], list[LambdaLR]]:
            self.strategy = TripleLRSGDStrategy(model)  # type: ignore
            groups = _partition(self.strategy, self)
            optimizer = SGD(
                [
                    {"params": groups[TripleLRSGDStrategy.BATCH_NORM_TAG]},
                    {
                        "params": groups[TripleLRSGDStrategy.WEIGHT_TAG],
                        "weight_decay": self.strategy.weight_decay,
                    },
                    {"params": groups[TripleLRSGDStrategy.BIAS_TAG]},
                ],
                lr=self.strategy.lr,
                momentum=self.strategy.momentum,
                nesterov=self.strategy.nesterov,
            )
            runtime = SimpleNamespace(
                group=lambda handle: optimizer.param_groups[handle.group_index]
            )
            self.strategy.attach(
                cast(Any, runtime),
                {
                    TripleLRSGDStrategy.BATCH_NORM_TAG: (GroupHandle(0, 0),),
                    TripleLRSGDStrategy.WEIGHT_TAG: (GroupHandle(0, 1),),
                    TripleLRSGDStrategy.BIAS_TAG: (GroupHandle(0, 2),),
                },
            )
            return [optimizer], [
                LambdaLR(optimizer, lr_lambda=self.strategy.lf)
            ]

        def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
            for i, param_group in enumerate(optimizer.param_groups):
                self.lr_list[i].append(param_group["lr"])

        def on_after_backward(self) -> None:
            self.strategy.update_parameters()

    model = DummyModel()

    dataset = torch.randn(model.core.loaders["train"].__len__(), 2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)  # type: ignore
    trainer = pl.Trainer(max_epochs=model.cfg.trainer.epochs)
    trainer.fit(model, dataloader)

    cases = [
        (0, 0, 0.0, None),
        (1, 0, 0.0, None),
        (2, 0, 0.1, None),
        (0, 100, 0.018, 3e-4),
        (1, 100, 0.018, 3e-4),
        (2, 100, 0.018, 3e-4),
        (0, -1, 0.0002, 5e-5),
        (1, -1, 0.0002, 5e-5),
        (2, -1, 0.0002, 5e-5),
        (0, 50, 0.0098, 1e-4),
        (1, 50, 0.0098, 1e-4),
        (2, 50, 0.0597, 1e-4),
        (0, 150, 0.0159, 3e-4),
        (1, 150, 0.0159, 3e-4),
        (2, 150, 0.0159, 3e-4),
    ]

    for group_idx, step, expected, tol in cases:
        value = model.lr_list[group_idx][step]
        if tol is None:
            assert value == expected
        else:
            assert abs(value - expected) < tol


def test_triple_lr_rules_classify_parameters():
    """The three rules classify parameters structurally - batch-norm
    weights before generic weights, biases separately - regardless of
    their `requires_grad` state (the partition is total; freezing is
    expressed through `requires_grad` alone).
    """
    model = torch.nn.Sequential(
        torch.nn.BatchNorm2d(3),
        torch.nn.Conv2d(3, 4, 1),
        torch.nn.Linear(4, 2),
    )
    batch_norm = model[0]
    convolution = model[1]
    linear = model[2]
    assert isinstance(batch_norm, torch.nn.BatchNorm2d)
    assert isinstance(convolution, torch.nn.Conv2d)
    assert isinstance(linear, torch.nn.Linear)
    assert batch_norm.weight is not None
    assert convolution.bias is not None

    batch_norm.weight.requires_grad_(False)
    convolution.bias.requires_grad_(False)
    linear.weight.requires_grad_(False)

    stub = SimpleNamespace(core=_Core(), cfg=_Cfg(), current_epoch=0)
    strategy = TripleLRSGDStrategy(cast(Any, stub))
    groups = _partition(strategy, model)

    def ids(tag: str) -> set[int]:
        return {id(parameter) for parameter in groups[tag]}

    assert ids(TripleLRSGDStrategy.BATCH_NORM_TAG) == {id(batch_norm.weight)}
    assert ids(TripleLRSGDStrategy.WEIGHT_TAG) == {
        id(convolution.weight),
        id(linear.weight),
    }
    assert ids(TripleLRSGDStrategy.BIAS_TAG) == {
        id(batch_norm.bias),
        id(convolution.bias),
        id(linear.bias),
    }
