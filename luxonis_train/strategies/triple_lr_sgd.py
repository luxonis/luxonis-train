import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from typing_extensions import override

import luxonis_train as lxt

from .base_strategy import BaseTrainingStrategy


@dataclass
class TripleLRScheduler:
    optimizer: Optimizer
    warmup_epochs: int
    warmup_bias_lr: float
    warmup_momentum: float
    lre: float
    cosine_annealing: bool
    epochs: int
    max_stepnum: int

    def __post_init__(self) -> None:
        self.warmup_stepnum = max(
            round(self.warmup_epochs * self.max_stepnum), 100
        )
        self.step = 0
        self.lrf = self.lre / self.optimizer.defaults["lr"]
        if self.cosine_annealing:
            self.lf = (
                lambda x: ((1 - math.cos(x * math.pi / self.epochs)) / 2)
                * (self.lrf - 1)
                + 1
            )
        else:
            self.lf = (
                lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.lrf)
                + self.lrf
            )

    def create_scheduler(self) -> LambdaLR:
        return LambdaLR(self.optimizer, lr_lambda=self.lf)

    def update_learning_rate(self, current_epoch: int) -> None:
        self.step = self.step % self.max_stepnum
        curr_step = self.step + self.max_stepnum * current_epoch

        if curr_step <= self.warmup_stepnum:
            for k, param in enumerate(self.optimizer.param_groups):
                warmup_bias_lr = self.warmup_bias_lr if k == 2 else 0.0
                param["lr"] = np.interp(
                    curr_step,
                    [0, self.warmup_stepnum],
                    [
                        warmup_bias_lr,
                        self.optimizer.defaults["lr"] * self.lf(current_epoch),
                    ],
                )
                if "momentum" in param:
                    momentum = cast(float, self.optimizer.defaults["momentum"])
                    self.optimizer.defaults["momentum"] = np.interp(
                        curr_step,
                        [0, self.warmup_stepnum],
                        [self.warmup_momentum, momentum],
                    )
        self.step += 1


@dataclass
class TripleLRSGD:
    model: nn.Module
    lr: float
    momentum: float
    weight_decay: float
    nesterov: bool

    def create_optimizer(self) -> Optimizer:
        batch_norm_weights, regular_weights, biases = [], [], []

        for module in self.model.modules():
            if hasattr(module, "bias") and isinstance(
                module.bias, nn.Parameter
            ):
                biases.append(module.bias)
            if isinstance(module, nn.BatchNorm2d):
                batch_norm_weights.append(module.weight)
            elif hasattr(module, "weight") and isinstance(
                module.weight, nn.Parameter
            ):
                regular_weights.append(module.weight)

        return SGD(
            [
                {
                    "params": batch_norm_weights,
                    "lr": self.lr,
                    "momentum": self.momentum,
                    "nesterov": self.nesterov,
                },
                {
                    "params": regular_weights,
                    "weight_decay": self.weight_decay,
                },
                {"params": biases},
            ],
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )


class TripleLRSGDStrategy(BaseTrainingStrategy):
    def __init__(
        self,
        pl_module: "lxt.LuxonisLightningModule",
        lr: float = 0.02,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        nesterov: bool = True,
        warmup_epochs: int = 3,
        warmup_bias_lr: float = 0.1,
        warmup_momentum: float = 0.8,
        lre: float = 0.0002,
        cosine_annealing: bool = True,
    ):
        """TripleLRSGD strategy.

        @type pl_module: pl.LightningModule
        @param pl_module: The pl_module to be used.
        @type params: dict
        @param params: The parameters for the strategy. Those are:
            - lr: The learning rate.
            - momentum: The momentum.
            - weight_decay: The weight decay.
            - nesterov: Whether to use nesterov.
            - warmup_epochs: The number of warmup epochs.
            - warmup_bias_lr: The warmup bias learning rate.
            - warmup_momentum: The warmup momentum.
            - lre: The learning rate for the end of the training.
            - cosine_annealing: Whether to use cosine annealing.
        """
        self.model = pl_module
        self.cfg = pl_module.cfg

        max_stepnum = math.ceil(
            len(self.model.core.loaders["train"]) / self.cfg.trainer.batch_size
        )

        self.optimizer = TripleLRSGD(
            model=self.model,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        ).create_optimizer()

        self.scheduler = TripleLRScheduler(
            optimizer=self.optimizer,
            warmup_epochs=warmup_epochs,
            warmup_bias_lr=warmup_bias_lr,
            warmup_momentum=warmup_momentum,
            lre=lre,
            cosine_annealing=cosine_annealing,
            epochs=self.cfg.trainer.epochs,
            max_stepnum=max_stepnum,
        )

    @override
    def configure_optimizers(self) -> tuple[list[Optimizer], list[LambdaLR]]:
        return [self.optimizer], [self.scheduler.create_scheduler()]

    @override
    def update_parameters(self) -> None:
        current_epoch = self.model.current_epoch
        self.scheduler.update_learning_rate(current_epoch)
