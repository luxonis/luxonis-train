# strategies/triple_lr_sgd.py
import math

import lightning.pytorch as pl
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer

from .base_strategy import BaseTrainingStrategy


class TripleLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: dict,
        epochs: int,
        max_stepnum: int,
    ) -> None:
        """TripleLRScheduler scheduler.

        @type optimizer: torch.optim.Optimizer
        @param optimizer: The optimizer to be used.
        @type params: dict
        @param params: The parameters for the scheduler.
        @type epochs: int
        @param epochs: The number of epochs to train for.
        @type max_stepnum: int
        @param max_stepnum: The maximum number of steps to train for.
        """
        self.optimizer = optimizer
        self.params = {
            "warmup_epochs": 3,
            "warmup_bias_lr": 0.1,
            "warmup_momentum": 0.8,
            "lre": 0.0002,
            "cosine_annealing": True,
        }
        if params:
            self.params.update(params)
        self.max_stepnum = max_stepnum
        self.warmup_stepnum = max(
            round(self.params["warmup_epochs"] * self.max_stepnum), 100
        )
        self.step = 0
        self.lrf = self.params["lre"] / self.optimizer.defaults["lr"]
        if self.params["cosine_annealing"]:
            self.lf = (
                lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2)
                * (self.lrf - 1)
                + 1
            )
        else:
            self.lf = (
                lambda x: max(1 - x / epochs, 0) * (1.0 - self.lrf) + self.lrf
            )

    def create_scheduler(self) -> LambdaLR:
        scheduler = LambdaLR(self.optimizer, lr_lambda=self.lf)
        return scheduler

    def update_learning_rate(self, current_epoch: int) -> None:
        self.step = self.step % self.max_stepnum
        curr_step = self.step + self.max_stepnum * current_epoch

        if curr_step <= self.warmup_stepnum:
            for k, param in enumerate(self.optimizer.param_groups):
                warmup_bias_lr = (
                    self.params["warmup_bias_lr"] if k == 2 else 0.0
                )
                param["lr"] = np.interp(
                    curr_step,
                    [0, self.warmup_stepnum],
                    [
                        warmup_bias_lr,
                        self.optimizer.defaults["lr"] * self.lf(current_epoch),
                    ],
                )
                if "momentum" in param:
                    self.optimizer.defaults["momentum"] = np.interp(
                        curr_step,
                        [0, self.warmup_stepnum],
                        [
                            self.params["warmup_momentum"],
                            self.optimizer.defaults["momentum"],
                        ],
                    )
        self.step += 1


class TripleLRSGD:
    def __init__(self, model: torch.nn.Module, params: dict) -> None:
        """TripleLRSGD optimizer.

        @type model: torch.nn.Module
        @param model: The model to be used.
        @type params: dict
        @param params: The parameters for the optimizer.
        """
        self.model = model
        self.params = {
            "lr": 0.02,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "nesterov": True,
        }
        if params:
            self.params.update(params)

    def create_optimizer(self) -> torch.optim.Optimizer:
        batch_norm_weights, regular_weights, biases = [], [], []

        for module in self.model.modules():
            if hasattr(module, "bias") and isinstance(
                module.bias, torch.nn.Parameter
            ):
                biases.append(module.bias)
            if isinstance(module, torch.nn.BatchNorm2d):
                batch_norm_weights.append(module.weight)
            elif hasattr(module, "weight") and isinstance(
                module.weight, torch.nn.Parameter
            ):
                regular_weights.append(module.weight)

        optimizer = SGD(
            [
                {
                    "params": batch_norm_weights,
                    "lr": self.params["lr"],
                    "momentum": self.params["momentum"],
                    "nesterov": self.params["nesterov"],
                },
                {
                    "params": regular_weights,
                    "weight_decay": self.params["weight_decay"],
                },
                {"params": biases},
            ],
            lr=self.params["lr"],
            momentum=self.params["momentum"],
            nesterov=self.params["nesterov"],
        )

        return optimizer


class TripleLRSGDStrategy(BaseTrainingStrategy):
    def __init__(self, pl_module: pl.LightningModule, params: dict):
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
        super().__init__(pl_module)
        self.model = pl_module
        self.params = params
        self.cfg = self.model.cfg

        max_stepnum = math.ceil(
            len(self.model.core.loaders["train"]) / self.cfg.trainer.batch_size
        )

        self.optimizer = TripleLRSGD(self.model, params).create_optimizer()
        self.scheduler = TripleLRScheduler(
            self.optimizer, params, self.cfg.trainer.epochs, max_stepnum
        )

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LambdaLR]]:
        return [self.optimizer], [self.scheduler.create_scheduler()]

    def update_parameters(self, *args, **kwargs) -> None:
        current_epoch = self.model.current_epoch
        self.scheduler.update_learning_rate(current_epoch)
