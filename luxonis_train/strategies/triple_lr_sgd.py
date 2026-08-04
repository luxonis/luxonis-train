import math

import numpy as np
from torch import nn
from typing_extensions import override

import luxonis_train as lxt
from luxonis_train.config.config import OptimizerConfig, SchedulerConfig
from luxonis_train.lightning.training_plan import StrategyRule

from .base_strategy import BaseTrainingStrategy


class TripleLRSGDStrategy(BaseTrainingStrategy):
    BATCH_NORM_TAG = "triple_lr/batch_norm_weights"
    WEIGHT_TAG = "triple_lr/weights"
    BIAS_TAG = "triple_lr/biases"

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

        Splits the model parameters into three SGD groups (batch-norm
        weights, other weights, biases; weight decay only on the
        weights) with a shared per-epoch learning-rate factor and a
        per-step linear warmup.

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
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.warmup_epochs = warmup_epochs
        self.warmup_bias_lr = warmup_bias_lr
        self.warmup_momentum = warmup_momentum
        self.lre = lre
        self.cosine_annealing = cosine_annealing

        self.max_stepnum = math.ceil(
            len(self.model.core.loaders["train"]) / self.cfg.trainer.batch_size
        )
        self.warmup_stepnum = max(
            round(self.warmup_epochs * self.max_stepnum), 100
        )
        self.step = 0
        self.lrf = self.lre / self.lr
        epochs = self.cfg.trainer.epochs
        if self.cosine_annealing:
            self.lf = lambda x: (
                ((1 - math.cos(x * math.pi / epochs)) / 2) * (self.lrf - 1) + 1
            )
        else:
            self.lf = lambda x: (
                max(1 - x / epochs, 0) * (1.0 - self.lrf) + self.lrf
            )

    def _sgd(self, **extra: float | bool) -> OptimizerConfig:
        return OptimizerConfig(
            name="SGD",
            params={
                "lr": self.lr,
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                **extra,
            },
        )

    @override
    def rules(self) -> list[StrategyRule]:
        # Batch-norm weights are tested before generic weights, so a
        # `BatchNorm2d.weight` lands in the batch-norm group.
        return [
            StrategyRule(
                tag=self.BATCH_NORM_TAG,
                selector=_is_batch_norm_weight,
                optimizer=self._sgd(),
            ),
            StrategyRule(
                tag=self.WEIGHT_TAG,
                selector=_is_weight,
                optimizer=self._sgd(weight_decay=self.weight_decay),
            ),
            StrategyRule(
                tag=self.BIAS_TAG,
                selector=_is_bias,
                optimizer=self._sgd(),
            ),
        ]

    @override
    def get_base_configs(self) -> tuple[OptimizerConfig, SchedulerConfig]:
        return self._sgd(), SchedulerConfig(
            name="LambdaLR",
            params={"lr_lambda": self.lf},  # type: ignore
        )

    @override
    def update_parameters(self) -> None:
        current_epoch = self.model.current_epoch
        self.step = self.step % self.max_stepnum
        curr_step = self.step + self.max_stepnum * current_epoch

        if curr_step <= self.warmup_stepnum:
            for tag, warmup_start_lr in (
                (self.BATCH_NORM_TAG, 0.0),
                (self.WEIGHT_TAG, 0.0),
                (self.BIAS_TAG, self.warmup_bias_lr),
            ):
                target_lr = self.lr * self.lf(current_epoch)
                for handle in self.group_handles.get(tag, ()):
                    self.runtime.group(handle)["lr"] = np.interp(
                        curr_step,
                        [0, self.warmup_stepnum],
                        [warmup_start_lr, target_lr],
                    )
        self.step += 1


def _is_batch_norm_weight(
    module: nn.Module,
    module_name: str,
    parameter: nn.Parameter,
    parameter_name: str,
) -> bool:
    _ = module_name, parameter
    return isinstance(module, nn.BatchNorm2d) and parameter_name == "weight"


def _is_weight(
    module: nn.Module,
    module_name: str,
    parameter: nn.Parameter,
    parameter_name: str,
) -> bool:
    _ = module, module_name, parameter
    return parameter_name == "weight"


def _is_bias(
    module: nn.Module,
    module_name: str,
    parameter: nn.Parameter,
    parameter_name: str,
) -> bool:
    _ = module, module_name, parameter
    return parameter_name == "bias"
