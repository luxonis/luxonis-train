"""An SGD strategy with a warmup and a decay of the learning rate.

The strategy splits the parameters into ``BatchNorm2d`` weights, other
weights, and biases.

"""

import math

import numpy as np
from torch import nn
from typing_extensions import override

import luxonis_train as lxt
from luxonis_train.config.config import OptimizerConfig, SchedulerConfig
from luxonis_train.lightning.training_plan import StrategyRule

from .base_strategy import BaseTrainingStrategy


class TripleLRSGDStrategy(BaseTrainingStrategy):
    r"""SGD over three parameter groups, with a warmup and a decay.

    The rules of the strategy split the parameters into three groups:

    - ``triple_lr/batch_norm_weights``: the ``weight`` of each
      ``BatchNorm2d``, without weight decay.
    - ``triple_lr/weights``: every other parameter named ``weight``,
      with ``weight_decay``.
    - ``triple_lr/biases``: every parameter named ``bias``, without
      weight decay.

    Every group uses SGD with ``lr``, ``momentum``, and ``nesterov``. A
    parameter with another name goes to the default rule, which uses the
    configs of `get_base_configs`.

    A ``LambdaLR`` scheduler sets the learning rate of each group to
    :math:`\text{lr} \cdot f(e)` in epoch :math:`e`, counted from ``0``.
    Let :math:`r = \text{lre} / \text{lr}`, and let :math:`E` be
    ``trainer.epochs``. With ``cosine_annealing``, the factor is:

    .. math::

        f(e) = 1 + (r - 1) \cdot \frac{1 - \cos(\pi e / E)}{2}

    Without ``cosine_annealing``, the factor falls linearly:

    .. math::

        f(e) = r + (1 - r) \cdot \max(1 - e / E, 0)

    Both factors go from ``1`` in the first epoch to :math:`r` in epoch
    :math:`E`.

    During the warmup, `update_parameters` sets the learning rate of
    the groups of the three rules on each step. The rate moves linearly
    from a start value to :math:`\text{lr} \cdot f(e)`. The bias group
    starts from ``warmup_bias_lr``, and the other groups start from
    ``0``.

    Example:
        The ``trainer`` section of a config:

        .. code-block:: yaml

            trainer:
              training_strategy:
                name: TripleLRSGDStrategy
                params:
                  lr: 0.02
                  lre: 0.0002
                  warmup_epochs: 3
                  cosine_annealing: true

    """

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
        """Store the settings and compute the length of the warmup.

        The number of batches in an epoch is
        ``ceil(len(pl_module.core.loaders["train"]) / trainer.batch_size)``.
        The warmup lasts ``warmup_epochs`` times that number of steps,
        rounded, and at least ``100`` steps.

        Args:
            pl_module (LuxonisLightningModule): The module to train. The
                strategy reads its ``cfg``, its ``core.loaders``, and its
                ``current_epoch``.
            lr (float): The base learning rate of every group.
            momentum (float): The SGD momentum of every group.
            weight_decay (float): The weight decay of the
                ``triple_lr/weights`` group.
            nesterov (bool): Whether SGD uses Nesterov momentum.
            warmup_epochs (int): The length of the warmup, in epochs.
            warmup_bias_lr (float): The learning rate of the bias group
                at the start of the warmup.
            warmup_momentum (float): The strategy stores the value, but
                does not use it.
            lre (float): The learning rate at the end of the training.
            cosine_annealing (bool): Whether the learning rate factor
                follows a cosine curve. With ``False``, it falls
                linearly.

        """
        self._model = pl_module
        self._cfg = pl_module.cfg
        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._nesterov = nesterov
        self._warmup_epochs = warmup_epochs
        self._warmup_bias_lr = warmup_bias_lr
        self._warmup_momentum = warmup_momentum
        self._lre = lre
        self._cosine_annealing = cosine_annealing

        self._max_stepnum = math.ceil(
            len(self._model.core.loaders["train"])
            / self._cfg.trainer.batch_size
        )
        self._warmup_stepnum = max(
            round(self._warmup_epochs * self._max_stepnum), 100
        )
        self._step = 0
        self._lrf = self._lre / self._lr
        epochs = self._cfg.trainer.epochs
        if self._cosine_annealing:
            self._lf = lambda x: (
                ((1 - math.cos(x * math.pi / epochs)) / 2) * (self._lrf - 1)
                + 1
            )
        else:
            self._lf = lambda x: (
                max(1 - x / epochs, 0) * (1.0 - self._lrf) + self._lrf
            )

    def _sgd(self, **extra: float | bool) -> OptimizerConfig:
        return OptimizerConfig(
            name="SGD",
            params={
                "lr": self._lr,
                "momentum": self._momentum,
                "nesterov": self._nesterov,
                **extra,
            },
        )

    @override
    def rules(self) -> list[StrategyRule]:
        """Return the three SGD rules of the strategy.

        The ``BatchNorm2d`` rule comes before the weight rule, so a
        ``BatchNorm2d`` weight goes to the batch-norm group. No rule sets
        a scheduler, so every group uses the ``LambdaLR`` of
        `get_base_configs`.

        Returns:
            list[StrategyRule]: The rules with the tags
            ``BATCH_NORM_TAG``, ``WEIGHT_TAG``, and ``BIAS_TAG``, in this
            order. Only the ``WEIGHT_TAG`` rule sets ``weight_decay``.

        """
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
                optimizer=self._sgd(weight_decay=self._weight_decay),
            ),
            StrategyRule(
                tag=self.BIAS_TAG,
                selector=_is_bias,
                optimizer=self._sgd(),
            ),
        ]

    @override
    def get_base_configs(self) -> tuple[OptimizerConfig, SchedulerConfig]:
        """Return the SGD config and the ``LambdaLR`` config.

        Returns:
            tuple[OptimizerConfig, SchedulerConfig]: The SGD config with
            ``lr``, ``momentum``, and ``nesterov``, without weight decay.
            The ``LambdaLR`` config, whose ``lr_lambda`` is the learning
            rate factor :math:`f` that the class describes.

        """
        return self._sgd(), SchedulerConfig(
            name="LambdaLR",
            params={"lr_lambda": self._lf},  # type: ignore
        )

    @override
    def update_parameters(self) -> None:
        r"""Set the warmup learning rates of the groups.

        `TrainingManager` calls the method after each backward pass. A
        call counts as one step, also with gradient accumulation. The
        step in the epoch is the call count modulo the number of batches
        in an epoch. The global step adds ``current_epoch`` times that
        number.

        While the global step is at most the warmup length, the method
        sets ``lr`` of each group of the three tags. At step ``0``, the
        value is the start value of the group. At the last warmup step,
        it is :math:`\text{lr} \cdot f(e)`, with :math:`e` equal to
        ``current_epoch``. Between the two steps, the value changes
        linearly. After the warmup, the method changes no group.

        """
        current_epoch = self._model.current_epoch
        self._step = self._step % self._max_stepnum
        curr_step = self._step + self._max_stepnum * current_epoch

        if curr_step <= self._warmup_stepnum:
            for tag, warmup_start_lr in (
                (self.BATCH_NORM_TAG, 0.0),
                (self.WEIGHT_TAG, 0.0),
                (self.BIAS_TAG, self._warmup_bias_lr),
            ):
                target_lr = self._lr * self._lf(current_epoch)
                for handle in self.group_handles.get(tag, ()):
                    self.runtime.group(handle)["lr"] = np.interp(
                        curr_step,
                        [0, self._warmup_stepnum],
                        [warmup_start_lr, target_lr],
                    )
        self._step += 1


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
