import math

import numpy as np
import torch


class TripleLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        params: dict,
        epochs: int,
        max_stepnum: int,
    ) -> None:
        """TripleLRScheduler is a custom learning rate scheduler that
        combines a cosine annealing.

        @type optimizer: torch.optim.Optimizer
        @param optimizer: The optimizer to be used
        @type parmas: dict
        @param parmas: The parameters to be used for the scheduler
        @type epochs: int
        @param epochs: The number of epochs to train for
        @type max_stepnum: int
        @param max_stepnum: The maximum number of steps to train for
        """
        self.optimizer = optimizer
        self.params = params
        self.max_stepnum = max_stepnum
        self.warmup_stepnum = max(
            round(self.params["warmup_epochs"] * self.max_stepnum), 1000
        )
        self.step = 0
        self.lrf = self.params["lre"] / self.optimizer.defaults["lr"]
        self.lf = (
            lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2)
            * (self.lrf - 1)
            + 1
        )

    def create_scheduler(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lf
        )
        return scheduler

    def update_learning_rate(self, current_epoch: int) -> None:
        """Update the learning rate based on the current epoch.

        @type current_epoch: int
        @param current_epoch: The current epoch
        """
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
