from typing import Literal

import lightning.pytorch as pl

import luxonis_train as lxt
from luxonis_train.callbacks.needs_checkpoint import NeedsCheckpoint
from luxonis_train.registry import CALLBACKS


@CALLBACKS.register()
class AIMETCallback(NeedsCheckpoint):
    def __init__(self, mode: Literal["PTQ", "QAT"] = "PTQ"):
        super().__init__()
        self.mode = mode

    def on_train_end(
        self, _: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        pl_module.core.quantize()
