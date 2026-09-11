"""Runs AIMET quantization when training ends."""

import lightning.pytorch as pl

import luxonis_train as lxt
from luxonis_train.callbacks.needs_checkpoint import NeedsCheckpoint
from luxonis_train.registry import CALLBACKS


@CALLBACKS.register()
class AIMETCallback(NeedsCheckpoint):
    """Quantize the model with AIMET when training ends.

    The ``exporter.aimet`` section of the config holds the options. The
    advanced techniques are slow: AdaRound alone can take from 40 minutes to
    several hours.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_train_end(
        self, _: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        pl_module.core.quantize(self.get_checkpoint(pl_module))
