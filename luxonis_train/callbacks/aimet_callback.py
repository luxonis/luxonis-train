"""Runs AIMET quantization when training ends."""

import lightning.pytorch as pl

import luxonis_train as lxt
from luxonis_train.callbacks.needs_checkpoint import NeedsCheckpoint
from luxonis_train.registry import CALLBACKS


@CALLBACKS.register()
class AIMETCallback(NeedsCheckpoint):
    """Quantize the model with AIMET when training ends.

    The callback passes the best checkpoint to `LuxonisModel.quantize`.
    That method runs post-training quantization and quantization-aware
    training. It writes the quantized ONNX model and its NN Archive to
    ``<run_save_dir>/aimet``. The ``exporter.aimet`` section of the
    config holds the options.

    `LuxonisLightningModule.configure_callbacks` adds this callback when
    ``exporter.aimet.active`` is set. The config does not have to list
    it. With that flag set, an entry in ``trainer.callbacks`` adds a
    second instance, and the two instances quantize the model twice.

    **The advanced techniques are slow.** AdaRound alone can take from
    40 minutes to several hours. The time depends on the size of the
    dataset and of the model.

    """

    def __init__(self, **kwargs):
        """Initialize the callback.

        Args:
            **kwargs (``Any``): Keyword arguments forwarded to
                `NeedsCheckpoint`. The ``preferred_checkpoint`` value
                ``"loss"`` selects the loss checkpoint. Every other value
                selects the metric checkpoint. Any other key raises
                ``TypeError``.

        """
        super().__init__(**kwargs)

    def on_train_end(
        self, _: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Quantize the model from the best checkpoint.

        Lightning calls this hook once when ``trainer.fit`` ends. The
        hook selects a checkpoint with `NeedsCheckpoint.get_checkpoint`
        and passes it to `LuxonisModel.quantize` with the options of
        ``exporter.aimet``. When no checkpoint exists, the quantization
        still runs, on the current weights of
        ``pl_module.core.lightning_module``. The quantization works on a
        deep copy of that module, so the module keeps its weights.

        Args:
            _ (``pl.Trainer``): The trainer. Unused.
            pl_module (LuxonisLightningModule): The model to quantize.

        """
        pl_module.core.quantize(self.get_checkpoint(pl_module))
