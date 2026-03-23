from typing import Literal

import lightning.pytorch as pl
from loguru import logger

import luxonis_train as lxt
from luxonis_train.callbacks.needs_checkpoint import NeedsCheckpoint


class AIMETCallback(NeedsCheckpoint):
    def __init__(self, mode: Literal["PTQ", "QAT"]):
        super().__init__()

    def on_train_end(
        self, _: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        onnx_path = pl_module.core._exported_models.get("onnx")
        if onnx_path is None:  # pragma: no cover
            checkpoint = self.get_checkpoint(pl_module)
            if checkpoint is None:
                logger.warning("Skipping model archiving.")
                return
            logger.info("Exported model not found. Exporting to ONNX...")
            pl_module.core.export(weights=checkpoint)
            onnx_path = pl_module.core._exported_models.get("onnx")

        if onnx_path is None:  # pragma: no cover
            logger.error(
                "Model executable not found and couldn't be created. "
                "Skipping AIMET."
            )
            return

        pl_module.core.quantize(onnx_path, mode)
