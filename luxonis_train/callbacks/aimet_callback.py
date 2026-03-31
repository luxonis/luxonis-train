import lightning.pytorch as pl
from aimet_torch.common.defs import QuantizationDataType, QuantScheme

import luxonis_train as lxt
from luxonis_train.callbacks.needs_checkpoint import NeedsCheckpoint
from luxonis_train.registry import CALLBACKS


@CALLBACKS.register()
class AIMETCallback(NeedsCheckpoint):
    def __init__(
        self,
        epochs: int = 4,
        quant_scheme: str | QuantScheme = QuantScheme.min_max,
        default_output_bw: int = 8,
        default_param_bw: int = 8,
        config_file: str | None = None,
        default_data_type: QuantizationDataType = QuantizationDataType.int,
    ):
        super().__init__()
        self.epochs = epochs
        self.quant_scheme = quant_scheme
        self.default_output_bw = default_output_bw
        self.default_param_bw = default_param_bw
        self.config_file = config_file
        self.default_data_type = default_data_type

    def on_train_end(
        self, _: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        pl_module.core.quantize(
            self.get_checkpoint(pl_module),
            self.epochs,
            self.quant_scheme,
            self.default_output_bw,
            self.default_param_bw,
            self.config_file,
            self.default_data_type,
        )
