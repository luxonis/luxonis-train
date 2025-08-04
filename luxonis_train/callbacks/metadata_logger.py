import lightning.pytorch as pl
import yaml

import luxonis_train as lxt
from luxonis_train.config import Config
from luxonis_train.registry import CALLBACKS


@CALLBACKS.register()
class MetadataLogger(pl.Callback):
    def __init__(self, hyperparams: list[str]):
        """Callback that logs training metadata.

        Metadata include all defined hyperparameters together with git
        hashes of luxonis-ml and luxonis-train packages. Also stores
        this information locally.

        @type hyperparams: list[str]
        @param hyperparams: List of hyperparameters to log.
        """
        super().__init__()
        self.hyperparams = hyperparams

    def on_fit_start(
        self, _: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        cfg: Config = pl_module.cfg

        hparams = {key: cfg.get(key) for key in self.hyperparams}

        pl_module.logger.log_hyperparams(hparams)
        with open(pl_module.save_dir / "metadata.yaml", "w") as f:
            yaml.safe_dump(hparams, f, default_flow_style=False)
