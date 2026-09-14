"""Logs chosen config values as hyperparameters and saves them to
``metadata.yaml``.
"""

import lightning.pytorch as pl
import yaml

import luxonis_train as lxt
from luxonis_train.config import Config
from luxonis_train.registry import CALLBACKS


@CALLBACKS.register()
class MetadataLogger(pl.Callback):
    """Callback that logs chosen config values as hyperparameters.

    When a fit starts, the callback reads each key of ``hyperparams``
    from the config. It logs the values with the logger of the model
    and saves them to ``metadata.yaml`` in the save directory of the
    model.

    The callback is in the ``CALLBACKS`` registry, so a config can add
    it:

    .. code-block:: yaml

        trainer:
          callbacks:
            - name: MetadataLogger
              params:
                hyperparams: ["trainer.epochs", "trainer.batch_size"]

    """

    def __init__(self, hyperparams: list[str]):
        """Initialize the callback.

        Args:
            hyperparams (list[str]): The config keys to log. A key
                separates its levels with dots, for example
                ``"trainer.epochs"``. A level of a list is an integer
                index, for example ``"model.nodes.0.name"``.

        """
        super().__init__()
        self.hyperparams = hyperparams

    def on_fit_start(
        self, _: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Log and save the chosen config values.

        Lightning calls this hook at the start of a fit. The hook reads
        each key of ``hyperparams`` with ``cfg.get``. A key that is not
        in the config gives ``None``. ``cfg.get`` raises ``ValueError``
        for a level of a list that is not an integer. The hook passes
        the dictionary of keys and values to ``log_hyperparams`` of
        ``pl_module.logger``. Then it writes the dictionary with
        ``yaml.safe_dump`` to ``metadata.yaml`` in
        ``pl_module.save_dir`` and replaces an existing file.
        ``yaml.safe_dump`` raises ``yaml.representer.RepresenterError``
        for a value that is not a plain YAML type, for example a nested
        config section.

        Args:
            _ (``pl.Trainer``): The trainer. Unused.
            pl_module (LuxonisLightningModule): The model. It gives the
                config, the logger, and the save directory.

        """
        cfg: Config = pl_module.cfg

        hparams = {key: cfg.get(key) for key in self.hyperparams}

        pl_module.logger.log_hyperparams(hparams)
        with open(pl_module.save_dir / "metadata.yaml", "w") as f:
            yaml.safe_dump(hparams, f, default_flow_style=False)
