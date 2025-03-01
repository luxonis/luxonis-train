import subprocess
from pathlib import Path

import lightning.pytorch as pl
import pkg_resources
import yaml

import luxonis_train as lxt
from luxonis_train.config import Config
from luxonis_train.utils.registry import CALLBACKS


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
        self,
        _: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
    ) -> None:
        cfg: Config = pl_module.cfg

        hparams = {key: cfg.get(key) for key in self.hyperparams}

        luxonis_ml_hash = self._get_editable_package_git_hash("luxonis_ml")
        if luxonis_ml_hash:  # pragma: no cover
            hparams["luxonis_ml"] = luxonis_ml_hash

        luxonis_train_hash = self._get_editable_package_git_hash(
            "luxonis_train"
        )
        if luxonis_train_hash:  # pragma: no cover
            hparams["luxonis_train"] = luxonis_train_hash

        pl_module.tracker.log_hyperparams(hparams)
        with open(pl_module.save_dir / "metadata.yaml", "w") as f:
            yaml.dump(hparams, f, default_flow_style=False)

    # TODO: Is this any useful?
    @staticmethod
    def _get_editable_package_git_hash(
        package_name: str,
    ) -> str | None:  # pragma: no cover
        """Get git hash of an editable package.

        @type package_name: str
        @param package_name: Name of the package.
        @rtype: str or None
        @return: Git hash of the package or None if the package is not
            installed in editable mode.
        """
        try:
            distribution = pkg_resources.get_distribution(package_name)
            if distribution.location is None:
                return None
            package_location = Path(distribution.location, package_name)

            git_dir = package_location / ".git"
            if git_dir.exists():
                git_command = ["git", "rev-parse", "HEAD"]
                try:
                    return subprocess.check_output(
                        git_command,
                        cwd=package_location,
                        stderr=subprocess.DEVNULL,
                        universal_newlines=True,
                    ).strip()
                except subprocess.CalledProcessError:
                    return None
            else:
                return None
        except pkg_resources.DistributionNotFound:
            return None
