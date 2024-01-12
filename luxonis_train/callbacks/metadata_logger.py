import os.path as osp
import subprocess

import lightning.pytorch as pl
import pkg_resources
import yaml

from luxonis_train.utils.registry import CALLBACKS


@CALLBACKS.register_module()
class MetadataLogger(pl.Callback):
    def __init__(self, hyperparams: list[str]):
        """Callback that logs training metadata.

        Metadata include all defined hyperparameters together with git hashes of
        luxonis-ml and luxonis-train packages. Also stores this information locally.

        @type hyperparams: list[str]
        @param hyperparams: List of hyperparameters to log.
        """
        super().__init__()
        self.hyperparams = hyperparams

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        cfg = pl_module.cfg

        hparams = {key: cfg.get(key) for key in self.hyperparams}

        # try to get luxonis-ml and luxonis-train git commit hashes (if installed as editable)
        luxonis_ml_hash = self._get_editable_package_git_hash("luxonis_ml")
        if luxonis_ml_hash:
            hparams["luxonis_ml"] = luxonis_ml_hash

        luxonis_train_hash = self._get_editable_package_git_hash("luxonis_train")
        if luxonis_train_hash:
            hparams["luxonis_train"] = luxonis_train_hash

        trainer.logger.log_hyperparams(hparams)  # type: ignore
        # also save metadata locally
        with open(osp.join(pl_module.save_dir, "metadata.yaml"), "w+") as f:
            yaml.dump(hparams, f, default_flow_style=False)

    def _get_editable_package_git_hash(self, package_name: str) -> str | None:
        try:
            distribution = pkg_resources.get_distribution(package_name)
            package_location = osp.join(distribution.location, package_name)

            # remove any additional folders in path (e.g. "/src")
            if "src" in package_location:
                package_location = package_location.replace("src", "")

            # Check if the package location is a Git repository
            git_dir = osp.join(package_location, ".git")
            if osp.exists(git_dir):
                git_command = ["git", "rev-parse", "HEAD"]
                try:
                    git_hash = subprocess.check_output(
                        git_command,
                        cwd=package_location,
                        stderr=subprocess.DEVNULL,
                        universal_newlines=True,
                    ).strip()
                    return git_hash
                except subprocess.CalledProcessError:
                    return None
            else:
                return None
        except pkg_resources.DistributionNotFound:
            return None
