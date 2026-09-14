"""Tests the best checkpoint when training ends."""

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger

import luxonis_train as lxt
from luxonis_train.registry import CALLBACKS
from luxonis_train.typing import View

from .needs_checkpoint import NeedsCheckpoint


@CALLBACKS.register()
class TestOnTrainEnd(NeedsCheckpoint):
    """Callback that tests the best checkpoint when training ends.

    The callback passes the checkpoint of `NeedsCheckpoint.get_checkpoint`
    to `LuxonisModel.test`. It always tries the best main metric first,
    and then the lowest validation loss. Its constructor does not accept
    ``preferred_checkpoint``.

    When ``trainer.smart_cfg_auto_populate`` is set,
    `Config.smart_auto_populate` adds this callback to
    ``trainer.callbacks`` if it is missing. `LuxonisModel.tune` removes it
    from the config of each trial.

    Attributes:
        view (``Literal["train", "val", "test"]``): The dataset view to
            test on.

    """

    def __init__(self, view: View = "test") -> None:
        """Initialize the callback.

        Args:
            view (``Literal["train", "val", "test"]``): The dataset view
                to test on. The test reads the PyTorch loader of this
                view. The logged keys start with ``test/`` for every
                view.

        """
        super().__init__()
        self.view: View = view

    def on_train_end(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Test the model on the best checkpoint.

        Lightning calls this hook once when ``trainer.fit`` ends. The
        hook selects a checkpoint with `NeedsCheckpoint.get_checkpoint`
        and passes it to `LuxonisModel.test` with ``view``. The test logs
        its values to the tracker. It does not finalize the tracker.
        `LuxonisModel.train` does that at the end of the run.

        When no checkpoint exists, the hook logs a warning. The test then
        uses the weights of the `LuxonisModel` constructor, or
        ``model.weights`` of the config. Without either, it uses the
        current weights of the module.

        After the test, the hook restores two things:

        - The test moves the module to the CPU. The hook moves
          ``pl_module`` back to its earlier device.
        - The test attaches new ``ModelCheckpoint`` callbacks from
          `LuxonisLightningModule.configure_callbacks`, each with an
          empty ``best_model_path``. The hook copies the earlier paths
          into them, matched by ``monitor``. Thus a later callback, such
          as `ConvertOnTrainEnd`, still finds the best checkpoint.

        The test loads the checkpoint into
        ``pl_module.core.lightning_module``, which is ``pl_module`` in a
        `LuxonisModel.train` run. It does not restore the earlier
        weights, so that module holds the checkpoint weights after the
        hook.

        Args:
            trainer (``pl.Trainer``): The trainer. The hook reads and
                updates its checkpoint callbacks.
            pl_module (LuxonisLightningModule): The model to test.

        """
        checkpoint = self.get_checkpoint(pl_module)
        if checkpoint is None:  # pragma: no cover
            logger.warning(
                "Best model checkpoint not found. Using last checkpoint for testing."
            )
        # `trainer.test` would delete the paths so we need to save them
        best_paths = {
            hash(callback.monitor): callback.best_model_path
            for callback in trainer.checkpoint_callbacks
            if isinstance(callback, ModelCheckpoint)
        }

        device_before = pl_module.device

        pl_module.core.test(
            weights=checkpoint,
            view=self.view,
            finalize_tracker=False,
        )

        # .test() moves pl_module to "cpu", we move it back to original device after
        pl_module.to(device_before)

        # Restore the paths
        for callback in trainer.checkpoint_callbacks:
            if (
                isinstance(callback, ModelCheckpoint)
                and hash(callback.monitor) in best_paths
            ):
                callback.best_model_path = best_paths[hash(callback.monitor)]
