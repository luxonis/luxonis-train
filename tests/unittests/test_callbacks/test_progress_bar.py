import pytest

from luxonis_train import LuxonisModel
from luxonis_train.callbacks import LuxonisRichProgressBar


def test_rich_progress_bar():
    progress_bar = LuxonisRichProgressBar()

    config = "configs/detection_light_model.yaml"
    model = LuxonisModel(config)

    try:
        progress_bar.on_train_epoch_end(
            model.pl_trainer, model.lightning_module
        )
    except Exception as e:
        pytest.fail(
            f"on_train_epoch_end raised an unexpected exception: {e!r}"
        )
