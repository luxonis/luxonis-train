from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.callbacks import LuxonisRichProgressBar
from luxonis_train.core import LuxonisModel


def test_rich_progress_bar(coco_dataset: LuxonisDataset, opts: Params):
    progress_bar = LuxonisRichProgressBar()

    config = "configs/detection_light_model.yaml"
    opts |= {
        "loader.params.dataset_name": coco_dataset.identifier,
    }
    model = LuxonisModel(config, opts)

    progress_bar.on_train_epoch_end(model.pl_trainer, model.lightning_module)
