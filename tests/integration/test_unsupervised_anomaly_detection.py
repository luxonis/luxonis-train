from pathlib import Path

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Kwargs

from luxonis_train.core import LuxonisModel


def get_config(image_size: tuple[int, int], batch_size: int) -> Kwargs:
    return {
        "model": {
            "name": "DREAM",
            "predefined_model": {
                "name": "AnomalyDetectionModel",
                "params": {"variant": "light"},
            },
        },
        "loader": {
            "name": "LuxonisLoaderPerlinNoise",
            "train_view": "train",
            "val_view": "val",
            "test_view": "val",
            "params": {
                "dataset_name": "dummy_mvtec",
                "anomaly_source_path": str(
                    Path("tests/data/COCO_people_subset")
                ),
            },
        },
        "trainer": {
            "preprocessing": {
                "train_image_size": image_size,
                "keep_aspect_ratio": False,
                "normalize": {"active": True},
            },
            "batch_size": batch_size,
            "epochs": 1,
            "n_workers": 0,
            "validation_interval": 10,
            "n_sanity_val_steps": 0,
        },
        "tracker": {
            "save_directory": "tests/integration/save-directory",
        },
    }


def test_anomaly_detection(
    anomaly_detection_dataset: LuxonisDataset,
    image_size: tuple[int, int],
    batch_size: int,
):
    model = LuxonisModel(
        get_config(image_size, batch_size),
        {"loader.params.dataset_name": anomaly_detection_dataset.identifier},
    )
    model.train()
    model.test()
