from pathlib import Path
from typing import Any

import pytest
from luxonis_ml.data import LuxonisDataset
from pytest_subtests import SubTests

from luxonis_train.core import LuxonisModel


@pytest.mark.parametrize(
    "config_name",
    [
        "classification_heavy_model",
        "classification_light_model",
        "segmentation_heavy_model",
        "segmentation_light_model",
        "detection_heavy_model",
        "detection_light_model",
        "embeddings_model",
        "keypoint_bbox_heavy_model",
        "keypoint_bbox_light_model",
        "ocr_recognition_light_model",
        "instance_segmentation_heavy_model",
        "instance_segmentation_light_model",
        "fomo_light_model",
    ],
)
def test_predefined_models(
    opts: dict[str, Any],
    config_name: str,
    coco_dataset: LuxonisDataset,
    cifar10_dataset: LuxonisDataset,
    toy_ocr_dataset: LuxonisDataset,
    embedding_dataset: LuxonisDataset,
    save_dir: Path,
    subtests: SubTests,
):
    config_file = f"configs/{config_name}.yaml"
    if config_name == "embeddings_model":
        dataset = embedding_dataset
    elif "ocr_recognition" in config_name:
        dataset = toy_ocr_dataset
    elif "classification" in config_name:
        dataset = cifar10_dataset
    else:
        dataset = coco_dataset

    opts |= {
        "loader.params.dataset_name": dataset.identifier,
        "tracker.run_name": config_name,
        "trainer.callbacks": [
            {"name": "ExportOnTrainEnd"},
            {"name": "ArchiveOnTrainEnd"},
            {"name": "TestOnTrainEnd"},
        ],
    }

    if config_name == "embeddings_model":
        opts |= {
            "loader.params.dataset_name": embedding_dataset.dataset_name,
            "tracker.save_directory": str(save_dir),
            "trainer.batch_size": 16,
            "trainer.preprocessing.train_image_size": [48, 64],
        }
    elif "ocr_recognition" in config_file:
        opts["trainer.preprocessing.train_image_size"] = [48, 320]

    with subtests.test("original_config"):
        model = LuxonisModel(config_file, opts)
        model.train()
    with subtests.test("saved_config"):
        opts["tracker.run_name"] = f"{config_name}_reload"
        model = LuxonisModel(
            str(save_dir / config_name / "training_config.yaml"), opts
        )
        model.test()
