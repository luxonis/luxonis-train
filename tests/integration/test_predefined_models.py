from pathlib import Path

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params
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
        "anomaly_detection_model",
    ],
)
def test_predefined_models(
    config_name: str,
    opts: Params,
    coco_dir: Path,
    coco_dataset: LuxonisDataset,
    cifar10_dataset: LuxonisDataset,
    toy_ocr_dataset: LuxonisDataset,
    embedding_dataset: LuxonisDataset,
    anomaly_detection_dataset: LuxonisDataset,
    subtests: SubTests,
):
    config_file = f"configs/{config_name}.yaml"
    coco_images = coco_dir / "person_val2017_subset"
    if config_name == "embeddings_model":
        dataset = embedding_dataset
    elif "ocr_recognition" in config_name:
        dataset = toy_ocr_dataset
    elif "classification" in config_name:
        dataset = cifar10_dataset
    elif "anomaly_detection" in config_name:
        opts |= {"loader.params.anomaly_source_path": str(coco_dir)}
        dataset = anomaly_detection_dataset
    else:
        dataset = coco_dataset

    opts |= {
        "model.name": config_name,
        "loader.params.dataset_name": dataset.identifier,
        "tracker.run_name": config_name,
    }

    if config_name == "embeddings_model":
        opts |= {
            "loader.params.dataset_name": embedding_dataset.dataset_name,
            "trainer.batch_size": 16,
            "trainer.preprocessing.train_image_size": [48, 64],
        }
    elif "ocr_recognition" in config_file:
        opts["trainer.preprocessing.train_image_size"] = [48, 320]

    model = LuxonisModel(config_file, opts)

    with subtests.test("train"):
        model.train()
        assert model.run_save_dir.exists()
        assert list(model.run_save_dir.iterdir())

    with subtests.test("export"):
        model.export()
        assert (model.run_save_dir / "export" / f"{config_name}.onnx").exists()

    with subtests.test("archive"):
        model.archive()
        assert (
            model.run_save_dir / "archive" / f"{config_name}.onnx.tar.xz"
        ).exists()

    with subtests.test("infer"):
        model.infer(save_dir=model.run_save_dir / "infer", view="test")
        assert (model.run_save_dir / "infer").exists()
        assert len(list((model.run_save_dir / "infer").iterdir())) == len(
            model.loaders["test"].loader
        )

    with subtests.test("annotate"):
        annotated_dataset = model.annotate(
            dir_path=coco_images,
            dataset_name="test_annotated_dataset",
            bucket_storage="local",
            delete_local=True,
        )
        assert isinstance(annotated_dataset, LuxonisDataset)
        assert len(annotated_dataset) == len(coco_dataset)

    with subtests.test("test-reload"):
        model_reload = LuxonisModel(
            str(model.run_save_dir / "training_config.yaml"),
            opts | {"tracker.run_name": f"{config_name}_reload"},
        )
        model_reload.test()
