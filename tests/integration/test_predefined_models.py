from pathlib import Path

import cv2
import numpy as np
import pytest
from luxonis_ml.data import LuxonisLoader
from luxonis_ml.typing import Params
from pytest_subtests import SubTests

from luxonis_train.core import LuxonisModel
from tests.conftest import LuxonisTestDataset


@pytest.mark.parametrize(
    "config_name",
    [
        "anomaly_detection_model",
        "embeddings_model",
        "fomo_light_model",
        "ocr_recognition_light_model",
        "classification_light_model",
        "detection_light_model",
        "instance_segmentation_light_model",
        "keypoint_bbox_light_model",
        "segmentation_light_model",
        "classification_heavy_model",
        "detection_heavy_model",
        "instance_segmentation_heavy_model",
        "keypoint_bbox_heavy_model",
        "segmentation_heavy_model",
    ],
)
def test_predefined_models(
    config_name: str,
    opts: Params,
    coco_dataset: LuxonisTestDataset,
    cifar10_dataset: LuxonisTestDataset,
    toy_ocr_dataset: LuxonisTestDataset,
    embedding_dataset: LuxonisTestDataset,
    anomaly_detection_dataset: LuxonisTestDataset,
    tempdir: Path,
    subtests: SubTests,
):
    config_file = f"configs/{config_name}.yaml"
    tempdir = tempdir / config_name
    tempdir.mkdir()

    if config_name == "embeddings_model":
        dataset = embedding_dataset
    elif "ocr_recognition" in config_name:
        dataset = toy_ocr_dataset
    elif "classification" in config_name:
        dataset = cifar10_dataset
    elif "anomaly_detection" in config_name:
        opts |= {
            "loader.params.anomaly_source_path": str(coco_dataset.source_path)
        }
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

    if config_name != "embeddings_model":
        with subtests.test("infer"):
            loader = LuxonisLoader(dataset)
            img_dir = tempdir / "images"
            video_path = tempdir / "video.avi"
            video_writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"XVID"), 1, (256, 256)
            )
            img_dir.mkdir()
            for i, (img, _) in enumerate(loader):
                assert isinstance(img, np.ndarray)
                img = cv2.resize(img, (256, 256))
                cv2.imwrite(str(img_dir / f"{i}.png"), img)
                video_writer.write(img)
            video_writer.release()

            for subtest in ["single_image", "image_dir", "video", "loader"]:
                with subtests.test(f"infer/{subtest}"):
                    save_dir = tempdir / f"infer_{subtest}"
                    if subtest == "single_image":
                        source = img_dir / "0.png"
                    elif subtest == "image_dir":
                        source = img_dir
                    elif subtest == "video":
                        source = video_path
                    else:
                        source = None

                    model.infer(source_path=source, save_dir=save_dir)

                    if subtest == "single_image":
                        assert len(list(save_dir.rglob("*.png"))) == 1
                    elif subtest == "image_dir":
                        assert len(list(save_dir.iterdir())) == len(loader)
                    elif subtest == "video":
                        assert len(list(save_dir.rglob("*.mp4"))) == 1
                    if subtest is None:
                        assert len(list(save_dir.iterdir())) == len(loader)

    # TODO: Support annotation for all models
    if (
        config_name
        not in {
            "embeddings_model",
            "anomaly_detection_model",
            "fomo_light_model",
        }
        or "heavy" in config_name
    ):
        with subtests.test("annotate"):
            model.annotate(
                dir_path=dataset.source_path,
                dataset_name="test_annotated_dataset",
                bucket_storage="local",
                delete_local=True,
            )

    with subtests.test("test-reload"):
        model_reload = LuxonisModel(
            str(model.run_save_dir / "training_config.yaml"),
            opts | {"tracker.run_name": f"{config_name}_reload"},
        )
        model_reload.test()
