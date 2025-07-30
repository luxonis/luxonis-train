import shutil
import sqlite3
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.typing import Params
from pytest_subtests import SubTests

from luxonis_train.core import LuxonisModel


@pytest.fixture
def infer_path(work_dir: Path, randint: int) -> Generator[Path]:
    path = work_dir / f"infer-save-directory-{randint}"
    path.mkdir(exist_ok=True)
    yield path
    shutil.rmtree(path)


def test_debug_mode(opts: Params):
    config_file = "configs/detection_light_model.yaml"
    opts = opts | {
        "loader.params.dataset_name": "invalid_dataset_name",
    }
    model = LuxonisModel(config_file, opts, debug_mode=True)
    model.train()


@pytest.mark.skipif(
    sys.platform == "win32", reason="Tuning not supported on Windows"
)
def test_tune(opts: Params, coco_dataset: LuxonisDataset, tempdir: Path):
    study_path = tempdir / "study_local.db"
    opts |= {
        "tuner.storage.database": str(study_path),
        "tuner.n_trials": 2,
        "tuner.params": {
            "trainer.optimizer.name_categorical": ["Adam", "SGD"],
            "trainer.optimizer.params.lr_float": [0.0001, 0.001],
            "trainer.batch_size_int": [4, 16, 4],
            "trainer.preprocessing.augmentations_subset": [
                ["Defocus", "Sharpen", "Flip", "Normalize", "invalid"],
                2,
            ],
            "model.losses.0.weight_uniform": [0.1, 0.9],
            "model.nodes.0.freezing.unfreeze_after_loguniform": [0.1, 0.9],
        },
        "loader.params.dataset_name": coco_dataset.identifier,
    }
    model = LuxonisModel("configs/example_tuning.yaml", opts)
    model.tune()
    assert study_path.exists()
    con = sqlite3.connect(study_path)
    cur = con.cursor()
    cur.execute("SELECT * FROM trial_params")
    # Should be 2 * 6 = 12, but the augmentation
    # subset parameters are not stored in the database
    assert len(cur.fetchall()) == 10


def test_infer(
    coco_dataset: LuxonisDataset,
    infer_path: Path,
    image_size: tuple[int, int],
    subtests: SubTests,
):
    loader = LuxonisLoader(coco_dataset)
    img_dir = Path("tests/data/img_dir")
    video_writer = cv2.VideoWriter(
        "tests/data/video.avi",  # type: ignore
        cv2.VideoWriter_fourcc(*"XVID"),
        1,
        (256, 256),
    )
    if img_dir.exists():  # pragma: no cover
        shutil.rmtree(img_dir)
    img_dir.mkdir()
    for i, (img, _) in enumerate(loader):
        assert isinstance(img, np.ndarray)
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(str(img_dir / f"{i}.jpg"), img)
        video_writer.write(img)
    video_writer.release()

    opts = {
        "loader.params.dataset_name": coco_dataset.identifier,
        "trainer.preprocessing.train_image_size": image_size,
        "trainer.preprocessing.augmentations": [],
    }
    model = LuxonisModel("configs/complex_model.yaml", opts)

    with subtests.test("single_image"):
        model.infer(source_path=img_dir / "0.jpg", save_dir=infer_path)
        assert len(list(infer_path.glob("*.png"))) == 3

    with subtests.test("image_dir"):
        model.infer(source_path=img_dir, save_dir=infer_path)
        assert len(list(infer_path.glob("*.png"))) == len(loader) * 3

    with subtests.test("video"):
        model.infer(source_path="tests/data/video.avi", save_dir=infer_path)
        assert len(list(infer_path.glob("*.mp4"))) == 3

    with subtests.test("loader"):
        model.infer(save_dir=infer_path, view="train")
        assert len(list(infer_path.glob("*.png"))) == len(loader) * 3 * 2

    with pytest.raises(ValueError, match="is not a valid file or directory"):
        model.infer(source_path="tests/data/invalid.jpg", save_dir=infer_path)


def test_archive(
    output_dir: Path, coco_dataset: LuxonisDataset, image_size: tuple[int, int]
):
    opts: Params = {
        "tracker.save_directory": str(output_dir),
        "loader.params.dataset_name": coco_dataset.identifier,
        "trainer.preprocessing.train_image_size": image_size,
    }
    model = LuxonisModel("tests/configs/archive_config.yaml", opts)
    model.archive()
    assert (
        Path(
            model.run_save_dir,
            "archive",
            model.cfg.archiver.name or model.cfg.model.name,
        )
        .with_suffix(".onnx.tar.xz")
        .exists()
    )


def test_weights_loading(cifar10_dataset: LuxonisDataset, opts: Params):
    config_file = "configs/classification_light_model.yaml"
    opts |= {
        "loader.params.dataset_name": cifar10_dataset.dataset_name,
    }

    model = LuxonisModel(config_file, opts)
    test_results = model.test()
    weights = model.get_min_loss_checkpoint_path()
    assert test_results == model.test(weights=weights)


def test_precision_fallback_to_bf16_on_cpu(
    coco_dataset: LuxonisDataset, opts: Params
):
    opts |= {
        "loader.params.dataset_name": coco_dataset.dataset_name,
        "trainer.precision": "16-mixed",
        "trainer.accelerator": "cpu",
    }

    model = LuxonisModel("configs/classification_light_model.yaml", opts)
    model.test()


@pytest.mark.parametrize(
    "config_name",
    [
        "segmentation_light_model",
        "detection_light_model",
        "keypoint_bbox_light_model",
        "instance_segmentation_light_model",
        "classification_light_model",
        "ocr_recognition_light_model",
    ],
)
# FIXME
def test_annotate(
    opts: dict[str, Any],
    config_name: str,
    coco_dataset: LuxonisDataset,
    cifar10_dataset: LuxonisDataset,
    toy_ocr_dataset: LuxonisDataset,
):
    config_file = f"configs/{config_name}.yaml"
    opts = opts | {
        "loader.params.dataset_name": (
            cifar10_dataset.identifier
            if "classification" in config_file
            else toy_ocr_dataset.identifier
            if "ocr_recognition" in config_file
            else coco_dataset.identifier
        )
    }
    model = LuxonisModel(config_file, opts)
    dir_path = Path(
        "tests", "data", "COCO_people_subset", "person_val2017_subset"
    ).absolute()
    annotated_dataset = model.annotate(
        dir_path=dir_path,
        dataset_name="test_annotated_dataset",
        bucket_storage="local",
        delete_local=True,
        delete_remote=True,
        team_id="test_team",
    )
    assert isinstance(annotated_dataset, LuxonisDataset)
    assert len(annotated_dataset) == len(coco_dataset)
