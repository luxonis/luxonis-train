import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.typing import Params
from pytest_subtests import SubTests

from luxonis_train.core import LuxonisModel

STUDY_PATH = Path("study_local.db")


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
def test_tune(opts: Params, coco_dataset: LuxonisDataset):
    opts["tuner.params"] = {
        "trainer.optimizer.name_categorical": ["Adam", "SGD"],
        "trainer.optimizer.params.lr_float": [0.0001, 0.001],
        "trainer.batch_size_int": [4, 16, 4],
        "trainer.preprocessing.augmentations_subset": [
            ["Defocus", "Sharpen", "Flip", "Normalize", "invalid"],
            2,
        ],
        "model.losses.0.weight_uniform": [0.1, 0.9],
        "model.nodes.0.freezing.unfreeze_after_loguniform": [0.1, 0.9],
    }
    opts["loader.params.dataset_name"] = coco_dataset.identifier
    opts["tuner.n_trials"] = 4
    model = LuxonisModel("configs/example_tuning.yaml", opts)
    model.tune()
    assert STUDY_PATH.exists()


def test_infer(
    coco_dataset: LuxonisDataset,
    tempdir: Path,
    image_size: tuple[int, int],
    subtests: SubTests,
):
    loader = LuxonisLoader(coco_dataset)
    img_dir = tempdir / "images"
    save_dir = tempdir / "infer_results"
    video_path = tempdir / "video.avi"
    video_writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"XVID"), 1, (256, 256)
    )
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

    for subtest in ["single_image", "image_dir", "video", "loader"]:
        with subtests.test(subtest):
            save_dir = tempdir / f"infer_{subtest}"
            if subtest == "single_image":
                source = img_dir / "0.jpg"
            elif subtest == "image_dir":
                source = img_dir
            elif subtest == "video":
                source = video_path
            else:
                source = None
            model.infer(source_path=source, save_dir=save_dir)

    with pytest.raises(ValueError, match="is not a valid file or directory"):
        model.infer(source_path="tests/data/invalid.jpg", save_dir=save_dir)


def test_archive(
    tempdir: Path,
    opts: Params,
    coco_dataset: LuxonisDataset,
):
    opts |= {"loader.params.dataset_name": coco_dataset.identifier}
    model = LuxonisModel("configs/detection_light_model.yaml", opts)
    model.archive(save_dir=tempdir)
    archive_name = model.cfg.archiver.name or model.cfg.model.name
    assert (tempdir / archive_name).with_suffix(".onnx.tar.xz").exists()


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
