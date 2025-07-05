import json
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

import cv2
import pytest
from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.typing import Params
from luxonis_ml.utils import environ
from pytest_subtests import SubTests
from tensorboard.backend.event_processing import event_accumulator

from luxonis_train.core import LuxonisModel

from .multi_input_modules import *

INFER_PATH = Path("tests/integration/infer-save-directory")
ONNX_PATH = Path("tests/integration/example_multi_input.onnx")
STUDY_PATH = Path("study_local.db")


@pytest.fixture
def infer_path() -> Path:
    if INFER_PATH.exists():
        shutil.rmtree(INFER_PATH)
    INFER_PATH.mkdir()
    return INFER_PATH


@pytest.fixture
def opts(output_dir: Path) -> dict[str, Any]:
    return {
        "trainer.epochs": 1,
        "trainer.batch_size": 1,
        "trainer.validation_interval": 1,
        "trainer.callbacks": [],
        "tracker.save_directory": str(output_dir),
        "tuner.n_trials": 4,
    }


@pytest.fixture(autouse=True)
def clear_files():
    yield
    STUDY_PATH.unlink(missing_ok=True)
    ONNX_PATH.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "config_name",
    [
        "classification_heavy_model",
        "classification_light_model",
        "segmentation_heavy_model",
        "segmentation_light_model",
        "detection_heavy_model",
        "detection_light_model",
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
    image_size: tuple[int, int],
    output_dir: Path,
    subtests: SubTests,
):
    config_file = f"configs/{config_name}.yaml"
    opts = opts | {
        "loader.params.dataset_name": (
            cifar10_dataset.identifier
            if "classification" in config_file
            else toy_ocr_dataset.identifier
            if "ocr_recognition" in config_file
            else coco_dataset.identifier
        ),
        "tracker.run_name": config_name,
        "trainer.epochs": 1,
        "trainer.preprocessing.train_image_size": image_size,
        "trainer.callbacks": [
            {"name": "ExportOnTrainEnd"},
            {"name": "ArchiveOnTrainEnd"},
            {"name": "TestOnTrainEnd"},
        ],
    }
    if "ocr_recognition" in config_file:
        opts["trainer.preprocessing.train_image_size"] = [48, 320]

    with subtests.test("original_config"):
        model = LuxonisModel(config_file, opts)
        model.train()
    with subtests.test("saved_config"):
        opts["tracker.run_name"] = f"{config_name}_reload"
        model = LuxonisModel(
            str(output_dir / config_name / "training_config.yaml"), opts
        )
        model.test()


def test_multi_input(opts: Params, infer_path: Path):
    config_file = "tests/configs/multi_input.yaml"
    model = LuxonisModel(config_file, opts)
    model.train()
    model.test(view="val")

    assert not ONNX_PATH.exists()
    model.export(str(ONNX_PATH.parent))
    assert ONNX_PATH.exists()

    assert len(list(infer_path.iterdir())) == 0
    model.infer(view="val", save_dir=infer_path)
    assert infer_path.exists()


def test_custom_tasks(
    opts: Params, parking_lot_dataset: LuxonisDataset, subtests: SubTests
):
    config_file = "tests/configs/parking_lot_config.yaml"
    opts = opts | {
        "loader.params.dataset_name": parking_lot_dataset.dataset_name,
        "trainer.preprocessing.train_image_size": [128, 160],
        "trainer.batch_size": 2,
    }
    del opts["trainer.callbacks"]
    model = LuxonisModel(config_file, opts)
    model.train()
    archive_path = Path(
        model.run_save_dir, "archive", model.cfg.model.name
    ).with_suffix(".onnx.tar.xz")
    correct_archive_config = json.loads(
        Path("tests/integration/parking_lot.json").read_text()
    )

    with subtests.test("test_archive"):
        assert archive_path.exists()
        with tarfile.open(archive_path) as tar:
            extracted_cfg = tar.extractfile("config.json")

            assert extracted_cfg is not None, (
                "Config JSON not found in the archive."
            )
            generated_config = json.loads(extracted_cfg.read().decode())

        # Sort the fields in the config to make the comparison consistent
        def sort_by_name(config: dict, keys: list[str]) -> None:
            for key in keys:
                if key in config["model"]:
                    config["model"][key] = sorted(
                        config["model"][key], key=lambda x: x["name"]
                    )

        keys_to_sort = ["inputs", "outputs", "heads"]
        sort_by_name(generated_config, keys_to_sort)
        sort_by_name(correct_archive_config, keys_to_sort)
        assert generated_config == correct_archive_config


@pytest.mark.skipif(
    environ.GOOGLE_APPLICATION_CREDENTIALS is None,
    reason="GCP credentials not set",
)
def test_parsing_loader():
    model = LuxonisModel("tests/configs/segmentation_parse_loader.yaml")
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
    model = LuxonisModel("configs/example_tuning.yaml", opts)
    model.tune()
    assert STUDY_PATH.exists()


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


def test_archive(output_dir: Path, coco_dataset: LuxonisDataset):
    opts: Params = {
        "tracker.save_directory": str(output_dir),
        "loader.params.dataset_name": coco_dataset.identifier,
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


def test_callbacks(opts: Params, coco_dataset: LuxonisDataset):
    config_file = "tests/configs/config_simple.yaml"
    opts = opts | {
        "trainer.use_rich_progress_bar": False,
        "trainer.seed": 42,
        "trainer.deterministic": "warn",
        "trainer.callbacks": [
            {
                "name": "MetadataLogger",
                "params": {
                    "hyperparams": ["trainer.epochs", "trainer.batch_size"],
                },
            },
            {"name": "TestOnTrainEnd"},
            {"name": "UploadCheckpoint"},
            {"name": "ExportOnTrainEnd"},
            {
                "name": "ExportOnTrainEnd",
                "params": {"preferred_checkpoint": "loss"},
            },
            {
                "name": "ArchiveOnTrainEnd",
                "params": {"preferred_checkpoint": "loss"},
            },
        ],
        "exporter.scale_values": [0.5, 0.5, 0.5],
        "exporter.mean_values": [0.5, 0.5, 0.5],
        "exporter.blobconverter.active": True,
        "loader.params.dataset_name": coco_dataset.identifier,
    }
    model = LuxonisModel(config_file, opts)
    model.train()


@pytest.mark.parametrize(
    ("lr_after_unfreeze", "expected_lrs"),
    [
        (
            0.001,
            {
                "lr-SGD": [
                    (0, 0.000100),
                    (1, 0.002080),
                ],
                "lr-SGD/pg1": [
                    (2, 0.010000),
                    (3, 0.008365),
                ],
                "lr-SGD/pg2": [
                    (2, 0.001000),
                    (3, 0.000837),
                ],
            },
        ),
        (
            0.01,
            {
                "lr-SGD": [
                    (0, 0.000100),
                    (1, 0.002080),
                ],
                "lr-SGD/pg1": [
                    (2, 0.010000),
                    (3, 0.008365),
                ],
                "lr-SGD/pg2": [
                    (2, 0.010000),
                    (3, 0.008365),
                ],
            },
        ),
    ],
)
def test_freezing_parametrized(
    lr_after_unfreeze: float,
    expected_lrs: dict[str, list[tuple[int, float]]],
    coco_dataset: LuxonisDataset,
):
    config_file = "configs/segmentation_light_model.yaml"
    opts = {
        "model.predefined_model.params": {
            "head_params": {
                "freezing": {
                    "active": True,
                    "unfreeze_after": 2,
                    "lr_after_unfreeze": lr_after_unfreeze,
                },
            },
        },
        "trainer.epochs": 4,
        "loader.params.dataset_name": coco_dataset.identifier,
        "loader.train_view": "val",
        "loader.val_view": "val",
        "loader.test_view": "val",
        "trainer.batch_size": 2,
        "trainer.optimizer": {
            "name": "SGD",
            "params": {
                "lr": 0.01,
                "momentum": 0.9,
                "nesterov": False,
                "weight_decay": 0.0005,
            },
        },
        "trainer.scheduler": {
            "name": "SequentialLR",
            "params": {
                "schedulers": [
                    {
                        "name": "LinearLR",
                        "params": {"start_factor": 0.01},
                    },
                    {
                        "name": "PolynomialLR",
                        "params": {"power": 0.8},
                    },
                ],
                "milestones": [2],
            },
        },
        "trainer.callbacks": [
            {"name": "LearningRateMonitor"},
        ],
    }
    model = LuxonisModel(config_file, opts)
    model.train()
    log_dir = model.lightning_module.logger.experiment["tensorboard"].log_dir

    ea = event_accumulator.EventAccumulator(
        str(Path(log_dir)),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    for tag, expected_seq in expected_lrs.items():
        actual_vals = ea.Scalars(tag)
        assert len(actual_vals) == len(expected_seq), (
            f"{tag}: expected {len(expected_seq)} entries, got {len(actual_vals)}"
        )
        for (exp_step, exp_val), actual in zip(expected_seq, actual_vals):
            act_step, act_val = actual.step, actual.value
            assert act_step == exp_step, (
                f"{tag}: expected step {exp_step}, got {act_step}"
            )
            assert abs(act_val - exp_val) < 1e-6, (
                f"{tag} @ step {act_step}: expected LR {exp_val:.6f}, got {act_val:.6f}"
            )


def test_smart_cfg_auto_populate(coco_dataset: LuxonisDataset):
    base_opts = {
        "loader.params.dataset_name": coco_dataset.dataset_name,
        "model.predefined_model.params.loss_params": {
            "iou_loss_weight": 14,
            "class_loss_weight": 1,
        },
    }

    config_path = "tests/configs/smart_cfg_populate_config.yaml"
    model = LuxonisModel(config_path, base_opts)

    assert model.cfg.trainer.scheduler is not None
    scheduler_params = model.cfg.trainer.scheduler.params
    assert scheduler_params["T_max"] == model.cfg.trainer.epochs

    augmentations = model.cfg.trainer.preprocessing.augmentations[0].params
    img_width, img_height = model.cfg.trainer.preprocessing.train_image_size
    assert augmentations["out_width"] == img_width
    assert augmentations["out_height"] == img_height

    batch_size = model.cfg.trainer.batch_size
    grad_accumulation = 64 // batch_size

    assert model.cfg.model.predefined_model is not None
    loss_params = model.cfg.model.predefined_model.params["loss_params"]
    expected_iou_weight = 2.5 * grad_accumulation
    expected_class_weight = 1.0 * grad_accumulation
    assert isinstance(loss_params, dict)

    assert loss_params["iou_loss_weight"] == expected_iou_weight
    assert loss_params["class_loss_weight"] == expected_class_weight


def test_weight_loading(coco_dataset: LuxonisDataset):
    config_file = "tests/configs/ddrnet.yaml"
    opts = {
        "loader.params.dataset_name": coco_dataset.dataset_name,
        "trainer.epochs": 1,
        "trainer.n_validation_batches": 1,
        "trainer.batch_size": 1,
        "loader.train_view": "val",
        "loader.val_view": "val",
        "loader.test_view": "val",
    }

    model = LuxonisModel(config_file, opts)
    model.train()
    weights = model.get_min_loss_checkpoint_path()
    model.test(weights=weights)


def test_precision_fallback_to_bf16_on_cpu(coco_dataset: LuxonisDataset):
    config_files = [
        "configs/segmentation_light_model.yaml",
        "configs/detection_light_model.yaml",
        "configs/keypoint_bbox_light_model.yaml",
        "configs/instance_segmentation_light_model.yaml",
    ]
    opts = {
        "loader.params.dataset_name": coco_dataset.dataset_name,
        "trainer.epochs": 1,
        "trainer.n_validation_batches": 1,
        "trainer.batch_size": 1,
        "loader.train_view": "val",
        "loader.val_view": "val",
        "loader.test_view": "val",
        "trainer.precision": "16-mixed",
        "trainer.accelerator": "cpu",
        "trainer.callbacks": [],
    }

    for config_file in config_files:
        model = LuxonisModel(config_file, opts)
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
def test_annotate_from_directory(
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


def test_debug_mode(opts: Params, subtests: SubTests):
    config_file = "configs/detection_light_model.yaml"
    opts = opts | {
        "loader.params.dataset_name": "invalid_dataset_name",
        "trainer.epochs": 1,
        "trainer.batch_size": 1,
    }
    model = LuxonisModel(config_file, opts, debug_mode=True)

    with subtests.test("train"):
        model.train()

    with subtests.test("test"):
        model.test()
