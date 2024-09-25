import json
import multiprocessing as mp
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import gdown
import numpy as np
import pytest
import torchvision
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.data.utils.data_utils import rgb_to_bool_masks
from luxonis_ml.utils import LuxonisFileSystem, environ

WORK_DIR = Path("tests", "data")


@pytest.fixture(scope="session")
def test_output_dir() -> Path:
    return Path("tests/integration/save-directory")


@pytest.fixture(scope="session", autouse=True)
def setup(test_output_dir: Path):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(WORK_DIR / "luxonisml", ignore_errors=True)
    shutil.rmtree(test_output_dir, ignore_errors=True)
    environ.LUXONISML_BASE_PATH = WORK_DIR / "luxonisml"
    test_output_dir.mkdir(exist_ok=True)


@pytest.fixture
def train_overfit() -> bool:
    return bool(os.getenv("LUXONIS_TRAIN_OVERFIT"))


@pytest.fixture(scope="session")
def parking_lot_dataset() -> LuxonisDataset:
    url = "gs://luxonis-test-bucket/luxonis-ml-test-data/D1_ParkingSlotTest"
    base_path = WORK_DIR / "D1_ParkingSlotTest"
    if not base_path.exists():
        base_path = LuxonisFileSystem.download(url, WORK_DIR)

    mask_brand_path = base_path / "mask_brand"
    mask_color_path = base_path / "mask_color"
    kpt_mask_path = base_path / "keypoints_mask_vehicle"

    def generator():
        filenames: dict[int, Path] = {}
        for base_path in [kpt_mask_path, mask_brand_path, mask_color_path]:
            for sequence_path in sorted(list(base_path.glob("sequence.*"))):
                frame_data = sequence_path / "step0.frame_data.json"
                with open(frame_data) as f:
                    data = json.load(f)["captures"][0]
                    frame_data = data["annotations"]
                    sequence_num = int(sequence_path.suffix[1:])
                    filename = data["filename"]
                    if filename is not None:
                        filename = sequence_path / filename
                        filenames[sequence_num] = filename
                    else:
                        filename = filenames[sequence_num]
                    W, H = data["dimension"]

                annotations = {
                    anno["@type"].split(".")[-1]: anno for anno in frame_data
                }

                bbox_classes = {}
                bboxes = {}

                for bbox_annotation in annotations.get(
                    "BoundingBox2DAnnotation", defaultdict(list)
                )["values"]:
                    class_ = (
                        bbox_annotation["labelName"].split("-")[-1].lower()
                    )
                    if class_ == "motorbiek":
                        class_ = "motorbike"
                    x, y = bbox_annotation["origin"]
                    w, h = bbox_annotation["dimension"]
                    instance_id = bbox_annotation["instanceId"]
                    bbox_classes[instance_id] = class_
                    bboxes[instance_id] = [x / W, y / H, w / W, h / H]
                    yield {
                        "file": filename,
                        "annotation": {
                            "type": "boundingbox",
                            "class": class_,
                            "x": x / W,
                            "y": y / H,
                            "w": w / W,
                            "h": h / H,
                            "instance_id": instance_id,
                        },
                    }

                for kpt_annotation in annotations.get(
                    "KeypointAnnotation", defaultdict(list)
                )["values"]:
                    keypoints = kpt_annotation["keypoints"]
                    instance_id = kpt_annotation["instanceId"]
                    class_ = bbox_classes[instance_id]
                    bbox = bboxes[instance_id]
                    kpts = []

                    if class_ == "motorbike":
                        keypoints = keypoints[:3]
                    else:
                        keypoints = keypoints[3:]

                    for kp in keypoints:
                        x, y = kp["location"]
                        kpts.append([x / W, y / H, kp["state"]])

                    yield {
                        "file": filename,
                        "annotation": {
                            "type": "detection",
                            "class": class_,
                            "task": class_,
                            "keypoints": kpts,
                            "instance_id": instance_id,
                            "boundingbox": {
                                "x": bbox[0],
                                "y": bbox[1],
                                "w": bbox[2],
                                "h": bbox[3],
                            },
                        },
                    }

                vehicle_type_segmentation = annotations[
                    "SemanticSegmentationAnnotation"
                ]
                mask = cv2.cvtColor(
                    cv2.imread(
                        str(
                            sequence_path
                            / vehicle_type_segmentation["filename"]
                        )
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                classes = {
                    inst["labelName"]: inst["pixelValue"][:3]
                    for inst in vehicle_type_segmentation["instances"]
                }
                if base_path == kpt_mask_path:
                    task = "vehicle_type-segmentation"
                elif base_path == mask_brand_path:
                    task = "brand-segmentation"
                else:
                    task = "color-segmentation"
                for class_, mask_ in rgb_to_bool_masks(
                    mask, classes, add_background_class=True
                ):
                    yield {
                        "file": filename,
                        "annotation": {
                            "type": "mask",
                            "class": class_,
                            "task": task,
                            "mask": mask_,
                        },
                    }
                if base_path == mask_color_path:
                    yield {
                        "file": filename,
                        "annotation": {
                            "type": "mask",
                            "class": "vehicle",
                            "task": "vehicle-segmentation",
                            "mask": mask.astype(bool)[..., 0]
                            | mask.astype(bool)[..., 1]
                            | mask.astype(bool)[..., 2],
                        },
                    }

    dataset = LuxonisDataset("_ParkingLot", delete_existing=True)
    dataset.add(generator())
    np.random.seed(42)
    dataset.make_splits()
    return dataset


@pytest.fixture(scope="session")
def coco_dataset() -> LuxonisDataset:
    dataset_name = "coco_test"
    url = "https://drive.google.com/uc?id=1XlvFK7aRmt8op6-hHkWVKIJQeDtOwoRT"
    output_zip = WORK_DIR / "COCO_people_subset.zip"

    if (
        not output_zip.exists()
        and not (WORK_DIR / "COCO_people_subset").exists()
    ):
        gdown.download(url, str(output_zip), quiet=False)

    parser = LuxonisParser(
        str(output_zip), dataset_name=dataset_name, delete_existing=True
    )
    return parser.parse(random_split=True)


@pytest.fixture(scope="session")
def cifar10_dataset() -> LuxonisDataset:
    dataset = LuxonisDataset("cifar10_test", delete_existing=True)
    output_folder = WORK_DIR / "cifar10"
    output_folder.mkdir(parents=True, exist_ok=True)
    cifar10_torch = torchvision.datasets.CIFAR10(
        root=output_folder, train=False, download=True
    )
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def CIFAR10_subset_generator():
        for i, (image, label) in enumerate(cifar10_torch):  # type: ignore
            if i == 1000:
                break
            path = output_folder / f"cifar_{i}.png"
            image.save(path)
            yield {
                "file": path,
                "annotation": {
                    "type": "classification",
                    "class": classes[label],
                },
            }

    dataset.add(CIFAR10_subset_generator())
    dataset.make_splits()
    return dataset


@pytest.fixture
def config(train_overfit: bool) -> dict[str, Any]:
    if train_overfit:  # pragma: no cover
        epochs = 100
    else:
        epochs = 1

    return {
        "tracker": {
            "save_directory": "tests/integration/save-directory",
        },
        "loader": {
            "train_view": "val",
            "params": {
                "dataset_name": "_ParkingLot",
            },
        },
        "trainer": {
            "batch_size": 4,
            "epochs": epochs,
            "n_workers": mp.cpu_count(),
            "validation_interval": epochs,
            "save_top_k": 0,
            "preprocessing": {
                "train_image_size": [256, 320],
                "keep_aspect_ratio": False,
                "normalize": {"active": True},
            },
            "callbacks": [
                {"name": "ExportOnTrainEnd"},
            ],
            "matmul_precision": "medium",
        },
    }
