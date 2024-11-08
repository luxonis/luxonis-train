import glob
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2
import numpy as np
from luxonis_ml.data import BucketStorage, LuxonisDataset

from luxonis_train.core import LuxonisModel

PathType = Union[str, Path]


def get_opts() -> dict[str, Any]:
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
                "train_image_size": [256, 256],
                "keep_aspect_ratio": False,
                "normalize": {"active": True},
            },
            "batch_size": 1,
            "epochs": 1,
            "num_workers": 0,
            "validation_interval": 10,
            "num_sanity_val_steps": 0,
        },
    }


def create_dummy_anomaly_detection_dataset(paths: Path):
    def random_square_mask(image_shape, num_squares=1):
        mask = np.zeros(image_shape, dtype=np.uint8)
        h, w = image_shape
        for _ in range(num_squares):
            top_left = (
                np.random.randint(0, w // 2),
                np.random.randint(0, h // 2),
            )
            bottom_right = (
                np.random.randint(w // 2, w),
                np.random.randint(h // 2, h),
            )
            cv2.rectangle(mask, top_left, bottom_right, 255, -1)
        return mask

    def dummy_generator(
        train_paths: List[PathType], test_paths: List[PathType]
    ):
        for path in train_paths:
            yield {
                "file": path,
                "annotation": {
                    "type": "rle",
                    "class": "object",
                    "height": 256,
                    "width": 256,
                    "counts": "0" * (256 * 256),
                },
            }

        for path in test_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            img_h, img_w, _ = img.shape
            mask = random_square_mask((img_h, img_w))
            poly = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]
            poly_normalized = [
                [(x / img_w, y / img_h) for x, y in contour.reshape(-1, 2)]
                for contour in poly
            ]
            yield {
                "file": path,
                "annotation": {
                    "type": "polyline",
                    "class": "object",
                    "points": [
                        pt for segment in poly_normalized for pt in segment
                    ],
                },
            }

    paths_total: List[PathType] = [
        Path(p) for p in glob.glob(str(paths), recursive=True)[:10]
    ]
    train_paths: List[PathType] = paths_total[:5]
    test_paths: List[PathType] = paths_total[5:]

    dataset = LuxonisDataset(
        "dummy_mvtec",
        bucket_storage=BucketStorage.LOCAL,
        delete_existing=True,
        delete_remote=True,
    )
    dataset.add(dummy_generator(train_paths, test_paths))
    definitions: Dict[str, List[PathType]] = {
        "train": train_paths,
        "val": test_paths,
    }
    dataset.make_splits(definitions=definitions)


def test_anomaly_detection():
    create_dummy_anomaly_detection_dataset(
        Path("tests/data/COCO_people_subset/person_val2017_subset/*")
    )
    config = get_opts()
    model = LuxonisModel(config)
    model.train()
    model.test()
