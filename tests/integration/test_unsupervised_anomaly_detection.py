from pathlib import Path

import cv2
import numpy as np
from luxonis_ml.data import BucketStorage, LuxonisDataset
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


def create_dummy_anomaly_detection_dataset(paths: Path):
    def random_square_mask(image_shape, n_squares=1):
        mask = np.zeros(image_shape, dtype=np.uint8)
        h, w = image_shape
        for _ in range(n_squares):
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

    def dummy_generator(train_paths: list[Path], test_paths: list[Path]):
        for path in train_paths:
            img = cv2.imread(str(path))
            img_h, img_w, _ = img.shape
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            yield {
                "file": path,
                "annotation": {
                    "class": "object",
                    "segmentation": {"mask": mask},
                },
            }

        for path in test_paths:
            img = cv2.imread(str(path))
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
                    "class": "object",
                    "segmentation": {
                        "height": img_h,
                        "width": img_w,
                        "points": [
                            pt for segment in poly_normalized for pt in segment
                        ],
                    },
                },
            }

    paths_total = list(paths.rglob("*.jpg"))
    train_paths = paths_total[:5]
    test_paths = paths_total[5:]

    dataset = LuxonisDataset(
        "dummy_mvtec",
        bucket_storage=BucketStorage.LOCAL,
        delete_existing=True,
        delete_remote=True,
    )
    dataset.add(dummy_generator(train_paths, test_paths))
    definitions = {
        "train": train_paths,
        "val": test_paths,
    }
    dataset.make_splits(definitions=definitions)  # type: ignore


# FIXME: Will break if it runs before the dataset is created
def test_anomaly_detection():
    create_dummy_anomaly_detection_dataset(
        Path("tests/data/COCO_people_subset/person_val2017_subset/*")
    )
    model = LuxonisModel(get_config())
    model.train()
    model.test()
