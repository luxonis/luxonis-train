import glob
from pathlib import Path
from typing import Any, List, Union

import cv2
import numpy as np
import pytest
from luxonis_ml.data import BucketStorage, LuxonisDataset

from luxonis_train.core import LuxonisModel

PathType = Union[str, Path]


def create_dummy_bbox_keypoint_dataset(paths: Path):
    def random_bboxes_with_center_keypoints(image_shape, num_bboxes=3):
        h, w = image_shape
        bboxes = []
        keypoints = []

        for _ in range(num_bboxes):
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            x2 = np.random.randint(w // 2, w)
            y2 = np.random.randint(h // 2, h)

            bbox = {
                "x": x1 / w,
                "y": y1 / h,
                "w": (x2 - x1) / w,
                "h": (y2 - y1) / h,
            }
            bboxes.append(bbox)

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            keypoints.append((center_x / w, center_y / h, 2))

        return bboxes, keypoints

    def dummy_generator(image_paths: List[Path]):
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            img_h, img_w, _ = img.shape

            bboxes, keypoints = random_bboxes_with_center_keypoints(
                (img_h, img_w)
            )

            for i, bbox in enumerate(bboxes):
                # Generate bounding box annotation
                yield {
                    "file": path,
                    "annotation": {
                        "type": "boundingbox",
                        "instance_id": i,
                        "class": "object",
                        "x": bbox["x"],
                        "y": bbox["y"],
                        "w": bbox["w"],
                        "h": bbox["h"],
                    },
                }
                yield {
                    "file": path,
                    "annotation": {
                        "type": "keypoints",
                        "instance_id": i,
                        "class": "object",
                        "keypoints": [keypoints[i]],
                    },
                }

    paths_total = [Path(p) for p in glob.glob(str(paths), recursive=True)[:10]]
    train_paths = paths_total[:5]
    test_paths = paths_total[5:]

    dataset = LuxonisDataset(
        "dummy_coco_bbox_keypoints",
        bucket_storage=BucketStorage.LOCAL,
        delete_existing=True,
        delete_remote=True,
    )
    dataset.add(dummy_generator(train_paths + test_paths))
    definitions: dict[str, list[PathType]] = {  # type: ignore
        "train": train_paths,
        "val": test_paths,
    }
    dataset.make_splits(definitions=definitions)


def get_opts_fomo_with_dummy_dataset(
    variant: str, dataset_name: str
) -> dict[str, Any]:
    return {
        "model": {
            "name": f"fomo_detection_{variant}",
            "predefined_model": {
                "name": "FOMOModel",
                "params": {"variant": variant},
            },
        },
        "loader": {
            "params": {
                "dataset_name": dataset_name  # Use the dummy dataset here
            },
            "train_view": "train",
            "val_view": "val",
            "test_view": "val",
        },
        "trainer": {
            "preprocessing": {
                "train_image_size": [384, 512],
                "keep_aspect_ratio": True,
                "normalize": {"active": True},
            },
            "batch_size": 8,
            "epochs": 200,
            "n_workers": 8,
            "validation_interval": 10,
            "n_log_images": 8,
            "callbacks": [
                {"name": "ExportOnTrainEnd"},
                {"name": "TestOnTrainEnd"},
            ],
            "optimizer": {"name": "Adam", "params": {"lr": 0.001}},
        },
    }


def train_and_test_fomo(
    config: dict[str, Any],
    opts: dict[str, Any],
):
    model = LuxonisModel(config, opts)
    model.train()


@pytest.mark.parametrize("variant", ["light", "heavy"])
def test_fomo_variants(variant: str):
    dataset_path = Path(
        "tests/data/COCO_people_subset/person_val2017_subset/*"
    )
    dataset_name = "dummy_coco_bbox_keypoints"
    create_dummy_bbox_keypoint_dataset(dataset_path)
    opts = get_opts_fomo_with_dummy_dataset(variant, dataset_name)
    config = {}
    train_and_test_fomo(config=config, opts=opts)
