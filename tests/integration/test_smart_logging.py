import os
from pathlib import Path

import cv2
import numpy as np
from luxonis_ml.data import LuxonisDataset
from tensorboard.backend.event_processing import event_accumulator

from luxonis_train import LuxonisModel


def create_image(i: int, dir: Path) -> Path:
    path = dir / f"img_{i}.jpg"
    if not path.exists():
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[0:10, 0:10] = np.random.randint(
            0, 255, (10, 10, 3), dtype=np.uint8
        )
        cv2.imwrite(str(path), img)
    return path


CONFIG = {
    "model": {
        "name": "detection_light",
        "nodes": [
            {
                "name": "EfficientRep",
                "params": {
                    "variant": "n",
                },
            },
            {
                "name": "RepPANNeck",
                "inputs": ["EfficientRep"],
                "params": {
                    "variant": "n",
                },
            },
            {
                "name": "EfficientBBoxHead",
                "task_name": "animals",
                "inputs": ["RepPANNeck"],
                "visualizers": [
                    {
                        "name": "BBoxVisualizer",
                    }
                ],
                "metrics": [
                    {
                        "name": "MeanAveragePrecision",
                    }
                ],
                "losses": [
                    {
                        "name": "AdaptiveDetectionLoss",
                    }
                ],
            },
            {
                "name": "SegmentationHead",
                "task_name": "objects",
                "inputs": ["RepPANNeck"],
                "visualizers": [
                    {
                        "name": "SegmentationVisualizer",
                    }
                ],
                "metrics": [{"name": "JaccardIndex"}],
                "losses": [
                    {
                        "name": "CrossEntropyLoss",
                    }
                ],
            },
        ],
    },
    "loader": {
        "test_view": "val",
        "params": {
            "dataset_name": "duplicates",
        },
    },
    "trainer": {
        "batch_size": 2,
        "n_log_images": 9,
    },
}


def test_smart_vis_logging(work_dir: Path):
    temp_dir = Path(work_dir) / "non_balanced"
    temp_dir.mkdir(parents=True, exist_ok=True)
    definitions = {"train": [], "val": []}

    def generator():
        for i in range(10):
            path = create_image(i, temp_dir)
            if i <= 5:
                yield {
                    "file": str(path),
                    "task_name": "animals",
                    "annotation": {
                        "class": "cat" if i in [0, 1] else "dog",
                        "boundingbox": {
                            "x": 0.5,
                            "y": 0.5,
                            "w": 0.1,
                            "h": 0.3,
                        },
                        "keypoints": {
                            "keypoints": [(0.5, 0.5, 1), (0.6, 0.6, 1)],
                        },
                    },
                }
            if (
                i > 5
            ):  # luxonis-ml: background class will be assigned based on 0-th sample (self._load_data(0) in luxonis_loader.py)
                mask = np.zeros((512, 512), dtype=np.uint8)
                mask[0:10, 0:10] = np.random.randint(
                    0, 2, (10, 10), dtype=np.uint8
                )
                yield {
                    "file": str(path),
                    "task_name": "objects",
                    "annotation": {
                        "class": "house" if i % 2 == 0 else "car",
                        "segmentation": {"mask": mask},
                    },
                }

                background_mask = 1 - mask
                yield {
                    "file": str(path),
                    "task_name": "objects",
                    "annotation": {
                        "class": "background",
                        "segmentation": {"mask": background_mask},
                    },
                }

            if i in [0, 1]:
                definitions["train"].append(str(path))
            else:
                definitions["val"].append(str(path))
            path = create_image(i, temp_dir)

        path = create_image(11, temp_dir)
        yield {
            "file": str(path),
        }
        definitions["val"].append(str(path))

    dataset = LuxonisDataset("non_balanced", delete_local=True)
    dataset.add(generator())
    dataset.make_splits(definitions)

    opts = {
        "loader.params.dataset_name": dataset.identifier,
    }

    model = LuxonisModel(CONFIG, opts)
    model.test()

    log_dir = model.lightning_module.logger.experiment["tensorboard"].log_dir

    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    image_tags = ea.Tags().get("images", [])

    expected_det = [
        f"test/visualizations/animals-EfficientBBoxHead/BBoxVisualizer/{i}"
        for i in range(9)
    ]
    expected_kpts = [
        f"test/visualizations/objects-SegmentationHead/SegmentationVisualizer/{i}"
        for i in range(9)
    ]

    expected = set(expected_det) | set(expected_kpts)

    assert set(image_tags) == expected, (
        f"Got image tags {image_tags!r}, but expected exactly {sorted(expected)!r}"
    )
