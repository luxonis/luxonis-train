from pathlib import Path

import cv2
import numpy as np
from luxonis_ml.data import DatasetIterator, LuxonisDataset
from luxonis_ml.typing import Params
from tensorboard.backend.event_processing import event_accumulator

from luxonis_train import LuxonisModel


def get_config(dataset_name: str) -> Params:
    return {
        "model": {
            "name": "test_smart_vis",
            "nodes": [
                {
                    "name": "EfficientRep",
                    "variant": "n",
                },
                {
                    "name": "RepPANNeck",
                    "inputs": ["EfficientRep"],
                    "variant": "n",
                },
                {
                    "name": "EfficientBBoxHead",
                    "task_name": "animals",
                    "inputs": ["RepPANNeck"],
                    "visualizers": [{"name": "BBoxVisualizer"}],
                    "metrics": [{"name": "MeanAveragePrecision"}],
                    "losses": [{"name": "AdaptiveDetectionLoss"}],
                },
                {
                    "name": "SegmentationHead",
                    "task_name": "objects",
                    "inputs": ["RepPANNeck"],
                    "visualizers": [{"name": "SegmentationVisualizer"}],
                    "metrics": [{"name": "JaccardIndex"}],
                    "losses": [{"name": "CrossEntropyLoss"}],
                },
            ],
        },
        "loader": {
            "test_view": "val",
            "params": {"dataset_name": dataset_name},
        },
        "trainer": {"batch_size": 2, "n_log_images": 9},
    }


def test_smart_vis_logging(tmp_path: Path):
    dataset = create_dataset(tmp_path)
    model = LuxonisModel(get_config(dataset.identifier))

    model.test()

    log_dir = model.lightning_module.logger.experiment["tensorboard"].log_dir

    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    image_tags = set(ea.Tags().get("images", []))

    expected_det = {
        f"test/visualizations/animals-EfficientBBoxHead/BBoxVisualizer/{i}"
        for i in range(9)
    }
    expected_kpts = {
        f"test/visualizations/objects-SegmentationHead/SegmentationVisualizer/{i}"
        for i in range(9)
    }

    assert image_tags == expected_det | expected_kpts


def create_image(i: int, dir: Path) -> Path:
    path = dir / f"img_{i}.jpg"
    if not path.exists():
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[0:10, 0:10] = np.random.randint(
            0, 255, (10, 10, 3), dtype=np.uint8
        )
        cv2.imwrite(str(path), img)
    return path


def create_dataset(tmp_path: Path) -> LuxonisDataset:
    definitions = {"train": [], "val": []}

    def generator() -> DatasetIterator:
        for i in range(10):
            path = create_image(i, tmp_path)
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
            if i > 5:
                # luxonis-ml: background class will be assigned based
                # on 0-th sample (self._load_data(0) in luxonis_loader.py)
                mask = np.zeros((64, 64), dtype=np.uint8)
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

        path = create_image(10, tmp_path)
        yield {"file": str(path), "task_name": "animals"}
        definitions["val"].append(str(path))

    dataset = LuxonisDataset("non_balanced", delete_local=True)
    dataset.add(generator())
    dataset.make_splits(definitions)
    return dataset
