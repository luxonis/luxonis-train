from luxonis_ml.typing import Params

from luxonis_train.nodes.backbones import __all__ as BACKBONES
from tests.conftest import LuxonisTestDataset, LuxonisTestDatasets

BACKBONES = [
    backbone
    for backbone in BACKBONES
    if backbone not in {"PPLCNetV3", "GhostFaceNet", "RecSubNet"}
]

PREDEFINED_MODELS = [
    ("anomaly_detection_model", None),
    ("embeddings_model", None),
    ("fomo_light_model", None),
    ("ocr_recognition_light_model", None),
    (
        "ocr_recognition_light_model",
        {
            "model.predefined_model.params.neck_params": {
                "mixer": "conv",
                "prenorm": True,
                "height": 8,
                "width": 5,
            },
        },
    ),
    ("classification_light_model", None),
    ("detection_light_model", None),
    ("instance_segmentation_light_model", None),
    ("keypoint_bbox_light_model", None),
    ("segmentation_light_model", None),
    ("classification_heavy_model", None),
    ("detection_heavy_model", None),
    ("instance_segmentation_heavy_model", None),
    ("keypoint_bbox_heavy_model", None),
    ("segmentation_heavy_model", None),
]


def prepare_predefined_model_config(
    config_name: str, opts: Params, test_datasets: LuxonisTestDatasets
) -> tuple[str, dict, LuxonisTestDataset]:
    """Prepares configuration and options for non-backbone models."""
    config_file = f"configs/{config_name}.yaml"

    # Choose dataset based on config name
    if config_name == "embeddings_model":
        dataset = test_datasets.embedding_dataset
    elif "ocr_recognition" in config_name:
        dataset = test_datasets.toy_ocr_dataset
    elif "classification" in config_name:
        dataset = test_datasets.cifar10_dataset
    elif "anomaly_detection" in config_name:
        opts |= {
            "loader.params.anomaly_source_path": str(
                test_datasets.coco_dataset.source_path
            )
        }
        dataset = test_datasets.anomaly_detection_dataset
    else:
        dataset = test_datasets.coco_dataset

    opts |= {
        "model.name": config_name,
        "loader.params.dataset_name": dataset.identifier,
        "tracker.run_name": config_name,
    }

    # Apply dataset-specific overrides
    if config_name == "embeddings_model":
        opts |= {
            "loader.params.dataset_name": test_datasets.embedding_dataset.dataset_name,
            "trainer.batch_size": 16,
            "trainer.preprocessing.train_image_size": [48, 64],
        }
    elif "ocr_recognition" in config_file:
        opts["trainer.preprocessing.train_image_size"] = [48, 320]

    return config_file, opts, dataset
