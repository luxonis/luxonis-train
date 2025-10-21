import json

import torch
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.config import Config
from luxonis_train.core import LuxonisModel


def test_callbacks(coco_dataset: LuxonisDataset, opts: Params):
    config_file = "configs/segmentation_light_model.yaml"
    opts |= {
        "rich_logging": False,
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
            {
                "name": "GradCamCallback",
                "params": {
                    "target_layer": 10,
                    "task": "segmentation",
                },
            },
        ],
        "exporter.scale_values": [0.5, 0.5, 0.5],
        "exporter.mean_values": [0.5, 0.5, 0.5],
        "exporter.blobconverter.active": True,
        "loader.params.dataset_name": coco_dataset.identifier,
    }
    model = LuxonisModel(config_file, opts, debug_mode=True)
    model.train()

    ckpt_path = model.get_best_metric_checkpoint_path()
    assert ckpt_path is not None, "No checkpoint found after training"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    assert "execution_order" in ckpt
    with open("tests/files/execution_order.json") as f:
        assert ckpt["execution_order"] == json.load(f)

    assert "config" in ckpt
    cfg = Config.get_config(ckpt["config"])
    assert model.cfg.model_dump() == cfg.model_dump()

    assert "dataset_metadata" in ckpt
    assert ckpt["dataset_metadata"] == {
        "classes": {"": {"person": 0}},
        "n_keypoints": {"": 17},
        "metadata_types": {},
    }
