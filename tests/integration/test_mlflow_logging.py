import os
import shutil
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch.nn as nn
from luxonis_ml.data import DatasetIterator, LuxonisDataset
from PIL import Image
from torch import Tensor

from luxonis_train import BaseHead, LuxonisModel, Tasks


@pytest.fixture(scope="session")
def set_env_vars():
    from luxonis_ml.utils.environ import environ

    environ.MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5001"


@pytest.fixture
def setup_mlflow(tmp_path):
    start_time = time.time()
    timeout = 60

    backend_store_uri = f"sqlite:///{tmp_path}/mlflow.db"
    artifact_root = tmp_path / "mlflow-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    mlflow_executable = shutil.which("mlflow")

    process = subprocess.Popen(
        [
            mlflow_executable,
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            "5001",
            "--backend-store-uri",
            backend_store_uri,
            "--default-artifact-root",
            str(artifact_root),
        ]
    )

    while True:
        try:
            mlflow.search_experiments()
            break
        except Exception:
            if time.time() - start_time > timeout:
                process.terminate()
                raise RuntimeError(
                    "MLflow server failed to start within 60 seconds"
                )
            time.sleep(0.5)
    yield
    process.terminate()
    time.sleep(1)


@pytest.mark.timeout(30)
def test_mlflow_logging(tmp_path, setup_mlflow, set_env_vars):
    def generator(tmp_path: Path) -> DatasetIterator:
        """Generate XOR dataset as images with 2 pixels representing XOR
        inputs."""

        data_dir = tmp_path / "xor_data"
        os.makedirs(data_dir, exist_ok=True)

        inputs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]

        outputs = [0, 1, 1, 0]

        for i, (x_values, y_value) in enumerate(zip(inputs, outputs)):
            pixel_values = [x * 255 for x in x_values]
            img_array = np.array(pixel_values, dtype=np.uint8).reshape(1, 2)

            img = Image.fromarray(img_array, mode="L")

            img_path = data_dir / f"xor_{i}.png"
            img.save(img_path)

            yield {
                "file": str(img_path),
                "annotation": {
                    "class": f"xor_{y_value}",
                },
            }

    dataset = LuxonisDataset("xor_dataset", delete_existing=True)
    dataset.add(generator(tmp_path))
    dataset.make_splits((1, 0, 0))

    class XORHead(BaseHead):
        task = Tasks.CLASSIFICATION

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.model = nn.Sequential(
                nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2)
            )

        def forward(self, x: Tensor) -> Tensor:
            x = x[:, 0, :, :].squeeze(1)
            return self.model(x)

    config = {
        "model": {
            "name": "xor_model",
            "nodes": [
                {
                    "name": "XORHead",
                    "losses": [{"name": "CrossEntropyLoss"}],
                    "metrics": [
                        {"name": "Accuracy", "is_main_metric": True},
                        {
                            "name": "ConfusionMatrix",
                        },
                        {"name": "F1Score", "params": {"average": None}},
                    ],
                    "visualizers": [{"name": "ClassificationVisualizer"}],
                }
            ],
        },
        "loader": {
            "name": "LuxonisLoaderTorch",
            "train_view": "train",
            "val_view": "train",
            "test_view": "train",
            "params": {
                "dataset_name": "xor_dataset",
                "bucket_storage": "local",
            },
        },
        "trainer": {
            "precision": "16-mixed",
            "preprocessing": {
                "train_image_size": [1, 2],
                "keep_aspect_ratio": False,
                "normalize": {
                    "active": True,
                    "params": {"mean": [0, 0, 0], "std": [1, 1, 1]},
                },
            },
            "batch_size": 4,
            "epochs": 30,
            "n_log_images": 4,
            "validation_interval": 15,
            "optimizer": {
                "name": "Adam",
                "params": {"lr": 0.1, "weight_decay": 0.01},
            },
            "scheduler": {
                "name": "StepLR",
                "params": {"step_size": 10, "gamma": 0.1},
            },
            "callbacks": [
                {"name": "TestOnTrainEnd"},
                {"name": "ExportOnTrainEnd"},
                {"name": "ArchiveOnTrainEnd"},
                {"name": "UploadCheckpoint"},
                {"name": "GPUStatsMonitor"},
                {"name": "DeviceStatsMonitor"},
            ],
        },
        "tracker": {
            "is_mlflow": True,
            "is_wandb": False,
            "is_tensorboard": True,
            "project_name": "xor_project",
            "run_name": "xor_run",
        },
    }
    model = LuxonisModel(config)
    model.train()

    client = mlflow.tracking.MlflowClient()
    experiments = mlflow.search_experiments()
    experiment_id = experiments[0].experiment_id
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time desc"],
        filter_string="",
    )
    latest_run = runs[0]
    run_id = latest_run.info.run_id

    all_artifacts = []

    def list_artifacts_recursively(path=""):
        artifacts = client.list_artifacts(run_id, path)
        for artifact in artifacts:
            artifact_path = artifact.path
            all_artifacts.append(artifact_path)
            if not artifact.is_dir:
                continue
            list_artifacts_recursively(artifact_path)

    list_artifacts_recursively()

    test_files = [
        "test/metrics/30/XORHead/confusion_matrix.json",
        "test/visualizations/XORHead/ClassificationVisualizer/30/0.png",
        "test/visualizations/XORHead/ClassificationVisualizer/30/1.png",
        "test/visualizations/XORHead/ClassificationVisualizer/30/2.png",
        "test/visualizations/XORHead/ClassificationVisualizer/30/3.png",
    ]

    for file_path in test_files:
        assert file_path in all_artifacts, (
            f"Missing test artifact: {file_path}"
        )

    validation_files = [
        "val/metrics/0/XORHead/confusion_matrix.json",
        "val/metrics/14/XORHead/confusion_matrix.json",
        "val/metrics/29/XORHead/confusion_matrix.json",
        "val/visualizations/XORHead/ClassificationVisualizer/0/0.png",
        "val/visualizations/XORHead/ClassificationVisualizer/0/1.png",
        "val/visualizations/XORHead/ClassificationVisualizer/0/2.png",
        "val/visualizations/XORHead/ClassificationVisualizer/0/3.png",
        "val/visualizations/XORHead/ClassificationVisualizer/14/0.png",
        "val/visualizations/XORHead/ClassificationVisualizer/14/1.png",
        "val/visualizations/XORHead/ClassificationVisualizer/14/2.png",
        "val/visualizations/XORHead/ClassificationVisualizer/14/3.png",
        "val/visualizations/XORHead/ClassificationVisualizer/29/0.png",
        "val/visualizations/XORHead/ClassificationVisualizer/29/1.png",
        "val/visualizations/XORHead/ClassificationVisualizer/29/2.png",
        "val/visualizations/XORHead/ClassificationVisualizer/29/3.png",
    ]

    for file_path in validation_files:
        assert file_path in all_artifacts, (
            f"Missing validation artifact: {file_path}"
        )

    model_files = [
        "best_val_metric.ckpt",
        "luxonis_train.log",
        "min_val_loss.ckpt",
        "training_config.yaml",
        "xor_model.onnx",
        "xor_model.onnx.tar.xz",
        "xor_model.yaml",
    ]

    for file_path in model_files:
        assert file_path in all_artifacts, (
            f"Missing model artifact: {file_path}"
        )

    train_loss = client.get_metric_history(run_id, "train/loss")
    assert len(train_loss) == 30, (
        f"Expected 30 train/loss metrics, but found {len(train_loss)}"
    )

    train_loss_xorhead = client.get_metric_history(
        run_id, "train/loss/XORHead/CrossEntropyLoss"
    )
    assert len(train_loss_xorhead) == 30, (
        f"Expected 30 train/loss/XORHead/CrossEntropyLoss metrics, but found {len(train_loss_xorhead)}"
    )

    expected_val_steps = [14, 29]

    val_loss = client.get_metric_history(run_id, "val/loss")
    assert len(val_loss) == len(expected_val_steps), (
        f"Expected {len(expected_val_steps)} val/loss metrics, but found {len(val_loss)}"
    )
    val_steps = sorted([m.step for m in val_loss])
    assert val_steps == expected_val_steps, (
        f"Expected val/loss steps {expected_val_steps}, but found {val_steps}"
    )

    for metric_name in [
        "val/metric/XORHead/Accuracy",
        "val/metric/XORHead/F1Score",
        "val/metric/XORHead/mcc",
        "val/metric/XORHead/MulticlassF1Score_xor_0",
        "val/metric/XORHead/MulticlassF1Score_xor_1",
    ]:
        metric_values = client.get_metric_history(run_id, metric_name)
        assert len(metric_values) == len(expected_val_steps), (
            f"Expected {len(expected_val_steps)} {metric_name} metrics, but found {len(metric_values)}"
        )
        metric_steps = sorted([m.step for m in metric_values])
        assert metric_steps == expected_val_steps, (
            f"Expected {metric_name} steps {expected_val_steps}, but found {metric_steps}"
        )

    for metric_name in [
        "test/loss/XORHead/CrossEntropyLoss",
        "test/loss",
        "test/metric/XORHead/Accuracy",
        "test/metric/XORHead/mcc",
        "test/metric/XORHead/F1Score",
        "test/metric/XORHead/MulticlassF1Score_xor_0",
        "test/metric/XORHead/MulticlassF1Score_xor_1",
    ]:
        metric_values = client.get_metric_history(run_id, metric_name)
        assert len(metric_values) == 1, (
            f"Expected 1 {metric_name} metric, but found {len(metric_values)}"
        )

    assert file_path in all_artifacts, f"Missing test artifact: {file_path}"
