import os
import shutil
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Final

import mlflow
import psutil
import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params
from luxonis_ml.utils.environ import environ
from mlflow.client import MlflowClient
from pytest_subtests import SubTests
from torch import Tensor, nn

from luxonis_train import BaseHead, LuxonisModel, Tasks

TIMEOUT: Final[int] = 60


@pytest.fixture(autouse=True)
def setup(tmp_path: Path):
    environ.MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
    os.environ["MLFLOW_TRACKING_URI"] = environ.MLFLOW_TRACKING_URI

    start_time = time.time()

    backend_store_uri = f"sqlite:///{tmp_path}/mlflow.db"
    artifact_root = tmp_path / "mlflow-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    mlflow_executable = shutil.which("mlflow")
    assert mlflow_executable, "MLflow executable not found in PATH"
    artifact_root_uri = artifact_root.as_uri()

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
            artifact_root_uri,
        ]
    )

    while True:
        try:
            mlflow.search_experiments()
            break
        except Exception as e:  # pragma: no cover
            if time.time() - start_time > TIMEOUT:
                process.kill()
                raise RuntimeError(
                    "MLflow server failed to start within 60 seconds"
                ) from e
            time.sleep(0.5)

    yield
    kill_process_tree(process.pid)


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


@pytest.mark.timeout(TIMEOUT)
def test_mlflow_logging(xor_dataset: LuxonisDataset, subtests: SubTests):
    model = LuxonisModel(
        get_config(), {"loader.params.dataset_name": xor_dataset.identifier}
    )
    model.train()

    client = mlflow.tracking.MlflowClient()
    experiments = mlflow.search_experiments()
    experiment_id = experiments[0].experiment_id
    run_id = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time desc"],
        filter_string='tags.mlflow.runName="xor_run"',
    )[0].info.run_id

    all_artifacts = list_artifacts(client, run_id)
    all_mlflow_logging_keys = model.get_mlflow_logging_keys()

    expected_files = {
        "test/metrics/10/XORHead/confusion_matrix.json",
        "test/visualizations/XORHead/ClassificationVisualizer/10/0.png",
        "test/visualizations/XORHead/ClassificationVisualizer/10/1.png",
        "test/visualizations/XORHead/ClassificationVisualizer/10/2.png",
        "val/metrics/0/XORHead/confusion_matrix.json",
        "val/metrics/4/XORHead/confusion_matrix.json",
        "val/metrics/9/XORHead/confusion_matrix.json",
        "val/visualizations/XORHead/ClassificationVisualizer/0/0.png",
        "val/visualizations/XORHead/ClassificationVisualizer/0/1.png",
        "val/visualizations/XORHead/ClassificationVisualizer/0/2.png",
        "val/visualizations/XORHead/ClassificationVisualizer/4/0.png",
        "val/visualizations/XORHead/ClassificationVisualizer/4/1.png",
        "val/visualizations/XORHead/ClassificationVisualizer/4/2.png",
        "val/visualizations/XORHead/ClassificationVisualizer/9/0.png",
        "val/visualizations/XORHead/ClassificationVisualizer/9/1.png",
        "val/visualizations/XORHead/ClassificationVisualizer/9/2.png",
        "best_val_metric.ckpt",
        "luxonis_train.log",
        "min_val_loss.ckpt",
        "training_config.yaml",
        "xor_model.onnx",
        "xor_model.onnx.tar.xz",
        "xor_model.yaml",
    }

    assert expected_files <= (set(all_mlflow_logging_keys["artifacts"]))

    for file_path in expected_files:
        assert file_path in all_artifacts

    for key, expected_steps in zip(
        [
            "train/loss",
            "train/loss/XORHead/CrossEntropyLoss",
            "val/loss",
            "val/metric/XORHead/Accuracy",
            "val/metric/XORHead/F1Score",
            "val/metric/XORHead/mcc",
            "val/metric/XORHead/MulticlassF1Score_xor_0",
            "val/metric/XORHead/MulticlassF1Score_xor_1",
            "test/loss",
            "test/loss/XORHead/CrossEntropyLoss",
            "test/metric/XORHead/Accuracy",
            "test/metric/XORHead/mcc",
            "test/metric/XORHead/F1Score",
            "test/metric/XORHead/MulticlassF1Score_xor_0",
            "test/metric/XORHead/MulticlassF1Score_xor_1",
        ],
        [
            set(range(10)),
            set(range(10)),
            {4, 9},
            {4, 9},
            {4, 9},
            {4, 9},
            {4, 9},
            {4, 9},
            {10},
            {10},
            {10},
            {10},
            {10},
            {10},
            {10},
        ],
        strict=True,
    ):
        with subtests.test(key):
            history = client.get_metric_history(run_id, key)
            assert len(history) == len(expected_steps)
            assert {m.step for m in history} == expected_steps
            assert key in all_mlflow_logging_keys["metrics"]


def get_config() -> Params:
    return {
        "model": {
            "name": "xor_model",
            "nodes": [
                {
                    "name": "XORHead",
                    "losses": [{"name": "CrossEntropyLoss"}],
                    "metrics": [
                        {"name": "Accuracy", "is_main_metric": True},
                        {"name": "ConfusionMatrix"},
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
        },
        "trainer": {
            "precision": "16-mixed",
            "preprocessing": {
                "train_image_size": [1, 2],
                "keep_aspect_ratio": False,
            },
            "batch_size": 4,
            "epochs": 10,
            "n_log_images": 3,
            "validation_interval": 5,
            "scheduler": {
                "name": "StepLR",
                "params": {"step_size": 10, "gamma": 0.1},
            },
            "callbacks": [
                {"name": "TestOnTrainEnd"},
                {"name": "ExportOnTrainEnd"},
                {"name": "ArchiveOnTrainEnd"},
                {"name": "UploadCheckpoint"},
                {"name": "DeviceStatsMonitor"},
            ],
        },
        "tracker": {
            "is_mlflow": True,
            "project_name": "xor_project",
            "run_name": "xor_run",
        },
    }


def kill_process_tree(pid: int) -> None:
    with suppress(psutil.NoSuchProcess):
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()

        _, alive = psutil.wait_procs(
            [parent, *parent.children(recursive=True)], timeout=5
        )
        for p in alive:  # pragma: no cover
            p.kill()


def list_artifacts(
    client: MlflowClient, run_id: str, path: str = ""
) -> list[str]:
    artifacts = client.list_artifacts(run_id, path)
    results = []
    for artifact in artifacts:
        artifact_path = artifact.path
        results.append(artifact_path)
        if artifact.is_dir:
            results.extend(list_artifacts(client, run_id, artifact_path))
    return results
