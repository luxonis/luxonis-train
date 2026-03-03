import math
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from luxonis_ml.typing import Params

from luxonis_train.config import Config
from luxonis_train.core import LuxonisModel
from tests.conftest import LuxonisTestDatasets
from tests.integration.backbone_model_utils import (
    prepare_predefined_model_config,
)

# Loss params with base weights (configs have them multiplied by
# accumulate_grad_batches=8 which inflates loss values in this test).
_DETECTION_LOSS_PARAMS = {
    "model.predefined_model.params.loss_params.iou_loss_weight": 2.5,
    "model.predefined_model.params.loss_params.class_loss_weight": 1.0,
}
_INSTANCE_SEG_LOSS_PARAMS = {
    "model.predefined_model.params.loss_params.bbox_loss_weight": 7.5,
    "model.predefined_model.params.loss_params.class_loss_weight": 0.5,
    "model.predefined_model.params.loss_params.dfl_loss_weight": 1.5,
}
_KEYPOINT_LOSS_PARAMS = {
    "model.predefined_model.params.loss_params.iou_loss_weight": 7.5,
    "model.predefined_model.params.loss_params.class_loss_weight": 0.5,
    "model.predefined_model.params.loss_params.regr_kpts_loss_weight": 12.0,
    "model.predefined_model.params.loss_params.vis_kpts_loss_weight": 1.0,
}

OVERFIT_MODELS = [
    ("classification_light_model", None, 1.0),
    ("detection_light_model", _DETECTION_LOSS_PARAMS, 5.0),
    ("segmentation_light_model", None, 1.0),
    # Instance seg sums 4 weighted losses (bbox 7.5 + seg 7.5 + dfl 1.5 + cls 0.5 = 17.0 total weight).
    # With 200 epochs at lr=1e-3 the total converges to ~6.4, not near zero.
    ("instance_segmentation_light_model", _INSTANCE_SEG_LOSS_PARAMS, 8.0),
    # Keypoint model sums 4 weighted losses (iou 7.5 + kpt_regr 12.0 + vis 1.0 + cls 0.5 = 21.0 total weight).
    # With 200 epochs at lr=1e-3 the total converges to ~9.1, not near zero.
    ("keypoint_bbox_light_model", _KEYPOINT_LOSS_PARAMS, 12.0),
    ("fomo_light_model", None, 1.0),
    ("anomaly_detection_model", None, 1.0),
    ("ocr_recognition_light_model", None, 1.0),
]


@pytest.fixture(autouse=True)
def reset_deterministic_state() -> Generator[None]:
    """Reset PyTorch deterministic state after test."""
    yield
    torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize(
    ("config_name", "extra_opts", "loss_threshold"),
    OVERFIT_MODELS,
    ids=[m[0] for m in OVERFIT_MODELS],
)
def test_overfit_convergence(
    config_name: str,
    extra_opts: dict | None,
    loss_threshold: float,
    opts: Params,
    test_datasets: LuxonisTestDatasets,
    save_dir: Path,
):
    """Train on a single batch and assert loss converges."""
    opts |= {
        "trainer.overfit_batches": 1,
        "trainer.seed": 42,
        "trainer.epochs": 200,
        "trainer.batch_size": 4,
        "trainer.validation_interval": 200,
        "trainer.smart_cfg_auto_populate": False,
        "trainer.training_strategy": None,
        "trainer.optimizer": {"name": "Adam", "params": {"lr": 0.001}},
        "trainer.scheduler": {"name": "ConstantLR"},
        "trainer.callbacks": [
            {"name": "TestOnTrainEnd", "active": False},
            {"name": "ExportOnTrainEnd", "active": False},
            {"name": "ArchiveOnTrainEnd", "active": False},
            {"name": "ConvertOnTrainEnd", "active": False},
            {"name": "UploadCheckpoint", "active": False},
        ],
        "tracker.save_directory": str(save_dir),
    }
    opts["trainer.deterministic"] = "warn"

    if extra_opts:
        opts |= extra_opts

    config_file, opts, _ = prepare_predefined_model_config(
        config_name, opts, test_datasets
    )

    cfg = Config.get_config(config_file, opts)
    model = LuxonisModel(cfg)
    model.train()

    final_loss = model.pl_trainer.callback_metrics["train/loss"].item()
    assert final_loss < loss_threshold, (
        f"{config_name} loss did not converge: {final_loss:.4f} >= {loss_threshold}"
    )


def test_metric_equivalence_callback_an_eval(
    opts: Params,
    test_datasets: LuxonisTestDatasets,
    save_dir: Path,
):
    """Verify TestOnTrainEnd and standalone eval produce the same
    results when EMACallback is active.

    This is because EMACallback can potentially replace the best val
    weights by the latest moving average weights.
    """
    config_name = "detection_light_model"
    opts |= {
        "trainer.overfit_batches": 1,
        "trainer.seed": 42,
        "trainer.epochs": 200,
        "trainer.batch_size": 4,
        "trainer.validation_interval": 50,
        "trainer.smart_cfg_auto_populate": False,
        "trainer.training_strategy": None,
        "trainer.optimizer": {"name": "Adam", "params": {"lr": 0.001}},
        "trainer.scheduler": {"name": "ConstantLR"},
        "trainer.callbacks": [
            {"name": "TestOnTrainEnd", "active": True},
            {
                "name": "EMACallback",
                "active": True,
                "params": {"decay": 0.9999},
            },
            {"name": "ExportOnTrainEnd", "active": False},
            {"name": "ArchiveOnTrainEnd", "active": False},
            {"name": "ConvertOnTrainEnd", "active": False},
            {"name": "UploadCheckpoint", "active": False},
        ],
        "tracker.save_directory": str(save_dir),
        "model.predefined_model.params.loss_params.iou_loss_weight": 2.5,
        "model.predefined_model.params.loss_params.class_loss_weight": 1.0,
    }
    opts["trainer.deterministic"] = "warn"

    config_file, opts, _ = prepare_predefined_model_config(
        config_name, opts, test_datasets
    )

    cfg = Config.get_config(config_file, opts)
    model = LuxonisModel(cfg)
    model.train()

    # Collect test metrics logged by TestOnTrainEnd
    callback_test_metrics = {
        k: v.item()
        for k, v in model.pl_trainer.callback_metrics.items()
        if k.startswith("test/")
    }

    # Run standalone test with the best checkpoint
    best_ckpt = model.get_best_metric_checkpoint_path()
    assert best_ckpt, "No best metric checkpoint found"

    standalone_results = model.test(view="test", weights=best_ckpt)
    assert standalone_results, "Standalone test returned no results"

    # Compare both
    for key in callback_test_metrics:
        if key not in standalone_results:
            continue
        callback_val = callback_test_metrics[key]
        standalone_val = standalone_results[key]
        # nan != nan in floating point, so handle it explicitly
        if math.isnan(callback_val) and math.isnan(standalone_val):
            continue
        assert callback_val == pytest.approx(standalone_val), (
            f"Metric mismatch for '{key}': "
            f"TestOnTrainEnd={callback_val:.6f} vs "
            f"standalone={standalone_val:.6f}"
        )
