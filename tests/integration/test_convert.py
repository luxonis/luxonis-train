from pathlib import Path

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel


def test_convert_basic(
    coco_dataset: LuxonisDataset, opts: Params, tmp_path: Path
):
    """Export + archive, without blobconverter or hubai exporter
    defined."""
    config_file = "configs/detection_light_model.yaml"
    opts |= {
        "loader.params.dataset_name": coco_dataset.identifier,
        "model.name": "test_convert_basic",
        "exporter.blobconverter.active": False,
        "exporter.hubai.active": False,
    }
    model = LuxonisModel(config_file, opts)

    save_dir = tmp_path / "convert_output"
    archive_path = model.convert(save_dir=save_dir)

    assert archive_path.exists(), "Archive was not created"
    assert archive_path.suffix == ".xz", "Archive should be a .xz file"

    onnx_path = model._exported_models.get("onnx")
    assert onnx_path is not None, "ONNX model was not exported"
    assert Path(onnx_path).exists(), "ONNX file does not exist"


def test_convert_with_blobconverter(
    coco_dataset: LuxonisDataset, opts: Params, tmp_path: Path
):
    config_file = "configs/detection_light_model.yaml"
    opts |= {
        "loader.params.dataset_name": coco_dataset.identifier,
        "model.name": "test_convert_blob",
        "exporter.blobconverter.active": True,
        "exporter.hubai.active": False,
        "exporter.scale_values": [255.0, 255.0, 255.0],
        "exporter.mean_values": [127.5, 127.5, 127.5],
    }
    model = LuxonisModel(config_file, opts)

    save_dir = tmp_path / "convert_blob_output"
    archive_path = model.convert(save_dir=save_dir)

    assert archive_path.exists(), "Archive was not created"

    blob_path = model._exported_models.get("blob")
    assert blob_path is not None, "Blob model was not created"
    assert Path(blob_path).exists(), "Blob file does not exist"
    assert Path(blob_path).suffix == ".blob", (
        "Blob file should have .blob extension"
    )


@pytest.mark.parametrize("platform", ["rvc2", "rvc3", "rvc4", "hailo"])
def test_convert_with_hubai(
    coco_dataset: LuxonisDataset, opts: Params, tmp_path: Path, platform: str
):
    config_file = "configs/detection_light_model.yaml"
    opts |= {
        "loader.params.dataset_name": coco_dataset.identifier,
        "model.name": f"test_convert_hubai_{platform}",
        "exporter.blobconverter.active": False,
        "exporter.hubai.active": True,
        "exporter.hubai.platform": platform,
    }
    model = LuxonisModel(config_file, opts)

    save_dir = tmp_path / f"convert_hubai_{platform}_output"
    archive_path = model.convert(save_dir=save_dir)

    assert archive_path.exists(), "Archive was not created"

    platform_archives = list(save_dir.glob("*.tar.xz"))
    assert len(platform_archives) > 0, (
        f"No platform-specific archive created for {platform}"
    )


def test_convert_saves_to_default_directory(
    coco_dataset: LuxonisDataset, opts: Params
):
    """Test that convert uses default save directory when not
    specified."""
    config_file = "configs/detection_light_model.yaml"
    opts |= {
        "loader.params.dataset_name": coco_dataset.identifier,
        "model.name": "test_convert_default_dir",
        "exporter.blobconverter.active": False,
        "exporter.hubai.active": False,
    }
    model = LuxonisModel(config_file, opts)

    archive_path = model.convert()

    assert archive_path.exists(), "Archive was not created"
