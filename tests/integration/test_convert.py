from pathlib import Path

import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.config.config import ExportConfig, HubAIExportConfig
from luxonis_train.core import LuxonisModel


def test_convert_basic(
    coco_dataset: LuxonisDataset, opts: Params, tmp_path: Path
):
    """Export + archive, without blobconverter or hubai exporter
    defined.
    """
    config_file = "configs/detection_light_model.yaml"
    opts |= {
        "loader.params.dataset_name": coco_dataset.identifier,
        "model.name": "test_convert_basic",
        "exporter.blobconverter.active": False,
        "exporter.hubai.active": False,
    }
    model = LuxonisModel(config_file, opts)

    save_dir = tmp_path / "convert_output"
    archive_path, _ = model.convert(save_dir=save_dir)

    assert archive_path.exists(), "Archive was not created"
    assert archive_path.suffix == ".xz", "Archive should be a .xz file"

    onnx_path = model._exported_models.get("onnx")
    assert onnx_path is not None, "ONNX model was not exported"
    assert Path(onnx_path).exists(), "ONNX file does not exist"


def test_convert_with_blobconverter(
    coco_dataset: LuxonisDataset,
    opts: Params,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    def fake_blobconverter_export(
        cfg: ExportConfig,
        scale_values: list[float] | None,
        mean_values: list[float] | None,
        reverse_channels: bool,
        export_path: Path | str,
        onnx_path: Path | str,
    ) -> Path:
        assert cfg.blobconverter.active
        assert scale_values == [255.0, 255.0, 255.0]
        assert mean_values == [127.5, 127.5, 127.5]
        assert reverse_channels is True
        onnx_path = Path(onnx_path)
        assert onnx_path.exists()

        blob_path = Path(export_path) / f"{onnx_path.stem}.blob"
        blob_path.write_bytes(b"blob")
        return blob_path

    monkeypatch.setattr(
        "luxonis_train.core.core.blobconverter_export",
        fake_blobconverter_export,
    )

    archive_path, conversion_artifacts = model.convert(save_dir=save_dir)

    assert archive_path.exists(), "Archive was not created"

    assert conversion_artifacts.get("blob") == model._exported_models.get(
        "blob"
    )
    blob_path = model._exported_models.get("blob")
    assert blob_path is not None, "Blob model was not created"
    assert Path(blob_path).exists(), "Blob file does not exist"
    assert Path(blob_path).suffix == ".blob", (
        "Blob file should have .blob extension"
    )


# TODO: reintroduce Hailo conversion when modelconv is released and hub-ai is updated accordingly
# TODO: reintroduce RVC3 conversion when remote-side conversion issue is resolved
@pytest.mark.parametrize("platform", ["rvc2", "rvc4"])
def test_convert_with_hubai(
    coco_dataset: LuxonisDataset,
    opts: Params,
    tmp_path: Path,
    platform: str,
    monkeypatch: pytest.MonkeyPatch,
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

    def fake_hubai_export(
        cfg: HubAIExportConfig,
        quantization_mode: str,
        archive_path: Path,
        export_path: Path,
        model_name: str,
        dataset_name: str | None = None,
    ) -> Path:
        assert cfg.platform == platform
        assert quantization_mode == model.cfg.exporter.quantization_mode
        assert model_name == model.cfg.model.name
        archive_path = Path(archive_path)
        assert archive_path.exists()

        output_path = (
            Path(export_path) / f"{model_name}_{platform.upper()}.tar.xz"
        )
        output_path.write_bytes(archive_path.read_bytes())
        return output_path

    monkeypatch.setattr(
        "luxonis_train.core.core.hubai_export", fake_hubai_export
    )

    archive_path, conversion_artifacts = model.convert(save_dir=save_dir)

    assert archive_path.exists(), "Base archive was not created"
    hubai_archive = conversion_artifacts.get("hubai_archive")
    assert hubai_archive is not None, "HubAI archive artifact was not recorded"
    assert model._exported_models.get("hubai_archive") == hubai_archive

    # HubAI conversion should create a platform-specific archive in addition
    # to the base ONNX archive. The platform archive has platform identifier
    # in its name
    all_archives = list(save_dir.glob("*.tar.xz"))
    platform_identifier = platform.upper()
    platform_archives = [
        p
        for p in all_archives
        if platform_identifier in p.name.upper() and p != archive_path
    ]
    assert platform_archives == [hubai_archive], (
        f"No platform-specific archive containing '{platform_identifier}' "
        f"found for {platform}. Archives found: {[p.name for p in all_archives]}"
    )


def test_convert_saves_to_default_directory(
    coco_dataset: LuxonisDataset, opts: Params
):
    """Test that convert uses default save directory when not
    specified.
    """
    config_file = "configs/detection_light_model.yaml"
    opts |= {
        "loader.params.dataset_name": coco_dataset.identifier,
        "model.name": "test_convert_default_dir",
        "exporter.blobconverter.active": False,
        "exporter.hubai.active": False,
    }
    model = LuxonisModel(config_file, opts)

    archive_path, _ = model.convert()

    assert archive_path.exists(), "Archive was not created"
