from pathlib import Path

import cv2
import numpy as np
import pytest
from luxonis_ml.data import LuxonisLoader
from luxonis_ml.typing import Params
from pytest_subtests import SubTests

from luxonis_train.core import LuxonisModel
from tests.conftest import LuxonisTestDatasets
from tests.integration.backbone_model_utils import (
    PREDEFINED_MODELS,
    prepare_predefined_model_config,
)


def test_model_construction():
    cfg = "configs/detection_light_model.yaml"
    model = LuxonisModel(
        cfg,
        {
            "model.predefined_model.include_losses": False,
            "model.predefined_model.include_metrics": False,
            "model.predefined_model.include_visualizers": False,
        },
        debug_mode=True,
    )
    for node in model.lightning_module.nodes.values():
        assert not node.losses
        assert not node.metrics
        assert not node.visualizers


@pytest.mark.parametrize(("config_name", "extra_opts"), PREDEFINED_MODELS)
def test_predefined_models(
    config_name: str,
    extra_opts: Params | None,
    opts: Params,
    test_datasets: LuxonisTestDatasets,
    tmp_path: Path,
    subtests: SubTests,
):
    config_file, opts, dataset = prepare_predefined_model_config(
        config_name, opts, test_datasets
    )
    extra_opts = extra_opts or {}
    tmp_path = tmp_path / config_name
    tmp_path.mkdir()

    model = LuxonisModel(config_file, opts | extra_opts)

    with subtests.test("train"):
        model.train()
        assert model.run_save_dir.exists()
        assert list(model.run_save_dir.iterdir())

    with subtests.test("export"):
        model.export()
        assert (model.run_save_dir / "export" / f"{config_name}.onnx").exists()

    with subtests.test("archive"):
        model.archive()
        assert (
            model.run_save_dir / "archive" / f"{config_name}.onnx.tar.xz"
        ).exists()

    if config_name != "embeddings_model":
        with subtests.test("infer"):
            loader = LuxonisLoader(dataset)
            img_dir = tmp_path / "images"
            video_path = tmp_path / "video.avi"
            video_writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"XVID"), 1, (256, 256)
            )
            img_dir.mkdir()
            for i, (img, _) in enumerate(loader):
                assert isinstance(img, np.ndarray)
                img = cv2.resize(img, (256, 256))
                cv2.imwrite(str(img_dir / f"{i}.png"), img)
                video_writer.write(img)
            video_writer.release()

            for subtest in ["single_image", "image_dir", "video", "loader"]:
                with subtests.test(f"infer/{subtest}"):
                    save_dir = tmp_path / f"infer_{subtest}"
                    if subtest == "single_image":
                        source = img_dir / "0.png"
                    elif subtest == "image_dir":
                        source = img_dir
                    elif subtest == "video":
                        source = video_path
                    else:
                        source = None

                    model.infer(source_path=source, save_dir=save_dir)

                    if subtest == "single_image":
                        assert len(list(save_dir.rglob("*.png"))) == 1
                    elif subtest == "image_dir":
                        assert len(list(save_dir.iterdir())) == len(loader)
                    elif subtest == "video":
                        assert len(list(save_dir.rglob("*.mp4"))) == 1
                    if subtest is None:
                        assert len(list(save_dir.iterdir())) == len(loader)

    # TODO: Support annotation for all models
    if (
        config_name
        not in {
            "embeddings_model",
            "anomaly_detection_model",
            "fomo_light_model",
        }
        or "heavy" in config_name
    ):
        with subtests.test("annotate"):
            model.annotate(
                dir_path=dataset.source_path,
                dataset_name="test_annotated_dataset",
                bucket_storage="local",
                delete_local=True,
            )

    with subtests.test("test-reload"):
        model_reload = LuxonisModel(
            str(model.run_save_dir / "training_config.yaml"),
            opts | {"tracker.run_name": f"{config_name}_reload"},
        )
        model_reload.test()
