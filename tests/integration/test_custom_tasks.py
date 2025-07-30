import json
import tarfile
from pathlib import Path

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params
from pytest_subtests import SubTests

from luxonis_train.core import LuxonisModel


def test_custom_tasks(
    opts: Params,
    parking_lot_dataset: LuxonisDataset,
    subtests: SubTests,
    tempdir: Path,
):
    cfg = "tests/configs/parking_lot_config.yaml"
    opts |= {
        "loader.params.dataset_name": parking_lot_dataset.dataset_name,
    }
    with subtests.test("create"):
        model = LuxonisModel(cfg, opts)

    with subtests.test("train"):
        model.train()

    with subtests.test("export"):
        model.export(tempdir)
        assert (tempdir / "parking_lot_model.onnx").exists()

    with subtests.test("test_archive"):
        archive_path = Path(
            model.run_save_dir, "archive", model.cfg.model.name
        ).with_suffix(".onnx.tar.xz")
        correct_archive_config = json.loads(
            Path("tests/integration/parking_lot.json").read_text()
        )

        assert archive_path.exists()
        with tarfile.open(archive_path) as tar:
            extracted_cfg = tar.extractfile("config.json")

            assert extracted_cfg is not None, (
                "Config JSON not found in the archive."
            )
            generated_config = json.loads(extracted_cfg.read().decode())

        # Sort the fields in the config to make the comparison consistent
        def sort_by_name(config: dict, keys: list[str]) -> None:
            for key in keys:
                if key in config["model"]:
                    config["model"][key] = sorted(
                        config["model"][key], key=lambda x: x["name"]
                    )

        keys_to_sort = ["inputs", "outputs", "heads"]
        sort_by_name(generated_config, keys_to_sort)
        sort_by_name(correct_archive_config, keys_to_sort)
        assert generated_config == correct_archive_config
