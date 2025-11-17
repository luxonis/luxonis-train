from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest
from luxonis_ml.typing import Params
from pytest_subtests import SubTests

from luxonis_train.core import LuxonisModel
from tests.conftest import LuxonisTestDatasets
from tests.integration.backbone_model_utils import (
    PREDEFINED_MODELS,
    prepare_predefined_model_config,
)
from tests.integration.test_combinations import BACKBONES, get_config


# for pyright safety
def get_opset_version(cfg: dict[str, Any]) -> int | None:
    exporter = cfg.get("exporter")
    if not isinstance(exporter, dict):
        return None

    onnx = exporter.get("onnx")
    if not isinstance(onnx, dict):
        return None

    opset = onnx.get("opset_version")
    return opset if isinstance(opset, int) else None


@pytest.mark.parametrize(
    ("config_name", "extra_opts"),
    [(b, {}) for b in BACKBONES] + PREDEFINED_MODELS,
)
@pytest.mark.parametrize("target_opset", [16])
def test_opset_bump_equivalence(
    config_name: str,
    extra_opts: Params | None,
    target_opset: int,
    opts: Params,
    subtests: SubTests,
    test_datasets: LuxonisTestDatasets,
    tmp_path: Path,
    dinov3_weights: Path,
    current_opset: int,
):
    """Tests whether or not bumping an opset version breaks model
    conversions.

    Performed tests:
        - Successful export of the model in previous opset version and newer opset version
        - Converted models can be executed and output formats are the same
        - Outputs are the same between both models on the same input
    """
    if target_opset == current_opset:
        pytest.skip(
            "Opset version is not being upgraded, skipping test for bumping opset version"
        )
    if config_name in BACKBONES:
        config = get_config(config_name, dinov3_weights)
        opts |= {
            "loader.params.dataset_name": test_datasets.parking_lot_dataset.identifier
        }
        opset = get_opset_version(config)
        if opset is not None and opset > current_opset:
            pytest.skip(
                f"Skipping for backbone {config_name} because the opset version is already set to higher than current opset"
            )
    else:
        config, opts, _ = prepare_predefined_model_config(
            config_name, opts, test_datasets
        )
        tmp_path = tmp_path / config_name
        tmp_path.mkdir()

    if extra_opts is None:
        extra_opts = {}

    def _export_model(opset_version: int) -> Path:
        merged_opts = opts | (extra_opts or {})
        merged_opts |= {"exporter": {"onnx": {"opset_version": opset_version}}}
        model = LuxonisModel(config, merged_opts)

        with subtests.test(f"export_opset_{opset_version}"):
            model.export(save_path=tmp_path)
            onnx_path = next(tmp_path.glob("*.onnx"))
            assert onnx_path.exists(), (
                f"Export failed for opset {opset_version}"
            )
        return onnx_path

    path_opset_current = _export_model(current_opset)
    path_opset_newer_version = _export_model(target_opset)

    sess_current_opset = ort.InferenceSession(
        str(path_opset_current), providers=["CPUExecutionProvider"]
    )
    sess_newer_opset = ort.InferenceSession(
        str(path_opset_newer_version), providers=["CPUExecutionProvider"]
    )

    inputs_current = sess_current_opset.get_inputs()
    assert len(inputs_current) == 1, "Expected a single model input"
    input_name = inputs_current[0].name
    input_shape = [
        d if isinstance(d, int) else 1 for d in inputs_current[0].shape
    ]
    dtype = np.float32

    # Random input with seed
    rng = np.random.default_rng(seed=3)
    random_input = rng.standard_normal(input_shape, dtype=dtype)

    with subtests.test("run_inference"):
        outputs_current = sess_current_opset.run(
            None, {input_name: random_input}
        )
        outputs_newer = sess_newer_opset.run(None, {input_name: random_input})
        assert len(outputs_current) == len(outputs_newer), (
            "Output count mismatch"
        )

    with subtests.test("compare_outputs"):
        for i, (out_current, out_newer) in enumerate(
            zip(outputs_current, outputs_newer, strict=True)
        ):
            # Convert any list or dict outputs to numpy arrays (pyright)
            def to_array(x: Any) -> np.ndarray:
                if hasattr(x, "to_dense"):
                    x = x.to_dense()
                if isinstance(x, dict):
                    x = np.concatenate(
                        [v for _, v in sorted(x.items())], axis=None
                    )
                elif isinstance(x, list):
                    x = np.concatenate([np.ravel(v) for v in x])
                return np.asarray(x)

            array_output_current = to_array(out_current)
            array_output_newer = to_array(out_newer)

            np.testing.assert_allclose(
                array_output_current,
                array_output_newer,
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Output {i} differs between opset {current_opset} and {target_opset}",
            )
