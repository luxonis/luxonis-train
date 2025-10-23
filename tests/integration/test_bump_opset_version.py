import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params
from pytest_subtests import SubTests

from luxonis_train.core import LuxonisModel
from tests.integration.test_combinations import BACKBONES, get_config

BACKBONES = [
    backbone
    for backbone in BACKBONES
    if backbone not in {"PPLCNetV3", "GhostFaceNet", "RecSubNet"}
]


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("target_opset", [16])
def test_opset_bump_equivalence(
    backbone: str,
    target_opset: int,
    opts: Params,
    subtests: SubTests,
    parking_lot_dataset: LuxonisDataset,
):
    """Tests whether or not bumping an opset version breaks model
    conversions.

    Performed tests:
        - Successful export of the model in previous opset version and newer opset version
        - Converted models can be executed and output formats are the same
        - Outputs are the same between both models on the same input
    """
    config = get_config(backbone)
    tmpdir = Path(tempfile.mkdtemp())
    opts |= {"loader.params.dataset_name": parking_lot_dataset.identifier}

    def _export_model(opset_version: int) -> Path:
        cfg = config.copy()
        cfg["exporter"] = {"onnx": {"opset_version": opset_version}}
        model = LuxonisModel(cfg, opts)

        with subtests.test(f"export_opset_{opset_version}"):
            model.export(save_path=tmpdir)
            onnx_path = next(tmpdir.glob("*.onnx"))
            assert onnx_path.exists(), (
                f"Export failed for opset {opset_version}"
            )
        return onnx_path

    path_opset_12 = _export_model(12)
    path_opset_newer_version = _export_model(target_opset)

    sess12 = ort.InferenceSession(
        str(path_opset_12), providers=["CPUExecutionProvider"]
    )
    sess16 = ort.InferenceSession(
        str(path_opset_newer_version), providers=["CPUExecutionProvider"]
    )

    inputs12 = sess12.get_inputs()
    assert len(inputs12) == 1, "Expected a single model input"
    input_name = inputs12[0].name
    input_shape = [d if isinstance(d, int) else 1 for d in inputs12[0].shape]
    dtype = np.float32

    # Random input with seed
    rng = np.random.default_rng(seed=3)
    random_input = rng.standard_normal(input_shape, dtype=dtype)

    with subtests.test("run_inference"):
        outputs12 = sess12.run(None, {input_name: random_input})
        outputs_newer = sess16.run(None, {input_name: random_input})
        assert len(outputs12) == len(outputs_newer), "Output count mismatch"

    with subtests.test("compare_outputs"):
        for i, (out12, out_newer) in enumerate(
            zip(outputs12, outputs_newer, strict=True)
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

            array_output_12 = to_array(out12)
            array_output_newer = to_array(out_newer)

            np.testing.assert_allclose(
                array_output_12,
                array_output_newer,
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Output {i} differs between opset 12 and 16",
            )
