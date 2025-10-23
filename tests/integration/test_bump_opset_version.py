import pytest
import tempfile
import numpy as np
import onnxruntime as ort
from pathlib import Path

from luxonis_train.nodes.backbones import __all__ as BACKBONES
from luxonis_train.core import LuxonisModel
from tests.integration.test_combinations import get_config
from luxonis_ml.typing import Params, ParamValue

BACKBONES = [
    backbone
    for backbone in BACKBONES
    if backbone not in {"PPLCNetV3", "GhostFaceNet", "RecSubNet"}
]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_opset_bump_equivalence(backbone: str, opts: Params, subtests, parking_lot_dataset):
    config = get_config(backbone)
    tmpdir = Path(tempfile.mkdtemp())
    opts |= {"loader.params.dataset_name": parking_lot_dataset.identifier}

    def _export_model(opset_version: int):
        cfg = config.copy()
        cfg["exporter"] = {"onnx": {"opset_version": opset_version}}
        model = LuxonisModel(cfg, opts)

        with subtests.test(f"export_opset_{opset_version}"):
            model.export(save_path=tmpdir)
            onnx_path = next(tmpdir.glob("*.onnx"))
            assert onnx_path.exists(), f"Export failed for opset {opset_version}"
        return onnx_path

    # this will fail if the export fails
    path_opset_12 = _export_model(12)
    path_opset_16 = _export_model(16)

    sess12 = ort.InferenceSession(str(path_opset_12), providers=["CPUExecutionProvider"])
    sess16 = ort.InferenceSession(str(path_opset_16), providers=["CPUExecutionProvider"])

    inputs12 = sess12.get_inputs()
    assert len(inputs12) == 1, "Expected a single model input"
    input_name = inputs12[0].name
    input_shape = [d if isinstance(d, int) else 1 for d in inputs12[0].shape]
    dtype = np.float32

    # Random input, here 0 is the seed to make the test reproducible
    rng = np.random.default_rng(0)
    random_input = rng.standard_normal(input_shape, dtype=dtype)

    # will pass if both models can be executed and if both outputs have the same structure
    with subtests.test("run_inference"):
        outputs12 = sess12.run(None, {input_name: random_input})
        outputs16 = sess16.run(None, {input_name: random_input})
        assert len(outputs12) == len(outputs16), "Output count mismatch"

    # this will fail if the outputs between opset 12 and 16 are not the same
    with subtests.test("compare_outputs"):
        for i, (out12, out16) in enumerate(zip(outputs12, outputs16)):
            np.testing.assert_allclose(
                out12,
                out16,
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Output {i} differs between opset 12 and 16",
            )
