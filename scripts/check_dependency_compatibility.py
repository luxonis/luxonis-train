"""Check the installed training and ONNX stack without a GPU or
dataset.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
from luxonis_ml.utils import environ

from luxonis_train.core import LuxonisModel


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(0)
    with TemporaryDirectory() as directory:
        path = Path(directory)
        environ.LUXONISML_BASE_PATH = path / "data"
        model = LuxonisModel(
            {
                "model": {
                    "name": "dependency-smoke",
                    "predefined_model": {
                        "name": "ClassificationModel",
                        "params": {"backbone_params": {"weights": "none"}},
                        "include_metrics": False,
                        "include_visualizers": False,
                    },
                },
                "loader": {
                    "params": {
                        "dataset_name": "dependency-smoke",
                        "n_classes": 3,
                    }
                },
                "trainer": {
                    "accelerator": "cpu",
                    "batch_size": 2,
                    "n_workers": 0,
                    "preprocessing": {"train_image_size": [32, 64]},
                },
                "tracker": {"save_directory": str(path / "runs")},
                "exporter": {"upload_to_run": False},
            },
            allow_empty_dataset=True,
        )
        inputs = torch.randn(1, 3, 32, 64)
        module = model.lightning_module.eval().set_export_mode(mode=True)
        with torch.no_grad():
            expected = [output.numpy() for output in module(inputs)]

        exported = model.export(path / "export", ignore_missing_weights=True)
        onnx.checker.check_model(exported)
        simplified, valid = onnxsim.simplify(onnx.load(exported))
        assert valid, "ONNX simplification failed"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(
            simplified.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        actual = session.run(
            None, {session.get_inputs()[0].name: inputs.numpy()}
        )
        for actual_output, expected_output in zip(
            actual, expected, strict=True
        ):
            assert isinstance(actual_output, np.ndarray)
            np.testing.assert_allclose(
                actual_output, expected_output, rtol=1e-4, atol=1e-5
            )
    print(
        "Model construction, ONNX export, simplification and inference passed."
    )


if __name__ == "__main__":
    main()
