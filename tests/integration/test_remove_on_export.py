from pathlib import Path

import onnxruntime as rt
import pytest
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel

ONNX_PATH = Path("tests/integration/ddrnet_segmentation.onnx")


@pytest.fixture(autouse=True)
def clear_files():
    yield
    ONNX_PATH.unlink(missing_ok=True)


def test_train_only_heads(coco_dataset: LuxonisDataset):
    config_file = "tests/configs/ddrnet.yaml"

    opts: Params = {"loader.params.dataset_name": coco_dataset.dataset_name}

    model = LuxonisModel(config_file, opts)
    results = model.test()

    name_to_check = "aux_segmentation_head"
    is_in_results = any(name_to_check in key for key in results)

    model.export(str(ONNX_PATH.parent))

    sess = rt.InferenceSession(str(ONNX_PATH))
    onnx_output_names = [output.name for output in sess.get_outputs()]
    is_in_output_names = any(
        name_to_check in name for name in onnx_output_names
    )

    assert is_in_results, (
        "'aux_segmentation_head' should be in the test results"
    )
    assert not is_in_output_names, (
        "'aux_segmentation_head' should not be in the ONNX output names"
    )
