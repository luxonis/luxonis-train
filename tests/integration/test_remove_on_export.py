import onnxruntime as rt
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel


def test_train_only_heads(coco_dataset: LuxonisDataset, opts: Params):
    config_file = "configs/segmentation_light_model.yaml"

    opts |= {"loader.params.dataset_name": coco_dataset.dataset_name}

    model = LuxonisModel(config_file, opts)
    results = model.test()

    name_to_check = "aux_segmentation_head"
    is_in_results = any(name_to_check in key for key in results)

    model.export()

    sess = rt.InferenceSession(
        model.run_save_dir / "export" / "segmentation_light.onnx"
    )
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
