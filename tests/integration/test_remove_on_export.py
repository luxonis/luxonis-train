import onnxruntime as rt
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel


def test_train_only_heads(coco_dataset: LuxonisDataset, opts: Params):
    opts |= {"loader.params.dataset_name": coco_dataset.dataset_name}

    model = LuxonisModel(get_config(), opts)
    results = model.test()

    name_to_check = "aux_segmentation_head"
    is_in_results = any(name_to_check in key for key in results)

    model.export()

    sess = rt.InferenceSession(
        model.run_save_dir / "export" / f"{model.cfg.model.name}.onnx"
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


def get_config() -> Params:
    return {
        "model": {
            "name": "ddrnet_segmentation",
            "nodes": [
                {"name": "DDRNet"},
                {
                    "name": "DDRNetSegmentationHead",
                    "alias": "segmentation_head",
                    "params": {"attach_index": -1},
                    "losses": [{"name": "CrossEntropyLoss"}],
                    "metrics": [{"name": "JaccardIndex"}],
                },
                {
                    "name": "DDRNetSegmentationHead",
                    "alias": "aux_segmentation_head",
                    "params": {"attach_index": -2},
                    "remove_on_export": True,
                    "losses": [{"name": "CrossEntropyLoss"}],
                },
            ],
        }
    }
