import onnx
import pytest
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel
from tests.conftest import LuxonisTestDatasets
from tests.integration.backbone_model_utils import (
    PREDEFINED_MODELS,
    prepare_predefined_model_config,
)


@pytest.mark.parametrize(("config_name", "extra_opts"), PREDEFINED_MODELS)
def test_unique_initializers_preserves_input_output_names(
    config_name: str,
    extra_opts: Params | None,
    opts: Params,
    test_datasets: LuxonisTestDatasets,
):
    """Test that unique_onnx_initializers flag does not change input/output names."""
    config_file, opts, _ = prepare_predefined_model_config(
        config_name, opts, test_datasets
    )
    extra_opts = extra_opts or {}

    model = LuxonisModel(config_file, opts | extra_opts)

    model.export(unique_onnx_initializers=False)
    onnx_path_normal = (
        model.run_save_dir / "export" / f"{model.cfg.model.name}.onnx"
    )
    model_normal = onnx.load(str(onnx_path_normal))

    normal_input_names = [inp.name for inp in model_normal.graph.input]
    normal_output_names = [out.name for out in model_normal.graph.output]

    export_path_unique = model.run_save_dir / "export_unique"
    model.export(
        save_path=export_path_unique,
        unique_onnx_initializers=True,
    )
    onnx_path_unique = export_path_unique / f"{model.cfg.model.name}.onnx"
    model_unique = onnx.load(str(onnx_path_unique))

    unique_input_names = [inp.name for inp in model_unique.graph.input]
    unique_output_names = [out.name for out in model_unique.graph.output]

    # Filter out initializer names from inputs (ONNX includes initializers in inputs)
    normal_initializer_names = {
        init.name for init in model_normal.graph.initializer
    }
    unique_initializer_names = {
        init.name for init in model_unique.graph.initializer
    }

    normal_actual_inputs = [
        name for name in normal_input_names if name not in normal_initializer_names
    ]
    unique_actual_inputs = [
        name for name in unique_input_names if name not in unique_initializer_names
    ]

    assert normal_actual_inputs == unique_actual_inputs, (
        f"Input names changed after applying unique_onnx_initializers. "
        f"Normal: {normal_actual_inputs}, Unique: {unique_actual_inputs}"
    )

    assert normal_output_names == unique_output_names, (
        f"Output names changed after applying unique_onnx_initializers. "
        f"Normal: {normal_output_names}, Unique: {unique_output_names}"
    )
