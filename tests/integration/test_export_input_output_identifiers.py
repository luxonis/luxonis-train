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
    """Test that unique_onnx_initializers flag does not change
    input/output names."""
    config_file, opts, _ = prepare_predefined_model_config(
        config_name, opts, test_datasets
    )
    extra_opts = extra_opts or {}

    # Export without unique initializers
    model_normal = LuxonisModel(
        config_file,
        opts | extra_opts | {"exporter.onnx.unique_onnx_initializers": False},
    )
    model_normal.export()
    onnx_path_normal = (
        model_normal.run_save_dir
        / "export"
        / f"{model_normal.cfg.model.name}.onnx"
    )
    onnx_model_normal = onnx.load(str(onnx_path_normal))

    normal_input_names = [inp.name for inp in onnx_model_normal.graph.input]
    normal_output_names = [out.name for out in onnx_model_normal.graph.output]

    # Export with unique initializers
    model_unique = LuxonisModel(
        config_file,
        opts | extra_opts | {"exporter.onnx.unique_onnx_initializers": True},
    )
    model_unique.export()
    onnx_path_unique = (
        model_unique.run_save_dir
        / "export"
        / f"{model_unique.cfg.model.name}.onnx"
    )
    onnx_model_unique = onnx.load(str(onnx_path_unique))

    unique_input_names = [inp.name for inp in onnx_model_unique.graph.input]
    unique_output_names = [out.name for out in onnx_model_unique.graph.output]

    # Filter out initializer names from inputs (ONNX includes initializers in inputs)
    normal_initializer_names = {
        init.name for init in onnx_model_normal.graph.initializer
    }
    unique_initializer_names = {
        init.name for init in onnx_model_unique.graph.initializer
    }

    normal_actual_inputs = [
        name
        for name in normal_input_names
        if name not in normal_initializer_names
    ]
    unique_actual_inputs = [
        name
        for name in unique_input_names
        if name not in unique_initializer_names
    ]

    assert normal_actual_inputs == unique_actual_inputs, (
        f"Input names changed after applying unique_onnx_initializers. "
        f"Normal: {normal_actual_inputs}, Unique: {unique_actual_inputs}"
    )

    assert normal_output_names == unique_output_names, (
        f"Output names changed after applying unique_onnx_initializers. "
        f"Normal: {normal_output_names}, Unique: {unique_output_names}"
    )
