import numpy as np
import onnx
import onnxruntime as ort
import pytest
from luxonis_ml.typing import Params

from luxonis_train.core import LuxonisModel
from tests.conftest import LuxonisTestDatasets
from tests.integration.backbone_model_utils import (
    PREDEFINED_MODELS,
    prepare_predefined_model_config,
)


@pytest.mark.parametrize(("config_name", "extra_opts"), PREDEFINED_MODELS)
def test_unique_initializers_creates_unique_names(
    config_name: str,
    extra_opts: Params | None,
    opts: Params,
    test_datasets: LuxonisTestDatasets,
):
    """Test that unique_onnx_initializers flag actually makes all
    initializer names unique."""
    config_file, opts, _ = prepare_predefined_model_config(
        config_name, opts, test_datasets
    )
    extra_opts = extra_opts or {}

    model = LuxonisModel(config_file, opts | extra_opts)

    model.export(unique_onnx_initializers=True)

    onnx_path = model.run_save_dir / "export" / f"{model.cfg.model.name}.onnx"
    onnx_model = onnx.load(str(onnx_path))

    initializer_names = [init.name for init in onnx_model.graph.initializer]

    assert len(initializer_names) == len(set(initializer_names)), (
        f"Initializer names are not unique for {config_name}. "
        f"Found {len(initializer_names)} initializers but only "
        f"{len(set(initializer_names))} unique names."
    )

    for name in initializer_names:
        assert "_unique_" in name, (
            f"Initializer '{name}' in {config_name} does not follow the expected "
            "naming convention (should contain '_unique_')"
        )


@pytest.mark.parametrize(("config_name", "extra_opts"), PREDEFINED_MODELS)
def test_unique_initializers_model_validity(
    config_name: str,
    extra_opts: Params | None,
    opts: Params,
    test_datasets: LuxonisTestDatasets,
):
    """Test that the model with unique initializers passes ONNX
    checker."""
    config_file, opts, _ = prepare_predefined_model_config(
        config_name, opts, test_datasets
    )
    extra_opts = extra_opts or {}

    model = LuxonisModel(config_file, opts | extra_opts)

    model.export(unique_onnx_initializers=True)

    onnx_path = model.run_save_dir / "export" / f"{model.cfg.model.name}.onnx"

    onnx.checker.check_model(str(onnx_path))


@pytest.mark.parametrize(("config_name", "extra_opts"), PREDEFINED_MODELS)
def test_unique_initializers_no_shared_weights(
    config_name: str,
    extra_opts: Params | None,
    opts: Params,
    test_datasets: LuxonisTestDatasets,
):
    """Test that no initializer is referenced by multiple nodes after
    transformation.

    Each initializer should be used at most once
    """
    config_file, opts, _ = prepare_predefined_model_config(
        config_name, opts, test_datasets
    )
    extra_opts = extra_opts or {}

    model = LuxonisModel(config_file, opts | extra_opts)

    model.export(unique_onnx_initializers=True)

    onnx_path = model.run_save_dir / "export" / f"{model.cfg.model.name}.onnx"
    onnx_model = onnx.load(str(onnx_path))

    initializer_names = {init.name for init in onnx_model.graph.initializer}

    usage_count: dict[str, int] = dict.fromkeys(initializer_names, 0)

    for node in onnx_model.graph.node:
        for input_name in node.input:
            if input_name in initializer_names:
                usage_count[input_name] += 1

    for name, count in usage_count.items():
        assert count <= 1, (
            f"Initializer '{name}' in {config_name} is used by {count} nodes, "
            "but should be used by at most 1 after unique_onnx_initializers"
        )


@pytest.mark.parametrize(("config_name", "extra_opts"), PREDEFINED_MODELS)
def test_unique_initializers_numerical_equivalence(
    config_name: str,
    extra_opts: Params | None,
    opts: Params,
    test_datasets: LuxonisTestDatasets,
):
    """Run inference on both models (exported with flag True and False)
    and test that the outputs are the same."""
    config_file, opts, _ = prepare_predefined_model_config(
        config_name, opts, test_datasets
    )
    extra_opts = extra_opts or {}

    model = LuxonisModel(config_file, opts | extra_opts)

    model.export(unique_onnx_initializers=False)
    onnx_path_normal = (
        model.run_save_dir / "export" / f"{model.cfg.model.name}.onnx"
    )

    export_path_unique = model.run_save_dir / "export_unique"
    model.export(save_path=export_path_unique, unique_onnx_initializers=True)
    onnx_path_unique = export_path_unique / f"{model.cfg.model.name}.onnx"

    model_normal = onnx.load(str(onnx_path_normal))
    session_normal = ort.InferenceSession(str(onnx_path_normal))
    session_unique = ort.InferenceSession(str(onnx_path_unique))

    # Test inputs based on model input specs
    test_inputs = {}
    for input_info in model_normal.graph.input:
        initializer_names = {
            init.name for init in model_normal.graph.initializer
        }
        if input_info.name in initializer_names:
            continue

        shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(1)  # Replace dynamic dims with 1

        elem_type = input_info.type.tensor_type.elem_type
        if elem_type == onnx.TensorProto.FLOAT:
            dtype = np.float32
        elif elem_type == onnx.TensorProto.INT64:
            dtype = np.int64
        elif elem_type == onnx.TensorProto.INT32:
            dtype = np.int32
        else:
            dtype = np.float32

        test_inputs[input_info.name] = np.random.randn(*shape).astype(dtype)

    # Run inference on both models and compare outputs

    outputs_normal = session_normal.run(None, test_inputs)
    outputs_unique = session_unique.run(None, test_inputs)

    assert len(outputs_normal) == len(outputs_unique), (
        f"Different number of outputs for {config_name}: "
        f"{len(outputs_normal)} vs {len(outputs_unique)}"
    )

    for i, (out_normal, out_unique) in enumerate(
        zip(outputs_normal, outputs_unique, strict=True)
    ):
        out_normal_arr = np.asarray(out_normal)  # for PyRight errors
        out_unique_arr = np.asarray(out_unique)  # for PyRight errors
        assert np.allclose(
            out_normal_arr, out_unique_arr, rtol=1e-5, atol=1e-6
        ), (
            f"Output {i} differs for {config_name}. "
            f"Max absolute difference: {np.max(np.abs(out_normal_arr - out_unique_arr))}"
        )
