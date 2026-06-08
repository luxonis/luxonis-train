import pytest
from luxonis_ml.typing import Params

from ._helpers import build_snapshot, config, tiny_head_node


@pytest.mark.parametrize(
    "finetuning",
    [
        {
            "parameters": [{"module_type": "Linear"}],
            "optimizer": {"name": "MissingOptimizer"},
        },
        {
            "parameters": [{"module_type": "Linear"}],
            "scheduler": {"name": "MissingScheduler"},
        },
    ],
)
def test_unknown_optimizer_or_scheduler_name_raises(
    finetuning: Params, opts: Params
):
    with pytest.raises(KeyError):
        build_snapshot(config([tiny_head_node(finetuning)]), opts)


def test_invalid_optimizer_parameter_group_keys_raise(opts: Params):
    with pytest.raises(
        TypeError,
        match="Invalid parameter group option\\(s\\) for optimizer 'Adam'",
    ):
        build_snapshot(
            config(
                [
                    tiny_head_node(
                        {
                            "parameters": [{"module_type": "Linear"}],
                            "optimizer": {"params": {"not_a_param": True}},
                        }
                    )
                ]
            ),
            opts,
        )


def test_invalid_scheduler_params_raise(opts: Params):
    with pytest.raises(TypeError):
        build_snapshot(
            config(
                [
                    tiny_head_node(
                        {
                            "parameters": [{"module_type": "Linear"}],
                            "scheduler": {"params": {"not_a_param": True}},
                        }
                    )
                ]
            ),
            opts,
        )


@pytest.mark.parametrize(
    ("finetuning", "expected_error", "match"),
    [
        ({"parameters": []}, ValueError, "at least one parameter pattern"),
        ({"parameters": [{"name": ""}]}, ValueError, "cannot be empty"),
        ({"parameters": [{"module_type": ""}]}, ValueError, "cannot be empty"),
        ({"parameters": [1]}, TypeError, "Parameter patterns must be"),
        (
            {"parameters": [{"name": "missing"}]},
            ValueError,
            "did not match any available trainable parameters",
        ),
    ],
)
def test_selector_validation_error_messages(
    finetuning: Params,
    expected_error: type[Exception],
    match: str,
    opts: Params,
):
    with pytest.raises(expected_error, match=match):
        build_snapshot(config([tiny_head_node(finetuning)]), opts)
