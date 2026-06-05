from typing import Any

import pytest
from luxonis_ml.typing import Params
from torch.optim import SGD

from luxonis_train.config.config import ParameterPattern

from ._helpers import (
    assert_no_duplicate_parameters,
    build_snapshot,
    config,
    find_group,
    matching_names,
    tiny_head_node,
)


@pytest.mark.parametrize(
    ("parameters", "expected_parts"),
    [
        (None, ("Head.",)),
        ("fc", ("Head.Linear.fc",)),
        ([{"name": "branch[12]\\.0"}], ("branch1.0", "branch2.0")),
        ([{"module_type": "Linear"}], ("Head.Linear.fc",)),
        ([{"name": "fc", "module_type": "Linear"}], ("Head.Linear.fc",)),
        ([{"name": "branch1\\.0"}, {"name": "fc"}], ("branch1.0", "fc")),
        ("BRANCH1", ("branch1",)),
        ([{"module_type": "linear"}], ("Head.Linear.fc",)),
    ],
)
def test_valid_parameter_selectors(
    parameters: Any, expected_parts: tuple[str, ...], opts: Params
):
    snapshot = build_snapshot(
        config(
            [
                tiny_head_node(
                    {
                        "parameters": parameters,
                        "optimizer": {
                            "name": "SGD",
                            "params": {"lr": 0.123},
                        },
                    }
                )
            ]
        ),
        opts,
    )

    expected_names = set().union(
        *(matching_names(snapshot, part) for part in expected_parts)
    )
    _, optimizer, group = find_group(snapshot, expected_names)
    assert isinstance(optimizer, SGD)
    assert group["lr"] == pytest.approx(0.123)
    assert_no_duplicate_parameters(snapshot)


@pytest.mark.parametrize(
    ("finetuning", "expected_error", "match"),
    [
        ({"parameters": []}, ValueError, "at least one parameter pattern"),
        ({"parameters": ""}, ValueError, "cannot be empty"),
        ({"parameters": [{"name": ""}]}, ValueError, "cannot be empty"),
        (
            {"parameters": [{"module_type": ""}]},
            ValueError,
            "cannot be empty",
        ),
        ({"parameters": [1]}, TypeError, "Parameter patterns must be"),
        ({"parameters": [object()]}, TypeError, "Parameter patterns must be"),
        ({"parameters": [{"name": "missing"}]}, ValueError, "did not match any"),
    ],
)
def test_invalid_parameter_selectors(
    finetuning: Params,
    expected_error: type[Exception],
    match: str,
    opts: Params,
):
    with pytest.raises(expected_error, match=match):
        build_snapshot(config([tiny_head_node(finetuning)]), opts)


@pytest.mark.parametrize(
    ("pattern", "module_type", "parameter_name", "expected"),
    [
        (ParameterPattern(name="fc"), "Linear", "fc.weight", True),
        (ParameterPattern(name="fc"), "Linear", "branch1.0.weight", False),
        (ParameterPattern(name="FC"), "Linear", "fc.weight", True),
        (
            ParameterPattern(name="branch[12]\\.0"),
            "Conv2d",
            "branch2.0.bias",
            True,
        ),
        (ParameterPattern(module_type="Linear"), "Linear", "fc.bias", True),
        (ParameterPattern(module_type="Conv2d"), "Linear", "fc.bias", False),
        (
            ParameterPattern(name="fc", module_type="Linear"),
            "Linear",
            "fc.weight",
            True,
        ),
        (
            ParameterPattern(name="fc", module_type="Conv2d"),
            "Linear",
            "fc.weight",
            False,
        ),
        (
            ParameterPattern(name="branch", module_type="Linear"),
            "Conv2d",
            "branch1.0.weight",
            False,
        ),
    ],
)
def test_parameter_pattern_matches(
    pattern: ParameterPattern,
    module_type: str,
    parameter_name: str,
    expected: bool,
):
    assert pattern.matches(module_type, parameter_name) is expected
