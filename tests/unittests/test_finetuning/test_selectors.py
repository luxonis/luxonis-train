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
        ({"module_type": "Linear"}, ("Head.Linear.fc",)),
        (ParameterPattern(name="fc"), ("Head.Linear.fc",)),
    ],
)
def test_valid_parameter_selectors(
    parameters: Any, expected_parts: tuple[str, ...], opts: Params
):
    r"""Every supported way of writing a ``parameters`` selector should
    normalize into a matching set that the finetuning builder can use.
    Each case configures one Head rule with the given selector, an SGD
    optimizer, and ``lr=0.123`` — the assertion is that exactly the
    expected params end up in that SGD group.

    Cases (in the order listed above):
        1. ``None`` — omitted selector means "everything under this
           node"; the group covers all Head params.
        2. Bare string ``'fc'`` — coerced to
           ``ParameterPattern(name='fc')``; matches the Linear.fc
           weight/bias.
        3. Regex in the name (``'branch[12]\\.0'``) — matches both
           ``branch1.0`` and ``branch2.0`` Conv2d layers.
        4. ``module_type='Linear'`` — selects by module class name only.
        5. Combined ``name`` + ``module_type`` (both AND'd) — must
           match both.
        6. List with two patterns — the union of both selections.
        7. Case-insensitive name matching (``'BRANCH1'`` finds
           ``branch1``).
        8. Case-insensitive module type matching (``'linear'`` finds
           ``Linear``).
        9. Bare dict (not wrapped in a list) — the config validator
           wraps single dicts into a one-element list.
        10. Pre-built ``ParameterPattern`` instance — passes through
            unchanged.
    """
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
        (
            {"parameters": [{"name": "missing"}]},
            ValueError,
            "did not match any",
        ),
    ],
)
def test_invalid_parameter_selectors(
    finetuning: Params,
    expected_error: type[Exception],
    match: str,
    opts: Params,
):
    """Malformed selectors should surface at config-validation time (or
    at optimizer build time, for the "no matches" case), not silently
    produce empty groups.

    Cases (in the order listed above):
        1. Empty list — no patterns supplied.
        2. Empty string — coerces to an empty-name pattern which is
           rejected by ``ParameterPattern`` validation.
        3. Empty ``name`` field — same validator, dict form.
        4. Empty ``module_type`` field — same validator, module-type
           form.
        5. Non-string / non-dict item (int) — the list validator
           rejects it as an unsupported type.
        6. Arbitrary object — same rejection path as case 5.
        7. Pattern that compiles fine but matches nothing (``name
           ='missing'``) — build_optimizers raises so the user knows
           the rule is a no-op rather than silently skipping it.
    """
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
    """Unit test for ``ParameterPattern.matches`` in isolation from the
    finetuning pipeline.

    Semantics: ``name`` and ``module_type`` are independent
    case-insensitive regex searches; when both are set both must
    match. Missing fields are ignored (contribute no constraint).

    Cases (in the order listed above):
        1. Name-only match on ``fc``.
        2. Name-only miss — pattern ``fc`` doesn't appear in
           ``branch1.0.weight``.
        3. Case-insensitive name (``FC`` → ``fc``).
        4. Regex alternation on the name across module types.
        5. Module-type-only match.
        6. Module-type-only miss.
        7. Name + module_type both match → true.
        8. Name matches but module_type doesn't → false (AND).
        9. Name doesn't match but module_type does → false (AND).
    """
    assert pattern.matches(module_type, parameter_name) is expected
