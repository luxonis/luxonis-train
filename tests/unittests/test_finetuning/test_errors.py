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
    """Optimizer and scheduler classes are looked up in the OPTIMIZERS /
    SCHEDULERS registries by name. Unknown names should raise
    ``KeyError`` at build time rather than silently falling back to a
    default.

    Cases:
        1. Rule references a non-existent optimizer name.
        2. Rule references a non-existent scheduler name.
    """
    with pytest.raises(KeyError):
        build_snapshot(config([tiny_head_node(finetuning)]), opts)


def test_invalid_optimizer_parameter_group_keys_raise(opts: Params):
    """Per-group options are validated against the target optimizer's
    known keys (``optimizer.defaults`` + ``params``).

    An unknown key surfaces as a ``TypeError`` naming the offending
    optimizer — this matters because torch would otherwise silently pass
    unknown keys through into internal state.
    """
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
    """Scheduler params are passed straight to the scheduler
    constructor, so an unknown keyword argument surfaces as a
    ``TypeError`` from Python itself — the finetuning code isn't
    swallowing or renaming it.
    """
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
    """Regression check on the *messages* produced by selector
    validation (as opposed to just the exception types covered in
    ``test_selectors.test_invalid_parameter_selectors``).

    The message wording is the only signal users get when a rule they
    wrote is malformed, so pinning them here catches accidental
    rewordings that would break tutorials and docs.
    """
    with pytest.raises(expected_error, match=match):
        build_snapshot(config([tiny_head_node(finetuning)]), opts)
