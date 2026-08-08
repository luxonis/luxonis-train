import builtins
import inspect
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from docutils import nodes
from docutils.core import publish_doctree
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Size, nn

from luxonis_train._pydoctor import LuxonisTrainSystem
from luxonis_train._shape_contract import (
    ShapeContractError,
    register_shape_contract_directive,
)
from luxonis_train._shape_contract import (
    parse_shape_contract as parse_contract_source,
)
from luxonis_train.nodes.backbones.dinov3.rope_position_encoding import (
    RopePositionEmbedding,
)
from luxonis_train.nodes.blocks import (
    ConvBlock,
    DropPath,
    EfficientDecoupledBlock,
    SqueezeExciteBlock,
)
from luxonis_train.nodes.blocks.blocks import PreciseDecoupledBlock
from luxonis_train.nodes.heads import (
    DDRNetSegmentationHead,
    EfficientBBoxHead,
    EfficientKeypointBBoxHead,
    PrecisionBBoxHead,
    PrecisionSegmentBBoxHead,
)
from luxonis_train.nodes.heads.base_detection_head import BaseDetectionHead
from luxonis_train.nodes.necks.reppan_neck import RepPANNeck
from luxonis_train.nodes.necks.svtr_neck.blocks import (
    Attention,
    ConvMixer,
    SVTRBlock,
)

from .contract_cases import (
    CONTRACT_CASES,
    UNSUPPORTED,
    CaseFactory,
    ContractCase,
)
from .shape_contracts import (
    Bindings,
    ShapeContract,
    ShapeSpec,
    assert_contract,
    assert_matches_contract,
    discover_module_types,
    documented_outputs,
    get_forward_owner,
    parse_forward_arguments,
    parse_forward_returns,
    parse_shape_contract,
)

MODULE_TYPES = discover_module_types()
FORWARD_MODULE_TYPES = [
    module_type
    for module_type in MODULE_TYPES
    if get_forward_owner(module_type) is not None
]


def _build_cases(factory: CaseFactory) -> list[ContractCase]:
    built = factory()
    if isinstance(built, ContractCase):
        return [built]
    return list(built)


def _as_output_mapping(result: Any, documented: dict[str, ShapeSpec]) -> Any:
    """Name what a forward returned after its documented outputs."""
    if isinstance(result, Mapping):
        return result
    if len(documented) == 1:
        return {next(iter(documented)): result}
    if isinstance(result, tuple) and len(result) == len(documented):
        return dict(zip(documented, result, strict=True))
    return result


@pytest.mark.parametrize(
    ("module_type", "factory"),
    CONTRACT_CASES.items(),
    ids=[module_type.__qualname__ for module_type in CONTRACT_CASES],
)
def test_documented_shapes_hold_at_runtime(
    module_type: type[nn.Module], factory: CaseFactory
):
    cases = _build_cases(factory)
    assert cases, f"{module_type.__qualname__} contributed no contract case"
    contract = parse_shape_contract(module_type)
    for index, case in enumerate(cases):
        documented = documented_outputs(contract, case.mode)
        outputs = case.outputs
        if outputs is None:
            with torch.no_grad():
                outputs = case.module(**case.inputs)
        try:
            assert_contract(
                module_type,
                case.inputs,
                _as_output_mapping(outputs, documented),
                case.bindings,
                mode=case.mode,
            )
        except AssertionError as error:
            raise AssertionError(
                f"Case {index} of {module_type.__qualname__} with mode "
                f"{case.mode} does not match its documented contract: "
                f"{error}"
            ) from error


def test_every_forward_has_a_runtime_contract_case():
    missing = sorted(
        f"{module_type.__module__}.{module_type.__qualname__}"
        for module_type in FORWARD_MODULE_TYPES
        if module_type not in CONTRACT_CASES and module_type not in UNSUPPORTED
    )
    assert not missing, (
        "The documented shape contracts of "
        + ", ".join(missing)
        + " are never executed. A new node needs a runtime contract case "
        "in tests/unittests/nodes/contract_cases, added to the CASES "
        "mapping of the group it belongs to, or an UNSUPPORTED entry "
        "explaining why its contract cannot be run."
    )


def test_unsupported_contracts_are_justified_and_disjoint():
    unjustified = sorted(
        module_type.__qualname__
        for module_type, reason in UNSUPPORTED.items()
        if not reason.strip()
    )
    assert not unjustified, (
        f"{unjustified} are marked unsupported without a reason"
    )
    claimed_twice = sorted(
        module_type.__qualname__
        for module_type in UNSUPPORTED
        if module_type in CONTRACT_CASES
    )
    assert not claimed_twice, (
        f"{claimed_twice} are marked unsupported while also having "
        "runtime contract cases"
    )


@pytest.mark.parametrize(
    "module_type",
    MODULE_TYPES,
    ids=lambda module_type: module_type.__qualname__,
)
def test_every_module_has_its_own_docstring(module_type: type[nn.Module]):
    assert module_type.__dict__.get("__doc__")


@pytest.mark.parametrize(
    "module_type",
    FORWARD_MODULE_TYPES,
    ids=lambda module_type: module_type.__qualname__,
)
def test_every_forward_documents_its_shapes(module_type: type[nn.Module]):
    parse_shape_contract(module_type)


@pytest.mark.parametrize(
    "module_type",
    FORWARD_MODULE_TYPES,
    ids=lambda module_type: module_type.__qualname__,
)
def test_every_forward_documents_its_arguments(module_type: type[nn.Module]):
    owner = get_forward_owner(module_type)
    assert owner is not None
    parameters = inspect.signature(owner.__dict__["forward"]).parameters
    expected = {name for name in parameters if name != "self"}
    documented = parse_forward_arguments(module_type)
    assert documented.keys() == expected
    assert all(documented.values())


@pytest.mark.parametrize(
    "module_type",
    FORWARD_MODULE_TYPES,
    ids=lambda module_type: module_type.__qualname__,
)
def test_every_forward_documents_its_return_value(
    module_type: type[nn.Module],
):
    assert parse_forward_returns(module_type)


def test_shape_documentation_is_not_on_classes():
    for module_type in MODULE_TYPES:
        assert ".. shape-contract::" not in (
            module_type.__dict__.get("__doc__") or ""
        )


def test_every_shape_contract_uses_structured_math_markup():
    checked: set[int] = set()
    for module_type in MODULE_TYPES:
        for member in module_type.__dict__.values():
            docstring = getattr(member, "__doc__", None)
            if not docstring or ".. shape-contract::" not in docstring:
                continue
            if id(member) in checked:
                continue
            checked.add(id(member))
            contract = docstring.split(".. shape-contract::", 1)[1]
            assert ":math:`" in contract
            assert "``" not in contract
            assert ".. list-table::" not in contract
            assert ".. container::" not in contract


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            (
                "Inputs\n    :math:`x`\n        ``B, C``\nOutputs\n"
                "    :math:`y`\n        :math:`(B, C)`"
            ),
            "must use math roles",
        ),
        (
            (
                "Outputs\n    :math:`y`\n        :math:`(B, C)`\nInputs\n"
                "    :math:`x`\n        :math:`(B, C)`"
            ),
            "out of order",
        ),
        (
            "Inputs\n    :math:`x`\n        :math:`(B, C)`",
            "must contain Outputs",
        ),
        (
            (
                "Outputs\n    :math:`y`\n        :math:`(B, C)`\n"
                "Inputs outputs\n    :math:`z`\n        :math:`(B, C)`"
            ),
            "collides with a reserved group name",
        ),
        (
            (
                "Outputs\n    :math:`y`\n        :math:`(B, C)`\n"
                "Inference outputs\n    :math:`z`\n        :math:`(B, C)`\n"
                "Export outputs\n    :math:`w`\n        :math:`(B, C)`"
            ),
            "sorted alphabetically",
        ),
        (
            ("Outputs\n    :math:`y`\n        :math:`(B, \\sqrt{C})`"),
            "Invalid math symbol",
        ),
        (
            ("Outputs\n    :math:`y` (:math:`i = 1`)\n        :math:`(B, C)`"),
            "must pair an indexed name",
        ),
        (
            ("Outputs\n    :math:`y` (maybe)\n        :math:`(B, C)`"),
            "must be a math name",
        ),
        (
            (
                "Constraints\n    - :math:`B > 0`\n"
                "Outputs\n    :math:`y`\n        :math:`(B, C)`"
            ),
            "out of order",
        ),
    ],
)
def test_shape_contract_directive_rejects_invalid_grammar(
    body: str,
    message: str,
):
    register_shape_contract_directive()
    source = ".. shape-contract::\n\n" + "\n".join(
        f"    {line}" for line in body.splitlines()
    )
    document = publish_doctree(source)
    errors = list(document.findall(nodes.system_message))
    assert errors
    assert message in errors[-1].astext()
    with pytest.raises(ShapeContractError):
        parse_contract_source(source, context="test contract")


def test_pyproject_registers_the_directive_for_every_pydoctor_run():
    """The docs build must not be the only path that knows the grammar.

    PyDoctor only learns about `shape-contract` through the system class
    named in `[tool.pydoctor]`. Without that entry a plain
    `uv run pydoctor` reports every contract in the package as a docstring
    syntax error, because the directive is never registered.
    """
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    parts = pyproject.read_text(encoding="utf-8").split("[tool.pydoctor]")
    assert len(parts) == 2, "pyproject.toml has no [tool.pydoctor] section"
    section = parts[1].split("\n[", 1)[0]

    declared = f"{LuxonisTrainSystem.__module__}.{LuxonisTrainSystem.__name__}"
    assert f'system-class = "{declared}"' in section
    # Forcing a single docformat would misparse the half of the package
    # that is still Epytext, so the global default stays Epytext and the
    # migrated packages opt in with their own `__docformat__`.
    assert 'docformat = "epytext"' in section


def test_shape_contract_supports_optional_entries():
    source = (
        ".. shape-contract::\n"
        "\n"
        "    Inputs\n"
        "        :math:`x`\n"
        "            :math:`(B, C)`\n"
        "\n"
        "    Outputs\n"
        "        :math:`\\mathrm{output}`\n"
        "            :math:`(B, C)`\n"
        "        :math:`\\mathrm{extra}` (optional)\n"
        "            :math:`(B, C)`\n"
    )
    contract = parse_contract_source(source, context="test contract")
    assert contract.outputs["extra"] == {"optional": ["B", "C"]}

    bindings: dict[str, int | Sequence[int]] = {}
    x = torch.zeros(2, 3)
    assert_matches_contract({"x": x}, contract.inputs, bindings, path="inputs")
    assert_matches_contract(
        {"output": x}, contract.outputs, bindings, path="outputs"
    )
    assert_matches_contract(
        {"output": x, "extra": None},
        contract.outputs,
        bindings,
        path="outputs",
    )
    assert_matches_contract(
        {"output": x, "extra": x},
        contract.outputs,
        bindings,
        path="outputs",
    )
    with pytest.raises(AssertionError, match="required"):
        assert_matches_contract(
            {"extra": x}, contract.outputs, bindings, path="outputs"
        )


def test_node_docstrings_use_google_sections():
    legacy_tokens = (
        "@param",
        "@type",
        "@return",
        "@rtype",
        "@raises",
        "@ivar",
    )
    for module_type in MODULE_TYPES:
        members = [
            module_type,
            *[value for _, value in inspect.getmembers(module_type)],
        ]
        for member in members:
            docstring = getattr(member, "__doc__", None)
            member_module = getattr(member, "__module__", "") or ""
            if not docstring or member_module.startswith("torch"):
                continue
            assert not any(token in docstring for token in legacy_tokens), (
                f"{module_type.__qualname__} contains legacy Epytext markup"
            )


@settings(max_examples=12)
@given(
    batch=st.integers(1, 3),
    height=st.integers(5, 24),
    width=st.integers(5, 24),
    in_channels=st.integers(1, 8),
    out_channels=st.integers(1, 8),
    kernel_size=st.sampled_from([1, 3, 5]),
    stride=st.integers(1, 2),
)
def test_conv_block_shape_contract(
    *,
    batch: int,
    height: int,
    width: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
):
    padding = kernel_size // 2
    module = ConvBlock(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
    ).eval()
    x = torch.randn(batch, in_channels, height, width)
    output = module(x)
    height_out = (height + 2 * padding - kernel_size) // stride + 1
    width_out = (width + 2 * padding - kernel_size) // stride + 1
    assert_contract(
        ConvBlock,
        {"x": x},
        {"output": output},
        {
            "B": batch,
            "C_in": in_channels,
            "C_out": out_channels,
            "H": height,
            "W": width,
            "H_out": height_out,
            "W_out": width_out,
        },
    )


@settings(max_examples=8)
@given(
    batch=st.integers(1, 3),
    height=st.integers(2, 12),
    width=st.integers(2, 12),
    channels=st.integers(4, 12),
    n_classes=st.integers(1, 6),
    reg_max=st.integers(1, 6),
)
def test_prediction_block_shape_contracts(
    *,
    batch: int,
    height: int,
    width: int,
    channels: int,
    n_classes: int,
    reg_max: int,
):
    x = torch.randn(batch, channels, height, width)
    precise = PreciseDecoupledBlock(
        channels,
        channels,
        channels,
        n_classes=n_classes,
        reg_max=reg_max,
    ).eval()
    features, classes, regressions = precise(x)
    assert_contract(
        PreciseDecoupledBlock,
        {"x": x},
        {
            "features": features,
            "classes": classes,
            "regressions": regressions,
        },
        {
            "B": batch,
            "C_in": channels,
            "H": height,
            "W": width,
            "n_classes": n_classes,
            "reg_max": reg_max,
        },
    )

    efficient = EfficientDecoupledBlock(channels, n_classes).eval()
    features, classes, regressions = efficient(x)
    assert_contract(
        EfficientDecoupledBlock,
        {"x": x},
        {
            "features": features,
            "classes": classes,
            "regressions": regressions,
        },
        {
            "B": batch,
            "C_in": channels,
            "H": height,
            "W": width,
            "n_classes": n_classes,
        },
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda channels: SqueezeExciteBlock(channels, max(channels // 2, 1)),
        lambda _: DropPath(0.25),
    ],
)
@settings(max_examples=8)
@given(
    batch=st.integers(1, 3),
    channels=st.integers(2, 12),
    height=st.integers(2, 16),
    width=st.integers(2, 16),
)
def test_preserving_block_shape_contracts(
    factory: Callable[[int], nn.Module],
    batch: int,
    channels: int,
    height: int,
    width: int,
):
    module = factory(channels).eval()
    x = torch.randn(batch, channels, height, width)
    assert_contract(
        type(module),
        {"x": x},
        {"output": module(x)},
        {"B": batch, "C_in": channels, "H": height, "W": width},
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda _, channels: Attention(channels, n_heads=2),
        lambda length, channels: ConvMixer(
            channels, height=2, width=length // 2, n_heads=2
        ),
        lambda _, channels: SVTRBlock(channels, n_heads=2),
    ],
)
@settings(max_examples=8)
@given(
    batch=st.integers(1, 3),
    length=st.integers(1, 8).map(lambda value: value * 2),
    channels=st.integers(1, 6).map(lambda value: value * 2),
)
def test_sequence_block_shape_contracts(
    factory: Callable[[int, int], nn.Module],
    batch: int,
    length: int,
    channels: int,
):
    module = factory(length, channels).eval()
    x = torch.randn(batch, length, channels)
    assert_contract(
        type(module),
        {"x": x},
        {"output": module(x)},
        {"B": batch, "L": length, "C": channels},
    )


@settings(max_examples=8)
@given(
    height=st.integers(1, 12),
    width=st.integers(1, 12),
    head_dim=st.sampled_from([4, 8, 12, 16]),
)
def test_rope_shape_contract(height: int, width: int, head_dim: int):
    module = RopePositionEmbedding(head_dim, num_heads=1)
    sin, cos = module(H=height, W=width)
    assert_contract(
        RopePositionEmbedding,
        {},
        {"sin": sin, "cos": cos},
        {"H": height, "W": width, "head_dim": head_dim},
    )


DETECTION_HEADS: list[tuple[type[BaseDetectionHead], dict[str, Any]]] = [
    (EfficientBBoxHead, {}),
    (EfficientKeypointBBoxHead, {}),
    (PrecisionBBoxHead, {"reg_max": 4}),
    (PrecisionSegmentBBoxHead, {"n_masks": 4, "n_proto": 8, "reg_max": 4}),
]


@pytest.mark.parametrize(
    ("head_type", "extra_kwargs"),
    DETECTION_HEADS,
    ids=[head_type.__qualname__ for head_type, _ in DETECTION_HEADS],
)
@settings(max_examples=4)
@given(batch=st.integers(1, 3))
def test_detection_head_shape_contracts(
    head_type: type[BaseDetectionHead],
    extra_kwargs: dict[str, Any],
    batch: int,
):
    sizes = [Size((8, 8, 8)), Size((8, 4, 4))]
    inputs = [torch.randn(batch, *size) for size in sizes]
    bindings: Bindings = {
        "B": batch,
        "N": 2,
        "C": (8, 8),
        "H": (8, 4),
        "W": (8, 4),
        "A": 80,
        "n_classes": 2,
        "n_keypoints": 2,
        "reg_max": 4,
        "n_masks": 4,
    }
    head = head_type(
        n_heads=2,
        in_sizes=sizes,
        original_in_shape=Size((3, 64, 64)),
        n_classes=2,
        n_keypoints=2,
        **extra_kwargs,
    )

    assert_contract(head_type, {"inputs": inputs}, head(inputs), bindings)

    head.eval()
    head.export = True
    assert_contract(
        head_type,
        {"inputs": inputs},
        head(inputs),
        bindings,
        mode="export",
    )


def _ddrnet_invalid() -> None:
    DDRNetSegmentationHead(
        inter_channels=6,
        inter_mode="pixel_shuffle",
        in_sizes=Size((128, 4, 4)),
        original_in_shape=Size((3, 8, 8)),
        n_classes=2,
    )


def _reppan_invalid() -> None:
    RepPANNeck(
        n_heads=2,
        in_sizes=[Size((8, 15, 15)), Size((8, 8, 8))],
        original_in_shape=Size((3, 63, 63)),
        weights="none",
    )


INVALID_FACTORIES: dict[type[nn.Module], Callable[[], object]] = {
    DDRNetSegmentationHead: _ddrnet_invalid,
    RepPANNeck: _reppan_invalid,
    Attention: lambda: Attention(8, n_heads=2, mixer="local"),
    SVTRBlock: lambda: SVTRBlock(8, n_heads=2, mixer="conv"),
    RopePositionEmbedding: lambda: RopePositionEmbedding(
        8,
        num_heads=1,
        base=None,
    ),
}


@pytest.mark.parametrize(
    ("module_type", "factory"),
    INVALID_FACTORIES.items(),
    ids=lambda value: value.__qualname__ if isinstance(value, type) else None,
)
def test_documented_invalid_shapes_raise_clear_errors(
    module_type: type[nn.Module], factory: Callable[[], object]
):
    shape_contract: ShapeContract = parse_shape_contract(module_type)
    assert len(shape_contract.errors) == 1
    error = shape_contract.errors[0]
    with pytest.raises(_exception_type(error.raises), match=error.match):
        factory()


def _exception_type(name: str) -> type[BaseException]:
    candidate = getattr(builtins, name, None)
    assert isinstance(candidate, type), (
        f"Documented exception {name!r} is not a built-in type"
    )
    assert issubclass(candidate, BaseException), (
        f"Documented exception {name!r} is not an exception type"
    )
    return candidate
