r"""The ``shape-contract`` docstring grammar.

This module is the single implementation of the grammar. It is used both
by the documentation build, where :class:`ShapeContractDirective` renders
and validates the contract, and by the test suite, where
:func:`parse_shape_contract` turns the very same markup into a
:class:`ShapeContract`. Both paths run :func:`parse_contract_tree`, so a
construct the tests understand is exactly the set the docs build accepts.

The directive body is a definition list of groups. The groups must appear
in this order:

``Inputs``
    Tensors accepted by the ``forward`` method.
``Outputs``
    Tensors returned in the default mode.
``<Mode> outputs``
    Tensors returned in a named mode, for example ``Export outputs``.
    The mode is a single capitalized word and must not collide with a
    reserved group name. Several mode groups must be sorted
    alphabetically.
``Constraints``
    A bullet list of math expressions that must hold.
``Symbols``
    Descriptions of the symbols used above.

Every tensor entry is a term with its shape as the definition::

    Inputs
        :math:`x`
            :math:`(B, C_{\mathrm{in}}, H, W)`

A term is a math name, optionally followed by a parenthesized math index
range, optionally followed by an ``(optional)`` marker::

    Outputs
        :math:`\mathrm{features}_{i}` (:math:`i = 0, \ldots, N - 1`)
            :math:`(B, C_{i}, H_{i}, W_{i})`
        :math:`\mathrm{masks}` (optional)
            :math:`(B, M, H, W)`

An indexed name must always be paired with an index range. The
``(optional)`` marker documents a mapping entry that may be absent, or
present but holding no tensors, in some configurations. Such an entry is
skipped by the test-side matcher instead of being reported as an
undocumented structure.

Shapes and constraints are LaTeX math. The supported constructs are
integer literals, symbols, ``\left\lfloor \frac{a}{b} \right\rfloor``,
``\left\lceil \frac{a}{b} \right\rceil``, ``\land``, ``\lor``, ``\le``,
``\ge``, ``\ne``, ``=``, ``<``, ``>``, ``+``, ``-``, ``\cdot`` and
``\operatorname{mod}``. Anything else is rejected.
"""

__docformat__ = "google"

import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass

from docutils import nodes
from docutils.core import publish_doctree
from docutils.parsers.rst import Directive, directives

ShapeSpec = int | str | list["ShapeSpec"] | dict[str, "ShapeSpec"]

DIRECTIVE_NAME = "shape-contract"
DIRECTIVE_MARKER = f".. {DIRECTIVE_NAME}::"

INPUTS_GROUP = "Inputs"
OUTPUTS_GROUP = "Outputs"
CONSTRAINTS_GROUP = "Constraints"
SYMBOLS_GROUP = "Symbols"
RESERVED_GROUPS = frozenset(
    {INPUTS_GROUP, OUTPUTS_GROUP, CONSTRAINTS_GROUP, SYMBOLS_GROUP}
)

OPTIONAL_MARKER = "(optional)"
OPTIONAL_KEY = "optional"

_MODE_GROUP = re.compile(
    r"(?P<mode>[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*) outputs"
)
_INDEXED_PATH = re.compile(r"(?P<name>\w+)\[(?P<index>\d+)\]")
_QUANTIFIER = re.compile(r"i = 0, \\ldots, (?P<count>.+) - 1")

_FRACTIONS = {
    (r"\left\lfloor \frac", r" \right\rfloor"): "floor_div",
    (r"\left\lceil \frac", r" \right\rceil"): "ceil_div",
}
_OPERATORS = [
    (r" \land ", "and"),
    (r" \lor ", "or"),
    (r" \le ", "<="),
    (r" \ge ", ">="),
    (r" \ne ", "!="),
    (" = ", "=="),
    (" < ", "<"),
    (" > ", ">"),
    (" + ", "+"),
    (" - ", "-"),
    (r" \cdot ", "*"),
    (r" \operatorname{mod} ", "%"),
]


class ShapeContractError(AssertionError):
    """Raised when a shape contract does not follow the grammar."""


@dataclass(frozen=True)
class ShapeError:
    """A documented failure mode of a ``forward`` method."""

    when: str
    raises: str
    match: str


@dataclass(frozen=True)
class ShapeContract:
    """The parsed contents of a ``shape-contract`` directive."""

    inputs: dict[str, ShapeSpec]
    outputs: dict[str, ShapeSpec]
    constraints: tuple[str, ...] = ()
    modes: dict[str, dict[str, dict[str, ShapeSpec]]] | None = None
    errors: tuple[ShapeError, ...] = ()


class ShapeContractDirective(Directive):
    """Render and validate a node shape contract."""

    has_content = True

    def run(self) -> list[nodes.Node]:
        contract = nodes.container(
            self.block_text,
            classes=[DIRECTIVE_NAME],
        )
        self.state.nested_parse(
            self.content,
            self.content_offset,
            contract,
        )
        try:
            parse_contract_tree(contract, decorate=True)
        except ShapeContractError as exc:
            raise self.error(str(exc)) from exc
        return [contract]


def register_shape_contract_directive() -> None:
    """Register the shape contract directive with Docutils."""
    directives.register_directive(DIRECTIVE_NAME, ShapeContractDirective)


def parse_shape_contract(
    source: str | Sequence[str], *, context: str = "docstring"
) -> ShapeContract:
    """Parse the shape contract embedded in a docstring.

    Args:
        source: A docstring, or its lines. The directive block is
            dedented before it is parsed, so either form works.
        context: Name of the documented object, used in error messages.

    Returns:
        The parsed contract, without its documented errors.

    Raises:
        ShapeContractError: If the docstring has no contract, or if the
            contract does not follow the grammar.

    """
    lines = source.splitlines() if isinstance(source, str) else list(source)
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() == DIRECTIVE_MARKER
        ),
        None,
    )
    if start is None:
        raise ShapeContractError(f"{context} has no shape-contract directive")

    register_shape_contract_directive()
    document = publish_doctree(textwrap.dedent("\n".join(lines[start:])))
    messages = list(document.findall(nodes.system_message))
    if messages:
        details = "; ".join(message.astext() for message in messages)
        raise ShapeContractError(
            f"Invalid shape contract for {context}: {details}"
        )
    contracts = [
        node
        for node in document.findall(nodes.container)
        if DIRECTIVE_NAME in node.get("classes", [])
    ]
    if len(contracts) != 1:
        raise ShapeContractError(f"{context} must have one shape contract")
    return parse_contract_tree(contracts[0])


def parse_contract_tree(
    contract: nodes.Element, *, decorate: bool = False
) -> ShapeContract:
    """Validate a parsed directive body and collect its shapes.

    Args:
        contract: The container holding the directive body.
        decorate: Whether to tag the nodes with rendering classes. Only
            the documentation build needs them.

    Returns:
        The parsed contract, without its documented errors.

    Raises:
        ShapeContractError: If the body does not follow the grammar.

    """
    if list(contract.findall(nodes.literal)):
        raise ShapeContractError(
            "shape-contract values must use math roles, not code literals"
        )
    groups = contract.children[0] if len(contract.children) == 1 else None
    if not isinstance(groups, nodes.definition_list):
        raise ShapeContractError(
            "shape-contract must contain a definition list of groups"
        )
    if decorate:
        groups["classes"].append("shape-groups")

    inputs: dict[str, ShapeSpec] = {}
    outputs: dict[str, ShapeSpec] = {}
    modes: dict[str, dict[str, dict[str, ShapeSpec]]] = {}
    constraints: tuple[str, ...] = ()

    seen: set[str] = set()
    previous_rank = -1
    previous_mode = ""
    n_output_groups = 0
    for item in groups.children:
        if not isinstance(item, nodes.definition_list_item):
            raise ShapeContractError(
                "shape-contract contains an invalid group"
            )
        term, definition = item.children
        name = term.astext().strip()
        key = name.casefold()
        if key in seen:
            raise ShapeContractError(
                f"shape-contract group {name!r} is duplicated"
            )
        seen.add(key)

        rank, group_kind, mode = _classify_group(name)
        if rank < previous_rank:
            raise ShapeContractError(
                f"shape-contract group {name!r} is out of order"
            )
        if mode is not None:
            if mode <= previous_mode:
                raise ShapeContractError(
                    f"shape-contract group {name!r} is out of order, "
                    "mode-specific outputs must be sorted alphabetically"
                )
            previous_mode = mode
        previous_rank = rank
        if decorate:
            item["classes"].append(f"shape-{nodes.make_id(name)}")

        if group_kind == "inputs":
            _parse_tensor_entries(name, definition, inputs, decorate=decorate)
        elif group_kind == "outputs":
            if mode is None:
                target = outputs
            else:
                target = modes.setdefault(mode, {"outputs": {}})["outputs"]
            _parse_tensor_entries(name, definition, target, decorate=decorate)
            n_output_groups += 1
        elif group_kind == "constraints":
            constraints = _parse_constraints(definition, decorate=decorate)
        else:
            _parse_symbols(definition, decorate=decorate)

    if n_output_groups == 0:
        raise ShapeContractError(
            "shape-contract must contain Outputs or a mode-specific "
            "outputs group"
        )
    return ShapeContract(inputs, outputs, constraints, modes or None)


def unwrap_optional(specification: ShapeSpec) -> tuple[bool, ShapeSpec]:
    """Split a shape specification into its optional flag and its value.

    Args:
        specification: A documented shape specification.

    Returns:
        Whether the specification is optional, and the wrapped value.

    """
    if isinstance(specification, dict) and set(specification) == {
        OPTIONAL_KEY
    }:
        return True, specification[OPTIONAL_KEY]
    return False, specification


def _classify_group(name: str) -> tuple[int, str, str | None]:
    if name == INPUTS_GROUP:
        return 0, "inputs", None
    if name == OUTPUTS_GROUP:
        return 1, "outputs", None
    if name == CONSTRAINTS_GROUP:
        return 3, "constraints", None
    if name == SYMBOLS_GROUP:
        return 4, "symbols", None

    match = _MODE_GROUP.fullmatch(name)
    if match is None:
        raise ShapeContractError(f"unknown shape-contract group {name!r}")
    mode = match.group("mode")
    if mode in RESERVED_GROUPS:
        raise ShapeContractError(
            f"shape-contract mode {mode!r} collides with a reserved group name"
        )
    return 2, "outputs", mode.casefold()


def _parse_tensor_entries(
    group_name: str,
    definition: nodes.Node,
    target: dict[str, ShapeSpec],
    *,
    decorate: bool = False,
) -> None:
    entries = _nested_definition_list(group_name, definition)
    if not entries.children:
        raise ShapeContractError(
            f"{group_name} must document at least one tensor"
        )
    seen: set[str] = set()
    for item in entries.children:
        if not isinstance(item, nodes.definition_list_item):
            raise ShapeContractError(
                f"{group_name} contains an invalid tensor entry"
            )
        term, shape = item.children
        name, quantifier, optional = _parse_tensor_term(group_name, term)
        if name in seen:
            raise ShapeContractError(
                f"{group_name} tensor {name!r} is documented twice"
            )
        seen.add(name)

        shapes = list(shape.findall(nodes.math))
        if len(shapes) != 1 or shape.astext().strip() != shapes[0].astext():
            raise ShapeContractError(
                f"{group_name} tensor {name!r} must have exactly one "
                "math-formatted shape"
            )
        _insert_shape_specification(
            target,
            _parse_math_path(name),
            _parse_shape_math(shapes[0].astext()),
            None if quantifier is None else _parse_quantifier(quantifier),
            optional=optional,
        )
        if decorate:
            item["classes"].append("shape-entry")


def _parse_tensor_term(
    group_name: str, term: nodes.Node
) -> tuple[str, str | None, bool]:
    names = list(term.findall(nodes.math))
    if len(names) not in {1, 2}:
        raise ShapeContractError(
            f"{group_name} tensor terms must contain one math name "
            "and at most one math index range"
        )
    name = names[0].astext()
    quantifier = names[1].astext() if len(names) == 2 else None

    text = term.astext().strip()
    optional = text.endswith(OPTIONAL_MARKER)
    expected = name
    if quantifier is not None:
        expected = f"{expected} ({quantifier})"
    if optional:
        expected = f"{expected} {OPTIONAL_MARKER}"
    if text != expected:
        raise ShapeContractError(
            f"{group_name} tensor term {text!r} must be a math name, "
            "an optional math index range and an optional "
            f"{OPTIONAL_MARKER!r} marker"
        )

    if name.endswith("_{i}") != (quantifier is not None):
        raise ShapeContractError(
            f"{group_name} tensor {name!r} must pair an indexed name "
            "with an index range"
        )
    return name, quantifier, optional


def _parse_constraints(
    definition: nodes.Node, *, decorate: bool = False
) -> tuple[str, ...]:
    bullets = definition.children[0] if len(definition.children) == 1 else None
    if not isinstance(bullets, nodes.bullet_list):
        raise ShapeContractError("Constraints must be a bullet list")
    if not bullets.children:
        raise ShapeContractError(
            "Constraints must contain at least one expression"
        )
    constraints: list[str] = []
    for item in bullets.children:
        if not isinstance(item, nodes.list_item):
            raise ShapeContractError("Constraints contains an invalid item")
        expressions = list(item.findall(nodes.math))
        if (
            len(expressions) != 1
            or item.astext().strip() != expressions[0].astext()
        ):
            raise ShapeContractError(
                "each shape-contract constraint must be one math expression"
            )
        constraints.append(
            str(_parse_math_expression(expressions[0].astext()))
        )
        if decorate:
            item["classes"].append("shape-constraint")
    return tuple(constraints)


def _parse_symbols(definition: nodes.Node, *, decorate: bool = False) -> None:
    entries = _nested_definition_list(SYMBOLS_GROUP, definition)
    if not entries.children:
        raise ShapeContractError("Symbols must document at least one symbol")
    seen: set[str] = set()
    for item in entries.children:
        if not isinstance(item, nodes.definition_list_item):
            raise ShapeContractError("Symbols contains an invalid entry")
        term, description = item.children
        symbols = list(term.findall(nodes.math))
        if len(symbols) != 1 or term.astext().strip() != symbols[0].astext():
            raise ShapeContractError(
                "symbol names must be one math expression"
            )
        symbol = symbols[0].astext()
        if symbol in seen:
            raise ShapeContractError(f"symbol {symbol!r} is documented twice")
        seen.add(symbol)
        _parse_math_symbol(symbol)
        if not description.astext().strip():
            raise ShapeContractError(f"symbol {symbol!r} has no description")
        if decorate:
            item["classes"].append("shape-symbol")


def _nested_definition_list(
    group_name: str, definition: nodes.Node
) -> nodes.definition_list:
    entries = definition.children[0] if len(definition.children) == 1 else None
    if not isinstance(entries, nodes.definition_list):
        raise ShapeContractError(
            f"{group_name} must contain a definition list"
        )
    return entries


def _insert_shape_specification(
    target: dict[str, ShapeSpec],
    path: str,
    specification: ShapeSpec,
    repeat: str | int | None,
    *,
    optional: bool = False,
) -> None:
    repeated = path.endswith("[i]")
    if repeat is not None:
        if not repeated:
            raise ShapeContractError("Repeated shapes must use an [i] path")
        target[path[:-3]] = _wrap_optional(
            {"repeat": repeat, "item": specification}, optional
        )
        return
    if repeated:
        raise ShapeContractError("An [i] path must include its range")

    indexed = _INDEXED_PATH.fullmatch(path)
    if indexed is None:
        target[path] = _wrap_optional(specification, optional)
        return
    if optional:
        raise ShapeContractError(
            f"An indexed shape cannot be optional: {path}"
        )
    name = indexed.group("name")
    index = int(indexed.group("index"))
    sequence = target.setdefault(name, [])
    if not isinstance(sequence, list) or len(sequence) != index:
        raise ShapeContractError(f"Non-consecutive documented shape: {path}")
    sequence.append(specification)


def _wrap_optional(specification: ShapeSpec, optional: bool) -> ShapeSpec:
    if not optional:
        return specification
    return {OPTIONAL_KEY: specification}


def _parse_quantifier(text: str) -> int | str:
    if text == "i = 0":
        return 1
    match = _QUANTIFIER.fullmatch(text)
    if match is None:
        raise ShapeContractError(f"Invalid tensor quantifier: {text}")
    return _parse_math_expression(match.group("count"))


def _parse_math_path(text: str) -> str:
    return ".".join(_parse_math_symbol(part) for part in text.split("."))


def _parse_math_symbol(text: str) -> str:
    # A role and an index share one subscript, as in
    # `C_{\mathrm{in}, i}`. Writing this as two subscripts would be
    # invalid LaTeX and would not render.
    role_indexed = re.fullmatch(
        r"(?P<base>.+)_\{\\mathrm\{(?P<role>[A-Za-z0-9_]+)\}, "
        r"(?P<index>i|\d+)\}",
        text,
    )
    if role_indexed:
        return (
            f"{_parse_math_symbol(role_indexed.group('base'))}_"
            f"{role_indexed.group('role')}"
            f"[{role_indexed.group('index')}]"
        )
    indexed = re.fullmatch(r"(?P<base>.+)_\{(?P<index>i|\d+)\}", text)
    if indexed:
        return (
            f"{_parse_math_symbol(indexed.group('base'))}"
            f"[{indexed.group('index')}]"
        )
    descriptive = re.fullmatch(
        r"(?P<base>.+)_\{\\mathrm\{(?P<subscript>[A-Za-z0-9_]+)\}\}",
        text,
    )
    if descriptive:
        return (
            f"{_parse_math_symbol(descriptive.group('base'))}_"
            f"{descriptive.group('subscript')}"
        )
    numbered = re.fullmatch(r"(?P<base>.+)_(?P<index>\d+)", text)
    if numbered:
        return (
            f"{_parse_math_symbol(numbered.group('base'))}"
            f"{numbered.group('index')}"
        )
    upright = re.fullmatch(r"\\mathrm\{(?P<name>[A-Za-z0-9_]+)\}", text)
    if upright:
        return upright.group("name")
    if re.fullmatch(r"[A-Za-z]", text):
        return text
    raise ShapeContractError(f"Invalid math symbol: {text}")


def _parse_shape_math(text: str) -> ShapeSpec:
    if not (text.startswith("(") and text.endswith(")")):
        return {"shape": _parse_math_expression(text)}
    dimensions = _split_dimensions(text[1:-1])
    return [_parse_math_expression(dimension) for dimension in dimensions]


def _split_dimensions(text: str) -> list[str]:
    dimensions: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1
        elif character == "," and depth == 0:
            dimensions.append(text[start:index].strip())
            start = index + 1
    dimensions.append(text[start:].strip())
    if not all(dimensions):
        raise ShapeContractError(f"Invalid tensor dimensions: {text}")
    return dimensions


def _parse_math_expression(text: str) -> int | str:
    """Translate a LaTeX math expression into a Python expression."""
    text = text.strip()
    try:
        return int(text)
    except ValueError:
        pass

    for (prefix, suffix), operator in _FRACTIONS.items():
        if text.startswith(prefix) and text.endswith(suffix):
            numerator, position = _parse_braced(text, len(prefix))
            denominator, position = _parse_braced(text, position)
            if position != len(text) - len(suffix):
                raise ShapeContractError(f"Invalid math fraction: {text}")
            left = _parse_math_expression(numerator)
            right = _parse_math_expression(denominator)
            if operator == "floor_div":
                return f"{left} // {right}"
            return f"ceil_div({left}, {right})"

    for math_operator, python_operator in _OPERATORS:
        split = _split_top_level(text, math_operator)
        if split is not None:
            left_text, right_text = split
            return (
                f"{_parse_math_expression(left_text)} {python_operator} "
                f"{_parse_math_expression(right_text)}"
            )
    return _parse_math_symbol(text)


def _parse_braced(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != "{":
        raise ShapeContractError(f"Expected a braced expression in: {text}")
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    raise ShapeContractError(f"Unclosed braced expression in: {text}")


def _split_top_level(text: str, operator: str) -> tuple[str, str] | None:
    depth = 0
    for index, character in enumerate(text):
        if character in "({[":
            depth += 1
        elif character in ")}]":
            depth -= 1
        elif depth == 0 and text.startswith(operator, index):
            return text[:index], text[index + len(operator) :]
    return None
