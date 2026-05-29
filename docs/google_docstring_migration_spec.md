# Google Docstring Migration Specification

## Status

Draft.

## Goal

Migrate `luxonis_train` API documentation from epytext-style docstrings to
Google-style docstrings, while improving documentation correctness,
consistency, formatting, examples, API documentation rendering, and doctest
coverage.

The migration must preserve runtime behavior. Code changes are allowed only
when needed to make documentation truthful, executable, or type-checkable.

## Scope

In scope:

- Convert package docstrings under `luxonis_train/**/*.py` from epytext to
  Google style.
- Add missing docstrings for public modules, classes, functions, methods,
  properties, constants, and public attributes.
- Improve incorrect, stale, incomplete, or inconsistent documentation found
  during the conversion.
- Add runnable doctest examples where examples help users understand the API.
- Add doctests to CI and coverage reporting.
- Switch pydoctor parsing from `epytext` to `google`.
- Use pydoctor-supported reStructuredText markup inside Google-style
  docstrings.
- Evaluate whether Ruff can replace `docformatter`; keep `docformatter` if
  Ruff cannot fully cover the existing docstring formatting role.

Out of scope:

- Broad behavior changes unrelated to documentation correctness.
- Rewriting examples into long tutorials inside source docstrings.
- Adding doctests that require cloud credentials, network access, GPU access,
  large model downloads, or non-deterministic external services.
- Migrating README files or standalone prose docs, except when needed to
  document the new workflow.

## Current State

- `pyproject.toml` configures `docformatter` with `style = "epytext"`.
- `.pre-commit-config.yaml` runs `docformatter --style=epytext`.
- `.github/workflows/ci.yaml` has a docs job that installs pydoctor and runs:

  ```bash
  pydoctor --docformat=epytext luxonis_train
  ```

- The codebase contains many epytext fields and inline markup forms, including
  `@param`, `@type`, `@return`, `@rtype`, `@raise`, `@ivar`, `@keyword`,
  `L{...}`, `C{...}`, and `U{...}`.
- Ruff is already the primary linter and code formatter.

## Documentation Style

### Canonical Format

All migrated docstrings must use Google style parsed by pydoctor with
`docformat = "google"`.

Use this section order when applicable:

1. One-line summary.
2. Extended description.
3. `Args:`
4. `Keyword Args:`
5. `Attributes:`
6. `Returns:` or `Yields:`
7. `Raises:`
8. `Warns:`
9. `Warnings:`
10. `Notes:`
11. `See Also:`
12. `Examples:`

Omit sections that do not apply. Do not create empty sections.

### Summary And Paragraphs

- Use triple double quotes.
- Keep the summary on one physical line when practical.
- End the summary with a period unless it is a question or exclamation.
- Put a blank line between the summary and any longer description.
- Use descriptive, API-focused wording.
- Document semantics and caller-visible behavior, not internal implementation
  details.
- Mention side effects, mutation of arguments, filesystem writes, network
  access, device movement, random behavior, and expensive operations when they
  are relevant to the caller.

### Types Are Required In Docstrings

Every documented argument, keyword argument, return value, yielded value,
attribute, property, and public constant must include a type in the docstring,
even when the Python signature already has a type annotation.

This repository intentionally overrides the common recommendation to avoid
duplicating annotated types. The duplication is required so the rendered API
docs remain explicit and readable in all contexts.

Use modern Python type spelling:

```python
def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a training configuration.

    Args:
        path (str | Path): Path to the YAML configuration file.
        overrides (list[str] | None): Optional CLI-style override values.

    Returns:
        Config: Parsed training configuration.
    """
```

Rules:

- Prefer the same type spelling as the signature.
- Use `str | None` instead of `Optional[str]` in new or migrated docs when the
  signature already uses PEP 604 style.
- Use concrete collection shapes where meaningful, such as
  `dict[str, Tensor]`, `tuple[Tensor, dict[str, Tensor]]`, or
  `Sequence[Path]`.
- Use domain aliases such as `Labels`, `Packet[Tensor]`, or `PathType` when
  those aliases are part of the public API.
- For `*args` and `**kwargs`, document the starred name:

  ```python
  Args:
      *args (Any): Positional arguments forwarded to the parent class.
      **kwargs (Any): Keyword arguments forwarded to the parent class.
  ```

- For callable parameters, document the callable signature when useful:

  ```python
  Args:
      transform (Callable[[Tensor], Tensor]): Transform applied to each image.
  ```

- For tensor data, include shape, dtype, device, coordinate convention, and
  value range when these are part of the API contract:

  ```python
  Args:
      boxes (Tensor): Bounding boxes with shape `[N, 4]` in `xyxy` format.
  ```

### Args And Keyword Args

Use `Args:` for regular parameters and `Keyword Args:` only for open-ended or
forwarded keyword arguments that are not explicit in the function signature.

Each item must use this form:

```text
name (type): Description.
```

For multi-line descriptions, indent continuation lines by four additional
spaces:

```python
Args:
    losses (dict[str, dict[str, Tensor]]): Loss tensors grouped by node name
        and loss name.
```

Do not document `self` or `cls`.

### Returns And Yields

Use `Returns:` for non-`None` return values. Use `Yields:` for generators and
iterators that yield values.

Each return section must include a type:

```python
Returns:
    tuple[Tensor, dict[str, Tensor]]: Final loss and individual loss values.
```

For tuple returns, describe the tuple explicitly instead of pretending each
tuple element is a separate return value:

```python
Returns:
    tuple[Tensor, Tensor]: A tuple `(images, labels)`, where `images` has shape
    `[B, C, H, W]` and `labels` contains task-specific targets.
```

Omit `Returns:` only when the function always returns `None` and the summary or
description already makes that clear.

### Raises And Warnings

Document exceptions that are part of the public interface:

```python
Raises:
    ValueError: If `view` is not one of `"train"`, `"val"`, or `"test"`.
```

Do not document incidental exceptions from violating the documented API
contract unless callers are expected to handle them.

Use `Warns:` for warnings emitted through the `warnings` module. Use
`Warnings:` for human-readable cautionary notes that are not emitted warnings.

### Attributes, Properties, And Constants

Use `Attributes:` in class docstrings for public instance attributes that users
can read or set directly:

```python
class TrainerState:
    """State tracked during training.

    Attributes:
        epoch (int): Current zero-based epoch.
        global_step (int): Number of optimizer steps completed.
    """
```

For module-level constants and class variables, choose one convention per
module:

- Prefer inline attribute docstrings immediately after the assignment when the
  constant is local to that module.
- Prefer a module-level `Attributes:` section when many public constants are
  grouped together.

Do not mix both conventions for the same group of constants.

Property docstrings must read like attribute documentation and must include the
type:

```python
@property
def n_classes(self) -> int:
    """int: Number of classes predicted by the head."""
```

### Cross-References And Inline Markup

Google-style docstrings are converted by pydoctor into reStructuredText before
rendering. Use pydoctor-compatible reStructuredText markup inside docstrings.

Use:

- `*italic text*` for emphasis.
- `**bold text**` for strong emphasis.
- Double backticks for inline literals, for example ``None`` or ``"train"``.
- Single backticks for pydoctor cross-references to Python API objects, for
  example `` `LuxonisLightning` `` or `` `torch.Tensor` ``.
- reStructuredText links for external URLs:

  ```text
  `Paper title <https://example.com/paper>`_
  ```

- Inline math with `:math:` when supported by pydoctor/docutils:

  ```text
  The loss is scaled by :math:`\lambda`.
  ```

- Block math for longer formulas:

  ```text
  .. math::

      L = L_{cls} + \lambda L_{box}
  ```

Use raw docstrings, `r"""..."""`, when backslashes must survive Python string
escaping, especially in math-heavy docstrings.

Avoid Markdown fenced code blocks inside Python docstrings. Use doctest prompts
or reStructuredText directives instead.

### Epytext Conversion Map

Use this mapping during migration:

| Epytext | Google / reST replacement |
| --- | --- |
| `@param x:` | `Args:` item `x (type): ...` |
| `@type x:` | Type in the `Args:` item and signature annotation |
| `@return:` | `Returns:` |
| `@rtype:` | Type in the `Returns:` item and signature annotation |
| `@raise Error:` | `Raises:` item `Error: ...` |
| `@keyword x:` | `Keyword Args:` item `x (type): ...` |
| `@ivar x:` | Class `Attributes:` item `x (type): ...` |
| `@cvar x:` | Class variable inline docstring or `Attributes:` item |
| `L{Object}` | Single-backtick cross-reference `` `Object` `` |
| `C{value}` | Double-backtick literal ``value`` |
| `U{text <url>}` | reST link `` `text <url>`_ `` |
| `@note:` | `Notes:` or `.. note::` |
| `@warning:` | `Warnings:` or `.. warning::` |
| `@see:` | `See Also:` |

## Examples And Doctests

### When To Add Examples

Add an `Examples:` section when at least one of these is true:

- The API is public and non-trivial.
- The function has important shape, dtype, coordinate, or config conventions.
- The function has a common usage pattern that is easy to get wrong.
- The function returns structured data.
- The docstring would otherwise need a long prose explanation that an example
  can clarify.

Do not add examples merely to satisfy a quota.

### Doctest Format

Runnable examples must use standard doctest prompts:

```python
Examples:
    >>> from luxonis_train.utils.ocr import Encoder
    >>> encoder = Encoder(["a", "b"], ignore_unknown=True)
    >>> encoder.encode(["ab"]).tolist()
    [1, 2]
```

Prefer examples that assert behavior without relying on verbose object reprs:

```python
Examples:
    >>> result = normalize_boxes(boxes, image_size=(100, 200))
    >>> result.shape
    torch.Size([2, 4])
    >>> bool((result >= 0).all())
    True
```

### Doctest Requirements

Doctest examples must be:

- Deterministic.
- CPU-only.
- Fast enough to run on every PR.
- Independent of external services, credentials, downloads, and persistent
  local state.
- Stable across supported Python versions.
- Explicit about imports.
- Small enough to keep the docstring readable.

Use `# doctest: +SKIP` only when an example is valuable as documentation but
cannot satisfy these requirements. Skipped doctests must be rare and must have a
short explanation in the surrounding prose.

Use doctest option flags sparingly. Prefer stable expected output over broad
matching.

### Recommended Doctest Patterns

Use exact expected output for simple values:

```python
Examples:
    >>> clamp(5, min_value=0, max_value=3)
    3
```

Use `ELLIPSIS` only for unavoidable variable fragments:

```python
Examples:
    >>> model
    LuxonisModel(...)
```

Use `NORMALIZE_WHITESPACE` only when output formatting is semantically
irrelevant.

For exceptions, use stable exception details:

```python
Examples:
    >>> parse_view("invalid")
    Traceback (most recent call last):
    ...
    ValueError: Unknown dataset view: invalid
```

For tensor outputs, prefer shape, dtype, and scalar assertions over full tensor
reprs.

## Tooling Specification

### Pydoctor

Add pydoctor configuration to `pyproject.toml`:

```toml
[tool.pydoctor]
add-package = ["luxonis_train"]
docformat = "google"
project-name = "luxonis-train"
project-url = "https://github.com/luxonis/luxonis-train"
html-output = "apidocs"
warnings-as-errors = true
intersphinx = [
  "https://docs.python.org/3/objects.inv",
  "https://pytorch.org/docs/stable/objects.inv",
]
```

Update the docs CI step to run pydoctor from configuration:

```bash
pydoctor
```

During early migration, `warnings-as-errors` may be temporarily disabled on a
dedicated migration branch only if the branch tracks and reduces the warning
count. The final migration must enable `warnings-as-errors = true`.

Do not enable `process-types` for Google docstrings.

### Ruff

Ruff should remain the primary Python formatter and linter.

Enable docstring code example formatting:

```toml
[tool.ruff.format]
docstring-code-format = true
docstring-code-line-length = "dynamic"
```

Ruff can format Python code examples inside docstrings, including doctest
examples, reStructuredText literal blocks, and reStructuredText `code-block`
directives. Ruff does not replace pydoctor parsing or documentation
correctness checks.

### Docformatter

Do not remove `docformatter` solely as part of this migration unless local
validation shows Ruff fully covers the needed behavior.

Current decision:

- Keep `docformatter` for now.
- Do not rely on it for Google field formatting.
- Re-evaluate after the first large migrated batch.

Replacement criteria:

- `ruff format` with `docstring-code-format = true` formats all code examples
  acceptably.
- `docformatter` produces no useful changes on migrated Google-style
  docstrings.
- Removing `docformatter` does not reduce enforcement of summary wrapping,
  blank-line behavior, closing quote placement, or docstring consistency.
- Pre-commit and CI remain deterministic.

If these criteria are not met, keep `docformatter` unchanged except for any
minimal adjustment required to avoid damaging Google-style docstrings.

### Pytest Doctests

Add doctest configuration to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
doctest_optionflags = ["ELLIPSIS", "NORMALIZE_WHITESPACE"]
```

Do not add `--doctest-modules` to the global `addopts` unless the full test
suite remains performant and stable. Prefer an explicit CI command so doctests
can be tracked separately.

Recommended local command:

```bash
pytest --doctest-modules luxonis_train
```

Recommended coverage command:

```bash
pytest --doctest-modules luxonis_train --cov=luxonis_train --cov-report=xml
```

### CI

Update `.github/workflows/ci.yaml`:

- Change the docs job from `pydoctor --docformat=epytext luxonis_train` to
  `pydoctor`.
- Add a dedicated doctest job or step that installs the normal test
  dependencies and runs doctests with coverage.
- Upload doctest coverage to Codecov with a distinct flag, such as `doctest`.
- Upload doctest JUnit results if Codecov test-results upload remains enabled.

Recommended new doctest job shape:

```yaml
doctests:
  needs:
    - config-test
  runs-on: ubuntu-latest
  steps:
    - name: Checkout
      uses: actions/checkout@v6
      with:
        ref: ${{ github.head_ref }}

    - name: Set up Python
      uses: actions/setup-python@v6
      with:
        python-version: "3.10"
        cache: pip

    - name: Install dependencies
      run: pip install -e .[dev]

    - name: Install dev version of LuxonisML
      if: startsWith(github.head_ref, 'release/') == false
      run: |
        pip uninstall luxonis-ml -y
        pip install "luxonis-ml[data,tracker] @ git+https://github.com/luxonis/luxonis-ml.git@main"

    - name: Run doctests
      run: >
        pytest --doctest-modules luxonis_train
        --cov=luxonis_train
        --cov-report=xml
        --junitxml=junit.xml
        -o junit_family=legacy
```

If doctests need secrets or cloud credentials, the examples are too heavy and
must be rewritten, skipped with justification, or moved to normal tests.

## Migration Workflow

### Phase 1: Inventory

- Count all epytext markers:

  ```bash
  rg -n "@(param|type|return|rtype|raise|raises|ivar|cvar|keyword|see|note|warning)|L\\{|C\\{|U\\{" luxonis_train
  ```

- Identify modules with missing public API docs.
- Identify docstrings with stale parameter names, stale types, wrong defaults,
  typoed types, or undocumented returns.
- Identify candidates for doctest examples.

### Phase 2: Tooling Foundation

- Add `[tool.pydoctor]` with `docformat = "google"`.
- Update docs CI to use pydoctor config.
- Enable Ruff docstring code formatting.
- Add an explicit doctest CI command.
- Add doctest coverage upload.
- Keep `docformatter` unless replacement criteria are met.

### Phase 3: Mechanical Conversion

- Convert epytext fields to Google sections.
- Convert epytext inline markup to reStructuredText markup.
- Preserve existing useful prose.
- Remove duplicated stale prose.
- Keep converted docstrings rendering cleanly in pydoctor.

### Phase 4: Correctness Pass

For every touched docstring:

- Compare documented parameters against the actual signature.
- Compare documented types against annotations and runtime behavior.
- Verify documented defaults.
- Verify returns, yields, exceptions, side effects, tensor shapes, coordinate
  formats, and config keys.
- Add missing docs for public APIs exposed by the module.
- Add doctest examples where useful and feasible.

### Phase 5: Validation

Run:

```bash
ruff format --check .
ruff check .
pydoctor
pytest --doctest-modules luxonis_train
pytest --cov
```

Run targeted unit tests for modules whose examples or documentation required
small code changes.

## Acceptance Criteria

The migration is complete when:

- No epytext field syntax remains in `luxonis_train/**/*.py`.
- No epytext inline markup remains in migrated docstrings.
- Pydoctor runs with `docformat = "google"` and no warnings.
- Public API docstrings use consistent Google sections.
- Every documented parameter, return, yield, property, attribute, and public
  constant includes a type in the docstring.
- All public API parameters that need explanation are documented.
- All non-`None` public returns are documented.
- Public exceptions that callers are expected to handle are documented.
- Doctest examples run in CI.
- Doctest coverage is included in Codecov uploads.
- Ruff formats Python code examples in docstrings.
- `docformatter` is either retained intentionally or removed only after meeting
  the replacement criteria above.
- CI remains green.

## Review Checklist

Use this checklist for migrated files:

- [ ] Summary is one line and ends with punctuation.
- [ ] Google sections are ordered consistently.
- [ ] Every `Args:` item has `name (type): description`.
- [ ] Every `Returns:` or `Yields:` item has `type: description`.
- [ ] Every `Attributes:` item has `name (type): description`.
- [ ] No stale parameter names.
- [ ] No stale defaults.
- [ ] No typoed or invalid type names.
- [ ] Tensor shapes, units, coordinate formats, and value ranges are documented
      where relevant.
- [ ] Side effects and mutations are documented.
- [ ] Cross-references use pydoctor-compatible backticks.
- [ ] Inline literals use double backticks.
- [ ] Math-heavy docstrings use raw strings where needed.
- [ ] Examples are useful, deterministic, and small.
- [ ] Doctest examples pass.
- [ ] Pydoctor renders without warnings.

## References

- Pydoctor documentation formats:
  https://pydoctor.readthedocs.io/en/latest/docformat/
- Pydoctor Google and NumPy support:
  https://pydoctor.readthedocs.io/en/latest/docformat/google-numpy.html
- Pydoctor configuration:
  https://pydoctor.readthedocs.io/en/latest/help.html
- Pydoctor reStructuredText support:
  https://pydoctor.readthedocs.io/en/stable/docformat/restructuredtext.html
- Ruff formatter:
  https://docs.astral.sh/ruff/formatter/
- Pytest doctest support:
  https://docs.pytest.org/en/stable/how-to/doctest.html
