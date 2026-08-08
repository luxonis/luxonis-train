# Contributing to LuxonisTrain

**This guide is intended for our internal development team.**
It outlines our workflow and standards for contributing to this project.

## Pre-requisites

Clone the repository and navigate to the root directory:

```bash
git clone git@github.com:luxonis/luxonis-train.git
cd luxonis-train
```

Install [uv](https://docs.astral.sh/uv/) and sync the development environment:

```bash
uv sync
```

> [!NOTE]
> This creates a `.venv` with the package installed in editable mode, together with the `dev` dependencies.

To also install the AIMET support, run:

```bash
uv sync --extra aimet
```

The `requirements*.txt` files are exports of `uv.lock` for users installing with pip.
Do not edit them manually. After changing the dependencies, run:

```bash
scripts/export_requirements.sh
```

> [!NOTE]
> A pre-commit hook runs the script for you - you only need to stage the refreshed files.

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality and consistency.
The hooks are run by [`prek`](https://github.com/j178/prek), a faster drop-in replacement for `pre-commit`.

1. Run `uv run prek install` in the root directory.
1. The hooks will now run automatically on `git commit`.
   - If a hook fails, it will print an error message and abort the commit.
   - Some hooks will also modify the files in-place to fix found issues.

To run all the hooks manually, use `uv run prek run --all-files`.

**Do not commit directly to `main`** - the `no-commit-to-branch` hook blocks it.

## Documentation

Most existing packages use
[Epytext](https://epydoc.sourceforge.net/epytext.html). The
`luxonis_train.nodes` package uses Google-style docstrings while the rest of
the project is migrated. The documentation is built for the whole package
with a global `--docformat=epytext`, and the package-level
`__docformat__ = "google"` declaration in `luxonis_train/nodes/__init__.py`
tells Pydoctor to use the other parser for the migrated package.

Every class derived from `torch.nn.Module` in `luxonis_train.nodes` must have
its own class docstring. Shape information describes invocation rather than
construction, so every repository-defined `forward` method must document its
arguments and return value in Google-style `Args` and `Returns` sections. Put
the machine-readable tensor contract in the custom `shape-contract`
reStructuredText directive after those sections:

```python
class ExampleBlock(nn.Module):
    """Apply a projection without changing the spatial resolution."""

    def forward(self, x: Tensor) -> Tensor:
        r"""Project the input features.

        Args:
            x: Feature map to project.

        Returns:
            Projected feature map.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{output}`
                    :math:`(B, C_{\mathrm{out}}, H, W)`
        """
```

The directive body is an ordered definition list. `Inputs` comes first when
the method consumes tensors, followed by `Outputs` and any mode-specific groups
such as `Export outputs` or `Inference outputs`. At least one output group is
required. `Constraints` and `Symbols` follow the output groups when needed.
Each tensor name and shape is a reStructuredText `math` role. Code literals are
not accepted in a shape contract.

Repeated tensors include their range next to the tensor name:

```rst
Inputs
    :math:`\mathrm{inputs}_{i}` (:math:`i = 0, \ldots, N - 1`)
        :math:`(B, C_{i}, H_{i}, W_{i})`
```

An indexed name must always be paired with an index range, and an index range
must always accompany an indexed name.

A tensor that only exists in some configurations is marked with a trailing
`(optional)`, after the index range if there is one:

```rst
Outputs
    :math:`\mathrm{output}`
        :math:`(B, C, H, W)`
    :math:`\mathrm{masks}` (optional)
        :math:`(B, M, H, W)`
```

The marker means the entry may be absent, or present but holding no tensors.
The runtime checks skip such an entry when it is missing instead of reporting
an undocumented structure, and still check its shape when it is present. A
single element of a fixed-length sequence, such as
:math:`\mathrm{outputs}_{0}`, cannot be marked optional.

Use :math:`B` for batch size, :math:`C` for channels, :math:`H` and :math:`W`
for spatial dimensions, and :math:`L` for sequence length. The subscripts
`in`, `out`, `image`, and `skip` describe a dimension's role. A subscript
:math:`i` selects the corresponding tensor in a sequence. These shared symbols
do not need to be repeated in every contract. Define contract-specific symbols,
such as :math:`N`, :math:`n_{\mathrm{classes}}`, or
:math:`\mathrm{patch}`, in the local `Symbols` group. A shape without
parentheses, such as :math:`S`, denotes an arbitrary-rank shape.

Put only meaningful dimension relationships in `Constraints`, one math
expression per bullet. Ordinary tensor dimensions are already required to be
positive by PyTorch and do not need repetitive positivity constraints. Document
shape-related exceptions in the normal Google-style `Raises` section so users
find them with the rest of the callable contract. An entry written as
`If <condition>. The message contains "<substring>".` is machine-readable, and
the tests can execute it against the failing call to check that the documented
error really is raised.

The grammar has a single implementation, `luxonis_train/_shape_contract.py`.
The documentation build and the test suite both go through it, so a contract that
the tests accept is exactly one the documentation build accepts, and there is
no second grammar to keep in sync. The directive validates the body and emits
standard Docutils nodes with semantic `shape-contract`, `shape-inputs`,
`shape-outputs`, `shape-entry`, `shape-constraint`, and `shape-symbol`
classes, which keeps the source readable while letting the tests and the
rendered documentation parse the same structure. A subclass that inherits a
repository-defined `forward` method also inherits its argument and shape
documentation.

The `[tool.pydoctor]` section of `pyproject.toml` names the custom PyDoctor
system class that registers the directive, so any invocation picks it up. For
a quick check that every docstring parses, run PyDoctor with no arguments:

```bash
uv run pydoctor
```

Do not pass `--docformat` on the command line. The package is mid-migration,
so the global docformat must stay Epytext and `luxonis_train.nodes` selects
Google through its own `__docformat__` declaration. Forcing
`--docformat google` reports every Epytext docstring in the rest of the
package as a syntax error.

CI builds the published site through `tools/build_pydoctor_docs.py`, which
adds the project metadata and the versioned output layout. Run it the same way
locally with:

```bash
uv run --group docs python tools/build_pydoctor_docs.py --mode current --output apidocs
```

### Shape Contract Tests

A contract is only useful if it matches what the module really does, so every
contract needs a runtime case under `tests/unittests/nodes/contract_cases/`.
A case constructs the module, feeds it concrete inputs, and records the symbol
bindings, for example the values of :math:`B` and :math:`C_{\mathrm{in}}`. The
tests then run the module, check the real tensors against the documented
shapes, and check that the `Constraints` hold for those bindings. A
mode-specific group such as `Export outputs` needs its own case that runs the
module in that mode.

A completeness test fails when a module in `luxonis_train.nodes` has no case,
so adding a node or a `forward` method means adding a case as well. Other
tests in the same directory check that every class has its own docstring, that
every repository-defined `forward` documents all its arguments and its return
value, and that no Epytext markup is left in the package.

## Type Checking

The codebase is type-checked using [pyright](https://github.com/microsoft/pyright), pinned in the `dev` dependency group to match CI. To run type checking, use the following command in the root project directory:

```bash
uv run pyright --warnings --project pyproject.toml
```

### Editor Support

- **PyCharm** - [Pyright](https://plugins.jetbrains.com/plugin/24145-pyright) extension
- **Visual Studio Code** - [Pyright](https://marketplace.visualstudio.com/items?itemName=ms-pyright.pyright) extension
- **NeoVim** - [LSP-Config](https://github.com/neovim/nvim-lspconfig) plugin with the [pyright configuration](https://github.com/neovim/nvim-lspconfig/blob/master/doc/server_configurations.md#pyright)

## Tests

We use [pytest](https://docs.pytest.org/en/stable/) for testing.
The tests are located in the `tests` directory. To run the tests with coverage, use the following command:

```bash
uv run pytest --cov=luxonis_train --cov-report=html
```

This command will run all tests and generate HTML coverage report.

> [!TIP]
> The coverage report will be saved to `htmlcov` directory.
> If you want to inspect the coverage in more detail, open `htmlcov/index.html` in a browser.

> [!TIP]
> You can choose to run only the unit-tests or only the integration tests by adding `-m unit` or `-m integration` to the `pytest` command.

> [!IMPORTANT]
> If a new feature is added, a new test should be added to cover it.
> The minimum overall test coverage for a PR to be merged is 90%.
> The minimum coverage for new files is 80%.

## GitHub Actions

Our GitHub Actions workflow is run when a new PR is opened.

1. First, the [pre-commit](#pre-commit-hooks) hooks must pass.
1. Next, the [type checking](#type-checking) and the
   [documentation build](#documentation) run in parallel. Both must pass.
1. If all previous checks pass, the [tests](#tests) are run.

> [!TIP]
> Review the GitHub Actions output if your PR fails.

> [!IMPORTANT]
> Successful completion of all the workflow checks is required for merging a PR.

## Making and Submitting Changes

1. Make changes in a new branch with a descriptive prefix such as `feat/` or `fix/`.
1. Test your changes locally.
1. Commit your changes (pre-commit hooks will run).
1. Push your branch and create a pull request.
1. The team will review and merge your PR.
