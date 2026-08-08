# Repository Guidelines

## Scope and Sources of Truth

Work in the current checkout and on the current branch unless the user
explicitly asks for another branch or worktree. `CONTRIBUTING.md` is the
reference for the development workflow: environment setup, hooks, type
checking, tests, and the full shape contract grammar. Read it, along with
`pyproject.toml` and the relevant GitHub Actions workflow, before changing
tooling. This file only carries the rules that are specific to agents or that
`CONTRIBUTING.md` does not state. Repository-specific instructions take
precedence over the workspace-level `CLAUDE.md`, whose pip and pre-commit
guidance predates this repository's uv migration.

Do not create a pull request, draft or otherwise, unless the user asks.
Preserve unrelated dirty and untracked work. Do not add
`from __future__ import annotations`. Do not use em dashes in code, comments,
documentation, commit messages, or user-facing text.

## Environment and Commands

Use uv as the command surface. Run tools through `uv run`, never as bare
`pytest`, `ruff`, or `pyright`. The lock file and the dependency declarations
in `pyproject.toml` are authoritative.

- Sync the environment the way CI does, with
  `uv sync --locked --extra aimet --group dev`.
- AIMET is an optional package extra, but changes must preserve compatibility
  with it by default. Do not disable or bypass AIMET support to simplify a
  change, and exercise the AIMET path when a change affects models, blocks,
  export, quantization, Torch integration, or dependency resolution.
- Validate the lock with `uv lock --check`, and build release artifacts with
  `uv build`.
- If the sandbox cannot write to the uv or tool caches, point them at a
  task-specific writable directory under `/tmp`.

## Documentation in `luxonis_train.nodes`

The project is mid-migration between docstring formats, and the two halves
must not be mixed up:

- `luxonis_train.nodes` uses Google-style docstrings. The format is declared
  by `__docformat__ = "google"` in `luxonis_train/nodes/__init__.py`.
- The rest of `luxonis_train` is still Epytext.
- The documentation build runs over the whole package with a global
  `--docformat=epytext`, so that package-level `__docformat__` declaration is
  the only thing that makes the nodes package parse as Google style. Do not
  remove it, and do not convert modules outside `luxonis_train.nodes` to
  Google style as a side effect of an unrelated change.

Inside `luxonis_train.nodes` the following are mandatory:

- Every class derived from `torch.nn.Module` has its own class docstring. An
  inherited docstring does not count.
- Every repository-defined `forward` method has Google-style `Args` and
  `Returns` sections, followed by a machine-readable `.. shape-contract::`
  block. `CONTRIBUTING.md` describes the grammar and
  `luxonis_train/_shape_contract.py` implements it; that module is the final
  authority on what is accepted.
- `Args` entries do not repeat the type from the signature. Write
  `x: Feature map to project.`, not `x (Tensor): Feature map to project.`.
- Every contract has a runtime case registered under
  `tests/unittests/nodes/contract_cases/`, which runs the module and checks
  the real tensors against the documented shapes. Adding a node or a `forward`
  method without a case fails the test suite, and so does a contract whose
  shapes do not match what the module actually returns.

## Code and Tests

Match nearby patterns and keep changes narrowly scoped. Ruff enforces a
79-character line length and `E501` is not in the ignore list, so the limit
applies inside docstrings, comments, and shape contracts too. The formatter
does not rewrap prose, so overlong documentation lines have to be split by
hand.

Add focused regression tests for behavior changes. Use the existing pytest
markers, such as `unit`, `predefined_light`, `predefined_heavy`,
`combinations`, and `misc`, when selecting suites. Set `trainer.n_workers` to
0 in small synthetic training tests so local and sandboxed runs do not stall
on data-loader worker processes.

## Validation

Before presenting final results, run every locally applicable check and make
sure everything you ran passes. At minimum, run the focused tests for the
changed behavior, the full prek suite, Pyright, and the documentation build
when the change touches docstrings. For repository-wide or dependency
changes, also run the config test and the test suites that do not require
unavailable credentials or GPU resources. CI replaces the locked LuxonisML
release with the requested source ref before Pyright and the tests, so
reproduce that override when validating cross-repository changes. Report any
external blocker clearly instead of presenting partial validation as
complete.
