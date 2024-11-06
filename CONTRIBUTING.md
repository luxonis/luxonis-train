# Contributing to LuxonisTrain

**This guide is intended for our internal development team.**
It outlines our workflow and standards for contributing to this project.

## Table Of Contents

- [Pre-requisites](#pre-requisites)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Documentation](#documentation)
- [Type Checking](#type-checking)
  - [Editor Support](#editor-support)
- [Tests](#tests)
- [GitHub Actions](#github-actions)
- [Making and Reviewing Changes](#making-and-reviewing-changes)

## Pre-requisites

Clone the repository and navigate to the root directory:

```bash
git clone git@github.com:luxonis/luxonis-train.git
cd luxonis-train
```

Install the development dependencies by running `pip install -r requirements-dev.txt` or install the package with the `dev` extra flag:

```bash
pip install -e .[dev]
```

> \[!NOTE\]
> This will install the package in editable mode (`-e`),
> so you can make changes to the code and run them immediately.

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality and consistency:

1. Install `pre-commit` (see [pre-commit.com](https://pre-commit.com/#install)).
1. Clone the repository and run `pre-commit install` in the root directory.
1. The `pre-commit` hook will now run automatically on `git commit`.
   - If the hook fails, it will print an error message and abort the commit.
   - Some hooks will also modify the files in-place to fix found issues.

## Documentation

We use the [Epytext](https://epydoc.sourceforge.net/epytext.html) markup language for documentation.
To verify that your documentation is formatted correctly, run the following command:

```bash
pydoctor --docformat=epytext luxonis_train
```

**Editor Support:**

- **PyCharm** - built in support for generating `epytext` docstrings
- **Visual Studio Code** - [AI Docify](https://marketplace.visualstudio.com/items?itemName=AIC.docify) extension offers support for `epytext`
- **NeoVim** - [vim-python-docstring](https://github.com/pixelneo/vim-python-docstring) supports `epytext` style

## Type Checking

The codebase is type-checked using [pyright](https://github.com/microsoft/pyright) `v1.1.380`. To run type checking, use the following command in the root project directory:

```bash
pyright --warnings --level warning --pythonversion 3.10 luxonis_train
```

**Editor Support:**

- **PyCharm** - [Pyright](https://plugins.jetbrains.com/plugin/24145-pyright) extension
- **Visual Studio Code** - [Pyright](https://marketplace.visualstudio.com/items?itemName=ms-pyright.pyright) extension
- **NeoVim** - [LSP-Config](https://github.com/neovim/nvim-lspconfig) plugin with the [pyright configuration](https://github.com/neovim/nvim-lspconfig/blob/master/doc/server_configurations.md#pyright)

## Tests

We use [pytest](https://docs.pytest.org/en/stable/) for testing.
The tests are located in the `tests` directory. To run the tests with coverage, use the following command:

```bash
pytest --cov=luxonis_train --cov-report=html
```

This command will run all tests and generate HTML coverage report.

> \[!TIP\]
> The coverage report will be saved to `htmlcov` directory.
> If you want to inspect the coverage in more detail, open `htmlcov/index.html` in a browser.

> \[!TIP\]
> You can choose to run only the unit-tests or only the integration tests by adding `-m unit` or `-m integration` to the `pytest` command.

> \[!IMPORTANT\]
> If a new feature is added, a new test should be added to cover it.
> The minimum overall test coverage for a PR to be merged is 90%.
> The minimum coverage for new files is 80%.

## GitHub Actions

Our GitHub Actions workflow is run when a new PR is opened.

1. First, the [pre-commit](#pre-commit-hooks) hooks must pass and the [documentation](#documentation) must be built successfully.
1. Next, the [type checking](#type-checking) is run.
1. If all previous checks pass, the [tests](#tests) are run.

> \[!TIP\]
> Review the GitHub Actions output if your PR fails.

> \[!IMPORTANT\]
> Successful completion of all the workflow checks is required for merging a PR.

## Making and Submitting Changes

1. Make changes in a new branch.
1. Test your changes locally.
1. Commit your changes (pre-commit hooks will run).
1. Push your branch and create a pull request.
1. The team will review and merge your PR.
