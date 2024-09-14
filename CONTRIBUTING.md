# Contributing to LuxonisTrain<a name="contributing-to-luxonistrain"></a>

**This guide is intended for our internal development team.**
It outlines our workflow and standards for contributing to this project.

## Table Of Contents

- [Requirements](#requirements)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Documentation](#documentation)
- [Type Checking](#type-checking)
  - [Editor Support](#editor-support)
- [Tests](#tests)
- [GitHub Actions](#github-actions)
- [Making and Reviewing Changes](#making-and-reviewing-changes)

## Requirements

Install the development dependencies by running `pip install -r requirements-dev.txt` or installing the package with the `dev` extra:

```bash
pip install -e .[dev]
```

> \[!NOTE\]
> This will install the package in editable mode (`-e`),
> so you can make changes to the code and run them immediately.

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality and consistency:

1. Install pre-commit (see [pre-commit.com](https://pre-commit.com/#install)).
1. Clone the repository and run `pre-commit install` in the root directory.
1. The pre-commit hook will now run automatically on `git commit`.
   - If the hook fails, it will print an error message and abort the commit.
   - It will also modify the files in-place to fix any issues it can.

## Documentation

We use the [Epytext](https://epydoc.sourceforge.net/epytext.html) markup language for documentation.
To verify that your documentation is formatted correctly, follow these steps:

1. Download [`get-docs.py`](https://github.com/luxonis/python-api-analyzer-to-json/blob/main/gen-docs.py) script
1. Run `python3 get-docs.py luxonis_ml` in the root directory.
   - If the script runs successfully and produces `docs.json` file, your documentation is formatted correctly.
   - **NOTE:** If the script fails, it might not give the specific error message. In that case, you can run
     the script for each file individually until you find the one that is causing the error.

## Type Checking

The codebase is type-checked using [pyright](https://github.com/microsoft/pyright) `v1.1.380`. To run type checking, use the following command in the root project directory:

```bash
pyright --warnings --level warning --pythonversion 3.10 luxonis_train
```

### Editor Support

- **PyCharm** - built in support for generating `epytext` docstrings
- **Visual Studie Code** - [AI Docify](https://marketplace.visualstudio.com/items?itemName=AIC.docify) extension offers support for `epytext`
- **NeoVim** - [vim-python-docstring](https://github.com/pixelneo/vim-python-docstring) supports `epytext` style

## Tests

We use [pytest](https://docs.pytest.org/en/stable/) for testing.
The tests are located in the `tests` directory. You can run the tests locally by running `pytest` in the root directory.

This command will run all tests generate HTML coverage report.

> \[!TIP\]
> The coverage report will be saved to `htmlcov` directory.
> If you want to inspect the coverage in more detail, open `htmlcov/index.html` in a browser.

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
> Successfull completion of all the workflow checks is required for merging a PR.

## Making and Submitting Changes

1. Make changes in a new branch.
1. Test your changes locally.
1. Commit (pre-commit hook will run).
1. Push to your branch and create a pull request.
1. The team will review and merge your PR.
