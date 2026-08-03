"""Resolver for packaged predefined-model config YAMLs.

Enables `luxonis_train train --model detection --variant light` (and
equivalents on other commands) by mapping `(model, variant)` to a YAML
shipped in the `luxonis_train.configs` package.
"""

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import yaml

if TYPE_CHECKING:
    from luxonis_train.config.predefined_models import BasePredefinedModel

# Anchored inside `luxonis_train` on purpose: a top-level `configs`
# package would be shadowed by any other `configs` directory on
# `sys.path` (a user's own project layout, most commonly) and would
# clobber files with any other distribution shipping that name.
CONFIGS_PACKAGE = "luxonis_train.configs"

VARIANT_ORDER = ("light", "medium", "heavy")
_EXCLUDED = frozenset(
    {
        "defaults.yaml",
        "complex_model.yaml",
        "example_export.yaml",
        "example_tuning.yaml",
    }
)
_SUFFIX = "_model.yaml"


class ResolvedPredefinedConfig(NamedTuple):
    """A packaged config together with the overrides needed to realize
    the request.

    `opts` is non-empty when the requested variant has no dedicated YAML
    and has to be selected on the predefined model instead.
    """

    path: Path
    opts: list[str]


def configs_dir() -> Path:
    """Return the directory holding the packaged preset YAMLs."""
    return Path(str(files(CONFIGS_PACKAGE)))


def _config_path(filename: str) -> Path:
    return configs_dir() / filename


def _iter_config_files() -> list[str]:
    root = files(CONFIGS_PACKAGE)
    return sorted(
        f.name
        for f in root.iterdir()
        if f.is_file() and f.name.endswith(_SUFFIX) and f.name not in _EXCLUDED
    )


def _parse(filename: str) -> tuple[str, str | None]:
    stem = filename[: -len(_SUFFIX)]
    for variant in VARIANT_ORDER:
        token = f"_{variant}"
        if stem.endswith(token):
            return stem[: -len(token)], variant
    return stem, None


def _filename(model: str, variant: str | None) -> str:
    return (
        f"{model}_{variant}{_SUFFIX}"
        if variant is not None
        else f"{model}{_SUFFIX}"
    )


def _sort_variants(variants: list[str | None]) -> list[str | None]:
    variants.sort(
        key=lambda v: (
            v is None,
            VARIANT_ORDER.index(v) if v in VARIANT_ORDER else 99,
        )
    )
    return variants


def list_predefined_models() -> dict[str, list[str | None]]:
    """Enumerate `(model, [variants])` pairs from packaged YAMLs.

    A variant of `None` means the model has a single unvarianted config.
    Only variants backed by a dedicated YAML are listed; see
    `list_variants` for everything the model's class can be asked for.
    """
    result: dict[str, list[str | None]] = {}
    for filename in _iter_config_files():
        model, variant = _parse(filename)
        result.setdefault(model, []).append(variant)
    for variants in result.values():
        _sort_variants(variants)
    return dict(sorted(result.items()))


def _default_variant(model: str) -> str | None:
    """Return the variant used when `--variant` is omitted."""
    return list_predefined_models()[model][0]


def default_config_path(model: str) -> Path:
    """Path to the YAML backing `model`'s default variant."""
    return _config_path(_filename(model, _default_variant(model)))


def _model_class(model: str) -> "type[BasePredefinedModel] | None":
    """Resolve the predefined-model class named by `model`'s default
    YAML.

    Returns `None` when the config does not use a predefined model or
    the class cannot be resolved; callers fall back to the filename-
    derived variants in that case.
    """
    if model not in list_predefined_models():
        return None
    try:
        data = yaml.safe_load(default_config_path(model).read_text())
        name = data["model"]["predefined_model"]["name"]
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return None

    # Imported lazily: this module is imported when the CLI starts,
    # while registering the predefined models pulls in torch.
    import luxonis_train.config.predefined_models  # noqa: F401
    from luxonis_train.config.predefined_versions import (
        resolve_predefined_class,
    )

    try:
        return resolve_predefined_class(name)
    except (KeyError, ValueError):
        return None


def list_variants(model: str) -> list[str | None]:
    """Every variant selectable for `model`.

    The packaged YAMLs only cover a couple of variants per model, but
    the backing predefined-model class usually declares more (e.g.
    `DetectionModel` has `medium` with no YAML of its own). All of them
    are reachable from `--variant`, so all of them are listed here.
    """
    variants = list(list_predefined_models().get(model, []))
    cls = _model_class(model)
    if cls is None:
        return variants
    try:
        default_variant, class_variants = cls.get_variants()
    except NotImplementedError:
        return variants
    for variant in class_variants:
        # A single unvarianted YAML already stands for the class default.
        if variant == default_variant and None in variants:
            continue
        if variant not in variants:
            variants.append(variant)
    return _sort_variants(variants)


def _variant_labels(model: str) -> str:
    return ", ".join(
        v if v is not None else "<default>" for v in list_variants(model)
    )


def resolve_predefined_config(
    model: str, variant: str | None
) -> ResolvedPredefinedConfig:
    """Resolve a `(model, variant)` pair to a packaged YAML.

    Variants without a dedicated YAML are served by the model's default
    config plus a `model.predefined_model.variant` override, so the
    `--model`/`--variant` form covers exactly the same variants as
    writing the config by hand.

    Raises `ValueError` with a listing of available options on miss.
    """
    available = list_predefined_models()
    if model not in available:
        raise ValueError(
            f"Unknown predefined model '{model}'. "
            f"Available: {', '.join(available)}."
        )
    file_variants = available[model]

    if variant is None:
        return ResolvedPredefinedConfig(default_config_path(model), [])
    if variant in file_variants:
        return ResolvedPredefinedConfig(
            _config_path(_filename(model, variant)), []
        )
    if variant in list_variants(model):
        return ResolvedPredefinedConfig(
            default_config_path(model),
            ["model.predefined_model.variant", variant],
        )
    raise ValueError(
        f"Variant '{variant}' is not available for model '{model}'. "
        f"Available variants: {_variant_labels(model)}."
    )
