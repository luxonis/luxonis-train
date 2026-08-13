"""Resolve packaged predefined-model configs."""

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import yaml

if TYPE_CHECKING:
    from luxonis_train.config.predefined_models import BasePredefinedModel

# A generic top-level `configs` package is easily shadowed on `sys.path`.
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
    """A packaged config path and its required overrides."""

    path: Path
    opts: list[str]


def parse_model_spec(model: str) -> tuple[str, str | None]:
    """Split a model name from its optional version suffix."""
    if ":" not in model:
        return model, None
    family, _, version = model.partition(":")
    digits = version[1:]
    if version.startswith("v") and digits.isascii() and digits.isdigit():
        return family, digits
    if version == "latest":
        return family, version
    raise ValueError(
        f"Malformed model spec '{model}'. Expected '<name>', "
        f"'<name>:vN' (e.g. detection:v1), or '<name>:latest'."
    )


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
    """List models and the variants backed by packaged YAMLs."""
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
    """Resolve the class used by a packaged config, if available."""
    if model not in list_predefined_models():
        return None
    try:
        data = yaml.safe_load(default_config_path(model).read_text())
        name = data["model"]["predefined_model"]["name"]
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return None

    # Registering predefined models pulls in torch, so keep this lazy.
    import luxonis_train.config.predefined_models  # noqa: F401
    from luxonis_train.config.predefined_versions import (
        resolve_predefined_class,
    )

    try:
        return resolve_predefined_class(name)
    except (KeyError, ValueError):
        return None


def list_variants(model: str) -> list[str | None]:
    """List every variant selectable for a packaged model."""
    variants = list(list_predefined_models().get(model, []))
    cls = _model_class(model)
    if cls is None:
        return variants
    try:
        _, class_variants = cls.get_variants()
    except NotImplementedError:
        return variants
    for variant in class_variants:
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
    """Resolve a model and variant to a packaged YAML and overrides."""
    model, version = parse_model_spec(model)
    available = list_predefined_models()
    if model not in available:
        raise ValueError(
            f"Unknown predefined model '{model}'. "
            f"Available: {', '.join(available)}."
        )
    file_variants = available[model]
    opts = []
    if version is not None and version != "latest":
        opts = ["model.predefined_model.version", version]

    if variant is None:
        return ResolvedPredefinedConfig(default_config_path(model), opts)
    # The variant is always pinned via an override so that an explicit
    # `--variant` takes precedence over `model.predefined_model.variant`
    # passed in `opts`, whether or not the variant has its own YAML.
    if variant in file_variants:
        return ResolvedPredefinedConfig(
            _config_path(_filename(model, variant)),
            [*opts, "model.predefined_model.variant", variant],
        )
    if variant in list_variants(model):
        return ResolvedPredefinedConfig(
            default_config_path(model),
            [*opts, "model.predefined_model.variant", variant],
        )
    raise ValueError(
        f"Variant '{variant}' is not available for model '{model}'. "
        f"Available variants: {_variant_labels(model)}."
    )
