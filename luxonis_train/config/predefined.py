"""Resolver for packaged predefined-model config YAMLs.

Enables `luxonis_train train --model detection --variant light` (and
equivalents on other commands) by mapping `(model, variant)` to a YAML
shipped in the `configs` package.
"""

from importlib.resources import files
from pathlib import Path

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


def _iter_config_files() -> list[str]:
    root = files("configs")
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


def list_predefined_models() -> dict[str, list[str | None]]:
    """Enumerate `(model, [variants])` pairs from packaged YAMLs.

    A variant of `None` means the model has a single unvarianted config.
    """
    result: dict[str, list[str | None]] = {}
    for filename in _iter_config_files():
        model, variant = _parse(filename)
        result.setdefault(model, []).append(variant)
    for variants in result.values():
        variants.sort(
            key=lambda v: (
                v is None,
                VARIANT_ORDER.index(v) if v in VARIANT_ORDER else 99,
            )
        )
    return dict(sorted(result.items()))


def _resolve_filename(model: str, variant: str | None) -> str:
    available = list_predefined_models()
    if model not in available:
        raise ValueError(
            f"Unknown predefined model '{model}'. "
            f"Available: {', '.join(available)}."
        )
    variants = available[model]
    if variant is not None:
        if variant not in variants:
            variant_labels = [
                v if v is not None else "<default>" for v in variants
            ]
            raise ValueError(
                f"Variant '{variant}' is not available for model '{model}'. "
                f"Available variants: {', '.join(variant_labels)}."
            )
        chosen = variant
    else:
        chosen = variants[0]
    return (
        f"{model}_{chosen}{_SUFFIX}"
        if chosen is not None
        else f"{model}{_SUFFIX}"
    )


def resolve_predefined_config(model: str, variant: str | None) -> Path:
    """Return the on-disk path to the packaged YAML for a `(model,
    variant)`.

    Raises `ValueError` with a listing of available options on miss.
    """
    filename = _resolve_filename(model, variant)
    return Path(str(files("configs") / filename))
