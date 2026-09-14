"""Look up the config files of the predefined models in the package.

The ``--model`` and ``--variant`` options of the ``luxonis_train``
commands, and the ``model`` and ``variant`` arguments of `LuxonisModel`,
go through `resolve_predefined_config`. A packaged file
``<model>_<variant>_model.yaml`` or ``<model>_model.yaml`` in
`luxonis_train.configs` defines a model name and its variants.

"""

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import yaml

if TYPE_CHECKING:
    from luxonis_train.config.predefined_models import BasePredefinedModel

# A generic top-level `configs` package is easily shadowed on `sys.path`.
CONFIGS_PACKAGE = "luxonis_train.configs"

# The variant names that a file name can end with, in sort order. Other
# variant names sort after them, and `None` sorts last.
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
    """A packaged config file and the overrides that go with it.

    Attributes:
        path (``Path``): The path of the packaged YAML file.
        opts (list[str]): Config overrides as alternating keys and
            values, such as ``model.predefined_model.variant`` followed
            by ``medium``. The list is empty when the file needs no
            override.

    """

    path: Path
    opts: list[str]


def parse_model_spec(model: str) -> tuple[str, str | None]:
    """Split a model name from its optional version suffix.

    The suffix follows a ``:``. It is ``v`` with ASCII digits, or
    ``latest``.

    Args:
        model (str): The model name, as ``"<name>"``,
            ``"<name>:v<N>"``, or ``"<name>:latest"``.

    Returns:
        tuple[str, str | None]: The name, and the version. The
        version is the digits without ``v``, ``"latest"``, or ``None``
        when ``model`` has no ``:``.

    Raises:
        ValueError: When the text after ``:`` is neither ``v`` with
            digits nor ``latest``.

    Example:
        >>> parse_model_spec("detection:v2")
        ('detection', '2')
        >>> parse_model_spec("detection")
        ('detection', None)

    """
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
    """Return the directory of the packaged config files.

    Returns:
        ``Path``: The directory of the `luxonis_train.configs` package.

    Example:
        >>> configs_dir().name
        'configs'

    """
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
            VARIANT_ORDER.index(v)
            if v in VARIANT_ORDER
            else len(VARIANT_ORDER),
        )
    )
    return variants


def list_predefined_models() -> dict[str, list[str | None]]:
    """List the packaged models and the variants that have a file.

    The function reads the names of the files in `configs_dir` that end
    with ``_model.yaml``. It skips ``complex_model.yaml`` and the other
    example files. A name that ends with ``_<variant>_model.yaml``, for a
    variant in `VARIANT_ORDER`, gives the model and that variant. Any
    other name ``<model>_model.yaml`` gives the model and the variant
    ``None``. The list does not hold the variants that only the model
    class declares. `list_variants` adds them.

    Returns:
        dict[str, list[str | None]]: Each model name mapped to its
        variants, in the order of `VARIANT_ORDER`. The first variant is
        the default. The models are in alphabetical order.

    Example:
        >>> models = list_predefined_models()
        >>> models["detection"], models["embeddings"]
        (['light', 'heavy'], [None])

    """
    result: dict[str, list[str | None]] = {}
    for filename in _iter_config_files():
        model, variant = _parse(filename)
        result.setdefault(model, []).append(variant)
    for variants in result.values():
        _sort_variants(variants)
    return dict(sorted(result.items()))


def _default_variant(model: str) -> str | None:
    """Return the variant that applies when ``--variant`` is omitted.

    Args:
        model (str): The model name, without a version suffix.

    Returns:
        str | None: The first variant of the model in
        `list_predefined_models`.

    Raises:
        KeyError: When ``model`` is not a packaged model.

    """
    return list_predefined_models()[model][0]


def default_config_path(model: str) -> Path:
    """Return the config file of the default variant of a model.

    The default variant is the first variant that
    `list_predefined_models` gives for ``model``.

    Args:
        model (str): The model name, without a version suffix.

    Returns:
        ``Path``: The path of the packaged YAML file.

    Raises:
        KeyError: When ``model`` is not a packaged model.

    Example:
        >>> default_config_path("detection").name
        'detection_light_model.yaml'

    """
    return _config_path(_filename(model, _default_variant(model)))


def class_family(model: str) -> str | None:
    """Return the class name in the default config file of a model.

    The function reads ``model.predefined_model.name`` from the file of
    `default_config_path`.

    Args:
        model (str): The model name, without a version suffix.

    Returns:
        str | None: The value of ``model.predefined_model.name``.
        ``None`` when ``model`` is not a packaged model. Also ``None``
        when the function cannot read the file, when the file is not
        valid YAML, or when the file has no such key.

    Example:
        >>> class_family("keypoint_bbox")
        'KeypointDetectionModel'
        >>> print(class_family("unknown"))
        None

    """
    try:
        data = yaml.safe_load(default_config_path(model).read_text())
        return data["model"]["predefined_model"]["name"]
    except (OSError, KeyError, TypeError, yaml.YAMLError):
        return None


def _model_class(model: str) -> "type[BasePredefinedModel] | None":
    """Resolve the latest version of the class of a packaged model.

    Args:
        model (str): The model name, without a version suffix.

    Returns:
        ``type[BasePredefinedModel] | None``: The class that
        `class_family` names. ``None`` when ``model`` is not a packaged
        model, or when the class name is missing or not registered.

    """
    if model not in list_predefined_models():
        return None
    name = class_family(model)
    if name is None:
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
    """List every variant that a packaged model accepts.

    The list holds the variants of `list_predefined_models`, and the
    variants that ``get_variants`` of the model class declares. The
    model class comes from `class_family`, in its latest registered
    version. The function imports the predefined models to find it.
    When the function cannot resolve the class, or the class has no
    variants, the list holds only the variants with a file.

    Args:
        model (str): The model name, without a version suffix.

    Returns:
        list[str | None]: The variants in the order of `VARIANT_ORDER`.
        ``None`` stands for a file without a variant name. The list is
        empty when ``model`` is not a packaged model.

    Example:
        >>> list_variants("detection")
        ['light', 'medium', 'heavy']
        >>> list_variants("anomaly_detection")
        ['light', 'heavy', None]

    """
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
    """Find the config file and the overrides for a model and a variant.

    The function selects the file and the overrides as follows:

    - A version other than ``latest`` becomes the override
      ``model.predefined_model.version``.
    - Without ``variant``, the file is the default config file of the
      model, as in `default_config_path`.
    - A variant with its own file selects that file.
    - A variant that only the model class declares selects the default
      config file.

    In the last two cases, ``variant`` also becomes the override
    ``model.predefined_model.variant``, so it replaces the variant that
    the file sets.

    Args:
        model (str): The model name, with an optional version suffix,
            as in `parse_model_spec`.
        variant (str | None): The variant. ``None`` selects the default
            config file and adds no variant override.

    Returns:
        ResolvedPredefinedConfig: The path of the file and the
        overrides.

    Raises:
        ValueError: When the version suffix is malformed, when ``model``
            is not a packaged model, or when ``variant`` is not in
            `list_variants`.

    Example:
        >>> path, opts = resolve_predefined_config("detection:v1", "medium")
        >>> path.name
        'detection_light_model.yaml'
        >>> opts
        ['model.predefined_model.version', '1',
         'model.predefined_model.variant', 'medium']

    """
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
