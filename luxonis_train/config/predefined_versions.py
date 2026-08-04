"""Version-aware lookup for predefined-model classes.

Registry keys follow the pattern ``ClassName:vN`` (e.g.
``DetectionModel:v1``). The prefix before ``:v`` is the "family" and is
what users refer to in `predefined_model.name` in a YAML config, or via
``--model detection:v1`` on the CLI.
"""

import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from luxonis_train.registry import MODELS

if TYPE_CHECKING:
    from luxonis_train.config.config import PredefinedModelConfig
    from luxonis_train.config.predefined_models import BasePredefinedModel


_VERSIONED_KEY = re.compile(r"^(?P<family>.+):v(?P<version>\d+)$")


def _split_family_version(key_or_name: str) -> tuple[str, int | None]:
    """``DetectionModel:v2`` -> ("DetectionModel", 2).

    Bare
    ``DetectionModel`` -> ("DetectionModel", None).
    """
    match = _VERSIONED_KEY.match(key_or_name)
    if match is None:
        return key_or_name, None
    return match.group("family"), int(match.group("version"))


def family_name(name: str) -> str:
    """Strip an optional ``:vN`` suffix off a predefined-model name.

    ``DetectionModel:v1`` and ``DetectionModel`` both name the
    ``DetectionModel`` family, so anything keying behaviour off the
    configured name has to compare against this rather than the raw
    string.
    """
    return _split_family_version(name)[0]


def _plain_key_version(registered: Any) -> int | None:
    version = getattr(registered, "_VERSION", None)
    if isinstance(version, int):
        return version
    return None


def list_versions(family: str) -> dict[int, str]:
    """Enumerate ``{version: registered_key}`` for one family.

    Shipped presets are registered under versioned keys. Custom presets
    loaded after startup may still be registered under their plain class
    name, so expose that plain key as its class' ``_VERSION``.
    """
    versions: dict[int, str] = {}
    for registered_key, registered in MODELS._module_dict.items():
        f, v = _split_family_version(registered_key)
        if f != family:
            continue
        if v is not None:
            versions[v] = registered_key
            continue
        plain_version = _plain_key_version(registered)
        if plain_version is not None:
            versions.setdefault(plain_version, registered_key)
    return dict(sorted(versions.items()))


def _resolve_predefined_key(name: str, version: int | str = "latest") -> str:
    explicit_family, explicit_version = _split_family_version(name)
    versions = list_versions(explicit_family)
    if not versions:
        known_families = sorted(
            {_split_family_version(k)[0] for k in MODELS._module_dict}
        )
        raise ValueError(
            f"No predefined model registered under family "
            f"'{explicit_family}'. Known families: {known_families}."
        )

    if explicit_version is not None:
        if version != "latest" and int(version) != explicit_version:
            raise ValueError(
                f"Explicit class name '{name}' conflicts with "
                f"version={version!r}. Use "
                f"`name: {explicit_family}, version: {version}` "
                "or drop the version arg."
            )
        chosen = explicit_version
    elif version == "latest":
        chosen = max(versions)
    else:
        chosen = int(version)

    if chosen not in versions:
        raise ValueError(
            f"Version {chosen} of predefined model '{explicit_family}' "
            f"is not available. Available versions: {sorted(versions)}."
        )
    return versions[chosen]


def resolve_predefined_class(
    name: str, version: int | str = "latest"
) -> type["BasePredefinedModel"]:
    """Look up a predefined-model class by ``(name, version)``.

    ``name`` may be a family (``DetectionModel``) or an explicit key
    with the ``:vN`` suffix (``DetectionModel:v1``). When the explicit
    form is used, ``version`` must be ``"latest"`` or match the pinned
    version, else a ``ValueError`` is raised.
    """
    return MODELS.get(_resolve_predefined_key(name, version))


def resolved_class_name(name: str, version: int | str = "latest") -> str:
    return _resolve_predefined_key(name, version)


def warn_on_predefined_model_mismatch(
    current: "PredefinedModelConfig | None", ckpt_predefined: Any
) -> None:
    """Log a warning if ``current`` and ``ckpt_predefined`` resolve to
    different concrete predefined-model classes.

    ``ckpt_predefined`` is whatever was stored under the checkpoint's
    ``predefined_model`` key. Missing / not-a-dict is treated as a no-op
    (pre-versioning checkpoints).
    """
    if not isinstance(ckpt_predefined, dict) or current is None:
        return
    if "name" not in ckpt_predefined:
        return
    try:
        current_class = resolved_class_name(current.name, current.version)
    except (KeyError, ValueError):
        # The config itself does not resolve. Config validation reports
        # that with a better message than we could here.
        return
    try:
        ckpt_class = resolved_class_name(
            ckpt_predefined["name"],
            ckpt_predefined.get("version", "latest"),
        )
    except (KeyError, ValueError) as e:
        # The architecture the checkpoint was trained with is gone
        # (renamed family, dropped version). Warn instead of staying
        # silent, otherwise the only symptom is an opaque state-dict
        # load failure later on.
        logger.warning(
            f"The checkpoint was trained with predefined model "
            f"`{ckpt_predefined['name']}` "
            f"(version={ckpt_predefined.get('version', 'latest')}), which "
            f"can no longer be resolved: {e} Loading it into "
            f"`{current_class}` may fail or silently drop weights."
        )
        return
    if ckpt_class != current_class:
        logger.warning(
            f"Predefined model version mismatch: config resolves to "
            f"`{current_class}`, but the checkpoint was trained with "
            f"`{ckpt_class}`. Pin `predefined_model.version` in the "
            "config to reproduce the checkpoint's architecture."
        )
