"""Version-aware lookup for predefined-model classes."""

import re
from typing import TYPE_CHECKING, Any

from loguru import logger

from luxonis_train.registry import MODELS

if TYPE_CHECKING:
    from luxonis_train.config.config import PredefinedModelConfig
    from luxonis_train.config.predefined_models import BasePredefinedModel


_VERSIONED_KEY = re.compile(
    r"^(?P<family>.+):(?:v(?P<version>\d+)|latest)$", re.ASCII
)


def _split_family_version(key_or_name: str) -> tuple[str, int | None]:
    """Split the optional ``:vN`` or ``:latest`` suffix from a registry
    key.
    """
    match = _VERSIONED_KEY.match(key_or_name)
    if match is None:
        return key_or_name, None
    version = match.group("version")
    return match.group("family"), int(version) if version else None


def family_name(name: str) -> str:
    """Strip an optional ``:vN`` or ``:latest`` suffix from a model
    name.
    """
    return _split_family_version(name)[0]


def _plain_key_version(registered: Any) -> int | None:
    version = getattr(registered, "_VERSION", None)
    return version if isinstance(version, int) else None


def list_versions(family: str) -> dict[int, str]:
    """Map available versions in a family to their registry keys."""
    versions: dict[int, str] = {}
    fallback: dict[int, str] = {}
    for registered_key, registered in MODELS._module_dict.items():
        f, v = _split_family_version(registered_key)
        if f != family:
            continue
        if v is not None:
            versions[v] = registered_key
        elif registered_key == family:
            # Classes registered manually under a bare name.
            plain_version = _plain_key_version(registered)
            if plain_version is not None:
                fallback[plain_version] = registered_key
    for version, registered_key in fallback.items():
        versions.setdefault(version, registered_key)
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
    """Look up a predefined-model class by name and version."""
    return MODELS.get(_resolve_predefined_key(name, version))


def resolved_class_name(name: str, version: int | str = "latest") -> str:
    return _resolve_predefined_key(name, version)


def warn_on_predefined_model_mismatch(
    current: "PredefinedModelConfig | None", ckpt_predefined: Any
) -> None:
    """Warn when config and checkpoint resolve to different classes."""
    if not isinstance(ckpt_predefined, dict) or current is None:
        return
    if "name" not in ckpt_predefined:
        return
    try:
        current_class = resolved_class_name(current.name, current.version)
    except (KeyError, ValueError):
        # Config validation reports this with better context.
        return
    try:
        ckpt_class = resolved_class_name(
            ckpt_predefined["name"],
            ckpt_predefined.get("version", "latest"),
        )
    except (KeyError, ValueError) as e:
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
