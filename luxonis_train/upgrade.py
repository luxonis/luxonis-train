"""Upgrade of an old config and of the installed package.

`upgrade_config` migrates a config from an older release to the schema
of the installed release. `Config.get_config
<luxonis_train.config.Config.get_config>` calls it for each config file
or dictionary that it loads. The ``luxonis_train upgrade config``
command writes the result to a file. `upgrade_installation` upgrades the
package with ``pip``.

"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import EllipsisType
from typing import Any

import yaml
from loguru import logger
from luxonis_ml.typing import Params, ParamValue, PathType
from semver import Version

import luxonis_train as lxt


@dataclass
class NestedDict:
    """A wrapper that addresses a nested dictionary with dotted keys.

    The key ``"trainer.optimizer.name"`` stands for
    ``config["trainer"]["optimizer"]["name"]``. A read of a missing key
    returns ``None``, and a write creates the missing dictionaries.
    The wrapper does not copy the dictionary, so each change goes into
    the wrapped dictionary.

    Attributes:
        _dict (``dict[str, Any]``): The wrapped dictionary. The
            constructor takes it as its only argument.

    Example:
        >>> from luxonis_train.upgrade import NestedDict
        >>> config = {"trainer": {"epochs": 10}}
        >>> cfg = NestedDict(config)
        >>> cfg["trainer.epochs"], cfg["trainer.batch_size"]
        (10, None)
        >>> cfg["model.name"] = "detector"
        >>> config
        {'trainer': {'epochs': 10}, 'model': {'name': 'detector'}}

    """

    _dict: dict[str, Any]

    def __contains__(self, key: str) -> bool:
        """Check if a dotted key exists.

        Args:
            key (str): A dotted key, for example ``"trainer.epochs"``.

        Returns:
            bool: ``True`` when each part of ``key`` exists and each
            value before the last part is a dictionary. A stored
            ``None`` counts as present.

        """
        keys = key.split(".")
        current = self._dict
        for k in keys:
            if not isinstance(current, dict):
                return False
            if k not in current:
                return False
            current = current[k]
        return True

    def __getitem__(self, key: str) -> Any:
        """Return the value under a dotted key.

        Args:
            key (str): A dotted key, for example ``"trainer.epochs"``.

        Returns:
            ``Any``: The value, or ``None`` when ``key`` does not exist.

        """
        if key not in self:
            return None
        keys = key.split(".")
        current = self._dict
        for k in keys:
            current = current[k]
        return current

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the value under a dotted key.

        The method creates each missing dictionary on the path. It
        replaces a value on the path that is not a dictionary with an
        empty dictionary.

        Args:
            key (str): A dotted key, for example ``"trainer.epochs"``.
            value (``Any``): The new value.

        """
        keys = key.split(".")
        current = self._dict
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value under a dotted key, or a default value.

        Args:
            key (str): A dotted key, for example ``"model.nodes"``.
            default (``Any``): The value to return when ``key`` does
                not exist.

        Returns:
            ``Any``: The value under ``key``, or ``default`` when ``key``
            does not exist. A stored ``None`` gives ``None``.

        """
        if key not in self:
            return default
        return self[key]

    def pop(self, key: str, default: Any = ...) -> Any:
        """Remove a dotted key and return its value.

        The method removes only the last part of ``key``. The parent
        dictionaries stay, also when they become empty.

        Args:
            key (str): A dotted key, for example
                ``"exporter.output_names"``.
            default (``Any``): The value to return when ``key`` does
                not exist. The default ``...`` means that there is no
                default value.

        Returns:
            ``Any``: The removed value, or ``default`` when ``key`` does
            not exist.

        Raises:
            KeyError: When ``key`` does not exist and ``default`` is
                ``...``.

        Example:
            >>> from luxonis_train.upgrade import NestedDict
            >>> config = {"exporter": {"output_names": ["boxes"]}}
            >>> cfg = NestedDict(config)
            >>> cfg.pop("exporter.output_names")
            ['boxes']
            >>> config
            {'exporter': {}}
            >>> cfg.pop("exporter.output_names", None) is None
            True

        """
        if key not in self:
            if default is not ...:
                return default
            raise KeyError(f"Key '{key}' not found in config.")
        keys = key.split(".")
        current = self._dict
        for k in keys[:-1]:
            current = current[k]
        return current.pop(keys[-1], default)

    def update(self, key: str, value: Any) -> None:
        """Set the value under a dotted key and log the change.

        The method logs a message at the ``INFO`` level. For a missing
        ``key``, the message says that the field is new. Otherwise, it
        shows the old value and the new value. Then the method sets the
        value as ``self[key] = value`` does.

        Args:
            key (str): A dotted key, for example ``"version"``.
            value (``Any``): The new value.

        """
        old_value = self[key]

        if key not in self:
            logger.info(f"Creating new field '{key}' with value `{value}`")
        else:
            logger.info(f"Updating field '{key}': `{old_value}` -> `{value}`")

        self[key] = value

    def replace(
        self,
        old_key: str,
        new_key: str,
        value: ParamValue | EllipsisType | None = ...,
    ) -> None:
        """Move a value to a new dotted key and log the move.

        The method does nothing when ``old_key`` does not exist.
        Otherwise, it removes ``old_key`` as `pop` does and sets
        ``new_key`` as ``self[new_key] = value`` does. Then it calls
        `log_change`.

        Args:
            old_key (str): The dotted key to remove.
            new_key (str): The dotted key to set.
            value (``ParamValue | EllipsisType | None``): The value for
                ``new_key``. The default ``...`` keeps the old value.
                Any other value, ``None`` included, replaces it.

        """
        if old_key not in self:
            return
        old_value = self.pop(old_key)
        keys = new_key.split(".")
        current = self._dict
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value if value is not ... else old_value
        self.log_change(old_key, new_key)

    @staticmethod
    def log_change(old_field: str, new_field: str) -> None:
        """Log at the ``INFO`` level that a config field has a new key.

        Args:
            old_field (str): The old dotted key.
            new_field (str): The new dotted key.

        """
        logger.info(f"Changed config field '{old_field}' to '{new_field}'")


def upgrade_config(config: PathType | Params) -> Params:
    """Migrate a config to the schema of the installed release.

    The function reads a file as JSON when its suffix is ``.json``, and
    as YAML otherwise. It does not write the file back. It does not copy
    a dictionary: it changes the dictionary in place and returns it.

    A config without ``version`` counts as version ``0.3.0``. The
    function always removes the deprecated ``config_version`` field.
    When ``version`` is the installed version or newer, the function
    makes no other change. Otherwise, it does these steps:

    - It renames these fields:

      - ``trainer.use_rich_progress_bar`` to ``rich_logging``.
      - ``preprocessing.train_rgb`` to ``preprocessing.color_space``.
        A true value gives ``"RGB"``, and a false value gives
        ``"BGR"``.
      - ``model.predefined_model.params.variant`` to
        ``model.predefined_model.variant``.
      - ``tuner.storage.storage_type`` to ``tuner.storage.backend``.
        ``"local"`` gives ``"sqlite"``, and any other value gives
        ``"postgresql"``.

    - It removes a ``tuner`` field with the value ``None``.
    - In each node of ``model.nodes``, it moves ``params.variant`` to
      ``variant``. For a ``FOMOHead``, it renames
      ``params.num_conv_layers`` to ``params.n_conv_layers``. It removes
      ``params.download_weights``, and when that value is true, it sets
      ``params.weights`` to ``"download"``.
    - It moves ``exporter.output_names`` to the
      ``params.export_output_names`` of the head, when the model has
      exactly one head. Otherwise, it logs an error and drops the
      names.
    - It moves each module of ``model.losses``, ``model.metrics``, and
      ``model.visualizers`` to the head that the ``attached_to`` field
      of the module names. The module goes into the ``losses``,
      ``metrics``, or ``visualizers`` list of the head, without
      ``attached_to``.
    - It sets ``version`` to the installed version.

    A node is a head when its ``name`` contains ``"Head"``. The
    function finds a head by its ``alias``, or by its ``name`` when the
    head has no alias.

    The function logs a message at the ``INFO`` level when it finds
    ``config_version``. It also logs one when the config is already
    current, and one when the upgrade starts. At the same level, it logs
    each renamed field, each moved ``params.variant``, each moved
    module, and the new ``version``. The removal of ``tuner``, the
    change of ``params.download_weights``, and the move of
    ``exporter.output_names`` have no ``INFO`` message.

    Args:
        config (``PathType | Params``): The path of a local YAML or JSON
            config file, or the config as a dictionary.

    Returns:
        ``Params``: The migrated config.

    Raises:
        ValueError: When a module in ``model.losses``,
            ``model.metrics``, or ``model.visualizers`` has no
            ``attached_to`` field, or when ``attached_to`` does not name
            a head.

    """
    cfg = _load_config(config)

    old_version = Version.parse(cfg.get("version", "0.3.0"))
    if "config_version" in cfg:
        logger.info("Found deprecated field 'config_version' in config.")
        cfg.pop("config_version")
    if old_version >= lxt.__semver__:
        logger.info(
            f"The config is already at the latest version"
            f"(v{old_version}) relative to the version of "
            f"luxonis-train (v{lxt.__version__})."
        )
        return cfg._dict
    logger.info(
        f"Upgrading the config from v{old_version} to v{lxt.__version__}"
    )

    _apply_field_replacements(cfg)
    heads = _migrate_nodes(cfg)
    _assign_export_output_names(cfg, heads)
    _migrate_attached_modules(cfg, heads)

    cfg.update("version", lxt.__version__)

    return cfg._dict


def upgrade_installation() -> None:
    """Upgrade the installed ``luxonis-train`` package from PyPI.

    If a newer release exists, upgrade ``pip``, ``luxonis_train``, and
    ``luxonis_ml[data]`` with the current Python interpreter. A failed
    version check is logged and leaves the installation unchanged.

    """
    latest_version = get_latest_version()
    if latest_version is None:
        logger.info("Failed to check for updates. Try again later.")
        return
    if latest_version == lxt.__semver__:
        logger.info(f"luxonis-train is up-to-date (v{lxt.__version__}).")
    else:
        subprocess.check_output(
            f"{sys.executable} -m pip install -U pip".split()
        )
        subprocess.check_output(
            f"{sys.executable} -m pip install -U luxonis_train".split()
        )
        subprocess.check_output(
            f"{sys.executable} -m pip install -U luxonis_ml[data]".split()
        )
        logger.info(
            f"luxonis-train updated from v{lxt.__version__} to v{latest_version}."
        )


def get_latest_version() -> Version | None:
    """Get the version of the latest ``luxonis-train`` release on PyPI.

    The function reads ``info.version`` from the PyPI JSON API. The
    request has a timeout of 5 seconds.

    Returns:
        ``semver.Version | None``: The latest version. It is ``None``
        when the request fails or when the response status is not
        ``200``. It is also ``None`` when the response body is not JSON
        with a valid ``info.version``.

    """
    import requests

    url = "https://pypi.org/pypi/luxonis_train/json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return None
        return Version.parse(response.json()["info"]["version"])
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None


def _load_config(config: PathType | Params) -> NestedDict:
    if isinstance(config, dict):
        return NestedDict(config)
    config = Path(config)
    if config.suffix == ".json":
        cfg = json.loads(config.read_text(encoding="utf-8"))
    else:
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    return NestedDict(cfg)


def _apply_field_replacements(cfg: NestedDict) -> None:
    cfg.replace(
        "trainer.use_rich_progress_bar",
        "rich_logging",
    )
    cfg.replace(
        "preprocessing.train_rgb",
        "preprocessing.color_space",
        "RGB" if cfg["preprocessing.train_rgb"] else "BGR",
    )
    cfg.replace(
        "model.predefined_model.params.variant",
        "model.predefined_model.variant",
    )
    cfg.replace(
        "tuner.storage.storage_type",
        "tuner.storage.backend",
        "sqlite"
        if cfg["tuner.storage.storage_type"] == "local"
        else "postgresql",
    )
    if "tuner" in cfg and cfg["tuner"] is None:
        cfg.pop("tuner")


def _migrate_nodes(cfg: NestedDict) -> dict[str, NestedDict]:
    nodes = cfg.get("model.nodes", [])
    assert isinstance(nodes, list)

    heads: dict[str, NestedDict] = {}
    for node in map(NestedDict, nodes):
        node_class = node["name"]
        if lxt.__semver__ >= Version(0, 4):
            node.replace("params.variant", "variant")
            if node_class == "FOMOHead":
                node.replace("params.num_conv_layers", "params.n_conv_layers")

        node_name = node["alias"] or node["name"]
        if "Head" in node["name"]:
            heads[node_name] = node
        if node.pop("params.download_weights", False):
            node["params.weights"] = "download"
    return heads


def _assign_export_output_names(
    cfg: NestedDict, heads: dict[str, NestedDict]
) -> None:
    export_output_names = cfg.pop("exporter.output_names", None)
    if export_output_names is None:
        return
    if len(heads) == 1:
        head = next(iter(heads.values()))
        if "params" not in head:
            head["params"] = {}
        head["params.export_output_names"] = export_output_names
    else:
        logger.error(
            "Multiple heads found in model, cannot assign "
            "'exporter.output_names' to a specific head."
        )


def _migrate_attached_modules(
    cfg: NestedDict, heads: dict[str, NestedDict]
) -> None:
    for key in ["metrics", "losses", "visualizers"]:
        modules: list[dict] = cfg.pop(f"model.{key}", [])
        for module in modules:
            if "attached_to" not in module:
                raise ValueError(
                    f"Module in 'model.{key}' is missing 'attached_to' field."
                )
            attached_to = module.pop("attached_to")
            if attached_to not in heads:
                raise ValueError(
                    f"Module in 'model.{key}' is attached to unknown head "
                    f"'{attached_to}'."
                )
            head = heads[attached_to]
            head._dict.setdefault(key, []).append(module)
            logger.info(
                f"Moved module from 'model.{key}' to head '{attached_to}'."
            )
