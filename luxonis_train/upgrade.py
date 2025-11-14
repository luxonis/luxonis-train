import subprocess
import sys
from dataclasses import dataclass
from types import EllipsisType
from typing import Any

import requests
from loguru import logger
from luxonis_ml.typing import Params, ParamValue
from semver import Version

import luxonis_train as lxt


@dataclass
class NestedDict:
    _dict: dict[str, Any]

    def __contains__(self, key: str) -> bool:
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
        if key not in self:
            return None
        keys = key.split(".")
        current = self._dict
        for k in keys:
            current = current[k]
        return current

    def __setitem__(self, key: str, value: Any) -> None:
        keys = key.split(".")
        current = self._dict
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def pop(self, key: str, default: Any = ...) -> Any:
        if key not in self:
            if default is not ...:
                return default
            raise KeyError(f"Key '{key}' not found in config.")
        keys = key.split(".")
        current = self._dict
        for k in keys[:-1]:
            current = current[k]
        return current.pop(keys[-1])

    def update(self, key: str, value: Any) -> None:
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
        value: ParamValue | None | EllipsisType = ...,
    ) -> None:
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
        logger.info(f"Changed config field '{old_field}' to '{new_field}'")


def upgrade_config(cfg: Params | NestedDict) -> Params:
    if not isinstance(cfg, NestedDict):
        cfg = NestedDict(cfg)

    if "config_version" in cfg:
        old_version = Version(0, 3)
        cfg.pop("config_version")
    elif "version" in cfg:
        old_version = Version.parse(cfg["version"])
    else:
        raise ValueError("The config does not contain the 'version' field")
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

    nodes = cfg["model"]["nodes"] or []
    assert isinstance(nodes, list)

    heads: dict[str, NestedDict] = {}
    for node in map(NestedDict, nodes):
        node.replace("params.variant", "variant")
        node_name = node["alias"] or node["name"]
        if "Head" in node["name"]:
            heads[node_name] = node

    if "exporter.output_names" in cfg:
        if len(heads) == 1:
            output_names = cfg.pop("exporter.output_names")
            head = next(iter(heads.values()))
            if "params" not in head:
                head["params"] = {}
            head["params.export_output_names"] = output_names
        else:
            logger.error(
                "Multiple heads found in model, cannot assign "
                "'exporter.output_names' to a specific head."
            )

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

    cfg.update("version", lxt.__version__)

    return cfg._dict


def upgrade_installation() -> None:
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
    url = "https://pypi.org/pypi/luxonis_train/json"
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        data = response.json()
        versions = list(data["releases"].keys())
        versions.sort(key=lambda s: [int(u) for u in s.split(".")])
        return Version.parse(versions[-1])
    return None
