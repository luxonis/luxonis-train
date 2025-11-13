from dataclasses import dataclass
from types import EllipsisType
from typing import Any

from loguru import logger
from luxonis_ml.typing import Params, ParamValue
from semver import Version

from luxonis_train.config import CONFIG_VERSION


@dataclass
class ConfigWrapper:
    config: dict[str, Any]

    def __contains__(self, key: str) -> bool:
        keys = key.split(".")
        current = self.config
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
        current = self.config
        for k in keys:
            current = current[k]
        return current

    def __setitem__(self, key: str, value: Any) -> None:
        keys = key.split(".")
        current = self.config
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
        current = self.config
        for k in keys[:-1]:
            current = current[k]
        return current.pop(keys[-1])

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
        current = self.config
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value if value is not ... else old_value
        self.log_change(old_key, new_key)

    @staticmethod
    def log_change(old_field: str, new_field: str) -> None:
        logger.info(f"Changed config field '{old_field}' to '{new_field}'")


def migrate_v1_to_v2(config: Params) -> Params:
    cfg = ConfigWrapper(config)
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
        "tuner.backend",
        "sqlite"
        if cfg["tuner.storage.storage_type"] == "local"
        else "postgresql",
    )

    nodes = cfg["model"]["nodes"] or []
    assert isinstance(nodes, list)

    heads: dict[str, ConfigWrapper] = {}
    for node in map(ConfigWrapper, nodes):
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
            head.config.setdefault(key, []).append(module)
            logger.info(
                f"Moved module from 'model.{key}' to head '{attached_to}'."
            )

    cfg["config_version"] = "2.0"

    return cfg.config


def migrate_config(
    cfg: Params, fr: Version | None = None, to: Version = CONFIG_VERSION
) -> Params:
    if fr is None:
        if "config_version" not in cfg:
            raise ValueError(
                "Config does not contain the 'config_version' field"
            )
        version = cfg["config_version"]
        if not isinstance(version, str):
            raise TypeError(
                f"'config_version' field must be a string, got {type(version)}"
            )
        fr = Version.parse(version)

    if fr == to:
        return cfg

    # TODO: Chain migration
    map = {
        (Version(1), Version(2)): migrate_v1_to_v2,
    }
    return map[(fr, to)](cfg)
