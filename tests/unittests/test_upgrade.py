import json
from pathlib import Path

import luxonis_train as lxt
from luxonis_train.__main__ import config as upgrade_config_command
from luxonis_train.upgrade import upgrade_config


def test_upgrade_keeps_json_config_readable(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"version": lxt.__version__, "model": {"name": "dummy"}})
    )

    upgrade_config_command(cfg_path)

    assert json.loads(cfg_path.read_text()) == {
        "version": lxt.__version__,
        "model": {"name": "dummy"},
    }


def test_upgrade_config_parses_json_file(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"version": lxt.__version__, "model": {"name": "dummy"}})
    )

    assert upgrade_config(cfg_path) == {
        "version": lxt.__version__,
        "model": {"name": "dummy"},
    }
