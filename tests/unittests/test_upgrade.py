import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from semver import Version

import luxonis_train as lxt
from luxonis_train.__main__ import config as upgrade_config_command
from luxonis_train.upgrade import get_latest_version, upgrade_config


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


def test_upgrade_moves_exporter_output_names_to_the_only_head():
    assert upgrade_config(
        {
            "version": "0.3.0",
            "model": {
                "name": "dummy",
                "nodes": [{"name": "EfficientBBoxHead"}],
            },
            "exporter": {"output_names": ["boxes", "scores"]},
        }
    ) == {
        "version": lxt.__version__,
        "model": {
            "name": "dummy",
            "nodes": [
                {
                    "name": "EfficientBBoxHead",
                    "params": {"export_output_names": ["boxes", "scores"]},
                }
            ],
        },
        "exporter": {},
    }


def test_get_latest_version_uses_pypi_resolved_version(
    monkeypatch: pytest.MonkeyPatch,
):
    response = Mock(status_code=200)
    response.json.return_value = {
        "info": {"version": "1.2.3"},
        "releases": {"not-a-version": []},
    }
    monkeypatch.setattr(requests, "get", Mock(return_value=response))

    assert get_latest_version() == Version(1, 2, 3)


def test_get_latest_version_returns_none_when_request_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        requests, "get", Mock(side_effect=requests.ConnectionError)
    )

    assert get_latest_version() is None


def test_get_latest_version_returns_none_for_invalid_version(
    monkeypatch: pytest.MonkeyPatch,
):
    response = Mock(status_code=200)
    response.json.return_value = {"info": {"version": "not-a-version"}}
    monkeypatch.setattr(requests, "get", Mock(return_value=response))

    assert get_latest_version() is None
