import pytest
from _pytest.config import Config
from _pytest.python import Function


def pytest_collection_modifyitems(items: list[Function]):
    for item in items:
        if "/unittests/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            # ensure unittests run before integration tests
            item.add_marker(pytest.mark.order(0))
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


def pytest_configure(config: Config):
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
