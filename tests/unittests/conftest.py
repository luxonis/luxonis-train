from collections.abc import Iterator

import pytest

from luxonis_train.registry import MODELS


@pytest.fixture(autouse=True)
def _isolate_models_registry() -> Iterator[None]:
    """Undo registry mutations made by a test.

    Defining a `BasePredefinedModel` subclass registers it globally, so
    a test that declares a throw-away model (a `FamilyV2`, say) would
    otherwise leak it into every test that runs afterwards.
    """
    snapshot = dict(MODELS._module_dict)
    try:
        yield
    finally:
        MODELS._module_dict.clear()
        MODELS._module_dict.update(snapshot)
