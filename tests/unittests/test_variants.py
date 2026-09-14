from copy import deepcopy

import pytest
from luxonis_ml.typing import Kwargs

from luxonis_train.variants import VariantBase, add_variant_aliases


def test_add_variant_aliases():
    variants = {
        "foo": {"param1": 1},
        "bar": {"param2": 2},
        "large": {"param3": 3},
    }
    aliased_variants = add_variant_aliases(
        deepcopy(variants), {"foo": ["f"], "bar": ["b", "baz"]}
    )
    assert aliased_variants == {
        "foo": {"param1": 1},
        "bar": {"param2": 2},
        "f": {"param1": 1},
        "b": {"param2": 2},
        "baz": {"param2": 2},
        "large": {"param3": 3},
    }
    yolo_aliased_variants = add_variant_aliases(variants, "yolo")
    assert yolo_aliased_variants == {
        "foo": {"param1": 1},
        "bar": {"param2": 2},
        "large": {"param3": 3},
        "l": {"param3": 3},
    }


def test_init_error_is_not_chained_to_the_variant_lookup():
    class Remote(VariantBase, register=False):
        @staticmethod
        def get_variants() -> tuple[str, dict[str, Kwargs]]:
            raise NotImplementedError

        def __init__(self, **kwargs):
            raise RuntimeError("gateway timeout")

    with pytest.raises(RuntimeError, match="gateway timeout") as exc_info:
        Remote(variant="default")

    assert exc_info.value.__context__ is None
