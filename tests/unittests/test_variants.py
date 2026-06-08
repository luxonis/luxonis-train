from luxonis_train.variants import add_variant_aliases


def test_add_variant_aliases():
    variants = {
        "foo": {"param1": 1},
        "bar": {"param2": 2},
        "large": {"param3": 3},
    }
    aliased_variants = add_variant_aliases(
        variants, {"foo": ["f"], "bar": ["b", "baz"]}
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
        "f": {"param1": 1},
        "bar": {"param2": 2},
        "b": {"param2": 2},
        "baz": {"param2": 2},
        "large": {"param3": 3},
        "l": {"param3": 3},
    }
