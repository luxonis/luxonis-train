from types import SimpleNamespace

from luxonis_train.callbacks.fail_on_no_train_batches import (
    _format_details,
    _merge_loader_details,
    _minimum_batch_count,
    _minimum_required_size,
)


def test_minimum_batch_count_for_int_limit():
    assert _minimum_batch_count(5) == 1
    assert _minimum_batch_count(0) is None


def test_minimum_batch_count_for_float_limit():
    assert _minimum_batch_count(0.25) == 4
    assert _minimum_batch_count(0.0) is None


def test_minimum_required_size_with_drop_last():
    batch_size, world_size = 8, 2
    assert _minimum_required_size(batch_size, True, world_size, 1.0) == 16


def test_minimum_required_size_without_drop_last():
    # ceil(1 / 0.5) = 2 batches -> (2 - 1) * 8 * 2 + 1
    batch_size, world_size = 8, 2
    assert _minimum_required_size(batch_size, False, world_size, 0.5) == 17


def test_minimum_required_size_needs_batch_size_and_drop_last():
    assert _minimum_required_size(None, True, 1, 1.0) is None
    assert _minimum_required_size(8, None, 1, 1.0) is None


def test_merge_loader_details_fills_missing_fields():
    loader = SimpleNamespace(dataset=[1, 2, 3], batch_size=4, drop_last=True)
    assert _merge_loader_details((None, None, None), loader) == (3, 4, True)


def test_merge_loader_details_keeps_known_fields():
    loader = SimpleNamespace(dataset=[1], batch_size=4, drop_last=False)
    assert _merge_loader_details((7, 2, True), loader) == (7, 2, True)


def test_format_details_renders_all_parts():
    message = _format_details(3, 8, 8, 1, True, 1.0)
    assert message == (
        "(details: dataset_size=3, min_required_size=8, missing=5; "
        "params: batch_size=8, world_size=1, drop_last=True, "
        "limit_train_batches=1.0)"
    )


def test_format_details_skips_unknown_parts():
    message = _format_details(None, None, None, 2, None, 0.5)
    assert message == (
        "(details: ; params: world_size=2, limit_train_batches=0.5)"
    )
