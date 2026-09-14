import pytest
import torch

from luxonis_train.attached_modules.visualizers.utils import _resize_to_match


def test_keep_smaller_matches_the_smaller_height():
    first, second = _resize_to_match(
        torch.zeros(3, 10, 20), torch.zeros(3, 40, 30), keep_size="smaller"
    )
    assert first.shape[-2] == second.shape[-2] == 10


def test_keep_larger_matches_the_larger_height():
    first, second = _resize_to_match(
        torch.zeros(3, 10, 20), torch.zeros(3, 40, 30), keep_size="larger"
    )
    assert first.shape[-2] == second.shape[-2] == 40


def test_keep_first_leaves_the_first_image_untouched():
    first, second = _resize_to_match(
        torch.zeros(3, 10, 20), torch.zeros(3, 40, 30), keep_size="first"
    )
    assert first.shape == (3, 10, 20)
    assert second.shape[-2] == 10


def test_keep_second_leaves_the_second_image_untouched():
    first, second = _resize_to_match(
        torch.zeros(3, 10, 20), torch.zeros(3, 40, 30), keep_size="second"
    )
    assert second.shape == (3, 40, 30)
    assert first.shape[-2] == 40


def test_resize_along_width_matches_the_widths():
    first, second = _resize_to_match(
        torch.zeros(3, 10, 20), torch.zeros(3, 40, 30), resize_along="width"
    )
    assert first.shape[-1] == second.shape[-1] == 30


def test_without_aspect_ratio_both_images_get_the_same_shape():
    first, second = _resize_to_match(
        torch.zeros(3, 10, 20),
        torch.zeros(3, 40, 30),
        keep_aspect_ratio=False,
    )
    assert first.shape == second.shape


def test_invalid_keep_size_is_rejected():
    with pytest.raises(ValueError, match="Invalid value for keep_size"):
        _resize_to_match(
            torch.zeros(3, 10, 20),
            torch.zeros(3, 40, 30),
            keep_size="biggest",  # type: ignore
        )


def test_invalid_resize_along_is_rejected():
    with pytest.raises(ValueError, match="Invalid value for resize_along"):
        _resize_to_match(
            torch.zeros(3, 10, 20),
            torch.zeros(3, 40, 30),
            resize_along="diagonal",  # type: ignore
        )
