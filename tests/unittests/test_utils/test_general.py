import os

import pytest

from luxonis_train.utils.general import infer_upscale_factor, safe_download


@pytest.mark.parametrize(
    ("in_size", "orig_size", "expected"),
    [
        ((1, 1), (1, 1), 0),
        ((1, 1), (2, 2), 1),
        ((2, 2), (1, 1), -1),
        ((2, 2), (4, 4), 1),
        ((4, 4), (2, 2), -1),
        ((4, 4), (8, 8), 1),
        ((8, 8), (4, 4), -1),
        ((2, 2), (16, 16), 3),
        ((16, 16), (4, 4), -2),
        (4, 8, 1),
    ],
)
def test_infer_upscale_factor(
    in_size: tuple[int, int] | int,
    orig_size: tuple[int, int] | int,
    expected: int,
):
    assert infer_upscale_factor(in_size, orig_size) == expected


@pytest.mark.parametrize(
    ("in_size", "orig_size"),
    [
        ((1, 1), (2, 1)),
        ((1, 1), (1, 2)),
        ((2, 3), (16, 16)),
        ((3, 2), (16, 16)),
        ((3, 3), (16, 16)),
    ],
)
def test_infer_upscale_factor_fail(
    in_size: tuple[int, int] | int,
    orig_size: tuple[int, int] | int,
):
    with pytest.raises(ValueError):
        infer_upscale_factor(in_size, orig_size)


def test_safe_download():
    url = "https://github.com/luxonis/luxonis-train/releases/download/v0.1.0-beta/efficientrep_n_coco.ckpt"
    local_path = safe_download(url=url, file="test.ckpt", dir=".", force=True)
    if local_path:
        assert os.path.isfile(local_path)
        os.remove(local_path)


def test_safe_download_failed():
    url = "fake_url.fake"
    local_path = safe_download(url=url, file="test.ckpt", dir=".", force=True)
    assert local_path is None
