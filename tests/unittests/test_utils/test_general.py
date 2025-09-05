import pytest
import torch
from pytest_subtests import SubTests

from luxonis_train.utils.general import (
    infer_upscale_factor,
    instances_from_batch,
    safe_download,
)


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
    ("in_size", "orig_size", "match_error"),
    [
        ((1, 1), (2, 1), "are different"),
        ((1, 1), (1, 2), "are different"),
        ((2, 3), (16, 16), "but width"),
        ((3, 2), (16, 16), "but height"),
        ((3, 3), (16, 16), "are not integers"),
    ],
)
def test_infer_upscale_factor_fail(
    in_size: tuple[int, int] | int,
    orig_size: tuple[int, int] | int,
    match_error: str,
):
    with pytest.raises(ValueError, match=match_error):
        infer_upscale_factor(in_size, orig_size)


def test_safe_download():
    url = "https://github.com/luxonis/luxonis-train/releases/download/v0.1.0-beta/efficientrep_n_coco.ckpt"
    local_path = safe_download(url=url, file="test.ckpt", dir=".", force=True)
    if local_path is not None:
        assert local_path.is_file()
        local_path.unlink()


def test_safe_download_failed():
    url = "fake_url.fake"
    local_path = safe_download(url=url, file="test.ckpt", dir=".", force=True)
    assert local_path is None


def test_instances_from_batch(subtests: SubTests):
    with subtests.test("bboxes"):
        bboxes = torch.tensor([[0, 1], [0, 2], [1, 3]])
        instances = [t.tolist() for t in instances_from_batch(bboxes)]
        assert len(instances) == 2
        assert instances == [[[1], [2]], [[3]]]
    with subtests.test("combined"):
        bboxes = torch.tensor([[0, 1], [0, 2], [1, 3]])
        keypoints = torch.tensor([[10], [20], [30]])
        instances = [
            (b.tolist(), k.tolist())
            for b, k in instances_from_batch(bboxes, keypoints)
        ]
        assert len(instances) == 2
        assert instances == [([[1], [2]], [[10], [20]]), ([[3]], [[30]])]
    with subtests.test("empty"):
        bboxes = torch.empty((0, 2))
        instances = [t.tolist() for t in instances_from_batch(bboxes)]
        assert instances == []
        instances = [
            t.tolist() for t in instances_from_batch(bboxes, batch_size=4)
        ]
        assert instances == [[] for _ in range(4)]
    with subtests.test("partially-empty"):
        tensor = torch.tensor(
            [
                [0.0000, 0.0000, 13.8655, 14.7887, 20.8700, 25.5167],
                [0.0000, 0.0000, 14.0510, 9.6272, 19.6425, 18.3628],
                [0.0000, 0.0000, 29.7840, 6.3282, 31.9525, 18.0485],
                [0.0000, 0.0000, 21.7350, 5.0170, 23.3745, 5.6698],
                [0.0000, 0.0000, 14.3695, 5.0098, 16.6020, 5.6003],
                [0.0000, 0.0000, 27.0940, 5.0891, 30.3860, 5.6106],
                [0.0000, 0.0000, 0.0555, 5.0289, 3.0025, 5.4869],
                [0.0000, 0.0000, 24.9735, 5.0392, 28.7320, 5.7285],
                [1.0000, 0.0000, 13.1480, 15.2740, 18.2235, 23.6970],
                [2.0000, 0.0000, 11.0897, 6.6301, 21.1002, 20.1511],
            ],
        )
        assert len(list(instances_from_batch(tensor))) == 3
        assert len(list(instances_from_batch(tensor, batch_size=4))) == 4
    with (
        subtests.test("fail"),
        pytest.raises(
            ValueError, match="All tensors must have the same length"
        ),
    ):
        list(
            instances_from_batch(
                torch.tensor([[0, 1], [0, 2], [1, 3]]),
                torch.tensor([[10], [20]]),
            )
        )
