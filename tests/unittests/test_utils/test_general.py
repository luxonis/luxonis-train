from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest
import torch
from pytest_subtests import SubTests

from luxonis_train import __version__
from luxonis_train.utils.general import (
    clean_url,
    decode_text_metadata_labels,
    get_attribute_check_none,
    get_batch_instances,
    get_signature,
    get_with_default,
    infer_upscale_factor,
    instances_from_batch,
    make_divisible,
    safe_download,
    to_shape_packet,
    url2file,
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


def test_safe_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_download(_url: str, file: str, progress: bool) -> None:
        assert progress is True
        Path(file).write_bytes(b"downloaded")

    monkeypatch.setattr(
        "luxonis_train.utils.general.torch.hub.download_url_to_file",
        fake_download,
    )

    url = "https://example.com/efficientrep_n_coco.ckpt"
    local_path = safe_download(
        url=url, file="test.ckpt", cache_dir=tmp_path, force=True
    )

    assert local_path is not None
    assert local_path.read_bytes() == b"downloaded"


def test_safe_download_local_and_cached_paths(tmp_path: Path) -> None:
    local_path = tmp_path / "local.bin"
    local_path.touch()

    assert safe_download(None) is None
    assert safe_download(local_path) == local_path
    assert safe_download(str(local_path)) == local_path

    cache_dir = tmp_path / "cache"
    cached_file = cache_dir / __version__ / "weights.bin"
    cached_file.parent.mkdir(parents=True)
    cached_file.write_bytes(b"cached")

    assert (
        safe_download(
            "https://example.com/weights.bin",
            file="weights.bin",
            cache_dir=cache_dir,
        )
        == cached_file
    )


def test_safe_download_remote_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_download(url: str, dest: Path) -> Path:
        calls.append((url, dest))
        dest.write_bytes(b"remote")
        return dest

    monkeypatch.setattr(
        "luxonis_train.utils.general.LuxonisFileSystem.download",
        fake_download,
    )
    remote_path = safe_download("s3://bucket/model.bin", cache_dir=tmp_path)
    assert remote_path is not None
    assert remote_path.is_file()
    assert calls[0][0] == "s3://bucket/model.bin"

    def fail_download(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise RuntimeError("download failed")

    monkeypatch.setattr(
        "luxonis_train.utils.general.torch.hub.download_url_to_file",
        fail_download,
    )
    assert (
        safe_download(
            "https://example.com/missing.bin",
            cache_dir=tmp_path,
            retry=0,
            force=True,
        )
        is None
    )


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


def test_general_small_helpers() -> None:
    assert make_divisible(17, 8) == 24
    assert clean_url(
        "https://example.com/a%2Fb/file.txt?token=secret"
    ).endswith("/a/b/file.txt")
    assert (
        url2file("https://example.com/path/file.txt?token=secret")
        == "file.txt"
    )

    packet = {
        "one": torch.zeros(2, 3),
        "many": [torch.zeros(1), torch.zeros(4, 5)],
    }
    assert to_shape_packet(packet) == {
        "one": torch.Size([2, 3]),
        "many": [torch.Size([1]), torch.Size([4, 5])],
    }

    assert get_with_default("value", "action", default="fallback") == "value"
    assert (
        get_with_default(None, "action", "caller", default="fallback")
        == "fallback"
    )

    def func(
        self: object, keep: str, drop: str, **kwargs: Any
    ) -> tuple[str, str, dict[str, Any]]:
        return keep, drop, kwargs

    assert list(get_signature(func, exclude={"drop"})) == ["keep"]

    class Holder:
        def __init__(self, value: int | None) -> None:
            self._value = value

    assert get_attribute_check_none(Holder(1), "value") == 1
    with pytest.raises(ValueError, match="attribute 'value' was not set"):
        get_attribute_check_none(Holder(None), "value")

    bboxes = torch.tensor([[0, 1], [1, 2], [0, 3]])
    payload = torch.tensor([[10], [20], [30]])
    assert get_batch_instances(0, bboxes).tolist() == [[1], [3]]
    assert get_batch_instances(1, bboxes, payload).tolist() == [[20]]

    empty = torch.empty((0, 2))
    empty_payload = torch.empty((0, 1))
    instances = list(instances_from_batch(empty, empty_payload, batch_size=2))
    assert len(instances) == 2
    assert all(
        b.shape == empty.shape and p.shape == empty.shape for b, p in instances
    )


def test_decode_text_metadata_labels():
    labels = {
        "category": np.array([1, 2]),
        "already_text": np.array(["ok"]),
        "codes": np.array([[65, 66, 0], [67, 0, 68]]),
        "float_codes": np.array([[65.0, 0.0]]),
        "bad_float": np.array([[65.5]]),
        "bad_negative": np.array([[-1]]),
        "bad_complex": np.array([[1 + 2j]]),
        "bad_object": np.array([[object()]], dtype=object),
        "empty": np.array([]),
    }
    metadata_types = {
        "already_text": str,
        "codes": str,
        "float_codes": str,
        "bad_float": str,
        "bad_negative": str,
        "bad_complex": str,
        "bad_object": str,
        "empty": str,
    }

    decoded = decode_text_metadata_labels(labels, metadata_types)

    assert decoded["category"] is labels["category"]
    assert decoded["already_text"].tolist() == ["ok"]
    assert decoded["codes"].tolist() == ["AB", "C"]
    assert decoded["float_codes"].tolist() == ["A"]
    assert decoded["bad_float"] is labels["bad_float"]
    assert decoded["bad_negative"] is labels["bad_negative"]
    assert decoded["bad_complex"] is labels["bad_complex"]
    assert decoded["bad_object"] is labels["bad_object"]
    assert decoded["empty"].size == 0
