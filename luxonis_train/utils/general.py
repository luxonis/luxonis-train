import math
import urllib.parse
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar, overload

import torch
from loguru import logger
from luxonis_ml.typing import PathType
from luxonis_ml.utils import LuxonisFileSystem
from torch import Size, Tensor

from luxonis_train import __version__
from luxonis_train.typing import Packet


def make_divisible(x: float, divisor: int) -> int:
    """Upward revision the value x to make it evenly divisible by the
    divisor.

    Equivalent to M{ceil(x / divisor) * divisor}.

    @type x: int | float
    @param x: Value to be revised.
    @type divisor: int
    @param divisor: Divisor.
    @rtype: int
    @return: Revised value.
    """
    return math.ceil(x / divisor) * divisor


def infer_upscale_factor(
    in_size: tuple[int, int] | int, orig_size: tuple[int, int] | int
) -> int:
    """Infer the upscale factor from the input shape and the original
    shape.

    @type in_size: tuple[int, int] | int
    @param in_size: Input shape as a tuple of (height, width) or just
        one of them.
    @type orig_size: tuple[int, int] | int
    @param orig_size: Original shape as a tuple of (height, width) or
        just one of them.
    @rtype: int
    @return: Upscale factor.
    @raise ValueError: If the C{in_size} cannot be upscaled to the
        C{orig_size}. This can happen if the upscale factors are not
        integers or are different.
    """

    def _infer_upscale_factor(in_size: int, orig_size: int) -> int | float:
        factor = math.log2(orig_size) - math.log2(in_size)
        if abs(round(factor) - factor) < 1e-6:
            return round(factor)
        return factor

    if isinstance(in_size, int):
        in_size = (in_size, in_size)
    if isinstance(orig_size, int):
        orig_size = (orig_size, orig_size)
    in_height, in_width = in_size
    orig_height, orig_width = orig_size

    width_factor = _infer_upscale_factor(in_width, orig_width)
    height_factor = _infer_upscale_factor(in_height, orig_height)

    # TODO: Better error messages, suggest possible solutions
    match (width_factor, height_factor):
        case (int(wf), int(hf)) if wf == hf:
            return wf
        case (int(wf), int(hf)):
            raise ValueError(
                f"Width and height upscale factors are different. "
                f"Width: {wf}, height: {hf}."
            )
        case (int(wf), float(hf)):
            raise ValueError(
                f"Width upscale factor is an integer, but height upscale factor is not. "
                f"Width: {wf}, height: {hf}."
            )
        case (float(wf), int(hf)):
            raise ValueError(
                f"Height upscale factor is an integer, but width upscale factor is not. "
                f"Width: {wf}, height: {hf}."
            )
        case (float(wf), float(hf)):
            raise ValueError(
                "Width and height upscale factors are not integers. "
                f"Width: {wf}, height: {hf}."
            )

    raise NotImplementedError(
        f"Unexpected case: {width_factor}, {height_factor}"
    )


def to_shape_packet(packet: Packet[Tensor]) -> Packet[Size]:
    """Converts a packet of tensors to a packet of shapes. Used for
    debugging purposes.

    @type packet: Packet[Tensor]
    @param packet: Packet of tensors.
    @rtype: Packet[Size]
    @return: Packet of shapes.
    """
    shape_packet: Packet[Size] = {}
    for name, value in packet.items():
        shape_packet[name] = (
            [x.shape for x in value]
            if isinstance(value, list)
            else value.shape
        )
    return shape_packet


T = TypeVar("T")


def get_with_default(
    value: T | None,
    action_name: str,
    caller_name: str | None = None,
    *,
    default: T,
) -> T:
    """Returns value if it is not C{None}, otherwise returns the default
    value and log an info.

    @type value: T | None
    @param value: Value to return.
    @type action_name: str
    @param action_name: Name of the action for which the default value
        is being used. Used for logging.
    @type caller_name: str | None
    @param caller_name: Name of the caller function. Used for logging.
    @type default: T
    @param default: Default value to return if C{value} is C{None}.
    @rtype: T
    @return: C{value} if it is not C{None}, otherwise C{default}.
    """
    if value is not None:
        return value

    msg = f"Default value of `{value}` is being used for {action_name}."

    if caller_name:
        msg = f"[{caller_name}] {msg}"

    logger.info(msg, stacklevel=2)
    return default


def safe_download(
    url: str,
    file: str | None = None,
    dir: PathType = ".cache/luxonis_train",
    retry: int = 3,
    force: bool = False,
) -> Path | None:
    """Downloads file from the web and returns either local path or None
    if downloading failed.

    @type url: str
    @param url: URL of the file you want to download.
    @type file: str | None
    @param file: Name of the saved file, if None infers it from URL.
        Defaults to None.
    @type dir: str
    @param dir: Directory to store downloaded file in. Defaults to
        '.cache_data'.
    @type retry: int
    @param retry: Number of retries when downloading. Defaults to 3.
    @type force: bool
    @param force: Whether to force redownload if file already exists.
        Defaults to False.
    @rtype: Path | None
    @return: Path to local file or None if downloading failed.
    """
    dir = Path(dir) / __version__
    dir.mkdir(parents=True, exist_ok=True)
    f = dir / (file or url2file(url))
    if f.is_file() and not force:
        logger.warning(f"File {f} is already cached, using that one.")
        return f
    uri = clean_url(url)
    logger.info(f"Downloading `{uri}` to `{f}`")
    for i in range(retry + 1):
        try:
            if "://" in url:
                protocol, _ = url.split("://")
                if protocol in {"s3", "gcs", "gs"}:
                    return LuxonisFileSystem.download(url, f)
            torch.hub.download_url_to_file(url, str(f), progress=True)
        except Exception:
            logger.exception(f"Download failed, retrying {i + 1}/{retry} ...")
        else:
            return f
    logger.warning("Download failed, retry limit reached.")
    return None


def clean_url(url: str) -> str:
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    # Pathlib turns :// -> :/, PurePosixPath for Windows
    url = str(PurePosixPath(url)).replace(":/", "://")
    # '%2F' to '/', split https://url.com/file.txt?auth
    return urllib.parse.unquote(url).split("?")[0]


def url2file(url: str) -> str:
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


def get_attribute_check_none(obj: object, attribute: str) -> Any:
    """Get private attribute from object and check if it is not None.

    Example:

        >>> class Person:
        ...     def __init__(self, age: int | None = None):
        ...         self._age = age
        ...
        ...     @property
        ...     def age(self):
        ...         return get_attribute_check_none(self, "age")

        >>> mike = Person(20)
        >>> print(mike.age)
        20

        >>> amanda = Person()
        >>> print(amanda.age)
        Traceback (most recent call last):
        ValueError: attribute 'age' was not set

    @type obj: object
    @param obj: Object to get attribute from.

    @type attribute: str
    @param attribute: Name of the attribute to get.

    @rtype: Any
    @return: Value of the attribute.

    @raise ValueError: If the attribute is None.
    """
    value = getattr(obj, f"_{attribute}")
    if value is None:
        raise ValueError(f"attribute '{attribute}' was not set")
    return value


def get_batch_instances(
    batch_index: int, bboxes: Tensor, payload: Tensor | None = None
) -> Tensor:
    """Get instances from batched data, where the batch index is
    encoded.

    as the first column of the bounding boxes.
    @type batch_index: int
    @param batch_index: Batch index.
    @type bboxes: Tensor
    @param bboxes: Tensor of bounding boxes. Must have the batch index
        as the first column.
    @type payload: Tensor | None
    @param payload: Additional tensor to be batched with the bounding
        boxes. This tensor is in the same batch order, but doesn't
        contain the batch index itself. If unset, returns the bounding
        box instances (without the batch index).
    @rtype: Tensor
    @return: Instances from the batched data.
    """
    if payload is None:
        return bboxes[bboxes[:, 0] == batch_index][:, 1:]
    return payload[bboxes[:, 0] == batch_index]


@overload
def instances_from_batch(
    bboxes: Tensor, *, batch_size: int | None = ...
) -> Iterator[Tensor]: ...


@overload
def instances_from_batch(
    bboxes: Tensor, *args: Tensor, batch_size: int | None = ...
) -> Iterator[tuple[Tensor, ...]]: ...


def instances_from_batch(
    bboxes: Tensor, *args: Tensor, batch_size: int | None = None
) -> Iterator[tuple[Tensor, ...]] | Iterator[Tensor]:
    """Generate instances from batched data, where the batch index is
    encoded as the first column of the bounding boxes.

    Example::
        >>> bboxes = torch.tensor([[0, 1], [0, 2], [1, 3]])
        >>> keypoints = torch.tensor([[0.1], [0.2], [0.3]])
        >>> for bbox, kpt in instances_from_batch(bboxes, keypoints):
        ...     print(bbox, kpt)
        tensor([[1], [2]]) tensor([[0.1], [0.2]])
        tensor([[3]]) tensor([[0.3]])

    @type bboxes: Tensor
    @param bboxes: Tensor of bounding boxes. Must have the batch index
        as the first column.
    @type *args: Tensor
    @param *args: Additional tensors to be batched with the bounding
        boxes. These tensors are in the same batch order, but don't
        contain the batch index themselves.
    @type batch_size: int
    @param batch_size: The batch size. Important in case of empty
        tensors. If provided and the tensors are empty, the generator
        will yield C{batch_size} empty tensors. If not provided, the
        generator will yield nothing. Defaults to C{None}.
    @rtype: Iterator[tuple[Tensor, ...]]
    @return: Generator of instances, where the first element is the
        bounding box tensor (with the batch index stripped) and the
        rest are the additional tensors (keypoints, masks, etc.).
    """
    if not all(len(arg) == len(bboxes) for arg in args):
        raise ValueError("All tensors must have the same length.")
    if not bboxes.numel():
        if batch_size is not None:
            for _ in range(batch_size):
                if not args:
                    yield torch.empty_like(bboxes)
                else:
                    yield tuple(
                        torch.empty_like(bboxes) for _ in [bboxes, *args]
                    )
        return
    for i in range(batch_size or int(bboxes[:, 0].max()) + 1):
        if not args:
            yield get_batch_instances(i, bboxes)
        else:
            yield tuple(
                get_batch_instances(i, bboxes, payload)
                for payload in [None, *args]
            )


class Counter:
    """Simple counter that can be used to generate unique IDs or
    indices."""

    def __init__(self, start: int = 0):
        self._count = start

    def __call__(self) -> int:
        current = self._count
        self._count += 1
        return current
