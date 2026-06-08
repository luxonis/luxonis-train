import inspect
import math
import urllib.parse
from collections.abc import Callable, Collection, Iterator
from inspect import Parameter
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar, overload

import numpy as np
import torch
from loguru import logger
from luxonis_ml.typing import PathType
from luxonis_ml.utils import LuxonisFileSystem
from torch import Size, Tensor

from luxonis_train import __version__
from luxonis_train.typing import Packet


def make_divisible(x: float, divisor: int) -> int:
    """Round ``x`` upward to make it evenly divisible by ``divisor``.

    Equivalent to :math:`ceil(x / divisor) * divisor`.

    Args:
        x (float): Value to revise.
        divisor (int): Divisor.

    Returns:
        int: Revised value.

    """
    return math.ceil(x / divisor) * divisor


def infer_upscale_factor(
    in_size: tuple[int, int] | int, orig_size: tuple[int, int] | int
) -> int:
    """Infer the upscale factor from input and original shapes.

    Args:
        in_size (tuple[int, int] | int): Input shape as ``(height, width)`` or
            a single size.
        orig_size (tuple[int, int] | int): Original shape as
            ``(height, width)`` or a single size.

    Returns:
        int: Upscale factor.

    Raises:
        ValueError: If ``in_size`` cannot be upscaled to ``orig_size`` because the
            upscale factors are not integers or are different.

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
    """Convert a packet of tensors to a packet of shapes.

    Used for debugging purposes.

    Args:
        packet (``Packet[Tensor]``): Packet of tensors.

    Returns:
        ``Packet[Size]``: Packet of shapes.

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
    """Get value with default.

    Returns ``value`` if it is not ``None``. Otherwise, logs that the default is
    being used and returns ``default``.

    Args:
        value (T | None): Value to return.
        action_name (str): Name of the action for which the default value is
            being used.
        caller_name (str | None): Name of the caller function, used for
            logging. Defaults to ``None``.
        default (T): Default value to return if ``value`` is ``None``.

    Returns:
        T: ``value`` if it is not ``None``; otherwise ``default``.

    """
    if value is not None:
        return value

    msg = f"Default value of `{value}` is being used for {action_name}."

    if caller_name:
        msg = f"[{caller_name}] {msg}"

    logger.info(msg, stacklevel=2)
    return default


def get_signature(
    func: Callable, exclude: Collection[str] | None = None
) -> dict[str, Parameter]:
    """Get a function signature without selected parameters.

    Args:
        func (``Callable``): Function to get the signature of.
        exclude (``Collection[str] | None``): ``Parameter`` names to exclude from the
            signature. Defaults to ``None``, which excludes ``"self"`` and
            ``"kwargs"``.

    Returns:
        ``dict[str, Parameter]``: ``Parameter`` names mapped to their
        `inspect.Parameter` objects.

    """
    exclude = set(exclude or [])
    exclude |= {"self", "kwargs"}
    signature = dict(inspect.signature(func).parameters)
    return {
        name: param for name, param in signature.items() if name not in exclude
    }


def safe_download(
    url: PathType | None,
    file: str | None = None,
    cache_dir: PathType = ".cache/luxonis_train",
    retry: int = 3,
    force: bool = False,
) -> Path | None:
    """Download a remote file and return its local path.

    Args:
        url (``PathType | None``): URL of the file to download. If ``None``,
            returns ``None``.
        file (str | None): Name of the saved file. If ``None``, the name is
            inferred from ``url``. Defaults to ``None``.
        cache_dir (`PathType <luxonis_ml.typing.PathType>`): Directory to store downloaded files in. Defaults
            to ``".cache/luxonis_train"``.
        retry (int): Number of retries when downloading. Defaults to ``3``.
        force (bool): Whether to force redownload if the file already exists.
            Defaults to ``False``.

    Returns:
        ``Path | None``: Local file path, or ``None`` if downloading failed.

    """
    if url is None or isinstance(url, Path):
        return url
    if LuxonisFileSystem.get_protocol(url) == "file":
        return Path(url)
    cache_dir = Path(cache_dir) / __version__
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / (file or url2file(url))
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

    Args:
        obj (object): Object to get the attribute from.
        attribute (str): Name of the attribute to get.

    Returns:
        ``Any``: Value of the attribute.

    Raises:
        ValueError: If the attribute is ``None``.

    Examples:
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

    """
    value = getattr(obj, f"_{attribute}")
    if value is None:
        raise ValueError(f"attribute '{attribute}' was not set")
    return value


def get_batch_instances(
    batch_index: int, bboxes: Tensor, payload: Tensor | None = None
) -> Tensor:
    """Get instances for one batch index from batched data.

    Args:
        batch_index (int): Batch index.
        bboxes (``Tensor``): Bounding boxes with the batch index in the first
            column.
        payload (``Tensor | None``): Additional tensor to select using the same
            batch order. If unset, returns bounding box instances without the
            batch index. Defaults to ``None``.

    Returns:
        ``Tensor``: Instances from the batched data.

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
    """Generate instances from batched data.

    The batch index is expected in the first column of ``bboxes``.

    Args:
        bboxes (``Tensor``): Bounding boxes with the batch index in the first
            column.
        *args (``Tensor``): Additional tensors in the same batch order. These
            tensors do not contain the batch index themselves.
        batch_size (int | None): Batch size. When tensors are empty, providing
            this value yields ``batch_size`` empty tensors. If omitted, empty
            input yields nothing. Defaults to ``None``.

    Yields:
        ``Tensor | tuple[Tensor, ...]``: Per-batch instances. When no extra tensors
        are provided, each item is a bounding box tensor with the batch index
        stripped. Otherwise, each item is a tuple containing the bounding boxes
        followed by the matching tensors from ``args``.

    Raises:
        ValueError: If any tensor in ``args`` has a different length than
            ``bboxes``.

    Examples:
        >>> bboxes = torch.tensor([[0, 1], [0, 2], [1, 3]])
        >>> keypoints = torch.tensor([[10], [20], [30]])
        >>> for bbox, kpt in instances_from_batch(bboxes, keypoints):
        ...     print(bbox.tolist(), kpt.tolist())
        [[1], [2]] [[10], [20]]
        [[3]] [[30]]

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


def decode_text_metadata_labels(
    labels: dict[str, np.ndarray],
    metadata_types: dict[str, type],
) -> dict[str, np.ndarray]:
    """Decode text metadata labels from character-code arrays."""
    decoded_labels: dict[str, np.ndarray] = {}

    for task, label in labels.items():
        if metadata_types.get(task) is not str:
            decoded_labels[task] = label
            continue

        arr = np.asarray(label)
        if arr.size == 0 or arr.dtype.kind in {"U", "S", "O"}:
            decoded_labels[task] = arr
            continue

        decoded_values: list[str] = []
        for row in np.atleast_2d(arr):
            chars: list[str] = []
            for value in np.asarray(row).reshape(-1):
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, float):
                    if not value.is_integer():
                        decoded_values = []
                        break
                    value = int(value)
                elif not isinstance(value, int):
                    decoded_values = []
                    break

                if value == 0:
                    break
                if value < 0:
                    decoded_values = []
                    break
                chars.append(chr(value))

            if not decoded_values and not chars and arr.size != 0:
                break
            decoded_values.append("".join(chars))

        decoded_labels[task] = (
            np.asarray(decoded_values) if decoded_values else arr
        )

    return decoded_labels


class Counter:
    """Simple counter that can be used to generate unique IDs or
    indices.
    """

    def __init__(self, start: int = 0):
        self._count = start

    def __call__(self) -> int:
        current = self._count
        self._count += 1
        return current
