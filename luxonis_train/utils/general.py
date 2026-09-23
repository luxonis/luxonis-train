"""Small helpers used across the package.

The helpers round channel counts, infer upscale factors, inspect
signatures, split batched instances, download files to a cache, and
decode text metadata labels.

"""

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
    r"""Round ``x`` up to the nearest multiple of ``divisor``.

    The function computes :math:`\lceil x / d \rceil \cdot d`, where
    :math:`d` is ``divisor``. The backbones and necks use it to round
    channel counts.

    Args:
        x (float): The value to round.
        divisor (int): The number the result is a multiple of.

    Returns:
        int: The smallest multiple of ``divisor`` that is not less than
        ``x``.

    Example:
        >>> make_divisible(37, 8)
        40
        >>> make_divisible(40, 8)
        40

    """
    return math.ceil(x / divisor) * divisor


def infer_upscale_factor(
    in_size: tuple[int, int] | int, orig_size: tuple[int, int] | int
) -> int:
    r"""Infer the number of doublings from one size to another.

    The function returns the exponent :math:`n` with
    :math:`o = i \cdot 2^{n}`, where :math:`i` is the input size and
    :math:`o` is the original size. The result is not the factor
    :math:`2^{n}` itself. :math:`n` is negative when the original size
    is smaller. The segmentation heads use :math:`n` as the number of
    upsampling steps, or compute the scale factor :math:`2^{n}` from it.

    Args:
        in_size (tuple[int, int] | int): The input size as
            ``(height, width)``, or one integer for both.
        orig_size (tuple[int, int] | int): The original size as
            ``(height, width)``, or one integer for both.

    Returns:
        int: The exponent :math:`n`.

    Raises:
        ValueError: When the width ratio or the height ratio is not a
            power of two, when the two exponents differ, or when a size
            is not positive.

    Examples:
        >>> infer_upscale_factor((32, 32), (128, 128))
        2
        >>> infer_upscale_factor(16, 64)
        2
        >>> infer_upscale_factor(64, 16)
        -2

        >>> infer_upscale_factor((32, 64), (128, 128))
        Traceback (most recent call last):
        ValueError: Width and height upscale factors are different. ...

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
    """Convert a packet of tensors to a packet of their shapes.

    `LuxonisOutput` uses it in its string form to show the output
    shapes of the nodes. The model uses it when it builds the nodes, to
    give each node the shapes of its inputs.

    Args:
        packet (``Packet[Tensor]``): A packet whose values are tensors
            or lists of tensors.

    Returns:
        ``Packet[Size]``: A packet with the same keys. A tensor becomes
        its `torch.Size`, and a list of tensors becomes a list of
        `torch.Size` objects.

    Example:
        >>> import torch
        >>> features = [torch.zeros(1, 3), torch.zeros(2, 4)]
        >>> to_shape_packet({"features": features, "boxes": torch.zeros(5, 4)})
        {'features': [torch.Size([1, 3]), torch.Size([2, 4])],
         'boxes': torch.Size([5, 4])}

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
    """Return ``value``, or ``default`` when ``value`` is ``None``.

    When the function returns ``default``, it logs an info message that
    names ``action_name``.

    Args:
        value (``T | None``): The value to return when it is not
            ``None``.
        action_name (str): What the value is for, named in the log
            message.
        caller_name (str | None): The name of the caller, used as a
            prefix of the log message. ``None`` adds no prefix.
        default (``T``): The value to return when ``value`` is ``None``.

    Returns:
        ``T``: ``value`` when it is not ``None``, else ``default``.

    Example:
        >>> get_with_default(0.4, "area factor", default=0.53)
        0.4

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
    """Get the parameters of a function without the excluded ones.

    The function always leaves out the parameters ``self`` and
    ``kwargs``. It also leaves out the names in ``exclude``.

    Args:
        func (``Callable``): The function or method to inspect.
        exclude (``Collection[str] | None``): More parameter names to
            leave out. ``None`` excludes only ``"self"`` and
            ``"kwargs"``.

    Returns:
        ``dict[str, Parameter]``: The remaining parameter names, in
        signature order, mapped to their `inspect.Parameter` objects.

    Example:
        >>> def forward(self, x, y=1, **kwargs): ...
        >>> list(get_signature(forward))
        ['x', 'y']
        >>> list(get_signature(forward, exclude=["y"]))
        ['x']

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
    """Download a remote file into the cache and return its local path.

    The function returns a `pathlib.Path` unchanged. It converts a
    ``str`` without a remote protocol to a `pathlib.Path`. It does not
    check that a local file exists.

    The function saves a remote file as ``cache_dir/<version>/<file>``,
    where ``<version>`` is the ``luxonis_train`` version. It creates
    that directory when it is missing. It reuses a file that is already
    there and logs a warning, unless ``force`` is ``True``. Before the
    download, it logs an info message with the local path and the URL.
    The logged URL comes from `clean_url` and has no query string.

    The function downloads ``s3``, ``gcs``, and ``gs`` URLs with
    ``LuxonisFileSystem.download``, and all other URLs with
    `torch.hub.download_url_to_file`. After a failed attempt, it logs
    the traceback and tries again, at most ``retry`` more times. When
    all attempts fail, it logs a warning and returns ``None``.

    Args:
        url (``PathType | None``): The URL or path of the file. ``None``
            returns ``None``.
        file (str | None): The name of the saved file. ``None`` takes
            the file name from ``url``.
        cache_dir (``PathType``): The root of the cache directory.
        retry (int): The number of repeated attempts after a failed
            download.
        force (bool): When ``True``, download again even when the file
            is in the cache.

    Returns:
        ``Path | None``: The local path of the file, or ``None`` when
        every attempt failed. For ``s3``, ``gcs``, and ``gs`` URLs, the
        path that ``LuxonisFileSystem.download`` returns.

    Example:
        >>> from pathlib import Path
        >>> safe_download("weights/model.ckpt") == Path("weights/model.ckpt")
        True
        >>> safe_download(None) is None
        True

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
    """Strip the query string from a URL and decode percent-escapes.

    The function first normalizes the URL with `pathlib.PurePosixPath`.
    This step collapses repeated slashes to one, removes the ``.``
    path components, and removes a trailing slash. The function decodes
    the escapes before it cuts the URL at the first ``?``, so an
    escaped ``%3F`` also cuts the URL.

    Args:
        url (str): The URL, for example
            ``"https://url.com/file%20a.txt?auth"``.

    Returns:
        str: The URL without the first ``?`` and the text after it, with
        the ``%XX`` escapes decoded, for example
        ``"https://url.com/file a.txt"``.

    Example:
        >>> clean_url("https://url.com/dir/file%20name.txt?token=abc")
        'https://url.com/dir/file name.txt'
        >>> clean_url("https://url.com//dir/")
        'https://url.com/dir'

    """
    # Pathlib turns :// -> :/, PurePosixPath for Windows
    url = str(PurePosixPath(url)).replace(":/", "://")
    # '%2F' to '/', split https://url.com/file.txt?auth
    return urllib.parse.unquote(url).split("?")[0]


def url2file(url: str) -> str:
    """Get the file name from a URL.

    Args:
        url (str): The URL, for example
            ``"https://url.com/file.txt?auth"``.

    Returns:
        str: The last path component of the URL after `clean_url`, for
        example ``"file.txt"``.

    Example:
        >>> url2file("https://url.com/dir/file.txt?token=abc")
        'file.txt'

    """
    return Path(clean_url(url)).name


def get_attribute_check_none(obj: object, attribute: str) -> Any:
    """Get the private attribute ``_<attribute>`` and reject ``None``.

    A property uses it to expose a value that the constructor can
    leave unset.

    Args:
        obj (object): The object that holds the attribute.
        attribute (str): The attribute name without the leading
            underscore.

    Returns:
        ``Any``: The value of ``obj._<attribute>``.

    Raises:
        AttributeError: When ``obj`` has no attribute ``_<attribute>``.
        ValueError: When the value is ``None``.

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
    """Select the rows of one image from batched instance data.

    Args:
        batch_index (int): The index of the image in the batch.
        bboxes (``Tensor``): The bounding boxes of the whole batch, of
            shape ``[N, C]``, with the batch index in the first column.
        payload (``Tensor | None``): A tensor of shape ``[N, ...]``
            with one row per row of ``bboxes``, in the same order.
            ``None`` selects from ``bboxes`` itself.

    Returns:
        ``Tensor``: The rows whose batch index equals ``batch_index``.
        From ``bboxes`` they come without the first column, of shape
        ``[n, C - 1]``. From ``payload`` they come with all columns,
        so a batch index column in ``payload`` stays.

    Example:
        >>> import torch
        >>> bboxes = torch.tensor([[0, 1], [0, 2], [1, 3]])
        >>> get_batch_instances(1, bboxes).tolist()
        [[3]]
        >>> payload = torch.tensor([10, 20, 30])
        >>> get_batch_instances(0, bboxes, payload).tolist()
        [10, 20]

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
) -> Iterator[Tensor | tuple[Tensor, ...]]:
    """Yield the instances of each image from batched instance data.

    The batch index is in the first column of ``bboxes``. The extra
    tensors in ``args`` have one row per row of ``bboxes``, in the same
    order. The function selects their rows with the batch index of
    ``bboxes`` and keeps all their columns. The object keypoint
    similarity metric passes the target keypoints as an extra tensor.

    The function yields one item for each image index from ``0`` to
    ``batch_size - 1``. When ``batch_size`` is ``None`` or ``0``, the
    number of items is the largest batch index plus one. In that case,
    the function yields no items for the images without instances at
    the end of the batch.

    When ``bboxes`` is empty, the function yields new empty tensors from
    `torch.empty_like`, with the shapes of the inputs. The bounding
    boxes keep the batch index column. Empty input yields no items when
    ``batch_size`` is ``None`` or ``0``.

    Args:
        bboxes (``Tensor``): The bounding boxes of the whole batch, of
            shape ``[N, C]``, with the batch index in the first column.
        *args (``Tensor``): Extra tensors of shape ``[N, ...]``, in the
            same order as ``bboxes``.
        batch_size (int | None): The number of images to yield. ``None``
            or ``0`` infers it from the largest batch index.

    Yields:
        ``Tensor | tuple[Tensor, ...]``: Without extra tensors, the
        bounding boxes of one image with the batch index column
        removed, of shape ``[n, C - 1]``. With extra tensors, a tuple
        of those bounding boxes followed by the matching rows of each
        tensor in ``args``.

    Raises:
        ValueError: When a tensor in ``args`` has a different length
            than ``bboxes``. The error occurs when the iteration starts,
            not when the code calls the function.

    Examples:
        >>> import torch
        >>> bboxes = torch.tensor([[0, 1], [0, 2], [1, 3]])
        >>> keypoints = torch.tensor([[10], [20], [30]])
        >>> for bbox, kpt in instances_from_batch(bboxes, keypoints):
        ...     print(bbox.tolist(), kpt.tolist())
        [[1], [2]] [[10], [20]]
        [[3]] [[30]]

        >>> [b.tolist() for b in instances_from_batch(bboxes, batch_size=3)]
        [[[1], [2]], [[3]], []]

    """
    if not all(len(arg) == len(bboxes) for arg in args):
        raise ValueError("All tensors must have the same length.")
    if not bboxes.numel():
        yield from _empty_batch_instances(bboxes, args, batch_size)
        return
    yield from _batched_instances(bboxes, args, batch_size)


def decode_text_metadata_labels(
    labels: dict[str, np.ndarray],
    metadata_types: dict[str, type],
) -> dict[str, np.ndarray]:
    """Decode the ``str`` metadata labels from character codes.

    `BaseLoaderTorch` converts a string label to a ``float32`` tensor
    of its character codes. When it collates a batch of
    ``metadata/text`` labels, it pads the shorter rows with ``0``. This
    function reverses that for every label whose type in
    ``metadata_types`` is ``str``:

    - Each row becomes the string of its codes up to the first ``0``.
    - A one-dimensional array counts as one row.
    - The codes can be integers or integer-valued floats.

    The function converts all other labels to arrays with
    ``np.asarray``.

    A ``str`` label stays unchanged in these cases:

    - Its array is empty or has a string or object dtype.
    - A row has a negative value before its first ``0``.
    - A row has a fractional value before its first ``0``.
    - A row has a value that is not an integer or a float before its
      first ``0``.

    Args:
        labels (``dict[str, np.ndarray]``): Label names mapped to their
            label arrays.
        metadata_types (dict[str, type]): Label names mapped to the type
            of their metadata values. The function does not decode a
            label that is missing here.

    Returns:
        ``dict[str, np.ndarray]``: The same label names. A decoded
        ``str`` label is an array of strings, one for each row.

    Example:
        >>> import numpy as np
        >>> labels = {
        ...     "text": np.array([[72.0, 105.0, 0.0], [79.0, 75.0, 33.0]]),
        ...     "count": np.array([1, 2]),
        ... }
        >>> decoded = decode_text_metadata_labels(
        ...     labels, {"text": str, "count": int}
        ... )
        >>> decoded["text"].tolist()
        ['Hi', 'OK!']
        >>> decoded["count"].tolist()
        [1, 2]

    """
    decoded_labels: dict[str, np.ndarray] = {}

    for task, label in labels.items():
        if metadata_types.get(task) is not str:
            decoded_labels[task] = np.asarray(label)
        else:
            decoded_labels[task] = _decode_text_label(label)

    return decoded_labels


class Counter:
    """A counter that returns the next integer on each call.

    The prediction writer that saves the inference renders keeps one
    counter for all batches. It uses the counter to number the saved
    render files, or to select the image path of each image.

    Example:
        >>> counter = Counter(start=5)
        >>> counter(), counter(), counter()
        (5, 6, 7)

    """

    def __init__(self, start: int = 0):
        """Initialize the counter.

        Args:
            start (int): The first value that the counter returns.

        """
        self._count = start

    def __call__(self) -> int:
        """Return the current value and advance the counter by one."""
        current = self._count
        self._count += 1
        return current


def _empty_batch_instances(
    bboxes: Tensor, args: tuple[Tensor, ...], batch_size: int | None
) -> Iterator[Tensor | tuple[Tensor, ...]]:
    if batch_size is None:
        return
    for _ in range(batch_size):
        if not args:
            yield torch.empty_like(bboxes)
            continue
        yield tuple(torch.empty_like(item) for item in (bboxes, *args))


def _batched_instances(
    bboxes: Tensor, args: tuple[Tensor, ...], batch_size: int | None
) -> Iterator[Tensor | tuple[Tensor, ...]]:
    n_batches = batch_size or int(bboxes[:, 0].max()) + 1
    for index in range(n_batches):
        if not args:
            yield get_batch_instances(index, bboxes)
            continue
        yield tuple(
            get_batch_instances(index, bboxes, payload)
            for payload in (None, *args)
        )


def _decode_text_label(label: np.ndarray) -> np.ndarray:
    arr = np.asarray(label)
    if arr.size == 0 or arr.dtype.kind in {"U", "S", "O"}:
        return arr
    decoded_values = []
    for row in np.atleast_2d(arr):
        decoded = _decode_text_row(row)
        if decoded is None:
            return arr
        decoded_values.append(decoded)
    return np.asarray(decoded_values)


def _decode_text_row(row: np.ndarray) -> str | None:
    chars = []
    for value in np.asarray(row).reshape(-1):
        value = value.item() if isinstance(value, np.generic) else value
        if isinstance(value, float):
            if not value.is_integer():
                return None
            value = int(value)
        if not isinstance(value, int) or value < 0:
            return None
        if value == 0:
            break
        chars.append(chr(value))
    return "".join(chars)
