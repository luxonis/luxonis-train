import logging
import math
from typing import TypeVar

from torch import Size, Tensor

from luxonis_train.utils.types import Packet

logger = logging.getLogger(__name__)


def make_divisible(x: int | float, divisor: int) -> int:
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
            return int(round(factor))
        return factor

    if isinstance(in_size, int):
        in_size = (in_size, in_size)
    if isinstance(orig_size, int):
        orig_size = (orig_size, orig_size)
    in_height, in_width = in_size
    orig_height, orig_width = orig_size

    width_factor = _infer_upscale_factor(in_width, orig_width)
    height_factor = _infer_upscale_factor(in_height, orig_height)

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
        shape_packet[name] = [x.shape for x in value]
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

    msg = f"Default value of {value} is being used for {action_name}."

    if caller_name:
        msg = f"[{caller_name}] {msg}"

    logger.info(msg, stacklevel=2)
    return default
