"""The rotary position embedding of DINOv3, changed to export to
ONNX.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Literal

import torch
from torch import Tensor, nn


class RopePositionEmbedding(nn.Module):
    r"""Axial rotary position embedding of DINOv3, without learnable weights.

    The module computes the sine and the cosine tables that the DINOv3
    attention uses to rotate the query and the key vectors. Each patch
    of an ``H x W`` grid gets the coordinates of its center. `forward`
    divides them as ``normalize_coords`` selects and maps each
    coordinate :math:`c` to :math:`2c - 1`. With ``"separate"`` or
    ``"max"``, the result is in ``[-1, 1]``. With ``"min"``, the
    coordinates of the longer axis can be above ``1``. Each angle
    depends on one axis only, so the axes do not mix. For the coordinate
    :math:`c` and the period :math:`p`, the angle is
    :math:`2 \pi c / p`.

    The periods come from one of two settings. :math:`D` is the head
    dimension, ``embed_dim // num_heads``.

    - ``base``: :math:`p_i = \text{base}^{2i / (D / 2)}` for
      :math:`i = 0, \dots, D / 4 - 1`.
    - ``min_period`` and ``max_period``: :math:`D / 4` periods in a
      geometric sequence from ``min_period`` to ``max_period``.

    The code comes from DINOv3. `forward` uses ``repeat`` instead of
    ``tile``, because ``tile`` does not export to ONNX.

    Attributes:
        periods (``Tensor``): The :math:`D / 4` periods, in a persistent
            buffer.

    Example:
        >>> rope = RopePositionEmbedding(64, num_heads=4)
        >>> [round(p, 2) for p in rope.periods.tolist()]
        [1.0, 3.16, 10.0, 31.62]
        >>> sin, cos = rope(H=2, W=3)
        >>> sin.shape, cos.shape
        (torch.Size([6, 16]), torch.Size([6, 16]))

        The second setting of the periods:

        >>> rope = RopePositionEmbedding(
        ...     64, num_heads=4, base=None, min_period=0.5, max_period=10.0
        ... )
        >>> [round(p, 2) for p in rope.periods.tolist()]
        [0.5, 1.36, 3.68, 10.0]

    """

    periods: Tensor

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        """Store the settings and compute the periods.

        Give either ``base``, or ``base=None`` with both ``min_period``
        and ``max_period``. When ``base`` is set, the constructor ignores
        a single ``min_period`` or ``max_period``.

        Args:
            embed_dim (int): The embedding dimension of the transformer.
                It must be a multiple of ``4 * num_heads``.
            num_heads (int): The number of attention heads.
            base (float | None): The base of the periods. ``None``
                selects the ``min_period`` and ``max_period`` setting.
            min_period (float | None): The smallest period. The module
                uses it only when ``base`` is ``None``.
            max_period (float | None): The largest period. The module
                uses it only when ``base`` is ``None``.
            normalize_coords (``Literal["min", "max", "separate"]``): The
                divisor of the patch coordinates. ``"separate"`` divides
                the rows by ``H`` and the columns by ``W``. ``"max"``
                divides both by ``max(H, W)``, and ``"min"`` divides both
                by ``min(H, W)``.
            shift_coords (float | None): In training mode, `forward` adds
                a random shift to each axis. Each axis gets its own
                shift, uniform in ``[-shift_coords, shift_coords]``.
                ``None`` adds no shift.
            jitter_coords (float | None): In training mode, `forward`
                multiplies each axis by its own random factor. The factor
                is log-uniform in ``[1 / jitter_coords, jitter_coords]``.
                ``None`` applies no jitter.
            rescale_coords (float | None): In training mode, `forward`
                multiplies both axes by one random factor. The factor is
                log-uniform in ``[1 / rescale_coords, rescale_coords]``.
                ``None`` applies no rescale.
            dtype (torch.dtype | None): The data type of the periods and
                the coordinates. ``None`` selects the default data type.
            device (torch.device | None): The device of the periods
                buffer. `forward` computes on the device of that buffer.

        Raises:
            AssertionError: When ``embed_dim`` is not a multiple of
                ``4 * num_heads``.
            ValueError: When ``base`` is ``None`` and one of the periods
                is ``None``, or when ``base`` and both periods are set.

        """
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (
            base is not None and both_periods
        ):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided."
            )

        D_head = embed_dim // num_heads
        self._base = base
        self._min_period = min_period
        self._max_period = max_period
        self._D_head = D_head
        self._normalize_coords = normalize_coords
        self._shift_coords = shift_coords
        self._jitter_coords = jitter_coords
        self._rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self._dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        """Compute the sine and the cosine tables for an ``H x W`` grid.

        The patches follow row-major order. For each patch, the method
        computes :math:`D / 4` angles from the row coordinate and then
        :math:`D / 4` angles from the column coordinate. It appends a
        copy of these :math:`D / 2` angles, which gives :math:`D` angles.
        In training mode, the method first shifts, jitters, and rescales
        the coordinates, as the constructor arguments enable.

        Args:
            H (int): The number of patch rows.
            W (int): The number of patch columns.

        Returns:
            ``tuple[Tensor, Tensor]``: The sine and the cosine of the
            angles, each of shape ``[H * W, D]``, where ``D`` is the head
            dimension.

        Raises:
            ValueError: When ``normalize_coords`` is not ``"min"``,
                ``"max"``, or ``"separate"``.

        """
        device = self.periods.device
        dtype = self._dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self._normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self._normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self._normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(
                f"Unknown normalize_coords: {self._normalize_coords}"
            )
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        )  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self._shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(
                -self._shift_coords, self._shift_coords
            )
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self._jitter_coords is not None:
            j = torch.tensor(self._jitter_coords, device=device, dtype=dtype)
            jmax_t = torch.log(j)  # was previously np.log
            jmin_t = -jmax_t  # tensor
            # uniform_ needs floats, so convert using .item()
            jitter_hw = (
                torch.empty(2, **dd)
                .uniform_(jmin_t.item(), jmax_t.item())
                .exp()
            )
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self._rescale_coords is not None:
            r = torch.tensor(self._rescale_coords, device=device, dtype=dtype)
            rmax_t = torch.log(r)
            rmin_t = -rmax_t
            rescale_hw = (
                torch.empty(1, **dd)
                .uniform_(rmin_t.item(), rmax_t.item())
                .exp()
            )
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = (
            2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        )  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.repeat(
            1, 2
        )  # [HW, D] This line was changed from angles = angles.tile(2), as angles.tile is not yet ONNX-convertible
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self) -> None:
        device = self.periods.device
        dtype = self._dtype
        if self._base is not None:
            periods = self._base ** (
                2
                * torch.arange(self._D_head // 4, device=device, dtype=dtype)
                / (self._D_head // 2)
            )  # [D//4]
        else:
            assert self._max_period is not None
            assert self._min_period is not None
            base = self._max_period / self._min_period
            exponents = torch.linspace(
                0, 1, self._D_head // 4, device=device, dtype=dtype
            )  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = (
                periods * self._max_period
            )  # range [min_period, max_period]
        self.periods.data = periods
