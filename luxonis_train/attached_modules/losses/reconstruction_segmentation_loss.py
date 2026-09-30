"""The DRAEM loss, which adds the L2 and the structural similarity
errors of the reconstruction to a focal loss over the anomaly mask.

The module also holds the structural similarity (SSIM) helpers that the
loss uses.

"""

from math import exp
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, amp, nn

from luxonis_train.nodes import DiscSubNetHead
from luxonis_train.tasks import Tasks

from .base_loss import BaseLoss
from .softmax_focal_loss import SoftmaxFocalLoss


class ReconstructionSegmentationLoss(BaseLoss):
    r"""Reconstruction and anomaly segmentation loss for
    `DiscSubNetHead`.

    The loss follows `DRAEM <https://arxiv.org/abs/2108.07610>`_. It
    compares the reconstructed image with the clean image, and the
    predicted anomaly segmentation with the anomaly mask.

    Inputs:
        - ``predictions`` (``Tensor``): :math:`\left[B, 2, H, W\right]`
          anomaly logits
        - ``reconstruction`` (``Tensor``): :math:`\left[B, 3, H,
          W\right]` reconstructed image
        - ``target_original_segmentation`` (``Tensor``): :math:`\left[B,
          3, H, W\right]` clean image
        - ``target_segmentation`` (``Tensor``): :math:`\left[B, 2, H,
          W\right]` one-hot anomaly mask

    Outputs:
        - ``Tensor``: scalar total loss, or :math:`\left[B, H, W\right]`
          when ``reduction`` is ``"none"``
        - ``dict[str, Tensor]``: sub-losses ``l2_loss``, ``ssim_loss``,
          ``focal_loss``

    Formula:
        :math:`r` is the reconstruction, :math:`x` the clean image,
        :math:`z` the logits, and :math:`y` the one-hot mask:

        .. math::

            L = \text{MSE}(r, x) + \left(1 - \text{SSIM}(r, x)\right)
            + \text{FL}(z, y)

        MSE is the mean over all elements. SSIM is the mean of the map
        from `ssim`, with an :math:`11 \times 11` Gaussian window. FL is
        `SoftmaxFocalLoss` with ``alpha``, ``gamma``, ``smooth``, and
        ``reduction``.

    References:
        - Source: This project.
        - License: Apache-2.0 (this project)

    Notes:
        The focal term comes from an internal `SoftmaxFocalLoss` without
        a node. The SSIM term runs in ``float32`` with autocast off, and
        it estimates the dynamic range from ``reconstruction``. The
        sub-losses are not detached.

    Example:
        Attached to a ``DiscSubNetHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: DiscSubNetHead
              inputs: [RecSubNet]
              losses:
                - name: ReconstructionSegmentationLoss

    Compatible with:
        - Used by: `AnomalyDetectionModel`
        - Nodes: `DiscSubNetHead`

    """

    node: DiscSubNetHead
    supported_tasks = [Tasks.ANOMALY_DETECTION]

    def __init__(
        self,
        alpha: float = 1,
        gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        smooth: float = 1e-5,
        **kwargs,
    ):
        r"""Initialize the L2, SSIM, and focal losses.

        Args:
            alpha (float): The factor :math:`\alpha` of
                `SoftmaxFocalLoss`. It scales the focal loss of every
                pixel.
            gamma (float): The focal exponent :math:`\gamma` of
                `SoftmaxFocalLoss`. A larger value gives less weight to the
                pixels that the head already classifies well.
            reduction (``Literal["none", "mean", "sum"]``): How
                `SoftmaxFocalLoss` reduces the loss of the pixels. With
                ``"none"``, the focal term and the total loss have the
                shape ``[B, H, W]``.
            smooth (float): The label smoothing of `SoftmaxFocalLoss`. Its
                constructor raises ``ValueError`` when ``smooth`` is not
                in ``[0, 1]``.
            **kwargs (``Any``): Keyword arguments forwarded to `BaseLoss`,
                such as ``node`` and ``final_loss_weight``.

        """
        super().__init__(**kwargs)
        self.loss_l2 = nn.MSELoss()
        self.loss_focal = SoftmaxFocalLoss(
            smooth=smooth, alpha=alpha, gamma=gamma, reduction=reduction
        )
        self.loss_ssim = SSIM()

    def forward(
        self,
        predictions: Tensor,
        reconstruction: Tensor,
        target_original_segmentation: Tensor,
        target_segmentation: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        r"""Compute the reconstruction and segmentation losses.

        `SoftmaxFocalLoss` raises ``ValueError`` when ``predictions`` has
        fewer than ``2`` channels, or when its shape is not the shape of
        ``target_segmentation``.

        Args:
            predictions (``Tensor``): Anomaly logits of shape
                ``[B, C, H, W]``, with ``C`` at least ``2``. The
                ``segmentation`` output of the node.
            reconstruction (``Tensor``): Reconstructed images of shape
                ``[B, 3, H, W]``. The ``reconstruction`` output of the
                node.
            target_original_segmentation (``Tensor``): Clean images of the
                shape of ``reconstruction``. The ``original_segmentation``
                label of the task.
            target_segmentation (``Tensor``): One-hot anomaly masks of the
                shape of ``predictions``. The ``segmentation`` label of the
                task.

        Returns:
            ``tuple[Tensor, dict[str, Tensor]]``: The sum of the three
            terms, and a dictionary that maps ``"l2_loss"``,
            ``"ssim_loss"``, and ``"focal_loss"`` to the terms. The L2 and
            SSIM terms are scalars. The focal term and the sum are
            scalars, or of shape ``[B, H, W]`` when ``reduction`` is
            ``"none"``.

        Example:
            A perfect reconstruction makes the L2 and SSIM terms ``0``.
            Zero logits give the focal term
            :math:`-(1 - 0.5)^2 \ln 0.5 \approx 0.1733`:

            >>> import torch
            >>> loss = ReconstructionSegmentationLoss()
            >>> image = torch.linspace(0, 1, 48).reshape(1, 3, 4, 4)
            >>> logits = torch.zeros(1, 2, 4, 4)
            >>> mask = torch.zeros(1, 2, 4, 4)
            >>> mask[:, 0] = 1
            >>> total, sub_losses = loss(logits, image, image, mask)
            >>> sorted(sub_losses)
            ['focal_loss', 'l2_loss', 'ssim_loss']
            >>> round(total.item(), 4)
            0.1733

        """
        l2 = self.loss_l2(reconstruction, target_original_segmentation)
        ssim = self.loss_ssim(reconstruction, target_original_segmentation)
        focal = self.loss_focal(predictions, target_segmentation)

        total_loss = l2 + ssim + focal

        sub_losses = {
            "l2_loss": l2,
            "ssim_loss": ssim,
            "focal_loss": focal,
        }

        return total_loss, sub_losses


class SSIM(nn.Module):
    r"""Structural dissimilarity loss, :math:`1 - \text{SSIM}`.

    The module calls `ssim` with a Gaussian window of the standard
    deviation ``1.5`` and returns one minus the result. It caches the
    window for the channel count of the last input.

    """

    def __init__(
        self,
        window_size: int = 11,
        size_average: bool = True,
        val_range: float | None = None,
    ):
        """Initialize the loss with a window for one channel.

        Args:
            window_size (int): Side of the square Gaussian window, in
                pixels.
            size_average (bool): Whether `forward` averages over the
                whole batch. When ``False``, it returns one value for each
                image.
            val_range (float | None): A dynamic range for `ssim`.
                `forward` does not pass it to `ssim`, so `ssim` always
                estimates the dynamic range from ``img1``.

        """
        super().__init__()
        self._window_size = window_size
        self._size_average = size_average
        self._val_range = val_range

        # Assume 1 channel for SSIM
        self._channel = 1
        self._window = create_window(window_size)

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        r"""Return one minus the SSIM of two image batches.

        The method turns autocast off and casts both batches to
        ``float32``. When ``img1`` has another channel count than the
        cached window, the method builds a new window and caches it.

        Args:
            img1 (``Tensor``): Images of shape ``[B, C, H, W]``. `ssim`
                estimates the dynamic range from them.
            img2 (``Tensor``): Images of the shape of ``img1``.

        Returns:
            ``Tensor``: :math:`1 - \text{SSIM}` as a scalar, or of shape
            ``[B]`` when ``size_average`` is ``False``. ``0`` for equal
            images.

        Example:
            >>> import torch
            >>> loss = SSIM()
            >>> image = torch.linspace(0, 1, 768).reshape(1, 3, 16, 16)
            >>> loss(image, image).item()
            0.0
            >>> round(loss(image, 1 - image).item(), 4)
            0.7642

        """
        device = img1.device
        with amp.autocast(device_type=device.type, enabled=False):
            img1 = img1.float()
            img2 = img2.float()

            (_, channel, _, _) = img1.size()
            if channel == self._channel and self._window.dtype == img1.dtype:
                window = self._window.to(device).clone()
            else:
                window = (
                    create_window(self._window_size, channel)
                    .to(device)
                    .type(img1.dtype)
                )
                self._window = window
                self._channel = channel

            s_score = ssim(
                img1,
                img2,
                window=window,
                window_size=self._window_size,
                size_average=self._size_average,
            )
            return 1.0 - s_score


def create_window(window_size: int, channel: int = 1) -> Tensor:
    """Build a normalized 2D Gaussian window for `ssim`.

    The window is the outer product of a 1D `gaussian` with itself. The
    standard deviation is always ``1.5``. Each channel gets the same
    window.

    Args:
        window_size (int): Side of the square window, in pixels.
        channel (int): Number of channels.

    Returns:
        ``Tensor``: A ``float32`` window of shape
        ``[channel, 1, window_size, window_size]``, the weight shape of
        a grouped convolution. The weights of each channel sum to ``1``.

    Example:
        >>> import torch
        >>> window = create_window(11, channel=3)
        >>> window.shape
        torch.Size([3, 1, 11, 11])
        >>> torch.allclose(window.sum(dim=(1, 2, 3)), torch.ones(3))
        True

    """
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def gaussian(window_size: int, sigma: float) -> Tensor:
    r"""Build a normalized 1D Gaussian of ``window_size`` samples.

    Sample :math:`i` gets the weight
    :math:`\exp\left(-\frac{(i - c)^2}{2 \sigma^2}\right)`, with the
    center :math:`c` at ``window_size // 2``. The function then divides
    the weights by their sum. For an even ``window_size``, the center is
    the right one of the two middle samples.

    Args:
        window_size (int): Number of samples.
        sigma (float): Standard deviation :math:`\sigma`, in samples.

    Returns:
        ``Tensor``: The weights, of shape ``[window_size]``, with the sum
        ``1``.

    Example:
        >>> [round(weight, 4) for weight in gaussian(3, 1.5).tolist()]
        [0.3078, 0.3844, 0.3078]

    """
    gauss = torch.tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def ssim(
    img1: Tensor,
    img2: Tensor,
    window_size: int = 11,
    window: Tensor | None = None,
    size_average: bool = True,
    val_range: float | None = None,
) -> Tensor:
    r"""Compute the structural similarity (SSIM) of two image batches.

    The window filters each channel of both batches, with zero padding.
    The filter gives the local means :math:`\mu_1` and
    :math:`\mu_2`, the local variances :math:`\sigma_1^2` and
    :math:`\sigma_2^2`, and the local covariance :math:`\sigma_{12}`.
    The SSIM map is

    .. math::

        \text{SSIM} = \frac{\left(2 \mu_1 \mu_2 + C_1\right)
        \left(2 \sigma_{12} + C_2\right)}
        {\left(\mu_1^2 + \mu_2^2 + C_1\right)
        \left(\sigma_1^2 + \sigma_2^2 + C_2\right)}

    with :math:`C_1 = (0.01 L)^2` and :math:`C_2 = (0.03 L)^2` for the
    dynamic range :math:`L`.

    Args:
        img1 (``Tensor``): Images of shape ``[B, C, H, W]``.
        img2 (``Tensor``): Images of the shape of ``img1``.
        window_size (int): Side of the window. The padding is always
            ``window_size // 2``, also when ``window`` has another size.
        window (``Tensor | None``): Filter weights of shape
            ``[C, 1, k, k]``, for example from `create_window`. When
            ``None``, the function builds a window with the side
            ``min(window_size, H, W)``.
        size_average (bool): Whether to return the mean of the whole
            map. When ``False``, the function returns the mean of each
            image.
        val_range (float | None): The dynamic range :math:`L`. When
            ``None``, the function estimates :math:`L` from ``img1`` as
            the maximum minus the minimum. The maximum is ``255`` when a
            value of ``img1`` is above ``128``, otherwise ``1``. The
            minimum is ``-1`` when a value of ``img1`` is below ``-0.5``,
            otherwise ``0``.

    Returns:
        ``Tensor``: The mean SSIM as a scalar, or of shape ``[B]`` when
        ``size_average`` is ``False``. ``1`` for equal images.

    Example:
        >>> import torch
        >>> image = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8)
        >>> ssim(image, image).item()
        1.0
        >>> round(ssim(image, 1 - image).item(), 4)
        0.0926

    """
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1

        min_val = -1 if torch.min(img1) < -0.5 else 0
        dynamic_range = max_val - min_val
    else:
        dynamic_range = val_range

    pad = window_size // 2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2
    )

    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)
