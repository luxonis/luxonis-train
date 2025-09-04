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
        """ReconstructionSegmentationLoss implements a combined loss
        function for reconstruction and segmentation tasks.

        It combines L2 loss for reconstruction, SSIM loss, and Focal
        loss for segmentation.

        @type alpha: float
        @param alpha: Weighting factor for the rare class in the focal loss. Defaults to C{1}.
        @type gamma: float
        @param gamma: Focusing parameter for the focal loss. Defaults to C{2.0}.
        @type smooth: float
        @param smooth: Label smoothing factor for the focal loss.  Defaults to C{0.0}.
        @type reduction: Literal["none", "mean", "sum"]
        @param reduction: Reduction type for the focal loss.. Defaults to C{"mean"}.
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
    def __init__(
        self,
        window_size: int = 11,
        size_average: bool = True,
        val_range: float | None = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        device = img1.device
        with amp.autocast(device_type=device.type, enabled=False):
            img1 = img1.float()
            img2 = img2.float()

            (_, channel, _, _) = img1.size()
            if channel == self.channel and self.window.dtype == img1.dtype:
                window = self.window.to(device)
            else:
                window = (
                    create_window(self.window_size, channel)
                    .to(device)
                    .type(img1.dtype)
                )
                self.window = window
                self.channel = channel

            s_score = ssim(
                img1,
                img2,
                window=window,
                window_size=self.window_size,
                size_average=self.size_average,
            )
            return 1.0 - s_score


def create_window(window_size: int, channel: int = 1) -> Tensor:
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def gaussian(window_size: int, sigma: float) -> Tensor:
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
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1

        min_val = -1 if torch.min(img1) < -0.5 else 0
        dynamic_range = max_val - min_val
    else:
        dynamic_range = val_range

    padd = window_size // 2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    )

    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)
