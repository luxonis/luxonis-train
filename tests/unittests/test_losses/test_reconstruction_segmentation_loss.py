import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.losses import (
    ReconstructionSegmentationLoss,
)
from luxonis_train.nodes import DiscSubNetHead
from luxonis_train.tasks import Tasks

from .test_utils import load_checkpoint


class DummyPrecisionSegmentBBoxHead(DiscSubNetHead, register=False):
    task = Tasks.ANOMALY_DETECTION
    original_in_shape: Size = Size([3, 256, 256])

    def forward(self, _: Tensor) -> Tensor: ...


def test_reconstruction_segmentation_loss():
    loss = ReconstructionSegmentationLoss(node=DummyPrecisionSegmentBBoxHead())
    (
        predictions,
        reconstructed,
        target_original_segmentation,
        target_segmentation,
        expected_sub_losses,
    ) = load_checkpoint("reconstruction_segmentation_loss_data.pt")
    result = loss(
        predictions,
        reconstructed,
        target_original_segmentation,
        target_segmentation,
    )[1]
    for key, value in result.items():
        assert torch.isclose(value, expected_sub_losses[key], atol=1e-3)
