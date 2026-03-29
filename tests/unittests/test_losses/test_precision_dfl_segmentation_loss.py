import pytest
import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.losses import PrecisionDFLSegmentationLoss
from luxonis_train.nodes import PrecisionSegmentBBoxHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet

from .test_utils import load_checkpoint


class DummyPrecisionSegmentBBoxHead(PrecisionSegmentBBoxHead, register=False):
    task = Tasks.INSTANCE_SEGMENTATION
    original_in_shape: Size = Size([3, 384, 512])
    n_classes: int = 1

    @property
    def input_shapes(self) -> list[Packet[Size]]:  # pragma: no cover
        return [
            {
                "features": [
                    Size([2, 32, 48, 64]),
                    Size([2, 64, 24, 32]),
                    Size([2, 128, 12, 16]),
                ]
            }
        ]

    @property
    def in_sizes(self) -> list[Size]:
        return [
            Size([2, 32, 48, 64]),
            Size([2, 64, 24, 32]),
            Size([2, 128, 12, 16]),
        ]

    def forward(self, _: Tensor) -> Tensor: ...


@pytest.mark.parametrize("skip_stal", [True, False])
def test_precision_segmentation_loss(skip_stal: bool):
    loss = PrecisionDFLSegmentationLoss(
        node=DummyPrecisionSegmentBBoxHead(), skip_stal=skip_stal
    )
    (
        features,
        prototypes,
        mask_coeficients,
        target_boundingbox,
        target_instance_segmentation,
        expected_sub_losses,
    ) = load_checkpoint("precision_dfl_segmentation_loss_data.pt")
    result = loss(
        features,
        prototypes,
        mask_coeficients,
        target_boundingbox,
        target_instance_segmentation,
    )[1]
    assert result.keys() == expected_sub_losses.keys()
    for key, value in result.items():
        assert value.ndim == 0
        assert torch.isfinite(value)
        if skip_stal:
            assert torch.isclose(value, expected_sub_losses[key], atol=1e-3)
