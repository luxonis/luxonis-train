import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.losses import (
    PrecisionDFLSegmentationLoss,
    WeightedPrecisionDFLSegmentationLoss,
)
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


def test_default_params_match_parent():
    """With default params, weighted loss should match the parent."""
    dummy_node = DummyPrecisionSegmentBBoxHead()
    parent_loss = PrecisionDFLSegmentationLoss(node=dummy_node)
    weighted_loss = WeightedPrecisionDFLSegmentationLoss(node=dummy_node)

    inputs = load_checkpoint("precision_dfl_segmentation_loss_data.pt")
    # Unpack only the model inputs (not expected_sub_losses)
    model_inputs = inputs[:-1]

    parent_result = parent_loss(*model_inputs)
    weighted_result = weighted_loss(*model_inputs)

    assert torch.isclose(parent_result[0], weighted_result[0], atol=1e-5)
    for key in parent_result[1]:
        assert torch.isclose(
            parent_result[1][key], weighted_result[1][key], atol=1e-5
        )


def test_mask_pos_weight_increases_seg_loss():
    """A mask_pos_weight > 1 should increase the seg sub-loss."""
    dummy_node = DummyPrecisionSegmentBBoxHead()
    base_loss = WeightedPrecisionDFLSegmentationLoss(
        node=dummy_node, mask_pos_weight=1.0,
    )
    weighted_loss = WeightedPrecisionDFLSegmentationLoss(
        node=dummy_node, mask_pos_weight=5.0,
    )

    model_inputs = load_checkpoint(
        "precision_dfl_segmentation_loss_data.pt"
    )[:-1]

    _, base_sub = base_loss(*model_inputs)
    _, weighted_sub = weighted_loss(*model_inputs)

    for key in ["class", "iou", "dfl"]:
        assert torch.isclose(base_sub[key], weighted_sub[key], atol=1e-5)

    assert weighted_sub["seg"] > base_sub["seg"]
