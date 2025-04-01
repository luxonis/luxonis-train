import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.losses import EfficientKeypointBBoxLoss
from luxonis_train.nodes import EfficientKeypointBBoxHead
from luxonis_train.tasks import Tasks

from .test_utils import load_checkpoint


class DummyEfficientKeypointBBoxHead(
    EfficientKeypointBBoxHead, register=False
):
    task = Tasks.INSTANCE_KEYPOINTS
    original_in_shape: Size = Size([3, 384, 512])
    input_shapes = [
        {
            "features": [
                Size([2, 32, 48, 64]),
                Size([2, 64, 24, 32]),
                Size([2, 128, 12, 16]),
            ]
        }
    ]
    in_sizes = [
        Size([2, 32, 48, 64]),
        Size([2, 64, 24, 32]),
        Size([2, 128, 12, 16]),
    ]
    n_classes: int = 1
    n_keypoints: int = 17

    def forward(self, _: Tensor) -> Tensor: ...


def test_efficient_keypoint_bbox_loss():
    loss = EfficientKeypointBBoxLoss(
        node=DummyEfficientKeypointBBoxHead(),
        iou_type="ciou",
        n_warmup_epochs=0,
    )
    (
        features,
        class_scores,
        distributions,
        keypoints_raw,
        target_boundingbox,
        target_keypoints,
        expected_sub_losses,
    ) = load_checkpoint("efficient_keypoint_bbox_loss_data.pt")
    result = loss(
        features,
        class_scores,
        distributions,
        keypoints_raw,
        target_boundingbox,
        target_keypoints,
    )[1]
    for key, value in result.items():
        assert torch.isclose(value, expected_sub_losses[key], atol=1e-3)
