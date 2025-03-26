import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.losses import PrecisionDFLDetectionLoss
from luxonis_train.nodes import PrecisionBBoxHead
from luxonis_train.tasks import Tasks

from .test_utils import load_checkpoint


class DummyPrecisionBBoxHead(PrecisionBBoxHead, register=False):
    task = Tasks.BOUNDINGBOX
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

    def forward(self, _: Tensor) -> Tensor: ...


def test_precision_detection_loss():
    loss = PrecisionDFLDetectionLoss(node=DummyPrecisionBBoxHead())
    features, target, expected_sub_losses = load_checkpoint(
        "precision_dfl_detection_loss_data.pt"
    )
    result = loss(features, target)[1]
    for key, value in result.items():
        assert torch.isclose(value, expected_sub_losses[key], atol=1e-3)
