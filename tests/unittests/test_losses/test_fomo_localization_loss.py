import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.losses import FOMOLocalizationLoss
from luxonis_train.nodes.heads import FOMOHead
from luxonis_train.tasks import Tasks

from .test_utils import load_checkpoint


class DummyFOMOHead(FOMOHead, register=False):
    task = Tasks.FOMO
    original_in_shape: Size = Size([3, 384, 512])
    input_shapes = [
        {
            "features": [
                Size([2, 32, 96, 128]),
                Size([2, 64, 48, 64]),
                Size([2, 128, 24, 32]),
                Size([2, 256, 12, 16]),
            ]
        }
    ]  # type: ignore
    n_classes: int = 1

    def forward(self, _: Tensor) -> Tensor: ...


def test_fomo_localization_loss():
    loss = FOMOLocalizationLoss(node=DummyFOMOHead())
    heatmap, target, expected_loss = load_checkpoint(
        "fomo_localization_loss_data.pt"
    )
    result = loss(heatmap, target)
    assert torch.isclose(result, expected_loss, atol=1e-3)
