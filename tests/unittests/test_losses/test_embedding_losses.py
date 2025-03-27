import torch
from torch import Size

from luxonis_train.attached_modules.losses import EmbeddingLossWrapper
from luxonis_train.nodes import GhostFaceNetHead
from luxonis_train.tasks import Tasks


class DummyGhostFaceNetHead(GhostFaceNetHead):
    Task = Tasks.EMBEDDINGS
    original_in_shape: Size = Size([3, 112, 112])
    input_shapes = [{"features": [Size([2, 3, 56, 56])]}]

    def forward(self, _: torch.Tensor) -> torch.Tensor: ...


def test_embedding_loss_wrapper():
    predictions = torch.full((10, 128), 0.5)
    target = torch.zeros(10, dtype=torch.long)
    loss_fn = EmbeddingLossWrapper(node=DummyGhostFaceNetHead())
    loss = loss_fn(predictions, target)
    assert torch.isfinite(loss)
