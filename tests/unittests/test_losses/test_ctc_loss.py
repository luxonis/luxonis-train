import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.losses import CTCLoss
from luxonis_train.nodes import OCRCTCHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet


class DummyOCRCTCHead(OCRCTCHead, register=False):
    task = Tasks.OCR
    original_in_shape: Size = Size([3, 384, 512])

    @property
    def input_shapes(self) -> list[Packet[Size]]:
        return [{"features": [Size([2, 128, 12, 16])]}]

    def __init__(self, alphabet: str, **kwargs):
        super().__init__(alphabet=alphabet.split(), **kwargs)
        self.alphabet = alphabet
        mapping = {c: i for i, c in enumerate(self.alphabet)}

        class DummyEncoder:
            def __init__(self, mapping: dict[str, int]):
                self.char_to_int = mapping

            def __call__(self, targets: Tensor) -> Tensor:
                return targets.view(-1)

        self._encoder = DummyEncoder(mapping)

    def forward(self, _: Tensor) -> Tensor: ...


def test_ctc_loss():
    B, T, C = 2, 4, 4
    predictions = torch.full((B, T, C), 0.5).log_softmax(2)
    targets = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long)
    expected_loss = torch.tensor(3.5993)
    loss_fn = CTCLoss(
        node=DummyOCRCTCHead(alphabet="abcdefghijklmnopqrstuvwxyz"),
        use_focal_loss=False,
    )
    loss = loss_fn(predictions, targets)
    assert torch.isclose(loss, expected_loss, atol=1e-3)
