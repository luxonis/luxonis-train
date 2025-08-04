import hashlib

import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.visualizers import OCRVisualizer
from luxonis_train.nodes import OCRCTCHead
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet


class DummyOCRNode(OCRCTCHead, register=False):
    task = Tasks.FOMO

    @property
    def input_shapes(self) -> list[Packet[Size]]:
        return [{"features": [Size([2, 128, 12, 16])]}]

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        return x

    def decoder(self, predictions: Tensor):
        """For this test, just treat each non-zero int as an ASCII
        character, then return a dummy probability of 1.0."""
        results = []
        for row in predictions:
            row = row[row != 0]
            text = "".join(chr(int(c)) for c in row)
            results.append((text, 1.0))
        return results


def test_ocr_visualizer():
    visualizer = OCRVisualizer(
        node=DummyOCRNode(alphabet=list("abcdefghijklmnopqrstuvwxyz"))
    )

    canvas = torch.zeros(2, 3, 100, 100, dtype=torch.uint8)

    predictions = torch.tensor(
        [
            [65, 66, 0, 0, 0],
            [88, 89, 90, 0, 0],
        ],
        dtype=torch.int,
    )

    targets = torch.tensor(
        [
            [72, 73, 0, 0, 0],
            [74, 75, 76, 0, 0],
        ],
        dtype=torch.int,
    )

    targets_viz, predictions_viz = visualizer(
        canvas.clone(),
        canvas.clone(),
        predictions,
        targets,
    )

    combined_viz = (
        torch.cat([targets_viz, predictions_viz], dim=0)
        .cpu()
        .numpy()
        .tobytes()
    )
    computed_hash = hashlib.sha256(combined_viz).hexdigest()

    assert (
        computed_hash
        == "fd2002331686f3579151061ec07367da5e3050bb60a085c1764f76486f5b2a91"
    )
