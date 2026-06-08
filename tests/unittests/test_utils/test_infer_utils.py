from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from luxonis_ml.typing import PathType

from luxonis_train.core.utils.infer_utils import infer_from_loader
from luxonis_train.lightning import LuxonisOutput


class _MockTrainer:
    def __init__(self, prediction: LuxonisOutput, callbacks: list[object]):
        self._prediction = prediction
        self.callbacks = callbacks

    def predict(
        self, lightning_module: object, loader: list[object], **_
    ) -> list[LuxonisOutput] | None:

        for batch_idx, _ in enumerate(loader):
            for callback in list(self.callbacks):
                write_on_batch_end = getattr(
                    callback, "write_on_batch_end", None
                )
                if write_on_batch_end is None:
                    continue
                write_on_batch_end(
                    self,
                    lightning_module,
                    self._prediction,
                    None,
                    None,
                    batch_idx,
                    0,
                )
        return None


def test_infer_from_loader_temporary_callback_does_not_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    saved_paths: list[str] = []
    monkeypatch.setattr(
        "luxonis_train.core.utils.infer_utils.cv2.imwrite",
        lambda path, _: saved_paths.append(Path(path).name) or True,
    )

    prediction = LuxonisOutput(
        outputs={},
        losses={},
        visualizations={
            "DiscSubNetHead": {
                "SegmentationVisualizer": torch.zeros((1, 3, 4, 4))
            }
        },
    )
    existing_callback = object()
    trainer = _MockTrainer(prediction, [existing_callback])
    model: Any = SimpleNamespace(
        pl_trainer=trainer,
        lightning_module=object(),
    )
    loader: Any = [object()]
    img_paths: list[PathType] = [tmp_path / "first.png"]

    infer_from_loader(model, loader, tmp_path, img_paths)

    assert trainer.callbacks == [existing_callback]
    assert saved_paths == ["first_DiscSubNetHead_SegmentationVisualizer.png"]

    trainer.predict(model.lightning_module, loader)

    assert trainer.callbacks == [existing_callback]
    assert saved_paths == ["first_DiscSubNetHead_SegmentationVisualizer.png"]

    infer_from_loader(model, loader, tmp_path, img_paths)

    assert trainer.callbacks == [existing_callback]
    assert saved_paths == [
        "first_DiscSubNetHead_SegmentationVisualizer.png",
        "first_DiscSubNetHead_SegmentationVisualizer.png",
    ]
