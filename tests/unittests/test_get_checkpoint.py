import pickle
from pathlib import Path
from typing import NoReturn

import pytest
import torch

from luxonis_train.core import LuxonisModel


def test_get_checkpoint_deletes_file_when_load_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    saved: list[Path] = []

    def save_checkpoint(path: str | Path, weights_only: bool) -> Path:
        saved.append(Path(path))
        saved[-1].write_bytes(b"")
        return saved[-1]

    def load(*_: object, **__: object) -> NoReturn:
        raise pickle.UnpicklingError("Weights only load failed.")

    model = LuxonisModel.__new__(LuxonisModel)
    monkeypatch.setattr(model, "save_checkpoint", save_checkpoint)
    monkeypatch.setattr(torch, "load", load)

    with pytest.raises(pickle.UnpicklingError):
        model.get_checkpoint()

    assert len(saved) == 1
    assert not saved[0].exists()
