from typing import TypedDict

import torch
from torch import Tensor, nn

from luxonis_train.config.config import (
    NormalizeAugmentationConfig,
    PreprocessingConfig,
)
from luxonis_train.core.utils.export_utils import (
    get_preprocessing,
    replace_weights,
)

_ORIGINAL_WEIGHT = 1.0
_REPLACEMENT_WEIGHT = 2.0


def test_get_preprocessing_skips_inactive_normalization():
    cfg = PreprocessingConfig(
        color_space="BGR",
        normalize=NormalizeAugmentationConfig(
            active=False,
            params={
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        ),
    )

    mean, scale, color_space = get_preprocessing(
        cfg, "Exporting to NN Archive"
    )

    assert mean is None
    assert scale is None
    assert color_space == "BGR"


def test_get_preprocessing_returns_scaled_active_normalization():
    cfg = PreprocessingConfig(
        normalize=NormalizeAugmentationConfig(
            active=True,
            params={
                "mean": [0.5, 0.25, 0.125],
                "std": [0.1, 0.2, 0.4],
            },
        ),
    )

    mean, scale, color_space = get_preprocessing(cfg)

    assert mean == [127.5, 63.75, 31.875]
    assert scale == [25.5, 51.0, 102.0]
    assert color_space == "RGB"


def test_replace_weights_restores_original_state():
    module = _CheckpointModule()
    checkpoint: _Checkpoint = {
        "state_dict": {"weight": torch.tensor([_REPLACEMENT_WEIGHT])}
    }

    with replace_weights(module, checkpoint):
        assert module.weight.item() == _REPLACEMENT_WEIGHT

    assert module.weight.item() == _ORIGINAL_WEIGHT


class _Checkpoint(TypedDict):
    state_dict: dict[str, Tensor]


class _CheckpointModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([_ORIGINAL_WEIGHT]))

    def load_checkpoint(self, ckpt: _Checkpoint) -> None:
        self.load_state_dict(ckpt["state_dict"])
