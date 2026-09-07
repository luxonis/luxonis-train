import torch
from torch import Size

from luxonis_train.nodes.backbones import PPLCNetV3


def test_recognition_backbone():
    backbone = _build_backbone(use_detection_backbone=False)
    out = backbone(torch.rand(2, 3, 48, 320))
    assert len(out) == 5
    assert out[-1].shape[-2:] == (1, 40)


def test_detection_backbone():
    backbone = _build_backbone(use_detection_backbone=True)
    out = backbone(torch.rand(2, 3, 48, 320))
    assert [f.shape[1] for f in out] == [15, 22, 53, 456]


def _build_backbone(use_detection_backbone: bool) -> PPLCNetV3:
    # `variant=` hides the injected parameters from pyright.
    _, variants = PPLCNetV3.get_variants()
    variant = variants["rec-light"]
    return PPLCNetV3(
        input_shapes=[{"features": [Size((3, 48, 320))]}],
        scale=variant["scale"],
        n_branches=variant["n_branches"],
        layer_params=variant["layer_params"],
        use_detection_backbone=use_detection_backbone,
        max_text_len=40,
    )
