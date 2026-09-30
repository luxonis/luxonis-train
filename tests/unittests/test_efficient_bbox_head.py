import pytest
from torch import Size

from luxonis_train.nodes import EfficientBBoxHead, EfficientKeypointBBoxHead


def _shapes(channels: list[int]) -> dict:
    sizes = [Size([1, c, 32 >> i, 32 >> i]) for i, c in enumerate(channels)]
    return {
        "input_shapes": [{"features": sizes}],
        "original_in_shape": Size([3, 256, 256]),
    }


# The output channels of `RepPANNeck` for the variants "n", "s", and "l".
@pytest.mark.parametrize(
    ("channels", "size"),
    [([32, 64, 128], "n"), ([64, 128, 256], "s"), ([128, 256, 512], "l")],
)
def test_weights_url_follows_the_input_channels(
    channels: list[int], size: str
):
    head = EfficientBBoxHead(n_classes=2, **_shapes(channels))
    assert head.get_weights_url() == (
        f"{{github}}/efficientbbox_head_{size}_coco.ckpt"
    )


def test_no_weights_for_other_input_channels():
    # "m" of `RepPANNeck` has no head checkpoint.
    with pytest.raises(ValueError, match="does not implement"):
        EfficientBBoxHead(
            n_classes=2, weights="download", **_shapes([96, 192, 384])
        )


def test_keypoint_head_uses_the_box_checkpoint():
    # The checkpoint keys match the box branches of the keypoint head.
    head = EfficientKeypointBBoxHead(
        n_classes=2, n_keypoints=3, **_shapes([32, 64, 128])
    )
    assert head.get_weights_url() == "{github}/efficientbbox_head_n_coco.ckpt"
