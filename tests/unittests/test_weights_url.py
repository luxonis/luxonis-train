import pytest
from torch import Size

from luxonis_train.nodes import EfficientRep, RepPANNeck

IMAGE = Size([3, 64, 64])
# The outputs of the "n" variant of `EfficientRep` for `IMAGE`.
FEATURES = [
    Size([1, 32, 16, 16]),
    Size([1, 64, 8, 8]),
    Size([1, 128, 4, 4]),
    Size([1, 256, 2, 2]),
]


def _build(
    node: type[EfficientRep | RepPANNeck], variant: str | None
) -> EfficientRep | RepPANNeck:
    features = FEATURES if node is RepPANNeck else [IMAGE]
    return node(
        input_shapes=[{"features": features}],
        original_in_shape=IMAGE,
        variant=variant,
    )


@pytest.mark.parametrize("node", [EfficientRep, RepPANNeck])
@pytest.mark.parametrize("variant", [None, "m", "medium"])
def test_download_needs_a_variant_with_weights(
    node: type[EfficientRep | RepPANNeck], variant: str | None
):
    with pytest.raises(ValueError, match="only with the variants"):
        _build(node, variant).load_checkpoint()


@pytest.mark.parametrize(
    ("node", "name"),
    [(EfficientRep, "efficientrep"), (RepPANNeck, "reppanneck")],
)
def test_alias_selects_the_checkpoint(
    node: type[EfficientRep | RepPANNeck], name: str
):
    url = _build(node, "small").get_weights_url()
    assert url == f"{{github}}/{name}_s_coco.ckpt"
