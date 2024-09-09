import torch

from luxonis_train.nodes.blocks import SqueezeExciteBlock, autopad


def test_autopad():
    assert autopad(1, 2) == 2
    assert autopad(2) == 1
    assert autopad((2, 4)) == (1, 2)


def test_squeeze_excite_block():
    se_block = SqueezeExciteBlock(64, 32)
    x = torch.rand(1, 64, 224, 224)
    assert se_block(x).shape == (1, 64, 224, 224)
