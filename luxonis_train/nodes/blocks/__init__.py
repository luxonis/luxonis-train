"""Layer blocks shared by more than one node.

A block that only one node uses lives beside that node instead.

Some blocks subclass `Reparameterizable`. Such a block trains as a
multi-branch module and folds into a single convolution before export,
which keeps the accuracy of training and the speed of inference.

"""

from contextlib import suppress

from .blocks import (
    DFL,
    AttentionRefinementBlock,
    BlockRepeater,
    ConvBlock,
    ConvStack,
    CSPStackRepBlock,
    DropPath,
    EfficientDecoupledBlock,
    FeatureFusionBlock,
    GeneralReparameterizableBlock,
    SegProto,
    SpatialPyramidPoolingBlock,
    SqueezeExciteBlock,
    UpscaleOnline,
    autopad,
)
from .resnet import ResNetBlock, ResNetBottleneck
from .unet import (
    EncoderBlock,
    SimpleDecoder,
    SimpleDecoderBlock,
    SimpleEncoder,
    UNetDecoder,
    UNetDecoderBlock,
    UNetEncoder,
    UpBlock,
)

with suppress(ImportError):
    from aimet_torch.v2.nn import QuantizationMixin

    QuantizationMixin.ignore(DropPath)
    QuantizationMixin.ignore(UpscaleOnline)

__all__ = [
    "DFL",
    "AttentionRefinementBlock",
    "BlockRepeater",
    "CSPStackRepBlock",
    "ConvBlock",
    "ConvStack",
    "DropPath",
    "EfficientDecoupledBlock",
    "EncoderBlock",
    "FeatureFusionBlock",
    "GeneralReparameterizableBlock",
    "ResNetBlock",
    "ResNetBottleneck",
    "SegProto",
    "SimpleDecoder",
    "SimpleDecoderBlock",
    "SimpleEncoder",
    "SpatialPyramidPoolingBlock",
    "SqueezeExciteBlock",
    "UNetDecoder",
    "UNetDecoderBlock",
    "UNetEncoder",
    "UpBlock",
    "UpscaleOnline",
    "autopad",
]
