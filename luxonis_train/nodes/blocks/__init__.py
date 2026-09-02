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
