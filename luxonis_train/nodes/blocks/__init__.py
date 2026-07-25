from contextlib import suppress

from .blocks import (
    DFL,
    AttentionRefinmentBlock,
    BlockRepeater,
    ConvBlock,
    ConvStack,
    CSPStackRepBlock,
    DropPath,
    EfficientDecoupledBlock,
    FeatureFusionBlock,
    GeneralReparametrizableBlock,
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
    "AttentionRefinmentBlock",
    "BlockRepeater",
    "CSPStackRepBlock",
    "ConvBlock",
    "ConvStack",
    "DropPath",
    "EfficientDecoupledBlock",
    "EncoderBlock",
    "FeatureFusionBlock",
    "GeneralReparametrizableBlock",
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
