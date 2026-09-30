"""Layer blocks that the nodes build on.

The package holds convolution blocks and stacks, RepVGG-style blocks,
attention and pooling blocks, and parts of the detection and
segmentation heads. It also holds the ResNet blocks and the U-Net
encoder and decoder blocks.

`GeneralReparameterizableBlock` subclasses `Reparameterizable`. It
trains with parallel branches and fuses them into a single convolution
when its node enters export mode.

When ``aimet_torch`` is installed, the import of the package tells its
quantization to ignore `DropPath` and `UpscaleOnline`.

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
