"""Backbones that turn an image into feature maps.

Most backbones return a list of feature maps. The stride increases from
one map to the next. A neck or a head then reads the maps. Some
backbones differ:

- `ContextSpatial` and `DDRNet` return features at 1/8 of the input
  size, for a segmentation head.
- `DinoV3` returns a list with one CLS embedding instead of feature
  maps when ``return_sequence`` is ``True``.
- `GhostFaceNet` targets face embeddings.
- `RecSubNet` returns a packet with an image reconstruction and the
  original input, for anomaly detection.

A backbone with variants takes ``variant`` to pick a size. Each backbone
docstring lists its variants and the parameters that each variant sets.

"""

from luxonis_train.nodes.backbones.dinov3.dinov3 import DinoV3

from .contextspatial import ContextSpatial
from .ddrnet import DDRNet
from .efficientnet import EfficientNet
from .efficientrep import EfficientRep
from .efficientvit import EfficientViT
from .ghostfacenet import GhostFaceNet
from .micronet import MicroNet
from .mobilenetv2 import MobileNetV2
from .mobileone import MobileOne
from .pplcnet_v3 import PPLCNetV3
from .recsubnet import RecSubNet
from .repvgg import RepVGG
from .resnet import ResNet
from .rexnetv1 import ReXNetV1_lite

__all__ = [
    "ContextSpatial",
    "DDRNet",
    "DinoV3",
    "EfficientNet",
    "EfficientRep",
    "EfficientViT",
    "GhostFaceNet",
    "MicroNet",
    "MobileNetV2",
    "MobileOne",
    "PPLCNetV3",
    "ReXNetV1_lite",
    "RecSubNet",
    "RepVGG",
    "ResNet",
]
