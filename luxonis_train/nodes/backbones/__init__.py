"""Backbones that turn an image into feature maps.

Most backbones return one feature map for each stride, which a neck or a
head then consumes. `DinoV3` also returns a single CLS embedding when
``return_sequence`` is true. `GhostFaceNet` targets face embeddings and
`RecSubNet` reconstructs an image without its anomalies.

Set ``variant`` to pick a size. Each backbone docstring lists the
variants it declares and the parameters each one sets.

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
