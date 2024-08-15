from enum import Enum


class ImplementedHeads(Enum):
    """Task categorization for the implemented heads."""

    ClassificationHead = "ClassificationParser"
    EfficientBBoxHead = "YoloDetectionNetwork"
    ImplicitKeypointBBoxHead = "YoloDetectionNetwork"
    EfficientKeypointBBoxHead = "YoloDetectionNetwork"
    SegmentationHead = "SegmentationParser"
    BiSeNetHead = "SegmentationParser"


class ImplementedHeadsIsSoxtmaxed(Enum):
    """Softmaxed output categorization for the implemented heads."""

    ClassificationHead = False
    EfficientBBoxHead = None
    ImplicitKeypointBBoxHead = None
    EfficientKeypointBBoxHead = None
    SegmentationHead = False
    BiSeNetHead = False
