from enum import Enum


class ImplementedHeads(Enum):
    """Task categorization for the implemented heads."""

    ClassificationHead = "ClassificationParser"
    EfficientBBoxHead = "YOLO"
    EfficientKeypointBBoxHead = "YoloDetectionNetwork"
    SegmentationHead = "SegmentationParser"
    BiSeNetHead = "SegmentationParser"
    DDRNetSegmentationHead = "SegmentationParser"


class ImplementedHeadsIsSoxtmaxed(Enum):
    """Softmaxed output categorization for the implemented heads."""

    ClassificationHead = False
    EfficientBBoxHead = None
    EfficientKeypointBBoxHead = None
    SegmentationHead = False
    BiSeNetHead = False
    DDRNetSegmentationHead = False
