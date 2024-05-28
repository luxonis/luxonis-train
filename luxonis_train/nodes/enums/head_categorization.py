from enum import Enum


class ImplementedHeads(Enum):
    """Task categorization for the implemented heads."""

    ClassificationHead = "Classification"
    EfficientBBoxHead = "ObjectDetectionYOLO"
    ImplicitKeypointBBoxHead = "KeypointDetectionYOLO"
    EfficientKeypointBBoxHead = "Keypoint"
    SegmentationHead = "Segmentation"
    BiSeNetHead = "Segmentation"


class ImplementedHeadsIsSoxtmaxed(Enum):
    """Softmaxed output categorization for the implemented heads."""

    ClassificationHead = False
    EfficientBBoxHead = None
    ImplicitKeypointBBoxHead = None
    EfficientKeypointBBoxHead = None
    SegmentationHead = False
    BiSeNetHead = False
