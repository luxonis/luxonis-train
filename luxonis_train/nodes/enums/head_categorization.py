from enum import Enum


class ImplementedHeads(Enum):
    """Task categorization for the implemented heads."""

    ClassificationHead = "Classification"
    EfficientBBoxHead = "ObjectDetectionYOLO"
    ImplicitKeypointBBoxHead = "KeypointDetectionYOLO"
    SegmentationHead = "Segmentation"
    BiSeNetHead = "Segmentation"
