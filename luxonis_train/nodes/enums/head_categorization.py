from enum import Enum

class ImplementedHeads(Enum):
    """ Task categorization for the implemented heads. """

    ClassificationHead = "Classification"
    EfficientBBoxHead = "ObjectDetectionYOLO" # TODO ObjectDetectionYOLO/ObjectDetectionSSD?
    ImplicitKeypointBBoxHead = "KeypointDetection"
    SegmentationHead = "Segmentation"
    BiSeNetHead = "Segmentation" # TODO: SemanticSegmentation?

    