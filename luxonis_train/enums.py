from enum import Enum


class TaskType(str, Enum):
    """Tasks supported by nodes in LuxonisTrain."""

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    BOUNDINGBOX = "boundingbox"
    KEYPOINTS = "keypoints"
    LABEL = "label"
    ARRAY = "array"
