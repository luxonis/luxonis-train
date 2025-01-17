from enum import Enum


class TaskType(str, Enum):
    """Tasks supported by nodes in LuxonisTrain."""

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    BOUNDINGBOX = "boundingbox"
    KEYPOINTS = "keypoints"
    ARRAY = "array"
