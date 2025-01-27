from dataclasses import dataclass
from enum import Enum
from functools import cached_property


class Task(str, Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    BOUNDINGBOX = "boundingbox"
    KEYPOINTS = "keypoints"
    POINTCLOUD = "pointcloud"
    EMBEDDINGS = "embeddings"
    ANOMALY_DETECTION = "anomaly_detection"
    OCR = "ocr"
    FOMO = "fomo"

    @cached_property
    def required_labels(self) -> set[str]:
        match self:
            case Task.CLASSIFICATION | Task.SEGMENTATION | Task.BOUNDINGBOX:
                return {self.value}
            case Task.ANOMALY_DETECTION:
                return {"segmentation", "original_segmentation"}
            case Task.POINTCLOUD:
                return {"keypoints"}
            case Task.KEYPOINTS:
                return {"keypoints", "boundingbox"}
            case Task.INSTANCE_SEGMENTATION:
                return {"instance_segmentation", "boundingbox"}
            case Task.FOMO:
                return {"boundingbox"}
            case Task.EMBEDDINGS:
                return {Metadata("id")}
            case Task.OCR:
                return {Metadata("text"), Metadata("text_length")}


@dataclass(unsafe_hash=True)
class Metadata(str):
    name: str
    # typ: UnionType | type

    @cached_property
    def value(self):
        return f"metadata/{self.name}"

    def __str__(self) -> str:
        return self.value
