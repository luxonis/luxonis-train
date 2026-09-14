"""Visualizers that draw the predictions of a node.

A visualizer returns one batch of images, or a pair of batches.
`combine_visualizations` puts a pair side by side, with the first batch
on the left, before the trainer logs it. In a pair, the first batch
shows the labels and the second the predictions.
`ClassificationVisualizer` returns the text images and the probability
plots instead, and `EmbeddingsVisualizer` returns a KDE plot and a
scatter plot.

Attach a visualizer in the ``visualizers`` field of a node.
``trainer.n_log_images`` caps how many images each node logs on a
validation or test epoch.

"""

from .base_visualizer import BaseVisualizer
from .bbox_visualizer import BBoxVisualizer
from .classification_visualizer import ClassificationVisualizer
from .embeddings_visualizer import EmbeddingsVisualizer
from .fomo_visualizer import FOMOVisualizer
from .instance_seg_keypoint_visualizer import InstanceSegKeypointVisualizer
from .instance_segmentation_visualizer import InstanceSegmentationVisualizer
from .keypoint_visualizer import KeypointVisualizer
from .ocr_visualizer import OCRVisualizer
from .segmentation_visualizer import SegmentationVisualizer
from .utils import (
    combine_visualizations,
    denormalize,
    draw_bounding_box_labels,
    draw_keypoint_labels,
    draw_segmentation_targets,
    get_color,
    get_denormalized_images,
    preprocess_images,
)

__all__ = [
    "BBoxVisualizer",
    "BaseVisualizer",
    "ClassificationVisualizer",
    "EmbeddingsVisualizer",
    "FOMOVisualizer",
    "InstanceSegKeypointVisualizer",
    "InstanceSegmentationVisualizer",
    "KeypointVisualizer",
    "OCRVisualizer",
    "SegmentationVisualizer",
    "combine_visualizations",
    "denormalize",
    "draw_bounding_box_labels",
    "draw_keypoint_labels",
    "draw_segmentation_targets",
    "get_color",
    "get_denormalized_images",
    "preprocess_images",
]
