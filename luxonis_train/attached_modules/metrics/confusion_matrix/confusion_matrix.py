"""The factory that returns the confusion matrix for the task of a
node.
"""

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.registry import METRICS
from luxonis_train.tasks import Tasks

from .detection_confusion_matrix import DetectionConfusionMatrix
from .fomo_confusion_matrix import FomoConfusionMatrix
from .instance_segmentation_confusion_matrix import (
    InstanceSegmentationConfusionMatrix,
)
from .recognition_confusion_matrix import RecognitionConfusionMatrix


@METRICS.register()  # type: ignore
class ConfusionMatrix:
    """Factory for the confusion matrix metric of a node.

    The class is in the `METRICS` registry, so a config can name it. A
    call does not return an instance of this class. It returns a new
    metric of the class that the task of the node selects:

    - ``Tasks.CLASSIFICATION`` and ``Tasks.SEGMENTATION``:
      `RecognitionConfusionMatrix`.
    - ``Tasks.BOUNDINGBOX`` and ``Tasks.INSTANCE_KEYPOINTS``:
      `DetectionConfusionMatrix`.
    - ``Tasks.INSTANCE_SEGMENTATION``:
      `InstanceSegmentationConfusionMatrix`.
    - ``Tasks.FOMO``: `FomoConfusionMatrix`.

    Inputs:
        - The inputs of the selected metric, see its class.

    Outputs:
        - The outputs of the selected metric. The metric for instance
          segmentation returns ``detection_mcc``,
          ``detection_confusion_matrix``, ``segmentation_mcc``, and
          ``segmentation_confusion_matrix``. The other metrics return
          ``mcc`` and ``confusion_matrix``.

    References:
        - Source: This project.
        - License: Apache-2.0 (this project)

    Notes:
        For a node with any other task, the call raises ``ValueError``.
        ``Tasks.INSTANCE_SEGMENTATION_KEYPOINTS`` is an example of such
        a task. When a confusion matrix is the main metric,
        `get_main_metric` selects its ``mcc`` value. The metric for
        instance segmentation has no such value.

    Example:
        Attached to a ``ClassificationHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: ClassificationHead
              inputs: [ResNet]
              metrics:
                - name: ConfusionMatrix

    Compatible with:
        - Used by:

          - `ClassificationModel`
          - `DetectionModel`
          - `FOMOModel`
          - `InstanceSegmentationModel`
          - `KeypointDetectionModel`
          - `SegmentationModel`

        - Nodes:

          - `BiSeNetHead`
          - `ClassificationHead`
          - `DDRNetSegmentationHead`
          - `EfficientBBoxHead`
          - `EfficientKeypointBBoxHead`
          - `FOMOHead`
          - `PrecisionBBoxHead`
          - `PrecisionSegmentBBoxHead`
          - `SegmentationHead`
          - `TransformerClassificationHead`
          - `TransformerSegmentationHead`

    """

    def __new__(
        cls, node: BaseNode, **kwargs
    ) -> (
        RecognitionConfusionMatrix
        | DetectionConfusionMatrix
        | InstanceSegmentationConfusionMatrix
    ):
        """Build the confusion matrix metric for a node.

        Args:
            node (BaseNode): The node that the metric attaches to. Its
                ``task`` selects the metric class, see the class
                docstring.
            **kwargs (``Any``): Keyword arguments forwarded to the
                constructor of the selected metric. Only
                `DetectionConfusionMatrix` and its subclasses accept
                ``iou_threshold``.

        Returns:
            RecognitionConfusionMatrix | DetectionConfusionMatrix | InstanceSegmentationConfusionMatrix:
            A new metric, attached to ``node``. A `FomoConfusionMatrix`
            is a `DetectionConfusionMatrix`.

        Raises:
            ValueError: When the task of ``node`` is ``None``, or when
                the factory does not support the task.

        Example:
            A ``SimpleNamespace`` stands in for the node.

            >>> from types import SimpleNamespace
            >>> from luxonis_train.tasks import Tasks
            >>> node = SimpleNamespace(
            ...     name="head", task=Tasks.FOMO, n_classes=2
            ... )
            >>> type(ConfusionMatrix(node=node)).__name__
            'FomoConfusionMatrix'

        """
        match node.task:
            case None:  # pragma: no cover
                raise ValueError(
                    f"Node {node.name} does not have the 'task' parameter set"
                )
            case Tasks.CLASSIFICATION | Tasks.SEGMENTATION:
                return RecognitionConfusionMatrix(node=node, **kwargs)
            case Tasks.BOUNDINGBOX | Tasks.INSTANCE_KEYPOINTS:
                return DetectionConfusionMatrix(node=node, **kwargs)
            case Tasks.INSTANCE_SEGMENTATION:
                return InstanceSegmentationConfusionMatrix(node=node, **kwargs)
            case Tasks.FOMO:
                return FomoConfusionMatrix(node=node, **kwargs)
            case _:  # pragma: no cover
                raise ValueError(
                    f"'ConfusionMatrix' does not support task '{node.task.name}'"
                )
