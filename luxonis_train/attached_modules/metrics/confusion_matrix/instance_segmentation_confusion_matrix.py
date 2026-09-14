"""The confusion matrix for instance segmentation, which reports one
matrix for the boxes and one for the masks.
"""

from torch import Tensor
from typing_extensions import override

from luxonis_train.tasks import Tasks

from .detection_confusion_matrix import DetectionConfusionMatrix
from .recognition_confusion_matrix import RecognitionConfusionMatrix
from .utils import preprocess_instance_masks


class InstanceSegmentationConfusionMatrix(
    DetectionConfusionMatrix, RecognitionConfusionMatrix
):
    r"""Box and mask confusion matrices for instance segmentation.

    The class combines `DetectionConfusionMatrix` for the boxes with
    `RecognitionConfusionMatrix` for the pixels of the masks. It has no
    constructor of its own. `DetectionConfusionMatrix.__init__` takes
    ``iou_threshold`` and creates the box matrix. Through ``super()``, it
    also runs `RecognitionConfusionMatrix.__init__`, which creates the
    pixel matrix.

    Inputs:
        - ``boundingbox`` (``list[Tensor]``): :math:`\left[M_i,
          6\right]` per image, ``[x1, y1, x2, y2, conf, class]``, pixels
        - ``instance_segmentation`` (``list[Tensor]``):
          :math:`\left[M_i, H, W\right]` per image, binary
        - ``target_boundingbox`` (``Tensor``): :math:`\left[N,
          6\right]`, ``[batch, class, x, y, w, h]``, ``xywh`` normalized
        - ``target_instance_segmentation`` (``Tensor``): :math:`\left[N,
          H, W\right]`, one per target box

    Outputs:
        - ``detection_mcc`` (``Tensor``): scalar MCC of the box matrix
        - ``detection_confusion_matrix`` (``Tensor``):
          :math:`\left[n_{classes} + 1, n_{classes} + 1\right]` box
          counts, last row and column are background
        - ``segmentation_mcc`` (``Tensor``): scalar MCC of the pixel
          matrix
        - ``segmentation_confusion_matrix`` (``Tensor``):
          :math:`\left[n_{classes}, n_{classes}\right]` pixel counts, or
          :math:`\left[2, 2\right]` for one class

    Formula:
        The box matrix follows `DetectionConfusionMatrix`. For the pixel
        matrix, `preprocess_instance_masks` merges the masks of one
        class in one image into one semantic mask.
        `RecognitionConfusionMatrix` then counts the pixels. With more
        than one class, each pixel gets the lowest class whose mask
        holds it. **A pixel in no mask counts as the first class, in the
        predictions and in the targets.** The first class has the index
        ``0``. With one class, each pixel is in the mask or not.

    References:
        - Source: This project.
        - License: Apache-2.0 (this project)

    Notes:
        The result has no ``mcc`` key, but `get_main_metric` selects
        ``mcc`` when a confusion matrix is the main metric. So this
        metric does not work as the main metric. **The reset does not
        clear the box matrix.** The inherited
        `RecognitionConfusionMatrix.reset` clears only the pixel matrix.
        So the box matrix of an epoch also holds the counts of all
        earlier epochs.

    Example:
        Attached to a ``PrecisionSegmentBBoxHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: PrecisionSegmentBBoxHead
              inputs: [RepPANNeck]
              metrics:
                - name: InstanceSegmentationConfusionMatrix

    Compatible with:
        - Nodes: `PrecisionSegmentBBoxHead`

    """

    supported_tasks = [Tasks.INSTANCE_SEGMENTATION]

    @override
    def update(
        self,
        boundingbox: list[Tensor],
        instance_segmentation: list[Tensor],
        target_boundingbox: Tensor,
        target_instance_segmentation: Tensor,
    ) -> None:
        """Add the boxes and the masks of one batch to both matrices.

        `DetectionConfusionMatrix.update` counts the boxes. It writes
        ``xyxy`` pixels into ``target_boundingbox``.
        `preprocess_instance_masks` then merges the masks of each class
        into semantic masks of shape ``[B, n_classes, H, W]``.
        `RecognitionConfusionMatrix.update` counts their pixels.

        Args:
            boundingbox (``list[Tensor]``): The predicted boxes of each
                image, of shape ``[M_i, 6]``, as
                ``[x1, y1, x2, y2, conf, class]`` in pixels. The length
                of the list is the batch size.
            instance_segmentation (``list[Tensor]``): The predicted masks
                of each image, of shape ``[M_i, H, W]``, one for each
                box. A nonzero value marks the object. ``H`` and ``W``
                are the height and width of
                `BaseAttachedModule.original_in_shape`.
            target_boundingbox (``Tensor``): The ``boundingbox`` label of
                the batch, of shape ``[N, 6]``, as
                ``[batch_index, class, x, y, w, h]``. The values are
                normalized, and ``x`` and ``y`` are the top-left corner.
            target_instance_segmentation (``Tensor``): The
                ``instance_segmentation`` label of the batch, of shape
                ``[N, H, W]``, one mask for each target box.

        """
        DetectionConfusionMatrix.update(self, boundingbox, target_boundingbox)
        RecognitionConfusionMatrix.update(
            self,
            *preprocess_instance_masks(
                boundingbox,
                instance_segmentation,
                target_boundingbox,
                target_instance_segmentation,
                self.n_classes,
                *self.original_in_shape[1:],
                device=self.device,
            ),
        )

    @override
    def compute(self) -> dict[str, Tensor]:
        """Return the MCC and the matrix of the boxes and the pixels.

        Returns:
            ``dict[str, Tensor]``: The dictionary holds:

            - ``"detection_mcc"`` and ``"detection_confusion_matrix"``:
              the ``"mcc"`` and the ``"confusion_matrix"`` of
              `DetectionConfusionMatrix.compute`.
            - ``"segmentation_mcc"`` and
              ``"segmentation_confusion_matrix"``: the ``"mcc"`` and the
              ``"confusion_matrix"`` of
              `RecognitionConfusionMatrix.compute`, over the pixels.

        Example:
            The batch has one image of ``2`` by ``2`` pixels, with one
            target box of class ``1``. The predicted box matches it, but
            the predicted mask misses one pixel of the target mask. The
            two pixels in no mask count as class ``0``. A
            ``SimpleNamespace`` stands in for the node.

            >>> import torch
            >>> from types import SimpleNamespace
            >>> node = SimpleNamespace(
            ...     task=None,
            ...     n_classes=2,
            ...     original_in_shape=torch.Size([3, 2, 2]),
            ... )
            >>> metric = InstanceSegmentationConfusionMatrix(node=node)
            >>> boxes = [torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.9, 1.0]])]
            >>> masks = [torch.tensor([[[1, 0], [0, 0]]])]
            >>> target = torch.tensor([[0, 1, 0.0, 0.0, 1.0, 1.0]])
            >>> target_masks = torch.tensor([[[1, 1], [0, 0]]])
            >>> metric.update(boxes, masks, target, target_masks)
            >>> result = metric.compute()
            >>> result["detection_confusion_matrix"].tolist()
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
            >>> result["segmentation_confusion_matrix"].tolist()
            [[2, 0], [1, 1]]

        """
        det_result = DetectionConfusionMatrix.compute(self)
        rec_result = RecognitionConfusionMatrix.compute(self)
        det_renamed = {
            "detection_mcc": det_result["mcc"],
            "detection_confusion_matrix": det_result["confusion_matrix"],
        }
        rec_renamed = {
            "segmentation_mcc": rec_result["mcc"],
            "segmentation_confusion_matrix": rec_result["confusion_matrix"],
        }

        return det_renamed | rec_renamed
