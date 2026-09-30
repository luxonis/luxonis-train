"""Mean average precision over instance masks and their boxes."""

from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision
from typing_extensions import override

from luxonis_train.attached_modules.metrics import BaseMetric
from luxonis_train.tasks import Tasks

from .utils import compute_metric_lists, postprocess_metrics


class MeanAveragePrecisionSegmentation(MeanAveragePrecision, BaseMetric):
    r"""Mean average precision metric for instance segmentation masks.

    The class is a subclass of the ``torchmetrics``
    ``MeanAveragePrecision``, with ``iou_type=("bbox", "segm")``. It
    evaluates the boxes and the masks of the instances. `update`
    converts the boxes and masks of the node to the input of
    ``torchmetrics``. `compute` adds the F1 scores and splits the
    per-class values.

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
        - ``segm_map`` (``Tensor``): scalar, the main metric
        - ``dict[str, Tensor]``: the other box and mask AP, AR, and F1
          values, see `compute`

    Formula:
        The mask metrics use the IoU of the predicted mask :math:`P` and
        the target mask :math:`G`, counted in pixels:

        .. math::

            \text{IoU} = \frac{|P \cap G|}{|P \cup G|}

        A prediction is a true positive when its IoU with a target of
        the same class is at least the threshold :math:`t`. For class
        :math:`c`, :math:`\text{AP}_{c,t}` is the area under the
        precision-recall curve. With the default thresholds
        :math:`T = \{0.5, 0.55, \ldots, 0.95\}` and the classes
        :math:`C`:

        .. math::

            \text{mAP} = \frac{1}{|C| \, |T|} \sum_{c \in C} \sum_{t \in T} \text{AP}_{c,t}

        The box metrics use the same steps with the box IoU.

    References:
        - Source: Wraps `torchmetrics
          <https://github.com/Lightning-AI/torchmetrics>`_ (Apache-2.0).
        - License: Apache-2.0 (this project)

    Notes:
        The
        `luxonis_train.attached_modules.metrics.mean_average_precision.MeanAveragePrecision`
        factory selects the ``"faster_coco_eval"`` backend. A config
        that names this class directly gets the ``torchmetrics``
        default, ``"pycocotools"``. The per-class values carry the class
        names of the node, with each space replaced by an underscore.

    Example:
        Attached to a ``PrecisionSegmentBBoxHead`` in ``model.nodes``:

        .. code-block:: yaml

            - name: PrecisionSegmentBBoxHead
              inputs: [RepPANNeck]
              metrics:
                - name: MeanAveragePrecisionSegmentation

    Compatible with:
        - Nodes: `PrecisionSegmentBBoxHead`

    """

    supported_tasks = [
        Tasks.INSTANCE_SEGMENTATION,
        Tasks.INSTANCE_SEGMENTATION_KEYPOINTS,
    ]

    def __init__(self, **kwargs):
        """Initialize the metric with ``iou_type=("bbox", "segm")``.

        Args:
            **kwargs (``Any``): Keyword arguments forwarded to the
                ``torchmetrics`` ``MeanAveragePrecision``, such as
                ``iou_thresholds``, ``max_detection_thresholds``,
                ``class_metrics``, and ``backend``. The other arguments,
                such as ``node``, reach `BaseMetric`. A name that no base
                class accepts raises ``ValueError``. An ``iou_type``
                argument raises ``TypeError``. Keep ``box_format`` at
                ``"xyxy"``, because `update` gives all boxes in the
                ``xyxy`` format. Keep ``extended_summary`` at ``False``.
                With ``True``, `compute` raises ``AttributeError`` after
                any `update`.

        """
        super().__init__(iou_type=("bbox", "segm"), **kwargs)

    @override
    def update(
        self,
        boundingbox: list[Tensor],
        instance_segmentation: list[Tensor],
        target_boundingbox: Tensor,
        target_instance_segmentation: Tensor,
    ) -> None:
        """Convert the boxes and masks of one batch and store them.

        `luxonis_train.attached_modules.metrics.mean_average_precision.utils.compute_metric_lists`
        builds one prediction and one target dictionary for each image.
        It converts the target boxes to the ``xyxy`` format and scales
        them to pixels with the height and width of
        `BaseAttachedModule.original_in_shape`. It casts the masks to
        ``bool``. The ``update`` method of ``torchmetrics``
        then checks the dictionaries and stores the boxes, the masks,
        the scores, and the labels. It gives a ``UserWarning`` when one
        image has more predictions than the largest value of
        ``max_detection_thresholds``, ``100`` by default.

        Args:
            boundingbox (``list[Tensor]``): The predicted boxes of each
                image, of shape ``[M_i, 6]``, as
                ``[x1, y1, x2, y2, score, class]`` in pixels.
            instance_segmentation (``list[Tensor]``): The predicted
                masks of each image, of shape ``[M_i, H, W]``, one for
                each predicted box.
            target_boundingbox (``Tensor``): The ``boundingbox`` label of
                the batch, of shape ``[N, 6]``, as
                ``[batch_index, class, x, y, w, h]``. The values are
                normalized, and ``x`` and ``y`` are the top-left corner.
            target_instance_segmentation (``Tensor``): The
                ``instance_segmentation`` label of the batch, of shape
                ``[N, H, W]``, one mask for each row of
                ``target_boundingbox``.

        """
        super().update(
            *compute_metric_lists(
                boundingbox,
                target_boundingbox,
                *self.original_in_shape[1:],
                masks=instance_segmentation,
                target_masks=target_instance_segmentation,
            )
        )

    @override
    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the box and mask mAP since the last reset.

        The ``compute`` method of ``torchmetrics`` runs the COCO
        evaluation two times, once on the boxes and once on the masks.
        This method moves each result to the device of the metric. It
        then gives the results to
        `luxonis_train.attached_modules.metrics.mean_average_precision.utils.postprocess_metrics`,
        which adds the F1 scores and splits the per-class values. A
        value is ``-1`` when it has no data, for example
        ``segm_map_small`` when no target is small.

        Returns:
            ``tuple[Tensor, dict[str, Tensor]]``: The scalar ``segm_map``
            and a dictionary of the other values. The dictionary holds
            the values that
            `luxonis_train.attached_modules.metrics.mean_average_precision.MeanAveragePrecisionBBox.compute`
            returns, two times:

            - With the prefix ``bbox_`` for the boxes, such as
              ``bbox_map``, ``bbox_mar_100``, and ``bbox_f1_large``.
            - With the prefix ``segm_`` for the masks, such as
              ``segm_map_50``, ``segm_mar_100``, and ``segm_f1_large``.
              The main value ``segm_map`` is not in the dictionary.

        """
        metrics = {k: v.to(self.device) for k, v in super().compute().items()}
        return postprocess_metrics(
            metrics, self.classes.inverse, "segm_map", self.device
        )
