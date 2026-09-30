"""The YOLOv8 instance segmentation head, which predicts mask
coefficients over a shared set of prototypes.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks import ConvBlock, SegProto
from luxonis_train.tasks import Task, Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import (
    apply_bounding_box_to_masks,
    non_max_suppression,
)

from .precision_bbox_head import PrecisionBBoxHead


class PrecisionSegmentBBoxHead(PrecisionBBoxHead):
    r"""Precision instance segmentation and detection head.

    The head adds masks to the boxes of `PrecisionBBoxHead`. Each
    detection gets a mask from a shared set of prototype masks.

    Inputs:
        - ``inputs`` (``list[Tensor]``): :math:`\left[B, C_i, H_i,
          W_i\right]` per scale, the last ``n_heads`` outputs of the
          input node

    Outputs:
        - train:

          - ``features`` (``list[Tensor]``): :math:`\left[B, 4 \cdot
            reg_{max} + n_{classes}, H_i, W_i\right]` per scale, logits
          - ``prototypes`` (``Tensor``): :math:`\left[B, n_{masks}, 2
            H_0, 2 W_0\right]`
          - ``mask_coefficients`` (``Tensor``): :math:`\left[B,
            n_{masks}, N\right]`, :math:`N = \sum_i H_i W_i`

        - eval:

          - ``features`` (``list[Tensor]``): :math:`\left[B, 4 \cdot
            reg_{max} + n_{classes}, H_i, W_i\right]` per scale, logits
          - ``prototypes`` (``Tensor``): :math:`\left[B, n_{masks}, 2
            H_0, 2 W_0\right]`
          - ``mask_coefficients`` (``Tensor``): :math:`\left[B,
            n_{masks}, N\right]`
          - ``boundingbox`` (``list[Tensor]``): :math:`\left[M_i,
            6\right]` per image, ``[x1, y1, x2, y2, score, class]``,
            pixels
          - ``instance_segmentation`` (``list[Tensor]``):
            :math:`\left[M_i, H, W\right]` per image, binary
          - ``detections_pre_nms`` (``Tensor``): :math:`\left[B, N, 5 +
            n_{classes} + n_{masks}\right]`, only when requested

        - export:

          - ``boundingbox`` (``list[Tensor]``): :math:`\left[B, 5 +
            n_{classes}, H_i, W_i\right]` per scale, DFL-decoded
          - ``masks`` (``list[Tensor]``): :math:`\left[B, n_{masks},
            H_i, W_i\right]` per scale, mask coefficients
          - ``prototypes`` (``Tensor``): :math:`\left[B, n_{masks}, 2
            H_0, 2 W_0\right]`

    References:
        - Source: Reimplemented from `Real-Time Flying Object Detection
          with YOLOv8 <https://arxiv.org/abs/2305.09972>`_ and `YOLOv6:
          A Single-Stage Object Detection Framework for Industrial
          Applications <https://arxiv.org/abs/2209.02976>`_.
        - License: Apache-2.0 (this project)

    Notes:
        Each scale gets a mask branch next to its `PreciseDecoupledBlock`.
        The branch predicts ``n_masks`` mask coefficients for each
        anchor point. `SegProto` builds ``n_masks`` prototypes from the
        first input feature map, at twice its resolution. In evaluation
        mode, NMS keeps the coefficients with each box.
        `refine_and_apply_masks` then combines the prototypes with the
        coefficients and crops the mask to the box. It also resizes the
        mask to the size of the model input image. A pixel is in the
        mask when its value is above ``0``. Training mode and export
        mode skip NMS and `refine_and_apply_masks`. The ``masks``
        output of export mode holds the mask coefficients.

    Variants:
        None. Configure the node through ``params``.

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: PrecisionSegmentBBoxHead
              inputs: [RepPANNeck]

    Compatible with:
        - Required labels:

          - ``boundingbox``
          - ``instance_segmentation``

        - Used by: `InstanceSegmentationModel`
        - Losses: `PrecisionDFLSegmentationLoss`
        - Metrics:

          - `ConfusionMatrix`
          - `MeanAveragePrecision`
          - `PrecisionRecallCurve`

        - Visualizers: `InstanceSegmentationVisualizer`
        - Export parser: ``YOLOExtendedParser``

    """

    task: Task = Tasks.INSTANCE_SEGMENTATION
    parser: str = "YOLOExtendedParser"

    def __init__(
        self,
        n_heads: Literal[2, 3, 4] = 3,
        n_masks: int = 32,
        n_proto: int = 64,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        **kwargs,
    ):
        """Initialize the mask branches and the prototype generator.

        The parent class builds the box branches. The mask branch of
        each scale has two ``3x3`` `ConvBlock` layers with batch norm
        and SiLU. A ``1x1`` convolution with ``n_masks`` output channels
        follows them. The hidden width of every mask branch is the
        larger of ``n_masks`` and ``in_channels[0] // 4``.
        ``in_channels[0]`` is the channel count of the first scale.
        `SegProto` reads the feature map of the first scale too.

        Args:
            n_heads (``Literal[2, 3, 4]``): Number of scales. The head
                reads the last ``n_heads`` outputs of the input node. An
                ``attach_index`` param replaces this selection. It must
                select a range or ``"all"``. An integer index makes the
                constructor raise ``TypeError``. When the head gets fewer
                outputs, it logs a warning and uses that number. When an
                ``attach_index`` selects more outputs, the constructor
                raises ``ValueError``. Defaults to ``3``.
            n_masks (int): Number of prototype masks. It is also the
                number of mask coefficients of each anchor point.
                Defaults to ``32``.
            n_proto (int): Number of hidden channels of the `SegProto`
                prototype generator. Defaults to ``64``.
            conf_thres (float): NMS keeps only the boxes whose maximum
                class score is above this value. The value must be in
                ``[0, 1]``. Defaults to ``0.25``.
            iou_thres (float): NMS removes a box when its IoU with a box
                of the same class and a higher score is above this
                value. The value must be in ``[0, 1]``. Defaults to
                ``0.45``.
            max_det (int): Maximum number of boxes that NMS keeps for
                each image. Defaults to ``300``.
            **kwargs (``Any``): Keyword arguments for `PrecisionBBoxHead`,
                such as ``reg_max``, and for `BaseNode`. They must hold
                ``original_in_shape``, the input sizes through
                ``input_shapes`` or ``in_sizes``, and the class count
                through ``n_classes`` or ``dataset_metadata``.

        """
        super().__init__(
            n_heads=n_heads,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            **kwargs,
        )

        mid_channels = max(self.in_channels[0] // 4, n_masks)

        self.segmentation_heads = nn.ModuleList(
            nn.Sequential(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    activation=nn.SiLU(),
                ),
                ConvBlock(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    activation=nn.SiLU(),
                ),
                nn.Conv2d(mid_channels, n_masks, 1, 1),
            )
            for in_channels in self.in_channels
        )

        self.proto = SegProto(self.in_channels[0], n_proto, n_masks)
        self._n_masks = n_masks

    def forward(self, inputs: list[Tensor]) -> Packet[Tensor]:
        r"""Run the branches and return the packet of the current mode.

        Each `PreciseDecoupledBlock` returns the features, the class
        logits, and the distance bin logits of one scale. `SegProto`
        builds the prototypes from the first feature map. The mask
        branch of each scale predicts the mask coefficients of its
        anchor points. Export mode has priority over training mode. The
        packet depends on the mode:

        - Export mode: ``"boundingbox"`` holds one map of shape
          ``[B, 5 + n_classes, H_i, W_i]`` for each scale. Its channels
          are the distances ``(l, t, r, b)`` in stride units, the
          maximum class score, and the class scores. `DFL` decodes the
          distances when ``reg_max`` is above ``1``. The scores are
          sigmoid probabilities. ``"masks"`` holds the mask coefficients
          of shape ``[B, n_masks, H_i, W_i]`` for each scale.
          ``"prototypes"`` holds the prototypes of shape
          ``[B, n_masks, 2 * H_0, 2 * W_0]``.
        - Training mode: ``"features"`` holds the regression and class
          logits of shape ``[B, 4 * reg_max + n_classes, H_i, W_i]`` for
          each scale. ``"prototypes"`` is as in export mode.
          ``"mask_coefficients"`` holds the coefficients of all ``N``
          anchor points, of shape ``[B, n_masks, N]``, where
          :math:`N = \sum_i H_i W_i`.
        - Evaluation mode: the keys of training mode and the NMS
          results. ``"boundingbox"`` holds a tensor of shape ``[M_i, 6]``
          for each image. Each row is ``[x1, y1, x2, y2, score, class]``,
          with the corners in the pixels of the model input image.
          ``"instance_segmentation"`` holds the masks of these boxes, of
          shape ``[M_i, H, W]``, with the values ``0`` and ``1``. ``H``
          and ``W`` come from ``original_in_shape``. For an
          image without boxes, ``M_i`` is ``0`` and the mask tensor has
          the dtype ``uint8``. After a call to
          `BaseDetectionHead.request_detections_pre_nms`, the packet
          also holds the NMS input ``"detections_pre_nms"``, of shape
          ``[B, N, 5 + n_classes + n_masks]``.

        Args:
            inputs (``list[Tensor]``): One feature map for each scale, of
                shape ``[B, C_i, H_i, W_i]``. ``H_0`` and ``W_0`` are the
                height and the width of the first map.

        Returns:
            ``Packet[Tensor]``: The packet of the current mode, with the
            keys that the description gives.

        Example:
            A new head is in training mode:

            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes import PrecisionSegmentBBoxHead
            >>> sizes = [Size([1, 8, 16, 16]), Size([1, 16, 8, 8])]
            >>> head = PrecisionSegmentBBoxHead(
            ...     n_heads=2,
            ...     n_masks=4,
            ...     n_proto=8,
            ...     n_classes=2,
            ...     input_shapes=[{"features": sizes}],
            ...     original_in_shape=Size([3, 64, 64]),
            ... )
            >>> out = head([torch.zeros(size) for size in sizes])
            >>> out["prototypes"].shape, out["mask_coefficients"].shape
            (torch.Size([1, 4, 32, 32]), torch.Size([1, 4, 320]))

            For zero inputs, a new head gives each class a score below
            ``conf_thres``. In evaluation mode, NMS thus keeps no box and
            no mask:

            >>> out = head.eval()([torch.zeros(size) for size in sizes])
            >>> boxes, masks = out["boundingbox"], out["instance_segmentation"]
            >>> boxes[0].shape, masks[0].shape
            (torch.Size([0, 6]), torch.Size([0, 64, 64]))

        """
        prototypes = self.proto(inputs[0])
        mask_coefficients = [
            head(x)
            for head, x in zip(self.segmentation_heads, inputs, strict=True)
        ]

        features_list, classes_list, regressions_list = self.forward_heads(
            inputs
        )

        if self.export:
            pred_bboxes = self._construct_raw_bboxes(
                classes_list, regressions_list
            )
            return {
                "boundingbox": pred_bboxes,
                "masks": mask_coefficients,
                "prototypes": prototypes,
            }

        mask_coefficients = torch.cat(
            [
                coefficient.view(coefficient.size(0), self._n_masks, -1)
                for coefficient in mask_coefficients
            ],
            dim=2,
        )

        if self.training:
            return {
                "features": features_list,
                "prototypes": prototypes,
                "mask_coefficients": mask_coefficients,
            }

        pred_bboxes = self._prepare_bbox_inference_output(
            classes_list, regressions_list
        )
        preds_combined = torch.cat(
            [pred_bboxes, mask_coefficients.permute(0, 2, 1)], dim=-1
        )
        preds = non_max_suppression(
            preds_combined,
            n_classes=self.n_classes,
            conf_thres=self._conf_thres,
            iou_thres=self.iou_thres,
            bbox_format="xyxy",
            max_det=self.max_det,
            predicts_objectness=False,
        )

        results = {
            "features": features_list,
            "prototypes": prototypes,
            "mask_coefficients": mask_coefficients,
            "boundingbox": [],
            self.task.main_output: [],
        }
        if self.keep_detections_pre_nms:
            results["detections_pre_nms"] = preds_combined

        for i, pred in enumerate(preds):
            height, width = self.original_in_shape[-2:]
            results[self.task.main_output].append(
                refine_and_apply_masks(
                    prototypes[i],
                    pred[:, 6:],
                    pred[:, :4],
                    height=height,
                    width=width,
                    upsample=True,
                )
            )
            results["boundingbox"].append(pred[:, :6])

        return results

    @property
    @override
    def export_output_names(self) -> list[str] | None:
        """The names of the outputs of the exported model.

        The head has ``2 * n_heads + 1`` outputs in the exported model.
        The ONNX export sorts them by the output key. The default names
        follow this order:

        - ``output1_yolov8`` to ``output{n_heads}_yolov8`` for the
          ``"boundingbox"`` maps.
        - ``output1_masks`` to ``output{n_heads}_masks`` for the
          ``"masks"`` coefficients.
        - ``protos_output`` for the ``"prototypes"``.

        The ``export_output_names`` param replaces the default names only
        when it holds exactly ``n_heads`` names. The head logs a warning
        each time it gives the default names. The value is never
        ``None``.

        **Warning:** A param with ``n_heads`` names has fewer names than
        the exported model has outputs. The ONNX export thus logs a
        warning and ignores the names. The NN Archive still lists these
        ``n_heads`` names as the outputs of the head.

        Example:
            A head with two scales gives five default names. The
            example turns the logger off, so the warning does not show:

            >>> from loguru import logger
            >>> from torch import Size
            >>> from luxonis_train.nodes import PrecisionSegmentBBoxHead
            >>> sizes = [Size([1, 8, 16, 16]), Size([1, 16, 8, 8])]
            >>> head = PrecisionSegmentBBoxHead(
            ...     n_heads=2,
            ...     n_classes=2,
            ...     input_shapes=[{"features": sizes}],
            ...     original_in_shape=Size([3, 64, 64]),
            ... )
            >>> logger.disable("luxonis_train")
            >>> head.export_output_names
            ['output1_yolov8', 'output2_yolov8', 'output1_masks',
             'output2_masks', 'protos_output']
            >>> logger.enable("luxonis_train")

        """
        return self.get_output_names(
            [f"output{i + 1}_yolov8" for i in range(self._n_heads)]
            + [f"output{i + 1}_masks" for i in range(self._n_heads)]
            + ["protos_output"]
        )  # export names are applied on sorted output names


def refine_and_apply_masks(
    mask_prototypes: Tensor,
    predicted_masks: Tensor,
    bounding_boxes: Tensor,
    height: int,
    width: int,
    upsample: bool = False,
) -> Tensor:
    r"""Build the mask of each detection and crop it to its box.

    The mask of a detection is :math:`\sum_k c_k P_k`, where :math:`P_k`
    is prototype :math:`k` and :math:`c_k` is mask coefficient :math:`k`
    of the detection. The function scales the boxes from the image size
    to the prototype size. It sets the mask pixels outside of each box
    to zero. With ``upsample``, it resizes the masks to ``height`` by
    ``width`` with bilinear interpolation. A pixel with the value
    :math:`x` is in the mask when :math:`x > 0`. This condition is equal
    to :math:`\sigma(x) > 0.5`, where :math:`\sigma` is the sigmoid.
    The interpolation can extend a mask a little past its box.

    Args:
        mask_prototypes (``Tensor``): The prototype masks, of shape
            ``[n_masks, h, w]``.
        predicted_masks (``Tensor``): The mask coefficients of each
            detection, of shape ``[N, n_masks]``.
        bounding_boxes (``Tensor``): The ``xyxy`` box of each detection,
            in the pixels of the ``height`` by ``width`` image, of shape
            ``[N, 4]``.
        height (int): The image height, in pixels.
        width (int): The image width, in pixels.
        upsample (bool): Whether to resize the masks to ``height`` by
            ``width``. Defaults to ``False``.

    Returns:
        ``Tensor``: A float tensor of masks with the values ``0`` and
        ``1``. Its shape is ``[N, height, width]`` with ``upsample``, and
        ``[N, h, w]`` without it. When ``predicted_masks`` or
        ``bounding_boxes`` has no rows, the function returns a ``uint8``
        tensor of shape ``[0, height, width]``.

    Example:
        The box covers the top-left quarter of the ``8x8`` image, so the
        mask keeps the top-left quarter of the ``4x4`` prototype:

        >>> import torch
        >>> prototypes = torch.ones(1, 4, 4)
        >>> coefficients = torch.tensor([[1.0]])
        >>> boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
        >>> masks = refine_and_apply_masks(
        ...     prototypes, coefficients, boxes, height=8, width=8
        ... )
        >>> masks[0].int().tolist()
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        >>> refine_and_apply_masks(
        ...     prototypes, coefficients, boxes, 8, 8, upsample=True
        ... ).shape
        torch.Size([1, 8, 8])

    """
    if predicted_masks.size(0) == 0 or bounding_boxes.size(0) == 0:
        return torch.zeros(
            0, height, width, dtype=torch.uint8, device=predicted_masks.device
        )

    channels, proto_h, proto_w = mask_prototypes.shape
    masks_combined = (
        predicted_masks @ mask_prototypes.float().view(channels, -1)
    ).view(-1, proto_h, proto_w)
    w_scale, h_scale = proto_w / width, proto_h / height
    scaled_boxes = bounding_boxes.clone()
    scaled_boxes[:, [0, 2]] *= w_scale
    scaled_boxes[:, [1, 3]] *= h_scale
    cropped_masks = apply_bounding_box_to_masks(masks_combined, scaled_boxes)
    if upsample:
        cropped_masks = F.interpolate(
            cropped_masks.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return (cropped_masks > 0).to(cropped_masks.dtype)
