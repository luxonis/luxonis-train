"""The YOLOv8 detection head, which regresses a distribution over
distance bins instead of a single distance.
"""

import math
from typing import Literal, cast

import torch
from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks import DFL
from luxonis_train.nodes.blocks.blocks import PreciseDecoupledBlock
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import (
    anchors_for_fpn_features,
    dist2bbox,
    non_max_suppression,
)

from .base_detection_head import BaseDetectionHead


class PrecisionBBoxHead(BaseDetectionHead):
    r"""Precision bounding box detection head.

    Inputs:
        - ``inputs`` (``list[Tensor]``): :math:`\left[B, C_i, H_i,
          W_i\right]` per scale, the last ``n_heads`` outputs of the
          input node

    Outputs:
        - train:

          - ``features`` (``list[Tensor]``): :math:`\left[B, 4 \cdot
            reg_{max} + n_{classes}, H_i, W_i\right]` per scale, logits

        - eval:

          - ``features`` (``list[Tensor]``): :math:`\left[B, 4 \cdot
            reg_{max} + n_{classes}, H_i, W_i\right]` per scale, logits
          - ``boundingbox`` (``list[Tensor]``): :math:`\left[M_i,
            6\right]` per image, ``[x1, y1, x2, y2, score, class]``,
            pixels
          - ``detections_pre_nms`` (``Tensor``): :math:`\left[B, N, 5 +
            n_{classes}\right]`, only when requested

        - export:

          - ``boundingbox`` (``list[Tensor]``): :math:`\left[B, 5 +
            n_{classes}, H_i, W_i\right]` per scale, DFL-decoded

    References:
        - Source: Reimplemented from `Real-Time Flying Object Detection
          with YOLOv8 <https://arxiv.org/abs/2305.09972>`_ and `YOLOv6:
          A Single-Stage Object Detection Framework for Industrial
          Applications <https://arxiv.org/abs/2209.02976>`_.
        - License: Apache-2.0 (this project)

    Notes:
        The head runs one `PreciseDecoupledBlock` for each scale. The
        regression branch predicts a distribution over ``reg_max``
        distance bins for each side of a box. `DFL` converts each
        distribution to the expected distance, in stride units. In
        evaluation mode, the head decodes the distances around the
        anchor points into boxes and runs NMS. Training mode skips this
        step, and it has priority over export mode. Export mode gives
        the distances and the sigmoid class scores of each scale,
        without NMS.

    Variants:
        None. Configure the node through ``params``.

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: PrecisionBBoxHead
              inputs: [RepPANNeck]

    Compatible with:
        - Required labels: ``boundingbox``
        - Losses: `PrecisionDFLDetectionLoss`
        - Metrics:

          - `ConfusionMatrix`
          - `MeanAveragePrecision`
          - `PrecisionRecallCurve`

        - Visualizers: `BBoxVisualizer`
        - Export parser: ``YOLO``

    """

    task = Tasks.BOUNDINGBOX

    def __init__(
        self,
        n_heads: Literal[2, 3, 4] = 3,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        reg_max: int = 16,
        **kwargs,
    ):
        """Initialize one decoupled block for each scale.

        The hidden width of every regression branch is the largest of
        ``16``, ``in_channels[0] // 4``, and ``4 * reg_max``. The hidden
        width of every class branch is the larger of ``in_channels[0]``
        and ``min(n_classes, 100)``. ``in_channels[0]`` belongs to the
        first scale. The constructor also sets ``grid_cell_offset`` to
        ``0.5`` and ``grid_cell_size`` to ``5.0``.
        `PrecisionDFLDetectionLoss` reads these two values from the node.

        Args:
            n_heads (``Literal[2, 3, 4]``): Number of scales. The head
                reads the last ``n_heads`` outputs of the input node. An
                ``attach_index`` param replaces this selection. When the
                head gets fewer outputs, it logs a warning and uses that
                number. When an ``attach_index`` selects more outputs,
                the head builds one block for each output but keeps only
                ``n_heads`` strides. The construction then fails, because
                `initialize_weights` raises ``ValueError``.
            conf_thres (float): NMS keeps only the boxes whose maximum
                class score is above this value. The value must be in
                ``[0, 1]``. Otherwise, NMS raises ``ValueError`` in
                evaluation mode.
            iou_thres (float): NMS removes a box when its IoU with a box
                of the same class and a higher score is above this
                value. The value must be in ``[0, 1]``. Otherwise, NMS
                raises ``ValueError`` in evaluation mode.
            max_det (int): Maximum number of boxes that NMS keeps for
                each image.
            reg_max (int): Number of distance bins for each side of a
                box. The regression branch gives ``4 * reg_max`` channels.
                With a value above ``1``, `DFL` converts the bins to a
                distance. With ``1``, the head uses the regression output
                as the distance directly.
            **kwargs (``Any``): Keyword arguments for `BaseNode`. They
                must hold ``original_in_shape``, the input sizes through
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
        self.reg_max = reg_max
        self.no = self.n_classes + reg_max * 4
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        reg_channels = max((16, self.in_channels[0] // 4, reg_max * 4))
        cls_channels = max(self.in_channels[0], min(self.n_classes, 100))

        self.heads = cast(
            list[PreciseDecoupledBlock],
            nn.ModuleList(
                PreciseDecoupledBlock(
                    in_channels=in_channels,
                    reg_channels=reg_channels,
                    cls_channels=cls_channels,
                    n_classes=self.n_classes,
                    reg_max=reg_max,
                )
                for in_channels in self.in_channels
            ),
        )

        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

    def forward_heads(
        self, inputs: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Run the decoupled block of each scale.

        The method pairs the blocks with ``inputs`` in order. It applies
        no sigmoid, so all outputs are logits. `PrecisionSegmentBBoxHead`
        also calls the method.

        Args:
            inputs (``list[Tensor]``): One feature map for each scale, of
                shape ``[B, C_i, H_i, W_i]``. The list must have one map
                for each block. Otherwise, ``zip`` raises ``ValueError``.

        Returns:
            ``tuple[list[Tensor], list[Tensor], list[Tensor]]``: Three
            lists with one tensor for each scale, in the order of
            ``inputs``:

            - the features: the distance bin logits and the class logits,
              joined along the channel axis, of shape
              ``[B, 4 * reg_max + n_classes, H_i, W_i]``;
            - the class logits, of shape ``[B, n_classes, H_i, W_i]``;
            - the distance bin logits, of shape
              ``[B, 4 * reg_max, H_i, W_i]``.

        Example:
            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes import PrecisionBBoxHead
            >>> sizes = [Size([1, 8, 32, 32]), Size([1, 16, 16, 16])]
            >>> head = PrecisionBBoxHead(
            ...     n_heads=2,
            ...     n_classes=3,
            ...     reg_max=4,
            ...     input_shapes=[{"features": sizes}],
            ...     original_in_shape=Size([3, 256, 256]),
            ... )
            >>> inputs = [torch.zeros(size) for size in sizes]
            >>> features, classes, regressions = head.forward_heads(inputs)
            >>> features[1].shape
            torch.Size([1, 19, 16, 16])
            >>> classes[1].shape[1], regressions[1].shape[1]
            (3, 16)

        """
        features_list = []
        classes_list = []
        regressions_list = []
        for head, x in zip(self.heads, inputs, strict=True):
            features, classes, regressions = head(x)
            features_list.append(features)
            classes_list.append(classes)
            regressions_list.append(regressions)
        return features_list, classes_list, regressions_list

    def forward(self, inputs: list[Tensor]) -> Packet[Tensor]:
        """Run the block of each scale and build the packet of the mode.

        `forward_heads` gives the features, the class logits, and the
        distance bin logits of each scale. Training mode has priority
        over export mode. The packet depends on the mode:

        - Training mode: ``"features"`` holds the distance bin logits and
          the class logits of shape
          ``[B, 4 * reg_max + n_classes, H_i, W_i]`` for each scale.
        - Export mode: ``"boundingbox"`` holds one map of shape
          ``[B, 5 + n_classes, H_i, W_i]`` for each scale. Its channels
          are the distances ``(l, t, r, b)`` in stride units, the maximum
          class score, and the class scores. `DFL` decodes the distances
          when ``reg_max`` is above ``1``. The scores are sigmoid
          probabilities.
        - Evaluation mode: ``"features"`` as in training mode, and the
          NMS result ``"boundingbox"``. The head decodes the distances
          around the anchor points into boxes in pixels and does not clip
          them to the image. Each image gets a tensor of shape
          ``[M_i, 6]`` with the rows ``[x1, y1, x2, y2, score, class]``.
          An image without boxes gets a tensor of shape
          ``[0, 5 + n_classes]``. After a call to
          `BaseDetectionHead.request_detections_pre_nms`, the packet also
          holds ``"detections_pre_nms"``. This is the NMS input of shape
          ``[B, N, 5 + n_classes]``: the ``xyxy`` box in pixels, a
          constant ``1``, and the class scores. ``N`` is the sum of
          ``H_i * W_i``.

        Args:
            inputs (``list[Tensor]``): One feature map for each scale, of
                shape ``[B, C_i, H_i, W_i]``.

        Returns:
            ``Packet[Tensor]``: The packet of the current mode.

        Example:
            A new head is in training mode:

            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes import PrecisionBBoxHead
            >>> sizes = [Size([1, 8, 32, 32]), Size([1, 16, 16, 16])]
            >>> head = PrecisionBBoxHead(
            ...     n_heads=2,
            ...     n_classes=3,
            ...     input_shapes=[{"features": sizes}],
            ...     original_in_shape=Size([3, 256, 256]),
            ... )
            >>> inputs = [torch.zeros(size) for size in sizes]
            >>> [f.shape for f in head(inputs)["features"]]
            [torch.Size([1, 67, 32, 32]), torch.Size([1, 67, 16, 16])]

            For zero inputs, a new head gives each class a score below
            ``conf_thres``. In evaluation mode, NMS thus keeps no box:

            >>> out = head.eval()(inputs)
            >>> sorted(out), out["boundingbox"][0].shape
            (['boundingbox', 'features'], torch.Size([0, 8]))

            Export mode gives one map for each scale:

            >>> head.export = True
            >>> [b.shape for b in head(inputs)["boundingbox"]]
            [torch.Size([1, 8, 32, 32]), torch.Size([1, 8, 16, 16])]

        """
        features_list, classes_list, regressions_list = self.forward_heads(
            inputs
        )

        if self.training:
            return {"features": features_list}

        if self.export:
            return {
                "boundingbox": self._construct_raw_bboxes(
                    classes_list, regressions_list
                )
            }

        detections_pre_nms = self._prepare_bbox_inference_output(
            classes_list, regressions_list
        )
        boxes = non_max_suppression(
            detections_pre_nms,
            n_classes=self.n_classes,
            conf_thres=self._conf_thres,
            iou_thres=self.iou_thres,
            bbox_format="xyxy",
            max_det=self.max_det,
            predicts_objectness=False,
        )

        packet: Packet[Tensor] = {
            "features": features_list,
            "boundingbox": boxes,
        }
        if self.keep_detections_pre_nms:
            packet["detections_pre_nms"] = detections_pre_nms
        return packet

    @override
    def initialize_weights(self, method: str | None = None) -> None:
        r"""Initialize the biases of the last layers of each scale.

        The method first calls `BaseNode.initialize_weights` with
        ``method``. Then, for each scale with the stride :math:`s`, it
        sets the bias of the last convolution of both branches. Each
        value of the regression bias becomes ``1``. Each value of the
        class bias becomes:

        .. math::

            b = \log \frac{5}{n_{classes} \left(H / s\right)^2}

        :math:`H` is the height in ``original_in_shape``. The method does
        not change the weights. For a zero input, a new head thus gives
        each class the sigmoid score :math:`\sigma(b)`. This score is
        close to :math:`5 / \left(n_{classes} (H / s)^2\right)`.

        Args:
            method (str | None): The method for
                `BaseNode.initialize_weights`. ``"yolo"`` changes the batch
                norm and activation settings. Other values skip that
                step.

        Example:
            The first scale has the stride ``8``. For ``H = 256`` and
            three classes, the class score of a new head is close to
            :math:`5 / \left(3 \cdot 32^2\right)`:

            >>> from torch import Size
            >>> from luxonis_train.nodes import PrecisionBBoxHead
            >>> sizes = [Size([1, 8, 32, 32]), Size([1, 16, 16, 16])]
            >>> head = PrecisionBBoxHead(
            ...     n_heads=2,
            ...     n_classes=3,
            ...     input_shapes=[{"features": sizes}],
            ...     original_in_shape=Size([3, 256, 256]),
            ... )
            >>> block = head.heads[0]
            >>> block.regression_branch[-1].bias.unique().tolist()
            [1.0]
            >>> score = block.classification_branch[-1].bias.sigmoid()
            >>> round(score[0].item(), 5), round(5 / (3 * 32**2), 5)
            (0.00162, 0.00163)

        """
        super().initialize_weights(method)
        for head, stride in zip(self.heads, self.stride, strict=True):
            reg_conv = head.regression_branch[-1]
            assert isinstance(reg_conv, nn.Conv2d)
            if reg_conv.bias is not None:
                nn.init.constant_(reg_conv.bias, 1.0)

            cls_conv = head.classification_branch[-1]
            assert isinstance(cls_conv, nn.Conv2d)
            if cls_conv.bias is not None:
                cls_conv.bias.data[: self.n_classes] = math.log(
                    5
                    / self.n_classes
                    / (self.original_in_shape[1] / stride) ** 2
                )

    @property
    @override
    def export_output_names(self) -> list[str] | None:
        """The names of the ``n_heads`` outputs of the exported model.

        The default names are ``output1_yolov8``, ``output2_yolov8``,
        and so on. The ``export_output_names`` param replaces them only
        when it holds exactly ``n_heads`` names. The head logs a warning
        each time it gives the default names. The value is never
        ``None``.

        """
        return self.get_output_names(
            [f"output{i + 1}_yolov8" for i in range(self._n_heads)]
        )

    @override
    def get_custom_head_config(self) -> Params:
        return super().get_custom_head_config() | {"subtype": "yolov8"}

    def _construct_raw_bboxes(
        self, classes_list: list[Tensor], regressions_list: list[Tensor]
    ) -> list[Tensor]:
        """Build the export map of each scale.

        Args:
            classes_list (``list[Tensor]``): Class logits of shape
                ``[B, n_classes, H_i, W_i]`` for each scale.
            regressions_list (``list[Tensor]``): Distance bin logits of
                shape ``[B, 4 * reg_max, H_i, W_i]`` for each scale.

        Returns:
            ``list[Tensor]``: One map of shape
            ``[B, 5 + n_classes, H_i, W_i]`` for each of the ``n_heads``
            scales. Its channels are the distances from ``dfl``, the
            maximum class score, and the class scores. The scores are
            sigmoid probabilities.

        """
        bboxes = []
        for i in range(self._n_heads):
            bbox = self.dfl(regressions_list[i])
            classes = classes_list[i].sigmoid()
            confidence = classes.max(1, keepdim=True)[0]
            # @shape: [N, 4 + 1 + n_classes, h_f, w_f]
            bboxes.append(torch.cat([bbox, confidence, classes], dim=1))
        return bboxes

    def _prepare_bbox_inference_output(
        self, classes_list: list[Tensor], regressions_list: list[Tensor]
    ) -> Tensor:
        """Decode the predictions into the input tensor of NMS.

        Args:
            classes_list (``list[Tensor]``): Class logits of shape
                ``[B, n_classes, H_i, W_i]`` for each scale.
            regressions_list (``list[Tensor]``): Distance bin logits of
                shape ``[B, 4 * reg_max, H_i, W_i]`` for each scale.

        Returns:
            ``Tensor``: Tensor of shape ``[B, N, 5 + n_classes]``, where
            ``N`` is the sum of ``H_i * W_i``. Each row holds the ``xyxy``
            box in pixels, a constant ``1``, and the sigmoid class scores.

        """
        raw_bboxes = self._construct_raw_bboxes(classes_list, regressions_list)
        bbox_distributions = []
        class_probabilities = []
        for raw_bbox in raw_bboxes:
            bs, _, h, w = raw_bbox.size()
            raw_bbox = raw_bbox.view(bs, -1, h * w)
            confidence = raw_bbox[:, :4, :]
            classes = raw_bbox[:, 5:, :]
            bbox_distributions.append(confidence)
            class_probabilities.append(classes)

        bbox_distributions = torch.cat(bbox_distributions, dim=2)
        class_probabilities = torch.cat(class_probabilities, dim=2)

        _, anchor_points, _, strides = anchors_for_fpn_features(
            raw_bboxes, self.stride, 0.5
        )

        pred_bboxes = dist2bbox(
            bbox_distributions,
            anchor_points.transpose(0, 1),
            out_format="xyxy",
            dim=1,
        ) * strides.transpose(0, 1)

        base_output = [
            # @shape: [N, H * W, 4]
            pred_bboxes.permute(0, 2, 1),
            torch.ones(
                (bbox_distributions.shape[0], pred_bboxes.shape[2], 1),
                dtype=pred_bboxes.dtype,
                device=pred_bboxes.device,
            ),
            # @shape: [N, H * W, n_classes]
            class_probabilities.permute(0, 2, 1),
        ]

        # @shape: [N, H * W, 4 + 1 + n_classes]
        return torch.cat(base_output, dim=-1)
