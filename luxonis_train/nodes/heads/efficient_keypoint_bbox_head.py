"""The YOLOv6 detection head with a keypoint branch for each scale."""

from typing import Literal

import torch
from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks import ConvBlock
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet
from luxonis_train.utils import anchors_for_fpn_features

from .efficient_bbox_head import EfficientBBoxHead


class EfficientKeypointBBoxHead(EfficientBBoxHead):
    r"""Efficient object and keypoint detection head.

    Inputs:
        - ``inputs`` (``list[Tensor]``): :math:`\left[B, C_i, H_i,
          W_i\right]` per scale

    Outputs:
        - train:

          - ``features`` (``list[Tensor]``): :math:`\left[B, C_i, H_i,
            W_i\right]` per scale
          - ``class_scores`` (``Tensor``): :math:`\left[B, N,
            n_{classes}\right]`
          - ``distributions`` (``Tensor``): :math:`\left[B, N,
            4\right]`, ``(l, t, r, b)`` in stride units
          - ``keypoints_raw`` (``Tensor``): :math:`\left[B, N, 3 *
            n_{keypoints}\right]`

        - eval:

          - ``features`` (``list[Tensor]``): :math:`\left[B, C_i, H_i,
            W_i\right]` per scale
          - ``class_scores`` (``Tensor``): :math:`\left[B, N,
            n_{classes}\right]`
          - ``distributions`` (``Tensor``): :math:`\left[B, N,
            4\right]`, ``(l, t, r, b)`` in stride units
          - ``keypoints_raw`` (``Tensor``): :math:`\left[B, N, 3 *
            n_{keypoints}\right]`
          - ``boundingbox`` (``list[Tensor]``): :math:`\left[M_i,
            6\right]` per image, ``[x1, y1, x2, y2, conf, class]``,
            pixels
          - ``keypoints`` (``list[Tensor]``): :math:`\left[M_i,
            n_{keypoints}, 3\right]` per image, ``(x, y, conf)``, pixels
          - ``detections_pre_nms`` (``Tensor``): :math:`\left[B, N, 5 +
            n_{classes} + 3 * n_{keypoints}\right]`, only when requested

        - export:

          - ``boundingbox`` (``list[Tensor]``): :math:`\left[B, 5 +
            n_{classes}, H_i, W_i\right]` per scale
          - ``keypoints`` (``list[Tensor]``): :math:`\left[B, 3 *
            n_{keypoints}, H_i * W_i\right]` per scale, ``(x, y, conf)``,
            pixels, ``conf`` as a logit

    References:
        - Source: Reimplemented from `YOLOv6: A Single-Stage Object
          Detection Framework for Industrial Applications
          <https://arxiv.org/abs/2209.02976>`_.
        - License: Apache-2.0 (this project)

    Notes:
        The head adds a keypoint branch for each scale to
        `EfficientBBoxHead`. It decodes the keypoints around the anchor
        points of the scale. NMS keeps the keypoints of each box that it
        keeps.

    Variants:
        None. Configure the node through ``params``.

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: EfficientKeypointBBoxHead
              inputs: [RepPANNeck]

    Compatible with:
        - Required labels:

          - ``boundingbox``
          - ``keypoints``

        - Used by: `KeypointDetectionModel`
        - Losses: `EfficientKeypointBBoxLoss`
        - Metrics:

          - `ConfusionMatrix`
          - `MeanAveragePrecision`
          - `ObjectKeypointSimilarity`
          - `PrecisionRecallCurve`

        - Visualizers: `KeypointVisualizer`
        - Export parser: ``YOLOExtendedParser``

    """

    parser = "YOLOExtendedParser"
    task = Tasks.INSTANCE_KEYPOINTS

    def __init__(
        self,
        n_heads: Literal[2, 3, 4] = 3,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        **kwargs,
    ):
        """Initialize the box head and the keypoint branches.

        Each keypoint branch has two ``3x3`` `ConvBlock` layers with
        batch norm and SiLU, then a ``1x1`` convolution with
        ``3 * n_keypoints`` output channels. The hidden layers of all
        branches have the larger of ``in_channels[0] // 4`` and
        ``3 * n_keypoints`` channels. ``in_channels[0]`` belongs to the
        first scale.

        Args:
            n_heads (``Literal[2, 3, 4]``): Number of scales. The head
                reads the last ``n_heads`` outputs of the input node. An
                ``attach_index`` param replaces this selection. The value
                is usually equal to the number of neck outputs. When the
                input node gives fewer outputs, the head logs a warning
                and uses that number. Defaults to ``3``.
            conf_thres (float): NMS keeps only the boxes whose maximum
                class score is above this value. The value must be in
                ``[0, 1]``. Defaults to ``0.25``.
            iou_thres (float): NMS removes a box when its IoU with a box of
                the same class and a higher score is above this value.
                The value must be in ``[0, 1]``. Defaults to ``0.45``.
            max_det (int): Maximum number of boxes that NMS keeps for each
                image. Defaults to ``300``.
            **kwargs (``Any``): Keyword arguments for `EfficientBBoxHead`,
                such as ``bias_init_p``, and for `BaseNode`, such as
                ``n_classes``, ``n_keypoints``, and ``input_shapes``.

        """
        super().__init__(
            n_heads=n_heads,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
            **kwargs,
        )

        self._n_keypoints_flat = self.n_keypoints * 3

        mid_channels = max(self.in_channels[0] // 4, self._n_keypoints_flat)
        self.keypoint_heads = nn.ModuleList(
            nn.Sequential(
                ConvBlock(
                    in_channels=self.in_channels[i],
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
                nn.Conv2d(
                    in_channels=mid_channels,
                    out_channels=self._n_keypoints_flat,
                    kernel_size=1,
                    stride=1,
                ),
            )
            for i in range(len(self.heads))
        )

    def forward(self, inputs: list[Tensor]) -> Packet[Tensor]:
        r"""Run the box and keypoint branches and build the packet.

        The box branches work as in `EfficientBBoxHead.forward`. The
        keypoint branch of each scale reads the input feature map of the
        scale. It predicts the raw values :math:`(t_x, t_y, t_c)` of
        every keypoint at every anchor point. The head decodes them
        with the anchor point :math:`(a_x, a_y)` in grid units and the
        stride :math:`s` of the scale:

        .. math::

            x = (2 t_x + a_x - 0.5) \, s, \qquad
            y = (2 t_y + a_y - 0.5) \, s, \qquad
            c = \sigma(t_c)

        Export mode has priority over training mode. The packet depends
        on the mode:

        - Export mode: ``"boundingbox"`` as in
          `EfficientBBoxHead.forward`, and ``"keypoints"`` with one
          tensor of shape ``[B, 3 * n_keypoints, H_i * W_i]`` for each
          scale. The tensor holds :math:`x`, :math:`y`, and :math:`t_c`
          for each keypoint. Export mode skips the sigmoid.
        - Training mode: ``"features"``, ``"class_scores"``, and
          ``"distributions"`` as in `EfficientBBoxHead.forward`, and
          ``"keypoints_raw"`` with the raw values of shape
          ``[B, N, 3 * n_keypoints]``.
        - Evaluation mode: the training keys and the NMS results.
          ``"boundingbox"`` holds a tensor of shape ``[M_i, 6]`` for each
          image, with the rows ``[x1, y1, x2, y2, score, class]``.
          ``"keypoints"`` holds the decoded keypoints of these boxes, of
          shape ``[M_i, n_keypoints, 3]``. Each keypoint holds
          :math:`(x, y, c)`. For an image without boxes, ``M_i`` is
          ``0``. After a call to
          `BaseDetectionHead.request_detections_pre_nms`, the packet
          also holds the NMS input ``"detections_pre_nms"``, of shape
          ``[B, N, 5 + n_classes + 3 * n_keypoints]``.

        Args:
            inputs (``list[Tensor]``): One feature map for each scale, of
                shape ``[B, C_i, H_i, W_i]``.

        Returns:
            ``Packet[Tensor]``: The packet of the current mode.

        Example:
            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes import EfficientKeypointBBoxHead
            >>> sizes = [Size([1, 8, 32, 32]), Size([1, 16, 16, 16])]
            >>> head = EfficientKeypointBBoxHead(
            ...     n_heads=2,
            ...     n_classes=3,
            ...     n_keypoints=5,
            ...     input_shapes=[{"features": sizes}],
            ...     original_in_shape=Size([3, 256, 256]),
            ... )
            >>> out = head([torch.zeros(size) for size in sizes])
            >>> out["keypoints_raw"].shape
            torch.Size([1, 1280, 15])

            A new head gives each class the score ``0.01``, which is
            below ``conf_thres``. In evaluation mode, NMS thus keeps no
            box and no keypoint:

            >>> out = head.eval()([torch.zeros(size) for size in sizes])
            >>> out["boundingbox"][0].shape, out["keypoints"][0].shape
            (torch.Size([0, 6]), torch.Size([0, 5, 3]))

        """
        features_list, classes_list, regressions_list = super()._forward(
            inputs
        )
        keypoints_list: list[Tensor] = []

        for head, x in zip(self.keypoint_heads, inputs, strict=True):
            keypoints_list.append(head(x))

        bs = features_list[0].shape[0]
        if self.export:
            packet = self._wrap_export(classes_list, regressions_list)
            keypoints = []
            for i, keypoint in enumerate(keypoints_list):
                keypoints.append(
                    self._distributions_to_keypoints(
                        keypoint.view(bs, self._n_keypoints_flat, -1),
                        features_list,
                        bs,
                        i,
                        apply_sigmoid=False,
                    )
                )
            return packet | {"keypoints": keypoints}

        class_scores = self._postprocess(classes_list)
        distributions = self._postprocess(regressions_list)
        keypoints_raw = self._postprocess(
            out.view(bs, self._n_keypoints_flat, -1) for out in keypoints_list
        )

        if self.training:
            return {
                "features": features_list,
                "class_scores": class_scores,
                "distributions": distributions,
                "keypoints_raw": keypoints_raw,
            }

        pred_keypoints = torch.cat(
            [
                self._distributions_to_keypoints(
                    keypoint.view(bs, self._n_keypoints_flat, -1),
                    features_list,
                    bs,
                    i,
                )
                for i, keypoint in enumerate(keypoints_list)
            ],
            dim=2,
        ).permute(0, 2, 1)

        _, anchor_points, _, stride_tensor = anchors_for_fpn_features(
            features_list,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=False,
        )
        detections_pre_nms = self._prepare_bbox_inference_output(
            features_list,
            class_scores,
            distributions,
            anchor_points,
            stride_tensor,
            tail=[pred_keypoints],
        )
        boxes, kpts = self._split_keypoint_detections(
            self._run_nms(detections_pre_nms)
        )
        packet: Packet[Tensor] = {
            "boundingbox": boxes,
            "keypoints": kpts,
            "features": features_list,
            "class_scores": class_scores,
            "distributions": distributions,
            "keypoints_raw": keypoints_raw,
        }
        if self.keep_detections_pre_nms:
            packet["detections_pre_nms"] = detections_pre_nms
        return packet

    @property
    @override
    def export_output_names(self) -> list[str] | None:
        """The names of the outputs of the exported model.

        The exported model has ``2 * n_heads`` outputs. The default
        names are ``output1_yolov6`` to ``output{n_heads}_yolov6`` for
        the boxes, then ``kpt_output1`` to ``kpt_output{n_heads}`` for
        the keypoints. The ``export_output_names`` param replaces them
        only when it holds exactly ``n_heads`` names. The value then has
        fewer names than the exported model has outputs. The head logs a
        warning each time it gives the default names. The value is never
        ``None``.

        """
        return self.get_output_names(
            [f"output{i + 1}_yolov6" for i in range(self._n_heads)]
            + [f"kpt_output{i + 1}" for i in range(self._n_heads)]
        )

    @override
    def get_custom_head_config(self) -> Params:
        return super().get_custom_head_config() | {
            "n_keypoints": self.n_keypoints
        }

    def _distributions_to_keypoints(
        self,
        keypoints: Tensor,
        features: list[Tensor],
        batch_size: int,
        index: int,
        apply_sigmoid: bool = True,
    ) -> Tensor:
        """Decode the raw keypoint values of one scale.

        Args:
            keypoints (``Tensor``): Raw values of shape
                ``[B, 3 * n_keypoints, H_i * W_i]``.
            features (``list[Tensor]``): Feature maps of all scales. The
                method builds the anchor points from them.
            batch_size (int): The batch size ``B``.
            index (int): Index of the scale.
            apply_sigmoid (bool): Whether to apply a sigmoid to the
                confidence. Defaults to ``True``.

        Returns:
            ``Tensor``: Tensor of shape ``[B, 3 * n_keypoints, H_i * W_i]``
            with ``(x, y, conf)`` for each keypoint. ``x`` and ``y`` are
            in pixels. ``conf`` is a probability when ``apply_sigmoid``
            is ``True``, and a logit otherwise.

        """
        _, anchor_points, n_anchors_list, _ = anchors_for_fpn_features(
            features,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            multiply_with_stride=False,
        )
        anchors = anchor_points.split(n_anchors_list, dim=0)
        keypoints = keypoints.view(batch_size, self.n_keypoints, 3, -1)
        grid_coords = (
            keypoints[:, :, :2] * 2.0 + (anchors[index].transpose(1, 0) - 0.5)
        ) * self.stride[index]

        conf_scores = keypoints[:, :, 2:3]

        if apply_sigmoid:
            conf_scores = conf_scores.sigmoid()

        return torch.cat((grid_coords, conf_scores), dim=2).view(
            batch_size, self._n_keypoints_flat, -1
        )

    def _split_keypoint_detections(
        self, detections: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Split the NMS output into boxes and keypoints.

        Args:
            detections (``list[Tensor]``): One tensor of shape
                ``[M_i, 6 + 3 * n_keypoints]`` for each image. An empty
                tensor can have a different number of columns.

        Returns:
            ``tuple[list[Tensor], list[Tensor]]``: The boxes of shape
            ``[M_i, 6]`` and the keypoints of shape
            ``[M_i, n_keypoints, 3]`` for each image.

        """
        bboxes = [detection[:, :6] for detection in detections]
        keypoints = [
            detection[:, 6:].reshape(-1, self.n_keypoints, 3)
            for detection in detections
        ]
        return bboxes, keypoints
