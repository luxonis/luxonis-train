"""The base class of the heads that predict bounding boxes.

It keeps the NMS settings and the strides of the scales. It also selects
the names of the exported outputs. Each subclass runs NMS itself, in
evaluation mode.

"""

import torch
from loguru import logger
from luxonis_ml.typing import Params
from torch import Size, Tensor
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead


class BaseDetectionHead(BaseHead):
    """Base class for YOLO-like detection heads with several scales.

    The head reads the last ``n_heads`` outputs of the input node, one
    feature map for each scale. For this selection, the constructor sets
    ``attach_index`` to ``(-n_heads - 1, -1)``. An ``attach_index`` in
    the node ``params`` replaces this value.

    A subclass stores one block for each scale in ``heads``. In
    evaluation mode, it runs NMS with ``conf_thres``, ``iou_thres``, and
    ``max_det``. After a call to `request_detections_pre_nms`, it also
    keeps the NMS input in its packet.

    Attributes:
        parser (str): The export parser, ``"YOLO"``. A subclass can
            replace it.
        n_heads (int): The number of scales that the head reads.
        conf_thres (float): The confidence threshold of NMS.
        iou_thres (float): The IoU threshold of NMS.
        max_det (int): The maximum number of boxes that NMS keeps for
            each image.
        stride (``Tensor``): The stride of each scale, an ``int32``
            tensor of shape ``[n_heads]``. `fit_stride_to_heads` computes
            it.

    Example:
        `EfficientBBoxHead` is a detection head. With two scales, it
        reads the last two of three feature maps:

        >>> from torch import Size
        >>> from luxonis_train.nodes import EfficientBBoxHead
        >>> sizes = [
        ...     Size([1, 4, 64, 64]),
        ...     Size([1, 8, 32, 32]),
        ...     Size([1, 16, 16, 16]),
        ... ]
        >>> head = EfficientBBoxHead(
        ...     n_heads=2,
        ...     n_classes=3,
        ...     input_shapes=[{"features": sizes}],
        ...     original_in_shape=Size([3, 256, 256]),
        ... )
        >>> head.attach_index, head.in_channels, head.stride.tolist()
        ((-3, -1), [8, 16], [8, 16])

    """

    parser = "YOLO"

    in_channels: list[int]
    in_sizes: list[Size]

    def __init__(
        self,
        n_heads: int,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
        **kwargs,
    ):
        """Set the NMS settings, the attach index, and the strides.

        The constructor reads `BaseNode.in_channels`,
        `BaseNode.in_sizes`, and `BaseNode.original_in_shape`. Thus
        ``kwargs`` must hold ``original_in_shape``, and ``input_shapes``
        or ``in_sizes``. When the head gets fewer than ``n_heads``
        feature maps, the constructor logs a warning and sets
        ``n_heads`` to that number. Without an ``attach_index`` in
        ``kwargs``, this check counts all outputs of the input node.

        Args:
            n_heads (int): The number of scales. The head reads the last
                ``n_heads`` outputs of the input node.
            conf_thres (float): The confidence threshold of NMS, in
                ``[0, 1]``.
            iou_thres (float): The IoU threshold of NMS, in ``[0, 1]``.
            max_det (int): The maximum number of boxes that NMS keeps for
                each image.
            **kwargs (``Any``): Keyword arguments for `BaseNode`. An
                ``attach_index`` among them replaces the default
                selection of the last ``n_heads`` outputs. It must select
                a range or ``"all"``. An integer index makes the
                constructor fail. This class annotates ``in_channels`` as
                ``list[int]``, so `BaseNode` raises `IncompatibleError`.
                A subclass with its own class annotations hides this
                annotation, for example `PrecisionSegmentBBoxHead`. For
                such a subclass, the constructor raises ``TypeError``
                instead.

        """
        super().__init__(**kwargs)

        self._n_heads = n_heads
        self._conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self._keep_detections_pre_nms = False

        if len(self.in_channels) < self._n_heads:
            logger.warning(
                f"Head '{self.name}' was set to use {self._n_heads} heads, "
                f"but received only {len(self.in_channels)} inputs. "
                f"Changing number of heads to {len(self.in_channels)}."
            )
            self._n_heads = len(self.in_channels)

        if "attach_index" not in kwargs:
            self.attach_index = (-self._n_heads - 1, -1)

        self.stride = self.fit_stride_to_heads()

    @property
    def keep_detections_pre_nms(self) -> bool:
        """Whether the head adds the pre-NMS candidates to its packet.

        It is ``False`` until a call to `request_detections_pre_nms`.
        When it is ``True``, the evaluation packet of a subclass also
        holds the ``"detections_pre_nms"`` key.

        """
        return self._keep_detections_pre_nms

    def request_detections_pre_nms(self) -> None:
        """Make the head add the pre-NMS candidates to its packet.

        After the call, the evaluation packet of a subclass also holds
        the ``"detections_pre_nms"`` key. Its value is the NMS input: a
        tensor of shape ``[B, N, 5 + n_classes]`` with one row for each
        of the ``N`` anchor points. Each row holds the ``xyxy`` box in
        pixels, a constant ``1``, and the class scores. A head that
        keeps more values with each box, such as keypoints or mask
        coefficients, adds columns after the class scores. The tensor
        can use much memory, so the head keeps it only on request.
        `PrecisionRecallCurve` calls this method in its constructor.

        Example:
            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes import EfficientBBoxHead
            >>> sizes = [Size([1, 8, 32, 32]), Size([1, 16, 16, 16])]
            >>> head = EfficientBBoxHead(
            ...     n_heads=2,
            ...     n_classes=3,
            ...     input_shapes=[{"features": sizes}],
            ...     original_in_shape=Size([3, 256, 256]),
            ... )
            >>> head.keep_detections_pre_nms
            False
            >>> head.request_detections_pre_nms()
            >>> out = head.eval()([torch.zeros(size) for size in sizes])
            >>> out["detections_pre_nms"].shape
            torch.Size([1, 1280, 8])

        """
        self._keep_detections_pre_nms = True

    def _forward(
        self, inputs: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        features_list: list[Tensor] = []
        classes_list: list[Tensor] = []
        regressions_list: list[Tensor] = []

        for head, x in zip(self.heads, inputs, strict=True):  # type: ignore
            features, classes, regressions = head(x)
            features_list.append(features)
            classes_list.append(torch.sigmoid(classes))
            regressions_list.append(regressions)
        return features_list, classes_list, regressions_list

    @override
    def get_custom_head_config(self) -> Params:
        """Return the NMS settings and the strides for the NN Archive.

        A subclass adds its own keys to this dictionary, for example
        ``"subtype"``.

        Returns:
            ``Params``: A dictionary with the keys ``"iou_threshold"``,
            ``"conf_threshold"``, ``"max_det"``, and ``"strides"``. They
            hold ``iou_thres``, ``conf_thres``, ``max_det``, and
            ``stride`` as a list with one integer for each scale.

        """
        return {
            "iou_threshold": self.iou_thres,
            "conf_threshold": self._conf_thres,
            "max_det": self.max_det,
            "strides": self.stride.tolist(),
        }

    def get_output_names(self, default: list[str]) -> list[str]:
        """Return the export output names or the ``default`` names.

        A subclass calls the method in its ``export_output_names``
        property. The method reads the ``export_output_names``
        constructor argument through `BaseNode.export_output_names`. It
        does not read the property of the subclass, which calls this
        method. It returns the names of the constructor argument when
        their number is ``n_heads``. Otherwise, it logs a warning and
        returns ``default``. It also logs a warning when the argument is
        ``None``.

        **Warning:** The method compares the number of names with
        ``n_heads``, not with the length of ``default``. A subclass with
        more than ``n_heads`` outputs thus never gets names for all its
        outputs from the argument.

        Args:
            default (list[str]): The names to return when the constructor
                argument is ``None`` or has the wrong length. The
                subclasses give names that DepthAI accepts.

        Returns:
            list[str]: The names of the constructor argument, or
            ``default``.

        Example:
            >>> from torch import Size
            >>> from luxonis_train.nodes import EfficientBBoxHead
            >>> sizes = [Size([1, 8, 32, 32]), Size([1, 16, 16, 16])]
            >>> def build_head(names):
            ...     return EfficientBBoxHead(
            ...         n_heads=2,
            ...         n_classes=3,
            ...         input_shapes=[{"features": sizes}],
            ...         original_in_shape=Size([3, 256, 256]),
            ...         export_output_names=names,
            ...     )
            >>> build_head(["small", "large"]).get_output_names(["a", "b"])
            ['small', 'large']

            One name for two scales gives the default names. The example
            turns the logger off, so the warning does not show:

            >>> from loguru import logger
            >>> logger.disable("luxonis_train")
            >>> build_head(["boxes"]).get_output_names(["a", "b"])
            ['a', 'b']
            >>> logger.enable("luxonis_train")

        """
        export_names = super().export_output_names
        if export_names is not None:
            if len(export_names) == self._n_heads:
                return export_names

            logger.warning(
                f"Number of provided output names ({len(export_names)}) "
                f"does not match number of heads ({self._n_heads}). "
                f"Using default names."
            )
        else:
            logger.warning(
                "No output names provided. "
                "Using names compatible with DepthAI."
            )
        return default

    def fit_stride_to_heads(self) -> Tensor:
        r"""Compute the stride of each scale from the input sizes.

        The stride of scale :math:`i` is
        :math:`s_i = \operatorname{round}(H / H_i)`, where :math:`H` is
        the height of the model input and :math:`H_i` is the height of
        the feature map. The method reads the first ``n_heads`` sizes of
        `BaseNode.in_sizes`. It takes :math:`H_i` from index ``2`` of
        each size, so the sizes must have the form ``[B, C, H, W]``. The
        constructor stores the result in ``stride``. The class example
        shows the result.

        Returns:
            ``Tensor``: An ``int32`` tensor of shape ``[n_heads]``.

        """
        return torch.tensor(
            [
                round(self.original_in_shape[1] / x[2])
                for x in self.in_sizes[: self._n_heads]
            ],
            dtype=torch.int,
        )
