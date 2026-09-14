"""The FCN segmentation head."""

from typing import Any

from luxonis_ml.typing import Params
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.blocks import UpBlock
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import infer_upscale_factor


class SegmentationHead(BaseHead):
    r"""Basic FCN segmentation head that upsamples to the image size.

    Inputs:
        - ``inputs`` (``Tensor``): :math:`\left[B, C, H/s, W/s\right]`

    Outputs:
        - ``segmentation`` (``Tensor``): :math:`\left[B, n_{classes}, H,
          W\right]` logits

    References:
        - Source: Adapted from `torchvision FCN
          <https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py>`_
          (BSD-3-Clause).
        - License: `BSD-3-Clause
          <https://github.com/pytorch/vision/blob/main/LICENSE>`_

    Notes:
        The stride :math:`s` must be :math:`2^n`, the same for the
        height and the width. The head applies :math:`n` upsampling
        steps. Each step is an `UpBlock`: a bilinear upsample by ``2``,
        a ``1x1`` convolution that halves the channels, and a ``3x3``
        `ConvBlock` with batch norm and ReLU. A last ``1x1`` convolution
        gives one logit map for each class. The mode does not change the
        output key or shape. This includes export mode.

    Variants:
        None. Configure the node through ``params``.

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: SegmentationHead
              inputs: [RepPANNeck]

    Compatible with:
        - Attach index: ``-1``, the last output of the input node
        - Required labels: ``segmentation``
        - Losses:

          - `BCEWithLogitsLoss`
          - `CrossEntropyLoss`
          - `OHEMLoss`
          - `SigmoidFocalLoss`
          - `SmoothBCEWithLogitsLoss`
          - `SoftmaxFocalLoss`

        - Metrics:

          - `Accuracy`
          - `ConfusionMatrix`
          - `DiceCoefficient`
          - `F1Score`
          - `JaccardIndex`
          - `MIoU`
          - `Precision`
          - `Recall`

        - Visualizers: `SegmentationVisualizer`
        - Export parser: ``SegmentationParser``

    """

    in_height: int
    in_width: int
    in_channels: int

    task = Tasks.SEGMENTATION
    parser: str = "SegmentationParser"

    def __init__(self, **kwargs: Any):
        r"""Build the upsampling steps and the class convolution.

        `infer_upscale_factor` gives the number of steps :math:`n` from
        the size of the feature map and the image size in
        ``original_in_shape``. It raises ``ValueError`` when a size
        ratio is not a power of two. It also raises it when the height
        ratio differs from the width ratio. Each step halves the
        channels and rounds down. The last step thus has
        :math:`\lfloor c / 2^n \rfloor` channels, where :math:`c` is the
        channel count of the feature map. When the feature map is not
        smaller than the image, :math:`n` is ``0`` or less. The head
        then has no upsampling step, and the logits keep the size of the
        feature map.

        The class annotates `BaseNode.in_channels`, `BaseNode.in_height`,
        and `BaseNode.in_width` as ``int``. An ``attach_index`` of
        ``"all"`` or a range gives lists instead. Such an
        ``attach_index`` thus makes the constructor raise
        `IncompatibleError`.

        Args:
            **kwargs (``Any``): Keyword arguments for `BaseNode`. They
                must hold ``original_in_shape``, the input sizes through
                ``input_shapes`` or ``in_sizes``, and the class count
                through ``n_classes`` or ``dataset_metadata``.

        """
        super().__init__(**kwargs)
        h, w = self.original_in_shape[1:]
        n_up = infer_upscale_factor((self.in_height, self.in_width), (h, w))

        modules: list[nn.Module] = []
        in_channels = self.in_channels
        for _ in range(int(n_up)):
            modules.append(
                UpBlock(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    kernel_size=2,
                    stride=2,
                    upsample_mode="conv_upsample",
                    interpolation_mode="bilinear",
                    align_corners=False,
                    use_norm=True,
                )
            )
            in_channels //= 2

        self.head = nn.Sequential(
            *modules, nn.Conv2d(in_channels, self.n_classes, kernel_size=1)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Compute the segmentation logits.

        Args:
            inputs (``Tensor``): The feature map of shape
                ``[B, C, H/s, W/s]``.

        Returns:
            ``Tensor``: The logits of shape ``[B, n_classes, H, W]``.
            `BaseNode.run` puts them under the ``"segmentation"`` key.

        Example:
            A feature map with the stride ``4`` gets two upsampling
            steps:

            >>> import torch
            >>> from torch import Size
            >>> from luxonis_train.nodes import SegmentationHead
            >>> head = SegmentationHead(
            ...     n_classes=3,
            ...     input_shapes=[{"features": [Size([1, 16, 8, 8])]}],
            ...     original_in_shape=Size([3, 32, 32]),
            ... )
            >>> head(torch.zeros(1, 16, 8, 8)).shape
            torch.Size([1, 3, 32, 32])

        """
        return self.head(inputs)

    @override
    def get_custom_head_config(self) -> Params:
        return {"is_softmax": False}
