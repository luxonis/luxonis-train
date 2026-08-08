from typing import Any

import torch.nn.functional as F
from torch import Size, Tensor, nn

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class TransformerSegmentationHead(BaseHead):
    """Semantic segmentation decoder head that takes feature maps as
    inputs.

    Section 6.3.2 of the DINOv3 paper (`https://arxiv.org/abs/2508.10104/
    <https://arxiv.org/abs/2508.10104/>`_) mentions a ViT-adapter
    without the injection followed by Mask2Former. In this
    implementation, Mask2Former is replaced by a simple convolutional
    head.

    Converts a list of [B, C, H, W] feature maps to segmentation logits
    [B, n_classes, H, W]
    """

    n_classes: int
    in_sizes: list[Size]

    task = Tasks.SEGMENTATION
    parser: str = "SegmentationParser"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        channels_list = [shape[1] for shape in self.in_sizes]

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, 256, kernel_size=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
                for c in channels_list
            ]
        )

        # Decoder head
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.n_classes, kernel_size=1),
        )

    def forward(self, x: list[Tensor]) -> Tensor:
        r"""Decode transformer feature maps into segmentation logits.

        Args:
            x: Feature tensor to decode.

        Returns:
            Segmentation logits at the original image resolution.

        .. shape-contract::

            Inputs
                :math:`x_{i}` (:math:`i = 0, \ldots, N - 1`)
                    :math:`(B, C_{i}, H_{i}, W_{i})`

            Outputs
                :math:`\mathrm{logits}`
                    :math:`(B, n_{\mathrm{classes}}, H_{\mathrm{image}}, W_{\mathrm{image}})`

            Symbols
                :math:`N`
                    Number of tensors in the feature sequence.
                :math:`n_{\mathrm{classes}}`
                    Number of predicted classes.

        """  # noqa: E501
        h, w = self.original_in_shape[1:]

        projected = []
        for i, feat in enumerate(x):
            feat = self.projections[i](feat)
            feat = F.interpolate(
                feat,
                size=(h // 4, w // 4),
                mode="bilinear",
                align_corners=False,
            )
            projected.append(feat)

        fused = sum(projected) / len(projected)

        logits = self.head(fused)

        return F.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
