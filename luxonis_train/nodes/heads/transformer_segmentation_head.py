from typing import Any

import torch.nn.functional as F
from torch import Size, Tensor, nn

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class TransformerSegmentationHead(BaseHead):
    """Semantic segmentation decoder head for transformer feature maps.

    Section 6.3.2 of the DINOv3 paper (`
    https://arxiv.org/abs/2508.10104/
    <https://arxiv.org/abs/2508.10104/>`_)
    mentions a ViT-adapter without the injection followed by Mask2Former.
    In this implementation, Mask2Former is replaced by a simple convolutional head.

    Converts a list of [B, C, H, W] feature maps to segmentation logits [B, n_classes, H, W].

    Metadata:
        - Node type: head
        - Registry name: ``TransformerSegmentationHead``
        - Task: segmentation
        - Attach index: None
        - Inputs: list of transformer feature-map tensors
        - Outputs: segmentation logits tensor

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Projects each transformer feature map to
          a common channel size, fuses by averaging, and upsamples logits
          to the original input resolution.

    Variants:
        - ``None``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - No predefined variants.

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
        """Decode transformer feature maps into segmentation logits.

        Args:
            x (``list[Tensor]``): List of successive feature maps of the same dimension.

        Returns:
            ``Tensor``: Segmentation logits.

        Notes:
            Steps: 1. Project each feature map to a channel dim of 256 using 1x1 convolutions. 2. Upsample the feature maps to 1/4 of the image size. 3. Fuse the projected feature maps through summation. 4. Apply segmentation head. 5. Upsample to original input resolution.

        """
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
