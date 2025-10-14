from typing import Any

import torch.nn.functional as F
from loguru import logger
from torch import Size, Tensor, nn

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class TransformerSegmentationHead(BaseHead):
    """Semantic segmentation decoder head for patch sequence from
    DINOv3.

    Converts [B, N, C] to segmentation map [B, n_classes, H, W]
    """

    in_sizes: Size
    in_height: int
    in_width: int
    n_classes: int

    task = Tasks.SEGMENTATION
    parser: str = "SegmentationParser"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.head = nn.Sequential(
            nn.LayerNorm(self.in_channels),
            nn.Linear(self.in_channels, self.n_classes),
        )

        if len(self.in_sizes) == 4:
            logger.warning(
                "The transformer segmentation head will not work "
                "with feature maps of dimension [B, C, H, W] as input. "
                "Please provide patch-level embeddings from "
                "transformer backbones in the format [B, N, C]"
            )

    @property
    def in_channels(self) -> int:
        """Extract embedding dim from self.in_sizes instead of
        input_shapes."""
        try:
            return self.in_sizes[-1]
        except Exception as e:
            raise RuntimeError(
                f"Could not determine in_channels from in_sizes: {self.in_sizes} — {e}"
            ) from e

    def forward(self, x: Tensor) -> Tensor:
        """
        @param x: Tensor of shape [B, N, C]
        @return: Segmentation logits of shape [B, n_classes, H, W].

        @note: Steps performed:
            1) Remove class token at position 0.
            2) Project patch tokens to class logits via LayerNorm + Linear.
            3) Infer patch grid dimensions (H_p x W_p) using image aspect ratio.
            4) Reshape [B, N, n_classes] → [B, n_classes, H_p, W_p].
            5) Upsample to original image resolution (H, W).
        """
        B, N, C = x.shape
        h, w = self.original_in_shape[1:]

        expected_N = (h // 16) * (w // 16)

        if expected_N != N:
            raise RuntimeError(
                f"Unexpected token count: got {N}, expected {expected_N}"
            )

        x = self.head(x)

        H_p = h // 16
        W_p = w // 16

        x = x.permute(0, 2, 1).reshape(B, self.n_classes, H_p, W_p)

        return F.interpolate(
            x, size=(h, w), mode="bilinear", align_corners=False
        )
