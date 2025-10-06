from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks


class TransformerSegmentationHead(BaseHead):
    in_height: int
    in_width: int
    in_channels: int
    n_classes: int

    task = Tasks.SEGMENTATION
    parser: str = "SegmentationParser"

    def __init__(self, **kwargs: Any):
        """Decoder head for patch sequence from DINOv3.

        Converts [B, N, C] to segmentation map [B, n_classes, H, W]
        """
        super().__init__(**kwargs)
        self.head = nn.Sequential(
            nn.LayerNorm(self.in_channels),
            nn.Linear(self.in_channels, self.n_classes),
        )

        if len(self.input_shapes[0]["features"]) == 4:
            logger.warning(
                "The transformer segmentation head will not work with feature maps of dimension [B, C, H, W] as input. Please provide patch-level embeddings from transformer backbones in the format [B, C, N]"
            )

        logger.warning(
            "In order to accurately calculate the patch size, this class assumes that the CLS token is in the given patch embeddings. Please make sure that the previously-defined transformer encoder does not remove the CLS token."
        )

    @property
    def in_channels(self) -> int:
        """Override to extract embedding dim from transformer output
        shape.

        Expected input_shapes: [{'features': [torch.Size([B, N, C])]}]
        """
        try:
            shape_dict = self.input_shapes[0]
            feature_shape = shape_dict["features"][0]

            return feature_shape[-1]
        except Exception as e:
            raise RuntimeError(
                f"Could not determine in_channels from input_shapes: {self.input_shapes} — {e}"
            ) from e

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N+1, C] = patch tokens including class token (CLS at position 0)

        Returns:
            Segmentation logits: [B, n_classes, H, W]

        Steps:
            1) Remove class token at position 0
            2) Project patch tokens to class logits via LayerNorm + Linear
            3) Infer patch grid dimensions (H_p x W_p) using image aspect ratio
            4) Reshape [B, N, n_classes] → [B, n_classes, H_p, W_p]
            5) Upsample to original image resolution (H, W)
        """
        B, N_with_cls, C = x.shape
        h, w = self.original_in_shape[1:]  # Original input resolution

        # Remove class token
        x = x[:, 1:, :]
        N = x.shape[1]

        x = self.head(x)

        aspect_ratio = w / h
        H_p = int((N / aspect_ratio) ** 0.5)
        W_p = N // H_p

        if H_p * W_p != N:
            logger.warning(
                f"Cannot reshape tokens into 2D grid. N={N}, "
                f"inferred H_p={H_p}, W_p={W_p}, product={H_p * W_p}. "
                f"Returning dummy output."
            )
            return torch.zeros(
                (B, self.n_classes, h, w), dtype=x.dtype, device=x.device
            )

        x = x.permute(0, 2, 1).reshape(
            B, self.n_classes, H_p, W_p
        )  # [B, n_classes, H_p, W_p]

        return F.interpolate(
            x, size=(h, w), mode="bilinear", align_corners=False
        )

    @override
    def get_custom_head_config(self) -> dict:
        return {"is_softmax": False}
