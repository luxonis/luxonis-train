import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Optional, Tuple
from typing_extensions import override
from loguru import logger

from luxonis_ml.typing import Params
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks

import torch.nn.functional as F


class TransformerSegmentationHead(BaseNode):
    in_height: int
    in_width: int
    in_channels: int
    n_classes: int

    task = Tasks.SEGMENTATION
    parser: str = "SegmentationParser"

    def __init__(self, **kwargs: Any):
        """
        Decoder head for patch sequence from DINOv3.
        Converts [B, N, C] to segmentation map [B, n_classes, H, W]
        """
        super().__init__(**kwargs)
        self.head = nn.Sequential(
            nn.LayerNorm(self.in_channels),
            nn.Linear(self.in_channels, self.n_classes),
        )

        if len(self.input_shapes[0]['features']) == 4:
            logger.warning(
                "The transformer segmentation head will not work with feature maps of dimension [B, C, H, W] as input. Please provide patch-level embeddings from transformer backbones in the format [B, C, N]")

        logger.warning(
            "In order to accurately calculate the patch size, this class assumes that the CLS token is in the given patch embeddings. Please make sure that the previously-defined transformer encoder does not remove the CLS token.")

    @property
    def in_channels(self) -> int:
        """
        Override to extract embedding dim from transformer output shape.
        Expected input_shapes: [{'features': [torch.Size([B, N, C])]}]
        """
        try:
            shape_dict = self.input_shapes[0]
            feature_shape = shape_dict["features"][0]

            return feature_shape[-1]
        except Exception as e:
            raise RuntimeError(f"Could not determine in_channels from input_shapes: {self.input_shapes} — {e}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Steps:
        1) Remove CLS token if it is there (if N = H_p * W_p + 1)
        2) Pass through LayerNorm and Linear layer to map embedding dim C to number of classes
        3) Infer patch grid dimensions H_p x W_p from original image size and patch size
        4) Reshape patch sequence [B, N, n_classes] into 2D grid [B, n_classes, H_p, W_p]
        5) Upsample logits to original image resolution [B, n_classes, H, W]
        """
        B, N, C = x.shape
        h, w = self.original_in_shape[1:]

        H_p = h // self.patch_size
        W_p = w // self.patch_size
        N_expected = H_p * W_p

        if N == N_expected + 1:
            x = x[:, 1:, :]
        elif N != N_expected:
            logger.warning(
                f"Token count N={N} does not match expected patch grid H_p*W_p={H_p}*{W_p}={N_expected}. "
                f"Skipping reshape and returning dummy segmentation output."
            )
            return torch.zeros((B, self.n_classes, h, w), dtype=x.dtype, device=x.device)

        x = self.head(x)

        x = x.permute(0, 2, 1).reshape(B, self.n_classes, H_p, W_p)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        return x

    @override
    def get_custom_head_config(self) -> dict:
        return {"is_softmax": False}
