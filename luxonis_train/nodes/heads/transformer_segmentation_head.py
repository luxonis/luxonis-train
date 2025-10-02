import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Optional, Tuple
from typing_extensions import override

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
        Args:
            x: [B, N, C] = patch tokens
        Returns:
            Segmentation logits: [B, n_classes, H, W]

        Steps:
            1) Pass through LayerNorm to normalize the features then a Linear layer to map embedding size C to number of classes
            2) Get original image input size
            3) Infer patch grid dimensions H_p x W_p
            4) Reshape patch sequence into 2d grids
            5) Upsample to original image resolution
        """
        B, N, C = x.shape
        h, w = self.original_in_shape[1:]

        x = self.head(x)

        aspect_ratio = w / h
        H_p = int(round((N / aspect_ratio) ** 0.5)) # note: for now it seems like I cannot export a model that ues round() to ONNX
        W_p = N // H_p
        assert H_p * W_p == N, f"Cannot reshape: N={N}, inferred H_p={H_p}, W_p={W_p}, product={H_p * W_p}"

        # Reshape to image grid: [B, n_classes, H_p, W_p]
        x = x.permute(0, 2, 1).reshape(B, self.n_classes, H_p, W_p)

        # Upsample to match original input image size
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x

    @override
    def get_custom_head_config(self) -> dict:
        return {"is_softmax": False}
