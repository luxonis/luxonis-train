import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal, TypedDict, Dict
from typing_extensions import override

from loguru import logger

from luxonis_train.nodes.blocks import UpBlock
from luxonis_train import BaseHead, BaseNode, BaseLoss
from luxonis_train import Tasks


class DinoV3(BaseNode):
    DINOv3Kwargs = Dict[str, str]

    def __init__(
            self,
            weights_link,
            variant: Literal[
                "vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16",
                "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
            ] = "vits16",
            weights: Literal["download", "none"] | None = "download",
            repo_dir: str = "facebookresearch/dinov3",
            return_sequence=False,
            **kwargs):
        """DinoV3 backbone

        Source: U{https://github.com/facebookresearch/dinov3}

        @param variant: Architecture variant of the DINOv3 backbone.
        @type variant: Literal of supported DINOv3 variants.

        @param weights: Whether to download pretrained weights ("download") or initialize randomly ("none").
        @type weights: Literal["download", "none"] or None

        @param repo_dir: Torch Hub repo to use. Defaults to the official "facebookresearch/dinov3".
        @type repo_dir: str

        @param return_sequence: If True, return the patch sequence directly to be processed by classification head. Otherwise, turn patch embeddings into [B, C, H, W] feature map to be passed to traditional heads
        @type return_sequence: bool
        """
        super().__init__(**kwargs)
        self.return_sequence = return_sequence

        if weights == "download":
            weights_url = weights_link
        else:
            weights_url = None

        self.backbone = self._get_backbone(
            variant=variant,
            weights=weights_url,
            repo_dir=repo_dir,
            **kwargs
        )

        logger.warning(
            "DinoV3 is not convertible for RVC2. If RVC2 is your target platform, please pick a different backbone."
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        """
        If self.return_sequence is True: return patch sequence directly in the format [B, N, C] to be passed to transformer heads
        If self.return sequence is False: convert patch-level embeddings to feature map to be passed to the classic heads
        """
        x = self.backbone.get_intermediate_layers(inputs, n=1)[0]

        if self.return_sequence:
            return [x]

        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return [x]

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, DINOv3Kwargs]]:
        return "vitb16", {
            "vits16": {"variant": "vits16"},
            "vits16plus": {"variant": "vits16plus"},
            "vitb16": {"variant": "vitb16"},
            "vitl16": {"variant": "vitl16"},
            "vith16plus": {"variant": "vith16plus"},
            "vit7b16": {"variant": "vit7b16"},
            "convnext_tiny": {"variant": "convnext_tiny"},
            "convnext_small": {"variant": "convnext_small"},
            "convnext_base": {"variant": "convnext_base"},
            "convnext_large": {"variant": "convnext_large"},
        }

    @staticmethod
    def _get_backbone(
            variant: Literal[
                "vits16", "vits16plus", "vitb16", "vitl16", "vith16plus", "vit7b16",
                "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
            ],
            weights: str,
            repo_dir: str = "facebookresearch/dinov3",
            **kwargs
    ):
        variant_to_hub_name = {
            "vits16": "dinov3_vits16",
            "vits16plus": "dinov3_vits16plus",
            "vitb16": "dinov3_vitb16",
            "vitl16": "dinov3_vitl16",
            "vith16plus": "dinov3_vith16plus",
            "vit7b16": "dinov3_vit7b16",
            "convnext_tiny": "dinov3_convnext_tiny",
            "convnext_small": "dinov3_convnext_small",
            "convnext_base": "dinov3_convnext_base",
            "convnext_large": "dinov3_convnext_large",
        }

        if variant not in variant_to_hub_name:
            raise ValueError(f"Unsupported variant: {variant}")

        model = torch.hub.load(
            repo_or_dir=repo_dir,
            model=variant_to_hub_name[variant],
            source="github",
            weights=weights,
            **kwargs
        )
        return model