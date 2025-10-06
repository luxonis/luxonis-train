import os
from typing import Literal, Protocol, cast

import torch
from dotenv import load_dotenv
from loguru import logger
from torch import Tensor
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode


class TransformerBackboneReturnsIntermediateLayers(Protocol):
    """Minimal interface for DINOv3 models.

    To properly declare the dinov3.models.vision_transformer.DinoVisionTransformer type, the Dinov3 repository needs to be cloned locally.
    """

    def get_intermediate_layers(
        self, x: Tensor, n: int
    ) -> tuple[Tensor, ...]: ...


class DinoV3(BaseNode):
    DINOv3Kwargs = dict[str, str]
    in_height: int
    in_width: int

    def __init__(
        self,
        weights_link: str = "",
        return_sequence: bool = False,
        variant: Literal[
            "vits16",
            "vits16plus",
            "vitb16",
            "vitl16",
            "vith16plus",
            "vit7b16",
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large",
        ] = "vits16",
        weights: Literal["download", "none"] | None = "download",
        repo_dir: str = "facebookresearch/dinov3",
        **kwargs,
    ):
        """DinoV3 backbone.

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

        weights_url = self._resolve_weights_url(weights_link)

        self.backbone, self.patch_size = self._get_backbone(
            variant=variant,
            weights=weights_url,
            repo_dir=repo_dir,
            **kwargs,
        )

        logger.warning(
            "DinoV3 is not convertible for RVC2. If RVC2 is your target platform, please pick a different backbone."
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        x = self.backbone.get_intermediate_layers(inputs, n=1)[0]

        if self.return_sequence:  # return patch sequence directly
            return [x]

        B, N, C = x.shape

        H = self.in_height // self.patch_size
        W = self.in_width // self.patch_size

        # Reshape from sequence to feature map
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
            "vits16",
            "vits16plus",
            "vitb16",
            "vitl16",
            "vith16plus",
            "vit7b16",
            "convnext_tiny",
            "convnext_small",
            "convnext_base",
            "convnext_large",
        ],
        weights: str | None,
        repo_dir: str = "facebookresearch/dinov3",
        **kwargs,
    ) -> tuple[TransformerBackboneReturnsIntermediateLayers, int]:
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
            **kwargs,
        )
        model = cast(TransformerBackboneReturnsIntermediateLayers, model)
        patch_size = getattr(model, "patch_size", 16)
        return model, patch_size

    def _resolve_weights_url(self, weights_link: str) -> str | None:
        """Resolve the URL or local path for pretrained weights.

        Priority:
            1. Use `weights_link` if it is a non-empty string.
            2. If empty, fall back to the `DINOV3_WEIGHTS` environment variable.
            3. If still missing, return None and log a warning.

        @param weights_link: Direct URL or file path to the weights. If empty, will attempt to read from environment.
        @type weights_link: str

        @return: URL or path to weights, or None if weights shouldn't be loaded.
        @rtype: str or None
        """
        if Path(".env").exists():
            load_dotenv()

        if weights_link and weights_link.strip():
            return weights_link

        env_weights = os.getenv("DINOV3_WEIGHTS")
        if env_weights and env_weights.strip():
            logger.info("Using DINOV3_WEIGHTS from environment.")
            return env_weights

        logger.warning(
            "No weights provided. Proceeding without pretrained weights."
        )
        return None
