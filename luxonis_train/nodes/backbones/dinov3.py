import os
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, cast

import torch
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


DINOv3Variant: TypeAlias = Literal[
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
]


class DinoV3(BaseNode):
    DINOv3Kwargs = dict[str, str]
    in_height: int
    in_width: int

    def __init__(
        self,
        weights_link: str = "",
        return_sequence: bool = False,
        variant: DINOv3Variant = "vits16",
        repo_dir: str = "",
        **kwargs,
    ):
        """DinoV3 backbone.

        Source: U{https://github.com/facebookresearch/dinov3}

        @param variant: Architecture variant of the DINOv3 backbone.
        @type variant: Literal of supported DINOv3 variants.

        @type weights: Literal["download", "none"] or None

        @param repo_dir: Torch Hub repo to use. Defaults to the official "facebookresearch/dinov3".
        @type repo_dir: str

        @param return_sequence: If True, return the patch sequence directly to be processed by classification head. Otherwise, turn patch embeddings into [B, C, H, W] feature map to be passed to traditional heads
        @type return_sequence: bool
        """
        super().__init__(**kwargs)

        self.return_sequence = return_sequence

        if not repo_dir:
            torch_home = Path(
                os.environ.get("TORCH_HOME", "~/.cache/torch")
            ).expanduser()
            repo_dir = str(torch_home / "hub" / "facebookresearch_dinov3_main")

            logger.info(
                f"Detected CI environment. Using local repo_dir: {repo_dir}"
            )
        else:
            repo_dir = "facebookresearch/dinov3"

        weights_url = self._resolve_weights_url(weights_link)

        self.backbone, self.patch_size = self._get_backbone(
            weights=weights_url,
            variant=variant,
            repo_dir=repo_dir,
            **kwargs,
        )

        logger.warning(
            "DinoV3 is not convertible for RVC2. If RVC2 is your target platform, please pick a different backbone."
        )

    def _is_ci(self) -> bool:
        """Detect if we're running in a CI environment."""
        return os.getenv("CI", "false").lower() == "true"

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
        return "vits16", {
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
        weights: str | None,
        variant: DINOv3Variant = "vits16",
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
            2. If empty, return None and log a warning.

        @param weights_link: Direct URL or file path to the weights.
        @type weights_link: str

        @return: URL or path to weights, or None if weights shouldn't be loaded.
        @rtype: str or None
        """
        if weights_link and weights_link.strip():
            return weights_link

        logger.warning(
            "No weights provided. Proceeding without pretrained weights."
        )
        return None
