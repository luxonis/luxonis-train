import os
from typing import Literal, Protocol, TypeAlias, cast

import torch
from loguru import logger
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode


class TransformerBackboneReturnsIntermediateLayers(Protocol):
    """Minimal interface for DINOv3 models.

    To properly declare the dinov3.models.vision_transformer.DinoVisionTransformer type, the Dinov3 repository needs to be cloned locally.
    """

    rope_embed: nn.Module
    embed_dim: int
    num_heads: int

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


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int, num_heads: int):
        super().__init__()
        head_dim = embed_dim // num_heads

        self.sin_embed = nn.Parameter(torch.zeros(seq_len, head_dim))
        self.cos_embed = nn.Parameter(torch.zeros(seq_len, head_dim))
        nn.init.trunc_normal_(self.sin_embed, std=0.02)
        nn.init.trunc_normal_(self.cos_embed, std=0.02)

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        return self.sin_embed, self.cos_embed


class DinoV3(BaseNode):
    DINOv3Kwargs = dict[str, str]
    in_height: int
    in_width: int

    def __init__(
        self,
        weights_link: str = "tests/data/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        return_sequence: bool = False,
        variant: DINOv3Variant = "vits16",
        repo_dir: str = "facebookresearch/dinov3",
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

        weights_url = self._resolve_weights_url(weights_link)

        self.backbone, self.patch_size = self._get_backbone(
            weights=weights_url,
            variant=variant,
            repo_dir=repo_dir,
            **kwargs,
        )
        seq_len = (self.in_height // self.patch_size) * (
            self.in_width // self.patch_size
        )
        self.backbone.rope_embed = AbsolutePositionalEmbedding(
            embed_dim=self.backbone.embed_dim,
            seq_len=seq_len,
            num_heads=self.backbone.num_heads,
        )

        logger.warning(
            "DinoV3 is not convertible for RVC2. If RVC2 is your target platform, please pick a different backbone."
        )

    def _is_ci(self) -> bool:
        """Detect if we're running in a CI environment."""
        return os.getenv("CI", "false").lower() == "true"

    def forward(self, inputs: Tensor) -> list[Tensor]:
        features = self.backbone.get_intermediate_layers(inputs, n=4)
        outs: list[Tensor] = []

        for x in features:
            if self.return_sequence:
                outs.append(x)  # B x N x C
            else:
                B, N_with_cls, C = x.shape

                H = self.in_height // self.patch_size
                W = self.in_width // self.patch_size

                assert x.shape[1] == H * W, (
                    f"Expected {H * W} tokens, got {x.shape[1]}"
                )

                # Reshape sequence to feature map
                x = x.permute(0, 2, 1).reshape(B, C, H, W)
                outs.append(x)

        return outs

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
            **kwargs,
        )
        model.load_state_dict(weights)
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
