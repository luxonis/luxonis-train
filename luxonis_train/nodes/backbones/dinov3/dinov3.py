import inspect
from typing import Literal, Protocol, TypeAlias, cast

import torch
from loguru import logger
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.backbones.dinov3.rope_position_encoding import (
    RopePositionEmbedding,
)
from luxonis_train.nodes.base_node import BaseNode


class TransformerBackboneReturnsIntermediateLayers(Protocol):
    """Minimal interface for DINOv3 models.

    To properly declare the dinov3.models.vision_transformer.DinoVisionTransformer type, the Dinov3 repository needs to be cloned locally.
    """

    embed_dim: int
    num_heads: int
    rope_embed: nn.Module

    def get_intermediate_layers(
        self, x: Tensor, n: int, norm: bool
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
        repo_dir: str = "facebookresearch/dinov3",
        **kwargs,
    ):
        """DinoV3 backbone.

        Source: U{https://github.com/facebookresearch/dinov3}

        @license: U{https://github.com/facebookresearch/dinov3?tab=License-1-ov-file#readme}

        @type weights_link: a weights link for the specific model, which needs to be requested here U{https://pytorch.org/get-started/locally/}

        @param return_sequence: If True, return the patch sequence [B, N, C] directly to be processed by transformer heads. Otherwise, turn patch embeddings into [B, C, H, W] feature map to be passed to traditional heads
        @type return_sequence: bool

        @param variant: Architecture variant of the DINOv3 backbone.
        @type variant: Literal of supported DINOv3 variants.

        @param repo_dir: "facebookresearch/dinov3" if the repository is not locally donwloaded or cached, "local" otherwise
        @type repo_dir: str
        """
        super().__init__(**kwargs)

        self.return_sequence = return_sequence

        self.backbone, self.patch_size = self._get_backbone(
            weights=weights_link,
            variant=variant,
            repo_dir=repo_dir,
            **kwargs,
        )

        self._replace_rope_embedding()

        logger.warning(
            "DinoV3 is not convertible for RVC2. If RVC2 is your target platform, please pick a different backbone."
        )

    def _replace_rope_embedding(self) -> None:
        """Replaces the default RoPE embedding in the DinoV3 backbone
        with a nearly-identical implementation that is ONNX-
        convertible."""
        old_rope = self.backbone.rope_embed
        new_rope_cls = RopePositionEmbedding

        init_params = inspect.signature(new_rope_cls.__init__).parameters
        param_names = [p for p in init_params if p != "self"]

        rope_kwargs = {}
        for name in param_names:
            if hasattr(self.backbone, name):
                rope_kwargs[name] = getattr(self.backbone, name)
            elif hasattr(old_rope, name):
                rope_kwargs[name] = getattr(old_rope, name)

        self.backbone.rope_embed = new_rope_cls(**rope_kwargs)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        features = self.backbone.get_intermediate_layers(
            inputs, norm=True, n=4
        )
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
        weights: str = "",
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

        if weights:
            model = torch.hub.load(
                repo_or_dir=repo_dir,
                model=variant_to_hub_name[variant],
                source="github",
                **kwargs,
            )
        else:
            model = torch.hub.load(
                weights=weights,
                repo_or_dir=repo_dir,
                model=variant_to_hub_name[variant],
                source="github",
                **kwargs,
            )
        model = cast(TransformerBackboneReturnsIntermediateLayers, model)
        patch_size = getattr(model, "patch_size", 16)
        return model, patch_size
