from typing import Literal, TypeAlias, cast

import torch
from loguru import logger
from torch import Tensor, nn

from luxonis_train.nodes.backbones.dinov3.rope_position_encoding import (
    RopePositionEmbedding,
)
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.typing import get_signature


class TransformerBackboneReturnsIntermediateLayers(nn.Module):
    """Minimal interface for DINOv3 models.

    To properly declare the
    dinov3.models.vision_transformer.DinoVisionTransformer
    type, the DINOv3 repository needs to be cloned locally.
    """

    embed_dim: int
    num_heads: int
    rope_embed: nn.Module

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: int,
        norm: bool,
        return_class_token: bool,
    ) -> list[Tensor] | list[tuple[Tensor, Tensor]]: ...


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
    """DINOv3 backbone: a self-supervised vision transformer
    encoder that learns strong, dense feature representations
    useful for various downstream tasks.

    Source: U{https://github.com/facebookresearch/dinov3}
    @license: U{https://github.com/facebookresearch/dinov3?
    tab=License-1-ov-file#readme}
    """

    in_height: int
    in_width: int

    def __init__(
        self,
        weights_link: str,
        return_sequence: bool = False,
        variant: DINOv3Variant = "vits16",
        repo_or_dir: str = "facebookresearch/dinov3",
        freeze_backbone: bool = False,
        depth: int = 4,
        **kwargs,
    ):
        """
        @param weights:link: a weights link for the specific model,
        which needs to be requested here U{https://pytorch.org/get-
        started/locally/}
        @type weights_link: string

        @param return_sequence: If True, return the patch sequence
        [B, N, C] directly to be processed by transformer heads.
        Otherwise, turn patch embeddings into [B, C, H, W] feature
        map to be passed to traditional heads
        @type return_sequence: bool

        @param variant: Architecture variant of the DINOv3 backbone.
        @type variant: Literal DINOv3Variant.

        @param repo_dir: "facebookresearch/dinov3" if the repository
        is not locally downloaded or cached, "local" otherwise
        @type repo_dir: str

        @param freeze_backbone: if True, freeze the backbone;
        this will lead to a transfer learning scenario where
        only the head contains trainable parameters
        @type freeze_backbone: bool

        @param depth: number of last layers that are taken
        from the transformer output and converted to feature maps
        @type depth: int
        """
        super().__init__(**kwargs)

        self.return_sequence = return_sequence
        self.depth = depth

        self.backbone, self.patch_size = self._get_backbone(
            weights=weights_link,
            variant=variant,
            repo_or_dir=repo_or_dir,
            **kwargs,
        )

        self._replace_rope_embedding()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        logger.warning(
            "DINOv3 is not convertible for RVC2. If RVC2 is your "
            "target platform, please pick a different backbone."
        )
        if (
            self.original_in_shape[-1] % self.patch_size != 0
            or self.original_in_shape[-2] % self.patch_size != 0
        ):
            logger.warning(
                f"Image dimensions should be divisible by {self.patch_size},"
                f"but got {self.original_in_shape}. "
                "This will cause inconsistent image sizes"
                f"as DINOv3 natively reshapes to multiples of {self.patch_size}."
            )

    def _replace_rope_embedding(self) -> None:
        """Replaces the default RoPE embedding in the DINOv3 backbone
        with a nearly-identical implementation that is ONNX-
        convertible.

        angles.tile(2) is not ONNX-convertible and was replaced by
        angles.repeat(1, 2)
        """
        old_rope = self.backbone.rope_embed

        init_params = get_signature(RopePositionEmbedding.__init__)
        param_names = [p for p in init_params if p != "self"]

        rope_kwargs = {}
        for name in param_names:
            if hasattr(self.backbone, name):
                rope_kwargs[name] = getattr(self.backbone, name)
            elif hasattr(old_rope, name):
                rope_kwargs[name] = getattr(old_rope, name)

        self.backbone.rope_embed = RopePositionEmbedding(**rope_kwargs)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        """If self.return_sequence is True, the CLS token is returned
        and this can be used for downstream classification tasks.

        Otherwise, the last `self.depth` layers of the network are
        returned, which can be used for downstream segmentation and
        other dense feature tasks
        """
        outs: list[Tensor] = []

        if self.return_sequence:
            features_with_cls = cast(
                list[tuple[Tensor, Tensor]],
                self.backbone.get_intermediate_layers(
                    inputs, norm=True, n=1, return_class_token=True
                ),
            )
            cls_tokens: list[Tensor] = [cls for _, cls in features_with_cls]
            outs.extend(cls_tokens)
        else:
            seq_features = cast(
                list[Tensor],
                self.backbone.get_intermediate_layers(
                    inputs, norm=True, n=self.depth, return_class_token=False
                ),
            )
            for x in seq_features:
                B, N, C = x.shape
                h, w = self.original_in_shape[1:]
                gh, gw = h // self.patch_size, w // self.patch_size
                assert gh * gw == N, f"Expected {gh * gw} tokens, got {N}"
                outs.append(x.permute(0, 2, 1).reshape(B, C, gh, gw))

        return outs

    @staticmethod
    def _get_backbone(
        weights: str,
        variant: DINOv3Variant = "vits16",
        repo_or_dir: str = "facebookresearch/dinov3",
        **kwargs,
    ) -> tuple[TransformerBackboneReturnsIntermediateLayers, int]:
        if variant not in DINOv3Variant.__args__:
            raise ValueError(f"Unsupported variant: {variant}")
        model_name = f"dinov3_{variant}"

        model = torch.hub.load(
            repo_or_dir=repo_or_dir,
            model=model_name,
            weights=weights,
            source="github",
            **kwargs,
        )

        model = cast(TransformerBackboneReturnsIntermediateLayers, model)
        patch_size = getattr(model, "patch_size", 16)
        return model, patch_size
