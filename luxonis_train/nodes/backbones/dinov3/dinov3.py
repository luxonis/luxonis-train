from typing import Literal, TypeAlias, cast

import torch
from loguru import logger
from luxonis_ml.typing import Kwargs
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.backbones.dinov3.rope_position_encoding import (
    RopePositionEmbedding,
)
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.utils import get_signature


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
    ) -> list[Tensor] | list[tuple[Tensor, Tensor]]:
        r"""Return intermediate transformer features.

        Args:
            x: Input image tensor.
            n: Number of intermediate layers to return.
            norm: Whether to normalize the returned features.
            return_class_token: Whether to include class tokens.

        Returns:
            Intermediate feature tensors, optionally paired with class tokens.

        .. shape-contract::

            Inputs
                :math:`x`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{features}_{i}` (:math:`i = 0, \ldots, N - 1`)
                    :math:`(B, L, C)`

            Symbols
                :math:`N`
                    Number of tensors in the feature sequence.

        """
        raise NotImplementedError(
            "DINOv3 backbones must implement get_intermediate_layers()"
        )


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
    """DINOv3 backbone: a self-supervised vision transformer encoder
    that learns strong, dense feature representations useful for various
    downstream tasks.

    Source: `DINOv3 <https://github.com/facebookresearch/dinov3>`_

    License:
        `DINOv3 License Agreement
        <https://github.com/facebookresearch/dinov3?tab=License-1-ov-file#readme>`_
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
        """Initialize the DINOv3 backbone.

        Args:
            weights_link: Link to the model weights.
            return_sequence: Whether to return the class token instead
                of dense feature maps.
            variant: Architecture variant.
            repo_or_dir: Torch Hub repository or local source directory.
            freeze_backbone: Whether to freeze all backbone parameters.
            depth: Number of final transformer layers returned for dense
                features.
            **kwargs: Base node arguments.

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
                "as DINOv3 natively reshapes to multiples of "
                f"{self.patch_size}."
            )

    def _replace_rope_embedding(self) -> None:
        """Replace the default RoPE embedding in the DINOv3 backbone
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
        r"""Encode an image as dense DINOv3 features or a class token.

        Args:
            inputs: Image batch to encode.

        Returns:
            Dense features from the final transformer layers, or the
            class token in sequence mode.

        .. shape-contract::

            Inputs
                :math:`\mathrm{inputs}`
                    :math:`(B, C_{\mathrm{in}}, H, W)`

            Outputs
                :math:`\mathrm{features}_{i}` (:math:`i = 0, \ldots, \mathrm{depth} - 1`)
                    :math:`(B, C_{\mathrm{out}}, \left\lfloor \frac{H}{\mathrm{patch}} \right\rfloor, \left\lfloor \frac{W}{\mathrm{patch}} \right\rfloor)`

            Sequence outputs
                :math:`\mathrm{features}_{i}` (:math:`i = 0`)
                    :math:`(B, C_{\mathrm{out}})`

            Constraints
                - :math:`H \operatorname{mod} \mathrm{patch} = 0`
                - :math:`W \operatorname{mod} \mathrm{patch} = 0`

            Symbols
                :math:`\mathrm{depth}`
                    Number of transformer layers returned.
                :math:`\mathrm{patch}`
                    DINOv3 patch size.

        """  # noqa: E501
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
            trust_repo=True,  # type: ignore
            **kwargs,
        )

        model = cast(TransformerBackboneReturnsIntermediateLayers, model)
        patch_size = getattr(model, "patch_size", 16)
        return model, patch_size

    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
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
