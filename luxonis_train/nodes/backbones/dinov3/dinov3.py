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
    """DINOv3 self-supervised vision transformer backbone.

    DINOv3 learns strong, dense feature representations useful for various
    downstream tasks and can return either dense feature maps or CLS token
    embeddings.

    Metadata:
        - Node type: backbone
        - Registry name: ``DinoV3``
        - Task: None
        - Attach index: ``-1``
        - Inputs: ``features`` tensor
        - Outputs: ``features`` list of tensors

    Provenance:
        - Source: ``facebookresearch/dinov3``
        - License: Unknown
        - Implementation notes: Loads DINOv3 through ``torch.hub`` and
          replaces RoPE with an ONNX-friendly local module.

    Variants:
        - ``"vits16"``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - ``variant``: ``"vits16"``
        - ``"vits16plus"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"vits16plus"``
        - ``"vitb16"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"vitb16"``
        - ``"vitl16"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"vitl16"``
        - ``"vith16plus"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"vith16plus"``
        - ``"vit7b16"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"vit7b16"``
        - ``"convnext_tiny"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"convnext_tiny"``
        - ``"convnext_small"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"convnext_small"``
        - ``"convnext_base"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"convnext_base"``
        - ``"convnext_large"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"convnext_large"``

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
            weights_link (str): Weights value passed to ``torch.hub.load``.
            return_sequence (bool): If True, return the CLS embedding [B, C] for downstream classification heads. Otherwise, turn patch embeddings into [B, C, H, W] feature maps to be passed to dense prediction heads.
            variant (DINOv3Variant): Architecture variant of the DINOv3 backbone.
            repo_or_dir (str): GitHub repository or local directory passed to ``torch.hub.load``. Defaults to ``facebookresearch/dinov3``.
            freeze_backbone (bool): If True, freeze the backbone so only downstream heads contain trainable parameters.
            depth (int): Number of last layers taken from the transformer output and converted to feature maps.
            **kwargs (``Any``): Keyword arguments forwarded to the parent class and ``torch.hub.load``.

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
        """Replace the default RoPE embedding in the DINOv3 backbone.

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
        """Run the DINOv3 backbone.

        If self.return_sequence is True, a list containing the CLS token
        embedding [B, C] is returned and this can be used for downstream
        classification tasks.

        Otherwise, the last ``self.depth`` layers of the network are
        returned as [B, C, H, W] feature maps, which can be used for
        downstream segmentation and other dense feature tasks.

        Args:
            inputs (``Tensor``): Input image tensor with shape [B, C, H, W].

        Returns:
            ``list[Tensor]``: CLS token embeddings or dense feature maps.

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
