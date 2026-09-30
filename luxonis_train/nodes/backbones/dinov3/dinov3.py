"""The DINOv3 backbone, a pretrained self-supervised model loaded
through ``torch.hub``.
"""

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
    """Minimal interface of the DINOv3 model that `DinoV3` uses.

    The class only gives a type to the model that ``torch.hub.load``
    returns. The real type is
    ``dinov3.models.vision_transformer.DinoVisionTransformer``. A
    declaration of that type needs a local clone of the DINOv3
    repository.

    `DinoV3.forward` calls ``get_intermediate_layers`` with
    ``norm=True`` and expects this contract. The method takes an image
    batch ``x`` of shape ``[B, C, H, W]``. It returns one entry for each
    of the last ``n`` blocks. ``norm`` selects whether the final norm
    applies to each output. Without ``return_class_token``, an entry is
    the patch tokens of the block, of shape ``[B, N, C]``. With it, an
    entry is a pair of the patch tokens and the CLS token, of shape
    ``[B, C]``.

    Attributes:
        embed_dim (int): The embedding dimension of the model.
        num_heads (int): The number of attention heads.
        rope_embed (``nn.Module``): The rotary position embedding.
            `DinoV3` replaces it with `RopePositionEmbedding`.

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
    r"""DINOv3 self-supervised vision transformer backbone.

    The node loads a pretrained DINOv3 model through ``torch.hub``. It
    returns the patch tokens of the last ``depth`` blocks as feature
    maps for a dense head. With ``return_sequence``, it returns the CLS
    token for a classification head instead.

    Inputs:
        - ``inputs`` (``Tensor``): :math:`\left[B, C, H, W\right]`

    Outputs:
        - ``features`` (``list[Tensor]``): ``depth`` x :math:`\left[B,
          C, H/16, W/16\right]`, or one :math:`\left[B, C\right]` CLS
          token when ``return_sequence``

    References:
        - Source: Paper: `DINOv3 <https://arxiv.org/abs/2508.10104>`_.
          Loads `facebookresearch/dinov3
          <https://github.com/facebookresearch/dinov3>`_ through
          ``torch.hub``.
        - License: Apache-2.0 (this project). The loaded code and
          weights are under the `DINOv3 License
          <https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md>`_.

    Notes:
        Loads DINOv3 through ``torch.hub`` and replaces RoPE with an
        ONNX-friendly local module. **The hub runs the code of the
        repository without a prompt.** The node needs
        ``original_in_shape``. It does not convert for RVC2.

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

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: DinoV3
              variant: vits16
              params:
                weights_link: "<path or URL to the DINOv3 weights>"

    Compatible with:
        - Attach index: ``-1``, the last output of the input node

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
        """Load the DINOv3 model and replace its RoPE module.

        The constructor loads the hub model ``dinov3_<variant>`` from
        GitHub with ``torch.hub.load`` and ``trust_repo=True``. The patch
        size comes from the ``patch_size`` attribute of the model, or is
        ``16`` when the model has no such attribute. The constructor then
        replaces ``rope_embed`` of the model with a
        `RopePositionEmbedding`, which exports to ONNX.

        The constructor always logs a warning that the node does not
        convert for RVC2. It logs a second warning when the height or the
        width of `BaseNode.original_in_shape` is not a multiple of the
        patch size.

        Args:
            weights_link (str): The path or URL of the pretrained
                weights. The constructor passes it as ``weights`` to
                ``torch.hub.load``.
            return_sequence (bool): Whether `forward` returns the CLS
                token, of shape ``[B, C]``, for a classification head.
                When ``False``, `forward` returns ``depth`` feature maps
                for a dense head.
            variant (``DINOv3Variant``): The DINOv3 model to load.
            repo_or_dir (str): The GitHub repository that holds the hub
                entry points, as ``"owner/name"`` or
                ``"owner/name:ref"``. The constructor always loads with
                ``source="github"``.
            freeze_backbone (bool): Whether to set ``requires_grad`` to
                ``False`` for all parameters of the loaded model. Then
                only the nodes after the backbone train.
            depth (int): The number of last blocks whose outputs become
                feature maps. `forward` ignores it with
                ``return_sequence``.
            **kwargs (``Any``): Keyword arguments forwarded to both
                `BaseNode` and ``torch.hub.load``.

        Raises:
            ValueError: When ``variant`` is not a ``DINOv3Variant``
                value.

        """
        super().__init__(**kwargs)

        self._return_sequence = return_sequence
        self._depth = depth

        self.backbone, self._patch_size = self._get_backbone(
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
            self.original_in_shape[-1] % self._patch_size != 0
            or self.original_in_shape[-2] % self._patch_size != 0
        ):
            logger.warning(
                f"Image dimensions should be divisible by {self._patch_size},"
                f"but got {self.original_in_shape}. "
                "This will cause inconsistent image sizes"
                f"as DINOv3 natively reshapes to multiples of {self._patch_size}."
            )

    def _replace_rope_embedding(self) -> None:
        """Replace the RoPE module with `RopePositionEmbedding`.

        The method reads each constructor argument of
        `RopePositionEmbedding` from the model first and from the old
        RoPE module second. An argument that neither of them has keeps
        its default. The new module computes its ``periods`` again from
        these arguments. It does not copy the buffer of the old module.
        The new module uses ``repeat(1, 2)`` instead of ``tile(2)``,
        because ``tile`` does not export to ONNX.

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
        """Return the CLS token or the feature maps of the last blocks.

        With ``return_sequence``, the method takes the normed output of
        the last block and returns its CLS token. Otherwise, it takes the
        normed outputs of the last ``depth`` blocks. It reshapes the
        patch tokens of each block into a feature map. The grid size
        comes from the height and width of `BaseNode.original_in_shape`,
        not from ``inputs``.

        Args:
            inputs (``Tensor``): Image batch of shape ``[B, C, H, W]``.

        Returns:
            ``list[Tensor]``: With ``return_sequence``, one CLS token of
            shape ``[B, C]``. Otherwise, ``depth`` feature maps of shape
            ``[B, C, H // p, W // p]``, where ``p`` is the patch size and
            ``C`` is the embedding dimension.

        Raises:
            AssertionError: When the number of patch tokens of a block is
                not ``(H // p) * (W // p)``.

        """
        outs: list[Tensor] = []

        if self._return_sequence:
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
                    inputs, norm=True, n=self._depth, return_class_token=False
                ),
            )
            for x in seq_features:
                B, N, C = x.shape
                h, w = self.original_in_shape[1:]
                gh, gw = h // self._patch_size, w // self._patch_size
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
        """Return the default variant name and the DINOv3 models.

        Each variant sets only ``variant``, to its own name. The
        constructor then loads the hub model ``dinov3_<variant>``.

        Returns:
            ``tuple[str, dict[str, Kwargs]]``: The name ``"vits16"``, and
            a dictionary that maps each of the ten ``DINOv3Variant``
            values to ``{"variant": name}``.

        Example:
            >>> from luxonis_train.nodes import DinoV3
            >>> default, variants = DinoV3.get_variants()
            >>> default, len(variants)
            ('vits16', 10)
            >>> variants["convnext_tiny"]
            {'variant': 'convnext_tiny'}

        """
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
