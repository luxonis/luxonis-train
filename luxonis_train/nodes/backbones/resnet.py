"""The ResNet backbone from ``torchvision``."""

from typing import Literal

import torchvision
from luxonis_ml.typing import Kwargs
from torch import Tensor
from torchvision.models import ResNet as TorchResNet
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode


class ResNet(BaseNode):
    r"""ResNet backbone that returns the four residual stage outputs.

    ResNet adds a shortcut connection around each residual block. The
    node wraps the ``torchvision`` ResNet of the depth that ``variant``
    selects. It runs the stem and returns the outputs of the residual
    stages ``layer1`` to ``layer4``.

    Inputs:
        - ``inputs`` (``Tensor``): :math:`\left[B, 3, H, W\right]`

    Outputs:
        - ``features`` (``list[Tensor]``): ``layer1`` to ``layer4``,
          strides 4, 8, 16, 32; 64, 128, 256, 512 channels for ``"18"``
          and ``"34"``, four times as many for the other depths

    References:
        - Source: Wraps `torchvision.models.resnet
          <https://docs.pytorch.org/vision/stable/models/resnet.html>`_
          (BSD-3-Clause). Paper: `Deep Residual Learning for Image
          Recognition <https://arxiv.org/abs/1512.03385>`_.
        - License: Apache-2.0 (this project)

    Notes:
        The input must have 3 channels. ``groups``,
        ``width_per_group``, and ``replace_stride_with_dilation`` accept
        values other than their defaults only for the depths ``"50"``,
        ``"101"``, and ``"152"``. The node keeps the unused ``avgpool``
        and ``fc`` layers of the ``torchvision`` model.

    Variants:
        - ``"18"``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - ``variant``: ``"18"``
        - ``"34"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"34"``
        - ``"50"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"50"``
        - ``"101"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"101"``
        - ``"152"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``variant``: ``"152"``

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: ResNet
              variant: 18

    Compatible with:
        - Attach index: ``-1``, the last output of the input node
        - Used by: `ClassificationModel`

    """

    def __init__(
        self,
        variant: Literal["18", "34", "50", "101", "152"] = "18",
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: tuple[bool, bool, bool] = (
            False,
            False,
            False,
        ),
        weights: Literal["download", "none"] | None = None,
        **kwargs,
    ):
        """Build the ``torchvision`` ResNet of the selected depth.

        Args:
            variant (``Literal["18", "34", "50", "101", "152"]``): Depth
                of the network. ``"18"`` and ``"34"`` use basic blocks.
                The other depths use bottleneck blocks, with four times
                as many output channels. Each variant of `get_variants`
                sets it to the name of the variant. Defaults to
                ``"18"``.
            zero_init_residual (bool): Whether to set the weight of the
                last batch norm in each residual block to zero. The
                residual branch of each block then starts with zero
                output, so only the shortcut reaches the final ReLU of
                the block. Pretrained weights replace this
                initialization. See `Accurate,
                Large Minibatch SGD <https://arxiv.org/abs/1706.02677>`_.
                Defaults to ``False``.
            groups (int): Number of groups of the ``3x3`` convolution in
                each bottleneck block. The depths ``"18"`` and ``"34"``
                accept only ``1``. For other values, ``torchvision``
                raises ``ValueError``. Defaults to ``1``.
            width_per_group (int): Number of channels per group in each
                bottleneck block. The ``3x3`` convolution of a block has
                ``int(planes * width_per_group / 64) * groups``
                channels. ``planes`` is 64, 128, 256, or 512 for
                ``layer1`` to ``layer4``. The depths ``"18"`` and
                ``"34"`` accept only ``64``. For other values,
                ``torchvision`` raises ``ValueError``. Defaults to
                ``64``.
            replace_stride_with_dilation (tuple[bool, bool, bool]): For
                ``layer2``, ``layer3``, and ``layer4``, whether to
                replace the stride ``2`` with a dilation. A stage with a
                dilation keeps the resolution of the stage before it.
                The depths ``"18"`` and ``"34"`` accept only ``False``.
                For ``True``, ``torchvision`` raises
                ``NotImplementedError``.
            weights (``Literal["download", "none"] | None``): The value
                ``"download"`` loads the ``DEFAULT`` ImageNet weights of
                ``torchvision``. Any other value keeps the random
                initialization. The value does not reach `BaseNode`, so
                a checkpoint URL or ``"yolo"`` has no effect.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseNode`.

        Raises:
            ValueError: When ``variant`` is not one of the five depths.

        """
        super().__init__(**kwargs)
        self.backbone = self._get_backbone(
            variant,
            weights="DEFAULT" if weights == "download" else None,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        """Run the stem and the four residual stages.

        The stem is ``conv1``, ``bn1``, ``relu``, and ``maxpool`` of the
        ``torchvision`` model, with a total stride of ``4``. The node
        skips ``avgpool`` and ``fc``.

        Args:
            inputs (``Tensor``): Image batch of shape ``[B, 3, H, W]``.

        Returns:
            ``list[Tensor]``: The outputs of ``layer1`` to ``layer4``,
            at the strides 4, 8, 16, and 32. They have 64, 128, 256,
            and 512 channels for the depths ``"18"`` and ``"34"``, and
            four times as many for the other depths. A stage with a
            dilation keeps the stride of the stage before it.

        Example:
            >>> import torch
            >>> from luxonis_train.nodes import ResNet
            >>> node = ResNet(variant="18")
            >>> [tuple(t.shape) for t in node(torch.zeros(1, 3, 64, 64))]
            [(1, 64, 16, 16), (1, 128, 8, 8), (1, 256, 4, 4), (1, 512, 2, 2)]

        """
        outs: list[Tensor] = []
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        outs.append(x)
        x = self.backbone.layer2(x)
        outs.append(x)
        x = self.backbone.layer3(x)
        outs.append(x)
        x = self.backbone.layer4(x)
        outs.append(x)

        return outs

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        """Return the default variant name and the ResNet depths.

        Each variant sets only ``variant``, to its own name. ``"18"`` is
        the default.

        Returns:
            ``tuple[str, dict[str, Kwargs]]``: The name ``"18"``, and a
            dictionary that maps each of ``"18"``, ``"34"``, ``"50"``,
            ``"101"``, and ``"152"`` to ``{"variant": name}``.

        Example:
            >>> from luxonis_train.nodes import ResNet
            >>> default, variants = ResNet.get_variants()
            >>> default, list(variants)
            ('18', ['18', '34', '50', '101', '152'])
            >>> variants["50"]
            {'variant': '50'}

        """
        return "18", {
            "18": {"variant": "18"},
            "34": {"variant": "34"},
            "50": {"variant": "50"},
            "101": {"variant": "101"},
            "152": {"variant": "152"},
        }

    @staticmethod
    def _get_backbone(
        variant: Literal["18", "34", "50", "101", "152"], **kwargs
    ) -> TorchResNet:
        variants = {
            "18": torchvision.models.resnet18,
            "34": torchvision.models.resnet34,
            "50": torchvision.models.resnet50,
            "101": torchvision.models.resnet101,
            "152": torchvision.models.resnet152,
        }
        if variant not in variants:
            raise ValueError(
                "ResNet model variant should be in "
                f"{list(variants.keys())}, got {variant}."
            )
        return variants[variant](**kwargs)
