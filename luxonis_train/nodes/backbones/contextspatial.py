"""The BiSeNet V1 context-spatial backbone and its two paths.

`SpatialPath` keeps the spatial detail of the image. `ContextPath` runs
another backbone for a large receptive field. `ContextSpatial` fuses the
outputs of the two paths.

"""

import torch.nn.functional as F
from luxonis_ml.typing import Kwargs
from torch import Tensor, nn

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.blocks import (
    AttentionRefinementBlock,
    ConvBlock,
    FeatureFusionBlock,
)
from luxonis_train.registry import NODES


class ContextSpatial(BaseNode):
    r"""BiSeNet V1 backbone that fuses a spatial and a context path.

    The node runs two paths on the same image and fuses their outputs:

    - `SpatialPath` keeps the detail. It gives 128 channels at 1/8 of
      the input size.
    - `ContextPath` runs a context backbone and refines its last two
      feature maps. It also gives 128 channels at 1/8 of the input
      size.
    - A `FeatureFusionBlock` fuses the two maps into 256 channels.

    `BiSeNetHead` can read the fused map.

    Inputs:
        - ``inputs`` (``Tensor``): :math:`\left[B, 3, H, W\right]`

    Outputs:
        - ``features`` (``list[Tensor]``): :math:`\left[B, 256, H/8,
          W/8\right]`

    References:
        - Source: Reimplemented from `BiSeNet: Bilateral Segmentation
          Network for Real-time Semantic Segmentation
          <https://arxiv.org/abs/1808.00897>`_.
        - License: Apache-2.0 (this project)

    Notes:
        The node builds `SpatialPath` for 3 input channels. The last two
        feature maps of the context backbone must have the strides 16
        and 32. Then all multiples of 32 for :math:`H` and :math:`W`
        work. Other sizes can give feature maps of different sizes, and
        PyTorch then raises ``RuntimeError``. In the training state, a
        batch must hold more than one image. The reason is that each
        batch norm after a global pooling sees one value per image.
        `ContextPath` creates its attention refinement blocks and its
        global context layer in the first forward call. Before that
        call, the state dictionary of the node does not hold them.

    Variants:
        None. Configure the node through ``params``.

    See Also:
        `The BiseNetv1 repository
        <https://github.com/taveraantonio/BiseNetv1>`_

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: ContextSpatial

    Compatible with:
        - Attach index: ``-1``, the last output of the input node

    """

    def __init__(
        self,
        context_backbone: str | nn.Module = "MobileNetV2",
        backbone_kwargs: Kwargs | None = None,
        **kwargs,
    ):
        """Build the two paths and the feature fusion block.

        Args:
            context_backbone (``str | nn.Module``): The backbone of
                `ContextPath`. A string names a node in
                `luxonis_train.registry.NODES`. The constructor builds
                that node with ``backbone_kwargs`` and ``kwargs``. An
                unknown name makes the registry raise ``KeyError``. A
                module goes to `ContextPath` as it is. The backbone must
                return a sequence of feature maps, and the last two
                entries must have the strides 16 and 32. Defaults to
                ``"MobileNetV2"``.
            backbone_kwargs (``Kwargs | None``): Keyword arguments for
                the backbone node. The constructor reads them only when
                ``context_backbone`` is a string. It merges ``kwargs``
                into this dictionary, and a key in ``kwargs`` replaces
                the same key here. A dictionary that is not empty
                changes in place, so the caller sees the merged keys.
                ``None`` and an empty dictionary start from a new empty
                dictionary.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseNode`. A backbone built from a name gets them too.

        """
        super().__init__(**kwargs)

        if isinstance(context_backbone, str):
            backbone_kwargs = backbone_kwargs or {}
            backbone_kwargs |= kwargs
            context_backbone = NODES.get(context_backbone)(**backbone_kwargs)

        self.context_path = ContextPath(context_backbone)
        self.spatial_path = SpatialPath(3, 128)
        self.ffm = FeatureFusionBlock(256, 256)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        """Fuse the spatial detail with the refined context features.

        `SpatialPath` and `ContextPath` both give 128 channels at 1/8 of
        the input size. `FeatureFusionBlock` concatenates the two maps
        and fuses them into 256 channels. The node ignores the second
        output of `ContextPath`.

        Args:
            inputs (``Tensor``): Image batch of shape ``[B, 3, H, W]``.
                All multiples of ``32`` for ``H`` and ``W`` work. Other
                sizes can give feature maps of different sizes, and
                PyTorch then raises ``RuntimeError``. In the training
                state, ``B`` must be larger than ``1``. Otherwise, a
                batch norm raises ``ValueError``.

        Returns:
            ``list[Tensor]``: A list with one fused feature map of shape
            ``[B, 256, ceil(H / 8), ceil(W / 8)]``.

        Example:
            >>> import torch
            >>> from luxonis_train.nodes import ContextSpatial
            >>> node = ContextSpatial()
            >>> [tuple(t.shape) for t in node(torch.zeros(2, 3, 64, 64))]
            [(2, 256, 8, 8)]

            A size that is not a multiple of 32 can work too:

            >>> [tuple(t.shape) for t in node(torch.zeros(2, 3, 62, 94))]
            [(2, 256, 8, 12)]

        """
        spatial_out = self.spatial_path(inputs)
        context16, _ = self.context_path(inputs)
        fm_fuse = self.ffm(spatial_out, context16)
        return [fm_fuse]


class SpatialPath(nn.Module):
    """Spatial path of `ContextSpatial` that keeps the image detail.

    Three strided `ConvBlock` layers reduce the input to 1/8 of its
    size. The first layer has a ``7x7`` kernel, and the other two have a
    ``3x3`` kernel. Each of them has stride ``2`` and 64 output
    channels. A ``1x1`` `ConvBlock` then maps the result to
    ``out_channels``. Every layer uses batch norm and ReLU.

    Example:
        >>> import torch
        >>> path = SpatialPath(3, 128)
        >>> path(torch.zeros(1, 3, 64, 64)).shape
        torch.Size([1, 128, 8, 8])

    """

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize the four convolutions.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        """
        super().__init__()
        intermediate_channels = 64
        self.conv_7x7 = ConvBlock(
            in_channels,
            intermediate_channels,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.conv_3x3_1 = ConvBlock(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv_3x3_2 = ConvBlock(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv_1x1 = ConvBlock(
            intermediate_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Reduce ``x`` to 1/8 of its size and map its channels.

        Args:
            x (``Tensor``): Input of shape ``[B, in_channels, H, W]``.

        Returns:
            ``Tensor``: Output of shape
            ``[B, out_channels, ceil(H / 8), ceil(W / 8)]``.

        """
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        return self.conv_1x1(x)


class ContextPath(nn.Module):
    """Context path of `ContextSpatial` on top of a backbone.

    The path reads the last two feature maps of the backbone. Below,
    ``f16`` and ``f32`` are these maps, at the strides 16 and 32. The
    path runs these steps:

    1. An `AttentionRefinementBlock` maps each of ``f16`` and ``f32`` to
       128 channels.
    2. A global average pool and a ``1x1`` `ConvBlock` turn ``f32`` into
       one context vector for each image. The path resizes the vector
       to the size of ``f32`` and adds it to the refined ``f32``.
    3. The path upsamples the sum by ``2``, applies a ``3x3``
       `ConvBlock`, and adds the result to the refined ``f16``.
    4. The path upsamples the second sum by ``2`` and applies a second
       ``3x3`` `ConvBlock`.

    The `ConvBlock` layers of the steps 2 to 4 use batch norm and ReLU.
    All resizes are bilinear with aligned corners.

    The attention blocks and the global context layer depend on the
    channel counts of the backbone. Thus the path creates them in the
    first `forward` call. Before that call, the state dictionary of the
    path does not hold them.

    """

    def __init__(self, backbone: nn.Module):
        """Store the backbone and build the fixed layers.

        The fixed layers are the two upsampling layers and the two
        ``3x3`` refinement blocks. `forward` builds the other layers.

        Args:
            backbone (``nn.Module``): Module that returns a sequence of
                feature maps. The path reads the last two entries. They
                must have the strides 16 and 32.

        """
        super().__init__()
        self.backbone = backbone

        self.up16 = nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=True
        )
        self.up32 = nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=True
        )

        self.refine16 = ConvBlock(128, 128, 3, 1, 1)
        self.refine32 = ConvBlock(128, 128, 3, 1, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Refine the last two backbone maps and merge them.

        The first call also creates the two attention blocks and the
        global context layer. It reads their input channels from the
        backbone outputs. The new layers use the default device and
        dtype of PyTorch, not those of ``x``. They start in the training
        state, also when the path is in the eval state.

        Args:
            x (``Tensor``): Input of the backbone, of shape
                ``[B, C, H, W]``. The stride-32 map, upsampled by ``2``,
                must have the size of the stride-16 map. All multiples
                of ``32`` for ``H`` and ``W`` meet this condition. In the
                training state, ``B`` must be larger than ``1``. The
                reason is that each batch norm after a global pooling
                sees one value per image.

        Returns:
            ``tuple[Tensor, Tensor]``: Two maps with 128 channels. The
            first is the final merged map, at twice the size of the
            stride-16 map. The second is the refined stride-32 branch
            after its upsampling and its ``3x3`` `ConvBlock`, at the
            size of the stride-16 map. When ``H`` and ``W`` are
            multiples of ``32``, the shapes are
            ``[B, 128, H / 8, W / 8]`` and ``[B, 128, H / 16, W / 16]``.

        Example:
            >>> import torch
            >>> from luxonis_train.nodes import MobileNetV2
            >>> path = ContextPath(MobileNetV2())
            >>> hasattr(path, "arm16")
            False
            >>> [tuple(t.shape) for t in path(torch.zeros(2, 3, 64, 64))]
            [(2, 128, 8, 8), (2, 128, 4, 4)]
            >>> hasattr(path, "arm16")
            True

        """
        *_, down16, down32 = self.backbone(x)

        if not hasattr(self, "arm16"):
            self.arm16 = AttentionRefinementBlock(down16.shape[1], 128)
            self.arm32 = AttentionRefinementBlock(down32.shape[1], 128)

            self.global_context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvBlock(down32.shape[1], 128, 1, 1, 0),
            )

        arm_down16 = self.arm16(down16)
        arm_down32 = self.arm32(down32)

        global_down32 = self.global_context(down32)
        global_down32 = F.interpolate(
            global_down32,
            size=down32.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        arm_down32 += global_down32
        arm_down32 = self.up32(arm_down32)
        arm_down32 = self.refine32(arm_down32)

        arm_down16 += arm_down32
        arm_down16 = self.up16(arm_down16)
        arm_down16 = self.refine16(arm_down16)

        return arm_down16, arm_down32
