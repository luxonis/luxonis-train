"""The MicroNet backbone for very low compute budgets."""

from typing import TypedDict

from torch import Tensor, nn
from typeguard import typechecked
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode

from .blocks import MicroBlock, Stem


class MicroNet(BaseNode):
    # TODO: Check docs, add source
    r"""MicroNet backbone for very low compute budgets.

    The backbone is a stem and a sequence of MicroBlocks. The stem
    splits a :math:`3 \times 3` convolution with a stride of ``2`` into
    a :math:`3 \times 1` and a grouped :math:`1 \times 3` convolution.
    The MicroBlocks use factorized depthwise convolutions, grouped
    :math:`1 \times 1` convolutions, channel shuffles, and Dynamic
    Shift-Max activations.
    `luxonis_train.nodes.backbones.micronet.micronet.LayerParamsDict`
    describes the parameters of one MicroBlock.

    Inputs:
        - ``inputs`` (``Tensor``): :math:`\left[B, 3, H, W\right]`

    Outputs:
        - ``features`` (``list[Tensor]``): one for each layer that
          ``out_indices`` selects, in layer order; strides 4, 8, 16, 32
          for ``"M1"`` and ``"M2"``, and 4, 8, 32, 32 for ``"M3"``

    References:
        - Source: Reimplemented from `MicroNet: Improving Image
          Recognition with Extremely Low FLOPs
          <https://arxiv.org/abs/2108.05894>`_.
        - License: Apache-2.0 (this project)

    Notes:
        The stem takes exactly 3 input channels. It gives
        ``stem_groups[0] * stem_groups[1]`` output channels.
        ``stem_channels`` must be equal to that product. In
        ``out_indices``, index ``0`` is the stem. Index ``i`` is the
        MicroBlock of ``layer_params[i - 1]``.

    Variants:
        - ``"M1"``:
            - Default: yes
            - Aliases: None
            - Parameters:
                - ``stem_channels``: ``6``
                - ``stem_groups``: ``(3, 2)``
                - ``init_a``: ``(1.0, 1.0)``
                - ``init_b``: ``(0.0, 0.0)``
                - ``out_indices``: ``[1, 2, 4, 7]``
            - Layers:
                - ``0``:
                    - ``out_channels``: ``8``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 6)``
                    - ``groups_2``: ``(2, 2)``
                    - ``dy_shift``: ``(2, 0, 1)``
                    - ``reduction_factor``: ``1``
                - ``1``:
                    - ``out_channels``: ``16``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 8)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``2``:
                    - ``out_channels``: ``16``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 16)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``3``:
                    - ``out_channels``: ``32``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(4, 4)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``4``:
                    - ``out_channels``: ``64``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``5``:
                    - ``out_channels``: ``96``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
                - ``6``:
                    - ``out_channels``: ``576``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(12, 12)``
                    - ``groups_2``: ``(0, 0)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
        - ``"M2"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``stem_channels``: ``8``
                - ``stem_groups``: ``(4, 2)``
                - ``init_a``: ``(1.0, 1.0)``
                - ``init_b``: ``(0.0, 0.0)``
                - ``out_indices``: ``[1, 3, 6, 9]``
            - Layers:
                - ``0``:
                    - ``out_channels``: ``12``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 8)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 0, 1)``
                    - ``reduction_factor``: ``1``
                - ``1``:
                    - ``out_channels``: ``16``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 12)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``2``:
                    - ``out_channels``: ``24``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 16)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``3``:
                    - ``out_channels``: ``32``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(6, 6)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``1``
                - ``4``:
                    - ``out_channels``: ``32``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
                - ``5``:
                    - ``out_channels``: ``64``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
                - ``6``:
                    - ``out_channels``: ``96``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
                - ``7``:
                    - ``out_channels``: ``128``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(12, 12)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
                - ``8``:
                    - ``out_channels``: ``768``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(16, 16)``
                    - ``groups_2``: ``(0, 0)``
                    - ``dy_shift``: ``(2, 2, 1)``
                    - ``reduction_factor``: ``2``
        - ``"M3"``:
            - Default: no
            - Aliases: None
            - Parameters:
                - ``stem_channels``: ``12``
                - ``stem_groups``: ``(4, 3)``
                - ``init_a``: ``(1.0, 0.5)``
                - ``init_b``: ``(0.0, 0.5)``
                - ``out_indices``: ``[1, 3, 8, 12]``
            - Layers:
                - ``0``:
                    - ``out_channels``: ``16``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 12)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``1``
                - ``1``:
                    - ``out_channels``: ``24``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 16)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``1``
                - ``2``:
                    - ``out_channels``: ``24``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(2, 2)``
                    - ``groups_1``: ``(0, 24)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``1``
                - ``3``:
                    - ``out_channels``: ``32``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(6, 6)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``1``
                - ``4``:
                    - ``out_channels``: ``32``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(4, 4)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``5``:
                    - ``out_channels``: ``64``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``6``:
                    - ``out_channels``: ``80``
                    - ``stride``: ``2``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(8, 8)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``7``:
                    - ``out_channels``: ``80``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(10, 10)``
                    - ``groups_2``: ``(8, 8)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``8``:
                    - ``out_channels``: ``120``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(10, 10)``
                    - ``groups_2``: ``(10, 10)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``9``:
                    - ``out_channels``: ``120``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``5``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(12, 12)``
                    - ``groups_2``: ``(10, 10)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``10``:
                    - ``out_channels``: ``144``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(12, 12)``
                    - ``groups_2``: ``(12, 12)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``
                - ``11``:
                    - ``out_channels``: ``864``
                    - ``stride``: ``1``
                    - ``kernel_size``: ``3``
                    - ``expand_ratio``: ``(1, 6)``
                    - ``groups_1``: ``(12, 12)``
                    - ``groups_2``: ``(0, 0)``
                    - ``dy_shift``: ``(0, 2, 0)``
                    - ``reduction_factor``: ``2``

    .. Generated by scripts/gen_component_docs.py. Do not edit.

    Example:
        A node entry in the ``model.nodes`` section of a config:

        .. code-block:: yaml

            - name: MicroNet
              variant: M1

    Compatible with:
        - Attach index: ``-1``, the last output of the input node

    """

    @typechecked
    def __init__(
        self,
        stem_channels: int = 6,
        stem_groups: tuple[int, int] = (3, 2),
        init_a: tuple[float, float] = (1.0, 1.0),
        init_b: tuple[float, float] = (0.0, 0.0),
        out_indices: list[int] | None = None,
        layer_params: list["LayerParamsDict"] | None = None,
        **kwargs,
    ):
        r"""Initialize the MicroNet backbone.

        The constructor builds the stem, then one `MicroBlock` for each
        entry of ``layer_params``. Each block takes the ``out_channels``
        of the previous block as its input channels. The first block
        takes ``stem_channels``. The default values of the arguments are
        the values of the ``"M1"`` variant.

        Args:
            stem_channels (int): The number of input channels of the
                first MicroBlock. The stem does not read this value. The
                value must be equal to ``stem_groups[0] * stem_groups[1]``,
                the output channels of the stem. Otherwise, `forward`
                raises ``RuntimeError``.
            stem_groups (tuple[int, int]): The channel layout of the
                stem. The first value is the number of output channels
                of the :math:`3 \times 1` convolution. It is also the
                number of groups of the :math:`1 \times 3` convolution and
                of the channel shuffle. The second value is the channel
                multiplier of the :math:`1 \times 3` convolution.
            init_a (tuple[float, float]): The offsets that Dynamic
                Shift-Max adds to the weights of the input features, one
                for each of its two branches. The MicroBlocks use them in
                the activations that the first two values of
                ``dy_shift`` select. Only ``init_a[0]`` has an effect,
                because `DYShiftMax` adds ``init_b[1]`` to the input
                weights of the second branch.
            init_b (tuple[float, float]): The offsets that Dynamic
                Shift-Max adds to the weights of the channel-shifted
                features, one for each of its two branches. The
                activations that use ``init_a`` also use them.
                ``init_b[1]`` also goes to the input weights of the
                second branch.
            out_indices (list[int] | None): The indices of the layers
                whose outputs `forward` returns. Index ``0`` is the stem.
                Index ``i`` is the MicroBlock of ``layer_params[i - 1]``.
                The node ignores negative indices and indices without a
                layer. ``None`` or an empty list selects ``[1, 2, 4, 7]``.
                This default does not change with ``layer_params``.
            layer_params (``list[LayerParamsDict] | None``): The
                parameters of the MicroBlocks, one dictionary for each
                block, in order. ``None`` or an empty list selects the
                seven MicroBlocks of the ``"M1"`` variant.
            **kwargs (``Any``): Keyword arguments forwarded to
                `BaseNode`.

        """
        super().__init__(**kwargs)
        out_indices = out_indices or [1, 2, 4, 7]
        layer_params = (
            layer_params or self.get_variants()[1]["M1"]["layer_params"]
        )

        self._out_indices = out_indices
        self.layers = nn.ModuleList([Stem(3, 2, stem_groups)])

        in_channels = stem_channels
        for params in layer_params:
            self.layers.append(
                MicroBlock(
                    in_channels,
                    init_a=init_a,
                    init_b=init_b,
                    **params,
                )
            )
            in_channels = params["out_channels"]

    def forward(self, inputs: Tensor) -> list[Tensor]:
        """Run the stem and the MicroBlocks on a batch of images.

        The method runs all layers in order. It also runs the layers
        after the highest selected index. With the ``out_indices`` of a
        variant, and a height and a width that are multiples of ``32``,
        the method returns these shapes:

        - ``"M1"``: ``[B, 8, H/4, W/4]``, ``[B, 16, H/8, W/8]``,
          ``[B, 32, H/16, W/16]``, and ``[B, 576, H/32, W/32]``.
        - ``"M2"``: ``[B, 12, H/4, W/4]``, ``[B, 24, H/8, W/8]``,
          ``[B, 64, H/16, W/16]``, and ``[B, 768, H/32, W/32]``.
        - ``"M3"``: ``[B, 16, H/4, W/4]``, ``[B, 24, H/8, W/8]``,
          ``[B, 80, H/32, W/32]``, and ``[B, 864, H/32, W/32]``.

        Args:
            inputs (``Tensor``): The input images, of shape
                ``[B, 3, H, W]``.

        Returns:
            ``list[Tensor]``: The outputs of the layers whose indices are
            in ``out_indices``, in layer order. The order of the indices
            and repeated indices do not change the result.

        Examples:
            >>> import torch
            >>> from luxonis_train.nodes.backbones import MicroNet
            >>> node = MicroNet(variant="M1")
            >>> features = node(torch.zeros(2, 3, 64, 64))
            >>> [tuple(feature.shape) for feature in features]
            [(2, 8, 16, 16), (2, 16, 8, 8), (2, 32, 4, 4), (2, 576, 2, 2)]

            Index ``0`` selects the output of the stem. The order of the
            indices does not change the order of the outputs.

            >>> node = MicroNet(out_indices=[7, 0])
            >>> features = node(torch.zeros(2, 3, 64, 64))
            >>> [tuple(feature.shape) for feature in features]
            [(2, 6, 32, 32), (2, 576, 2, 2)]

        """
        outs: list[Tensor] = []
        for i, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if i in self._out_indices:
                outs.append(inputs)
        return outs

    @override
    @staticmethod
    def get_variants() -> tuple[str, dict[str, "MicroNetVariantDict"]]:
        """Return the default variant name and the three MicroNet variants.

        Each variant sets ``stem_channels``, ``stem_groups``, ``init_a``,
        ``init_b``, ``out_indices``, and ``layer_params``. ``"M2"`` and
        ``"M3"`` have more MicroBlocks and more channels than ``"M1"``.
        The ``Variants`` section of `MicroNet` lists all values. Each
        call builds new dictionaries.

        Returns:
            tuple[str, dict[str, MicroNetVariantDict]]: The name of the
            default variant, ``"M1"``, and a dictionary that maps
            ``"M1"``, ``"M2"``, and ``"M3"`` to their constructor
            arguments.

        Example:
            >>> from luxonis_train.nodes.backbones import MicroNet
            >>> default, variants = MicroNet.get_variants()
            >>> default, sorted(variants)
            ('M1', ['M1', 'M2', 'M3'])
            >>> [len(params["layer_params"]) for params in variants.values()]
            [7, 9, 12]

        """
        return "M1", {
            "M1": {
                "stem_channels": 6,
                "stem_groups": (3, 2),
                "init_a": (1.0, 1.0),
                "init_b": (0.0, 0.0),
                "out_indices": [1, 2, 4, 7],
                "layer_params": [
                    {
                        "out_channels": 8,
                        "stride": 2,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 6),
                        "groups_2": (2, 2),
                        "dy_shift": (2, 0, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 16,
                        "stride": 2,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 8),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 16,
                        "stride": 2,
                        "kernel_size": 5,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 16),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 32,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (4, 4),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 64,
                        "stride": 2,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (8, 8),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 96,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (8, 8),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 576,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (1, 6),
                        "groups_1": (12, 12),
                        "groups_2": (0, 0),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                ],
            },
            "M2": {
                "stem_channels": 8,
                "stem_groups": (4, 2),
                "init_a": (1.0, 1.0),
                "init_b": (0.0, 0.0),
                "out_indices": [1, 3, 6, 9],
                "layer_params": [
                    {
                        "out_channels": 12,
                        "stride": 2,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 8),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 0, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 16,
                        "stride": 2,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 12),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 24,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 16),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 32,
                        "stride": 2,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (6, 6),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 32,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (4, 4),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 64,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (8, 8),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 96,
                        "stride": 2,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (8, 8),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 128,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (1, 6),
                        "groups_1": (12, 12),
                        "groups_2": (8, 8),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 768,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (1, 6),
                        "groups_1": (16, 16),
                        "groups_2": (0, 0),
                        "dy_shift": (2, 2, 1),
                        "reduction_factor": 2,
                    },
                ],
            },
            "M3": {
                "stem_channels": 12,
                "stem_groups": (4, 3),
                "init_a": (1.0, 0.5),
                "init_b": (0.0, 0.5),
                "out_indices": [1, 3, 8, 12],
                "layer_params": [
                    {
                        "out_channels": 16,
                        "stride": 2,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 12),
                        "groups_2": (4, 4),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 24,
                        "stride": 2,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 16),
                        "groups_2": (4, 4),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 24,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (2, 2),
                        "groups_1": (0, 24),
                        "groups_2": (4, 4),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 32,
                        "stride": 2,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (6, 6),
                        "groups_2": (4, 4),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 1,
                    },
                    {
                        "out_channels": 32,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (4, 4),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 64,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (8, 8),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 80,
                        "stride": 2,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (8, 8),
                        "groups_2": (8, 8),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 80,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (10, 10),
                        "groups_2": (8, 8),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 120,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (10, 10),
                        "groups_2": (10, 10),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 120,
                        "stride": 1,
                        "kernel_size": 5,
                        "expand_ratio": (1, 6),
                        "groups_1": (12, 12),
                        "groups_2": (10, 10),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 144,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (1, 6),
                        "groups_1": (12, 12),
                        "groups_2": (12, 12),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                    {
                        "out_channels": 864,
                        "stride": 1,
                        "kernel_size": 3,
                        "expand_ratio": (1, 6),
                        "groups_1": (12, 12),
                        "groups_2": (0, 0),
                        "dy_shift": (0, 2, 0),
                        "reduction_factor": 2,
                    },
                ],
            },
        }


class LayerParamsDict(TypedDict):
    r"""The parameters of one MicroBlock in ``layer_params``.

    `MicroNet` passes the keys to `MicroBlock` as keyword arguments,
    together with the input channels of the block, ``init_a``, and
    ``init_b``. A block first expands the channels to
    ``in_channels * expand_ratio[0] * expand_ratio[1]``. A lite or a
    full block then projects them to ``out_channels``. ``groups_1`` and
    ``groups_2`` select one of three block types:

    - A *lite* block, when ``groups_1[0]`` is ``0``. A factorized
      depthwise convolution expands the channels. A grouped
      :math:`1 \times 1` convolution projects them.
    - A *transition* block, when ``groups_2[1]`` is ``0`` and
      ``groups_1[0]`` is not ``0``. A grouped :math:`1 \times 1`
      convolution expands the channels. The block has no depthwise
      convolution and no projection. Its layers do not read
      ``out_channels``, ``stride``, or ``kernel_size``.
    - A *full* block, in all other cases. A grouped :math:`1 \times 1`
      convolution expands the channels. A factorized depthwise
      convolution follows. A second grouped :math:`1 \times 1`
      convolution projects the channels.

    The block adds its input to its output when ``stride`` is ``1`` and
    ``out_channels`` is equal to the input channels.

    Attributes:
        out_channels (int): The number of channels after the projection.
            The next block takes this value as its input channels. For a
            transition block, the value must be equal to the expanded
            number of channels.
        stride (int): The stride of the depthwise convolution. ``2``
            halves the height and the width.
        kernel_size (int): The kernel size :math:`k` of the depthwise
            convolution. The block splits the convolution into a
            :math:`k \times 1` and a :math:`1 \times k` convolution.
        expand_ratio (tuple[int, int]): The two channel multipliers of
            the expansion. A lite block applies one multiplier in each
            half of its depthwise convolution. The other blocks apply
            the product in the expansion :math:`1 \times 1` convolution.
        groups_1 (tuple[int, int]): The groups before the projection.
            The first value is the number of groups of the expansion
            :math:`1 \times 1` convolution. ``0`` selects a lite block.
            The second value is the number of groups of the Dynamic
            Shift-Max activations before the projection. The activation
            after the depthwise convolution of a full block is an
            exception. When the value is not ``1``, the groups of that
            activation are the expanded channels divided by the value.
            In a lite or a full block, the value also sets the groups of
            one channel shuffle before the projection. In a transition
            block, it sets the groups of the only Dynamic Shift-Max.
        groups_2 (tuple[int, int]): The groups of the projection and of
            the layers after it. The first value is the number of groups
            of the projection :math:`1 \times 1` convolution. The second
            value sets the groups of the last Dynamic Shift-Max and of one
            channel shuffle after the projection. ``0`` selects a
            transition block when ``groups_1[0]`` is not ``0``.
        dy_shift (tuple[int, int, int]): The activations after the
            expansion :math:`1 \times 1` convolution, after the depthwise
            convolution, and after the last convolution. In the first two
            positions, ``0`` or a negative value selects ``ReLU6``. ``2``
            selects Dynamic Shift-Max with the maximum of two branches.
            Another positive value selects Dynamic Shift-Max with one
            branch. In the last position, a positive value selects
            Dynamic Shift-Max with one branch. Other values select no
            activation. A lite block ignores the first value. A
            transition block reads only the last value. Values other
            than ``0`` also add channel shuffles to a lite or a full
            block. In a lite block, the last value adds its shuffle only
            when ``out_channels`` is even.
        reduction_factor (int): The reduction of the squeeze network in
            Dynamic Shift-Max. Its hidden layer has
            ``channels // (8 * reduction_factor)`` units, where
            ``channels`` is the number of input channels of the
            activation. Dynamic Shift-Max rounds this number to a
            multiple of ``4``, with a minimum of ``4``. After the
            projection of a lite block, the activation divides by
            ``4 * reduction_factor`` instead. After the projection of a
            full block, it does so only when ``out_channels`` is smaller
            than the expanded channels.

    """

    out_channels: int
    stride: int
    kernel_size: int
    expand_ratio: tuple[int, int]
    groups_1: tuple[int, int]
    groups_2: tuple[int, int]
    dy_shift: tuple[int, int, int]
    reduction_factor: int


class MicroNetVariantDict(TypedDict):
    """The constructor arguments of one MicroNet variant.

    `MicroNet.get_variants` maps each variant name to one of these
    dictionaries. The keys are parameters of the `MicroNet` constructor.
    The ``__init__`` docstring of `MicroNet` describes them in full.

    Attributes:
        stem_channels (int): The number of input channels of the first
            MicroBlock. It must be equal to the output channels of the
            stem, ``stem_groups[0] * stem_groups[1]``.
        stem_groups (tuple[int, int]): The channel layout of the stem.
        init_a (tuple[float, float]): The Dynamic Shift-Max offsets for
            the weights of the input features.
        init_b (tuple[float, float]): The Dynamic Shift-Max offsets for
            the weights of the channel-shifted features.
        out_indices (list[int]): The indices of the layers whose outputs
            the node returns. Index ``0`` is the stem.
        layer_params (``list[LayerParamsDict]``): The parameters of the
            MicroBlocks, in order.

    """

    stem_channels: int
    stem_groups: tuple[int, int]
    init_a: tuple[float, float]
    init_b: tuple[float, float]
    out_indices: list[int]
    layer_params: list[LayerParamsDict]
