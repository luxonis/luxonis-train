"""The token mixers and the transformer block of the SVTR neck."""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import MLP
from typeguard import typechecked

from luxonis_train.nodes.blocks import DropPath


class ConvMixer(nn.Module):
    """Token mixer of `SVTRBlock` that uses a grouped convolution.

    The mixer reshapes a sequence of tokens into a map of ``height`` by
    ``width`` tokens. A convolution with ``n_heads`` groups and a bias
    mixes each token with its neighbors. The mixer then flattens the map
    back into a sequence, row by row.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.necks.svtr_neck.blocks import ConvMixer
        >>> mixer = ConvMixer(8, height=2, width=4, n_heads=2)
        >>> mixer(torch.zeros(1, 8, 8)).shape
        torch.Size([1, 8, 8])

    """

    @typechecked
    def __init__(
        self,
        dim: int,
        height: int,
        width: int,
        n_heads: int,
        kernel_size: tuple[int, int] = (3, 3),
    ):
        """Build the grouped convolution.

        Args:
            dim (int): The number of channels of each token. It must be a
                multiple of ``n_heads``.
            height (int): The height of the token map.
            width (int): The width of the token map.
            n_heads (int): The number of groups of the convolution.
            kernel_size (tuple[int, int]): The height and the width of
                the kernel. The padding is half of each value, rounded
                down. Thus only odd values keep the size of the map.

        """
        super().__init__()
        self._height = height
        self._width = width
        self._dim = dim
        self.local_mixer = nn.Conv2d(
            dim,
            dim,
            kernel_size,
            1,
            (kernel_size[0] // 2, kernel_size[1] // 2),
            groups=n_heads,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Mix each token with its neighbors in the token map.

        Args:
            x (``Tensor``): The tokens, of shape
                ``[B, height * width, dim]``. With another number of
                tokens, ``reshape`` raises ``RuntimeError``.

        Returns:
            ``Tensor``: The mixed tokens, of the shape of ``x`` for an odd
            kernel size.

        """
        x = x.permute(0, 2, 1).reshape(
            [x.size(0), self._dim, self._height, self._width]
        )
        x = self.local_mixer(x)
        return x.flatten(2).permute(0, 2, 1)


class Attention(nn.Module):
    r"""Multi-head self-attention over a sequence of tokens.

    One linear layer computes the queries :math:`Q`, the keys :math:`K`,
    and the values :math:`V` of all heads. Each head computes

    .. math::

        \text{softmax}\left(s \, Q K^T + M\right) V

    where :math:`s` is the scale and :math:`M` is the mask. A linear
    layer then merges the heads. Dropout follows the softmax and the
    merge.

    The ``"global"`` mixer has no mask, so each token attends to every
    token. The ``"local"`` mixer lets a token attend only to the tokens
    in a window of ``kernel_size`` around it. For odd sizes, the window
    is centered on the token. The mask is ``0`` in the window and
    :math:`-\infty` outside. It has the shape ``[1, 1, N, N]``, where
    ``N`` is ``height * width``.

    **Warning:** The mask is a plain attribute, not a buffer.
    `torch.nn.Module.to` does not move it, and the state dictionary does
    not hold it. The mask stays on the CPU, so the ``"local"`` mixer
    works only for an input on the CPU.

    Example:
        With a ``3x3`` window, token ``7`` in row ``1`` and column ``2``
        sees three columns of the ``3x5`` map. Only the tokens that it
        sees get a gradient from its output:

        >>> import torch
        >>> from luxonis_train.nodes.necks.svtr_neck.blocks import Attention
        >>> attention = Attention(
        ...     8, height=3, width=5, n_heads=2, mixer="local", kernel_size=3
        ... )
        >>> tokens = torch.randn(1, 15, 8, requires_grad=True)
        >>> attention(tokens)[0, 7].sum().backward()
        >>> seen = tokens.grad[0].abs().sum(dim=1) > 0
        >>> seen.view(3, 5).int().tolist()
        [[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]]

    """

    @typechecked
    def __init__(
        self,
        dim: int,
        height: int | None = None,
        width: int | None = None,
        n_heads: int = 8,
        mixer: Literal["global", "local"] = "global",
        kernel_size: tuple[int, int] | int = (7, 11),
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        r"""Build the linear layers and the mask of the local mixer.

        Args:
            dim (int): The number of channels of each token. It must be a
                multiple of ``n_heads``. When ``qk_scale`` is ``None`` or
                ``0``, a value below ``n_heads`` makes the constructor
                raise ``ZeroDivisionError``.
            height (int | None): The height of the token map. The
                ``"local"`` mixer needs it. The ``"global"`` mixer
                ignores it.
            width (int | None): The width of the token map. The
                ``"local"`` mixer needs it. The ``"global"`` mixer ignores
                it.
            n_heads (int): The number of attention heads. Each head gets
                ``dim // n_heads`` channels.
            mixer (``Literal["global", "local"]``): The attention type.
                ``"global"`` has no mask. ``"local"`` builds the window
                mask.
            kernel_size (``tuple[int, int] | int``): The height and the
                width of the window of the ``"local"`` mixer. An integer
                gives a square window. The ``"global"`` mixer ignores it.
            qk_scale (float | None): The scale :math:`s` of the queries.
                ``None`` or ``0`` gives :math:`1 / \sqrt{d}`, where
                :math:`d` is ``dim // n_heads``.
            attn_drop (float): The dropout probability of the attention
                weights.
            proj_drop (float): The dropout probability after the linear
                layer that merges the heads.

        Raises:
            ValueError: When ``mixer`` is ``"local"`` and ``height`` or
                ``width`` is ``None``.

        """
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self._n_heads = n_heads
        self._dim = dim
        self._head_dim = dim // n_heads
        self._scale = qk_scale or 1 / math.sqrt(self._head_dim)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self._mask = None
        if mixer == "local":
            if height is None or width is None:
                raise ValueError(
                    "Height and width must be provided when using "
                    "'Attention' with the 'local' mixer."
                )
            kernel_height, kernel_width = kernel_size
            mask = torch.ones(
                (
                    height * width,
                    height + kernel_height - 1,
                    width + kernel_width - 1,
                ),
                dtype=torch.float32,
            )
            for h in range(height):
                for w in range(width):
                    mask[
                        h * width + w,
                        h : h + kernel_height,
                        w : w + kernel_width,
                    ] = 0.0
            mask_paddle = mask[
                :,
                kernel_height // 2 : height + kernel_height // 2,
                kernel_width // 2 : width + kernel_width // 2,
            ].flatten(1)
            mask_inf = torch.full(
                (height * width, height * width),
                -math.inf,
                dtype=torch.float32,
            )
            mask = torch.where(mask_paddle < 1, mask_paddle, mask_inf)
            self._mask = mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the self-attention to a sequence of tokens.

        Args:
            x (``Tensor``): The tokens, of shape ``[B, N, dim]``. With the
                ``"local"`` mixer, ``N`` must be ``height * width``, and
                ``x`` must be on the CPU. On another device, the addition
                of the CPU mask raises ``RuntimeError``.

        Returns:
            ``Tensor``: The attended tokens, of shape ``[B, N, dim]``.

        """
        batch_size = x.shape[0]
        qkv = (
            self.qkv(x)
            .reshape((batch_size, -1, 3, self._n_heads, self._head_dim))  # 0
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0] * self._scale, qkv[1], qkv[2]

        attn = q @ k.permute(0, 1, 3, 2)
        if self._mask is not None:
            attn += self._mask
        attn = F.log_softmax(attn, dim=-1).exp()
        attn: Tensor = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 1, 3).reshape((batch_size, -1, self._dim))
        x = self.proj(x)
        return self.proj_drop(x)


class SVTRBlock(nn.Module):
    r"""Transformer block of SVTR with a token mixer and an MLP.

    The block has two residual branches. The first branch runs the
    mixer:

    - ``"global"`` and ``"local"`` build an `Attention`.
    - ``"conv"`` builds a `ConvMixer`.

    The second branch runs an MLP: a linear layer to
    ``int(dim * mlp_ratio)`` features, ``act_layer``, dropout, a linear
    layer back to ``dim``, and dropout. When ``drop_path`` is above
    ``0``, `DropPath` drops each branch for random samples in training
    mode.

    ``prenorm`` sets the place of the two norm layers. :math:`f` is a
    branch:

    - ``True``: after each residual sum,
      :math:`x \leftarrow \text{norm}\left(x + f(x)\right)`.
    - ``False``: before each branch,
      :math:`x \leftarrow x + f\left(\text{norm}(x)\right)`.

    **The flag name does not match the usual term:** ``prenorm=True``
    puts the norm layers after the sums.

    Example:
        >>> import torch
        >>> from luxonis_train.nodes.necks.svtr_neck.blocks import SVTRBlock
        >>> block = SVTRBlock(8, n_heads=2, mixer="conv", height=2, width=4)
        >>> block(torch.zeros(1, 8, 8)).shape
        torch.Size([1, 8, 8])

    """

    @typechecked
    def __init__(
        self,
        dim: int,
        n_heads: int,
        height: int | None = None,
        width: int | None = None,
        mixer: Literal["global", "local", "conv"] = "global",
        mixer_kernel_size: tuple[int, int] = (7, 11),
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        epsilon: float = 1e-6,
        prenorm: bool = True,
    ):
        r"""Build the mixer, the MLP, and the two norm layers.

        Args:
            dim (int): The number of channels of each token. It must be a
                multiple of ``n_heads``.
            n_heads (int): The number of attention heads, or the number
                of convolution groups of the ``"conv"`` mixer.
            height (int | None): The height of the token map. The
                ``"local"`` and ``"conv"`` mixers need it.
            width (int | None): The width of the token map. The
                ``"local"`` and ``"conv"`` mixers need it.
            mixer (``Literal["global", "local", "conv"]``): The token
                mixer. `Attention` raises ``ValueError`` for ``"local"``
                without ``height`` and ``width``.
            mixer_kernel_size (tuple[int, int]): The window of the
                ``"local"`` mixer, or the kernel of the ``"conv"`` mixer.
                The ``"global"`` mixer ignores it.
            mlp_ratio (float): The number of hidden features of the MLP,
                as a multiple of ``dim``.
            qk_scale (float | None): The scale of the attention queries.
                ``None`` or ``0`` gives :math:`1 / \sqrt{d}`, where
                :math:`d` is ``dim // n_heads``. The ``"conv"`` mixer
                ignores it.
            dropout (float): The dropout probability of the two dropout
                layers of the MLP. The ``"global"`` and ``"local"`` mixers
                also apply it to their output.
            attn_drop (float): The dropout probability of the attention
                weights. The ``"conv"`` mixer ignores it.
            drop_path (float): The probability that `DropPath` drops a
                branch for a sample in training mode. ``0.0`` adds no
                `DropPath`.
            act_layer (``type[nn.Module]``): The activation class of the
                MLP. The MLP calls it without arguments.
            norm_layer (``type[nn.Module]``): The norm class. The block
                calls it as ``norm_layer(dim, eps=epsilon)``.
            epsilon (float): The ``eps`` of both norm layers.
            prenorm (bool): ``True`` puts the norm layers after the
                residual sums. ``False`` puts them before the branches.

        Raises:
            ValueError: When ``mixer`` is ``"conv"`` and ``height`` or
                ``width`` is ``None``.

        """
        super().__init__()
        self.norm1 = norm_layer(dim, eps=epsilon)
        if mixer in ("global", "local"):
            self.mixer = Attention(
                dim,
                n_heads=n_heads,
                mixer=mixer,
                height=height,
                width=width,
                kernel_size=mixer_kernel_size,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=dropout,
            )
        elif mixer == "conv":
            if height is None or width is None:
                raise ValueError(
                    "Height and width must be provided when using "
                    "'SVTRBlock' with the 'conv' mixer."
                )
            self.mixer = ConvMixer(
                dim,
                n_heads=n_heads,
                height=height,
                width=width,
                kernel_size=mixer_kernel_size,
            )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        self.norm2 = norm_layer(dim, eps=epsilon)
        self.mlp = MLP(
            in_channels=dim,
            hidden_channels=[int(dim * mlp_ratio), dim],
            activation_layer=act_layer,
            dropout=dropout,
        )
        self._prenorm = prenorm

    def forward(self, x: Tensor) -> Tensor:
        """Run the mixer branch and then the MLP branch.

        Args:
            x (``Tensor``): The tokens, of shape ``[B, N, dim]``. The
                ``"local"`` and ``"conv"`` mixers need ``N`` to be
                ``height * width``.

        Returns:
            ``Tensor``: The tokens after both residual branches, of shape
            ``[B, N, dim]``.

        """
        if self._prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
