import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import MLP
from typeguard import typechecked

from luxonis_train.nodes.blocks import DropPath


class ConvMixer(nn.Module):
    @typechecked
    def __init__(
        self,
        dim: int,
        height: int,
        width: int,
        n_heads: int,
        kernel_size: tuple[int, int] = (3, 3),
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.dim = dim
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
        x = x.permute(0, 2, 1).reshape(
            [x.size(0), self.dim, self.height, self.width]
        )
        x = self.local_mixer(x)
        return x.flatten(2).permute(0, 2, 1)


class Attention(nn.Module):
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
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = qk_scale or 1 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mask = None
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
            self.mask = mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        qkv = (
            self.qkv(x)
            .reshape((batch_size, -1, 3, self.n_heads, self.head_dim))  # 0
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q @ k.permute(0, 1, 3, 2)
        if self.mask is not None:
            attn += self.mask
        attn = F.log_softmax(attn, dim=-1).exp()
        attn: Tensor = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 1, 3).reshape((batch_size, -1, self.dim))
        x = self.proj(x)
        return self.proj_drop(x)


class SVTRBlock(nn.Module):
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
        self.prenorm = prenorm

    def forward(self, x: Tensor) -> Tensor:
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
