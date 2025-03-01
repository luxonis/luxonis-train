from typing import Literal

import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import DropPath


class Im2Seq(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, _ = x.shape
        assert H == 1
        return x.squeeze(2).permute(0, 2, 1)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class ConvMixer(nn.Module):
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
        x = x.permute(0, 2, 1).reshape([0, self.dim, self.height, self.width])
        x = self.local_mixer(x)
        return x.flatten(2).permute(0, 2, 1)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        height: int | None = None,
        width: int | None = None,
        n_heads: int = 8,
        mixer: Literal["global", "local", "conv"] = "global",
        kernel_size: tuple[int, int] = (7, 11),
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.height = height
        self.width = width
        if mixer == "local":
            if self.height is None or self.width is None:
                raise ValueError(
                    "Height and width must be provided when using "
                    "'Attention' with 'Local' mixer."
                )
            hk = kernel_size[0]
            wk = kernel_size[1]
            mask = torch.ones(
                (
                    self.height * self.width,
                    self.height + hk - 1,
                    self.width + wk - 1,
                ),
                dtype=torch.float32,
            )
            for h in range(self.height):
                for w in range(self.width):
                    mask[h * self.width + w, h : h + hk, w : w + wk] = 0.0
            mask_paddle = mask[
                :,
                hk // 2 : self.height + hk // 2,
                wk // 2 : self.width + wk // 2,
            ].flatten(1)
            mask_inf = torch.full(
                (self.height * self.width, self.height * self.width),
                float("-inf"),
                dtype=torch.float32,
            )
            mask = torch.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.mask = mask.unsqueeze(0).unsqueeze(0)
        self.mixer = mixer

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        qkv = (
            self.qkv(x)
            .reshape((batch_size, -1, 3, self.n_heads, self.head_dim))  # 0
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q.matmul(k.permute(0, 1, 3, 2))
        if self.mixer == "local":
            attn += self.mask
        attn = nn.functional.log_softmax(attn, dim=-1).exp()
        attn = self.attn_drop(attn)

        x = (
            (attn.matmul(v))
            .permute(0, 2, 1, 3)
            .reshape((batch_size, -1, self.dim))
        )
        x = self.proj(x)
        return self.proj_drop(x)


class SVTRBlock(nn.Module):
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
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        epsilon: float = 1e-6,
        prenorm: bool = True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=epsilon)
        if mixer in {"global", "local"}:
            self.mixer = Attention(
                dim,
                n_heads=n_heads,
                mixer=mixer,
                height=height,
                width=width,
                kernel_size=mixer_kernel_size,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif mixer == "conv":
            if height is None or width is None:
                raise ValueError(
                    "Height and width must be provided when using "
                    "'SVTRBlock' with 'Conv' mixer."
                )
            self.mixer = ConvMixer(
                dim,
                n_heads=n_heads,
                height=height,
                width=width,
                kernel_size=mixer_kernel_size,
            )
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        self.norm2 = norm_layer(dim, eps=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
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
