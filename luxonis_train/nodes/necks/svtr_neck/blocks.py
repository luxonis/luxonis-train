import torch
from torch import Tensor, nn

from luxonis_train.nodes.blocks import ConvModule, DropPath


class Im2Seq(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == 1
        x = x.squeeze(axis=2)
        x = x.permute([0, 2, 1])
        return x


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
        x = self.drop(x)
        return x


class ConvMixer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        HW: tuple[int, int] | None = (8, 25),
        local_k: tuple[int, int] = (3, 3),
    ):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(
            dim,
            dim,
            local_k,  # type: ignore
            1,
            (local_k[0] // 2, local_k[1] // 2),
            groups=num_heads,
            bias=True,
        )

    def forward(self, x):
        h = self.HW[0] # type: ignore
        w = self.HW[1] # type: ignore
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mixer: str = "Global",
        HW: tuple[int, int] | None = None,
        local_k: tuple[int, int] = (7, 11),
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == "Local" and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones(
                (H * W, H + hk - 1, W + wk - 1), dtype=torch.float32 # type: ignore
            )  # type: ignore
            for h in range(H):
                for w in range(W):
                    mask[h * W + w, h : h + hk, w : w + wk] = 0.0
            mask_paddle = mask[
                :, hk // 2 : H + hk // 2, wk // 2 : W + wk // 2
            ].flatten(1)
            mask_inf = torch.full(
                (H * W, H * W), float("-inf"), dtype=torch.float32 # type: ignore
            )  # type: ignore
            mask = torch.where(mask_paddle < 1, mask_paddle, mask_inf)
            self.mask = mask.unsqueeze(0).unsqueeze(0)
        self.mixer = mixer

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        qkv = (
            self.qkv(x)
            .reshape((batch_size, -1, 3, self.num_heads, self.head_dim))  # 0
            .permute((2, 0, 3, 1, 4))
        )
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q.matmul(k.permute((0, 1, 3, 2)))
        if self.mixer == "Local":
            attn += self.mask
        attn = nn.functional.log_softmax(attn, dim=-1).exp()
        attn = self.attn_drop(attn)

        x = (
            (attn.matmul(v))
            .permute((0, 2, 1, 3))
            .reshape((batch_size, -1, self.dim))
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SVTRBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mixer: str = "Global",
        local_mixer: tuple[int, int] = (7, 11),
        HW: tuple[int, int] | None = None,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] | str = "nn.LayerNorm",
        epsilon: float = 1e-6,
        prenorm: bool = True,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == "Global" or mixer == "Local":
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif mixer == "Conv":
            self.mixer = ConvMixer(
                dim, num_heads=num_heads, HW=HW, local_k=local_mixer
            )
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm2 = norm_layer(dim)
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


class EncoderWithSVTR(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dims: int = 64,
        depth: int = 2,
        hidden_dims: int = 120,
        use_guide: bool = False,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path: float = 0.0,
        kernel_size: tuple[int, int] = (3, 3),
        qk_scale: float | None = None,
    ):
        super(EncoderWithSVTR, self).__init__()
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvModule(
            in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=kernel_size[0] // 2,
            bias=True,
            activation=nn.ReLU(),
        )
        self.conv2 = ConvModule(
            in_channels // 8,
            hidden_dims,
            kernel_size=1,
            bias=True,
            activation=nn.ReLU(),
        )

        self.svtr_block = nn.ModuleList(
            [
                SVTRBlock(
                    dim=hidden_dims,
                    num_heads=num_heads,
                    mixer="Global",
                    HW=None,
                    mlp_ratio=mlp_ratio,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.ReLU,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    norm_layer="nn.LayerNorm",
                    epsilon=1e-05,
                    prenorm=False,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dims, eps=1e-6)
        self.conv3 = ConvModule(
            hidden_dims,
            in_channels,
            kernel_size=1,
            bias=True,
            activation=nn.ReLU(),
        )
        self.conv4 = ConvModule(
            2 * in_channels,
            in_channels // 8,
            kernel_size=kernel_size,
            padding=kernel_size[0] // 2,
            bias=True,
            activation=nn.ReLU(),
        )

        self.conv1x1 = ConvModule(
            in_channels // 8,
            dims,
            kernel_size=1,
            bias=True,
            activation=nn.ReLU(),
        )
        self.out_channels = dims
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_guide:
            z = x.clone().detach()
        else:
            z = x
        h = z

        z = self.conv1(z)
        z = self.conv2(z)

        B, C, H, W = z.shape
        z = z.flatten(2).permute(0, 2, 1)
        for blk in self.svtr_block:
            z = blk(z)
        z = self.norm(z)

        z = z.reshape([B, H, W, C]).permute(0, 3, 1, 2)
        z = self.conv3(z)
        z = torch.cat((h, z), dim=1)
        z = self.conv1x1(self.conv4(z))
        return z
