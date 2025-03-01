import torch
from torch import Tensor, nn

from luxonis_train.nodes import BaseNode
from luxonis_train.nodes.blocks import ConvModule
from luxonis_train.nodes.necks.svtr_neck.blocks import SVTRBlock


class SVTRNeck(BaseNode[list[Tensor], list[Tensor]]):
    """SVTR neck.

    @see: U{Adapted from <https://github.com/PaddlePaddle/PaddleOCR/
        blob/main/ppocr/modeling/necks/rnn.py>}
    @see: U{Original code <https://github.com/PaddlePaddle/PaddleOCR>}
    @license: U{Apache License, Version 2.0
        <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE >}
    @see: U{Adapted from <https://github.com/PaddlePaddle/PaddleOCR/
        blob/main/ppocr/modeling/necks/rnn.py>}
    @see: U{Original code <https://github.com/PaddlePaddle/PaddleOCR>}
    @license: U{Apache License, Version 2.0
        <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE >}
    """

    in_channels: int

    # TODO: @sokovnin: Document
    def __init__(
        self,
        dims: int = 64,
        depth: int = 2,
        mid_channels: int = 120,
        use_guide: bool = False,
        n_heads: int = 8,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path: float = 0.0,
        kernel_size: tuple[int, int] = (3, 3),
        qk_scale: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.use_guide = use_guide
        self.conv1 = ConvModule(
            self.in_channels,
            self.in_channels // 8,
            kernel_size=kernel_size,
            padding=kernel_size[0] // 2,
            bias=True,
            activation=nn.ReLU(),
        )
        self.conv2 = ConvModule(
            self.in_channels // 8,
            mid_channels,
            kernel_size=1,
            bias=True,
            activation=nn.ReLU(),
        )

        self.svtr_block = nn.ModuleList(
            [
                SVTRBlock(
                    dim=mid_channels,
                    n_heads=n_heads,
                    mixer="global",
                    height=None,
                    width=None,
                    mlp_ratio=mlp_ratio,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.ReLU,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    norm_layer=nn.LayerNorm,
                    epsilon=1e-05,
                    prenorm=False,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(mid_channels, eps=1e-6)
        self.conv3 = ConvModule(
            mid_channels,
            self.in_channels,
            kernel_size=1,
            bias=True,
            activation=nn.ReLU(),
        )
        self.conv4 = ConvModule(
            2 * self.in_channels,
            self.in_channels // 8,
            kernel_size=kernel_size,
            padding=kernel_size[0] // 2,
            bias=True,
            activation=nn.ReLU(),
        )

        self.conv1x1 = ConvModule(
            self.in_channels // 8,
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
        return self.conv1x1(self.conv4(z))
