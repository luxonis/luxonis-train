from torch import Tensor

from luxonis_train.nodes.base_node import BaseNode

from .blocks import EncoderWithSVTR, Im2Seq


class SVTRNeck(BaseNode[list[Tensor], list[Tensor]]):
    in_channels: int

    def __init__(self, **kwargs):
        """Initializes the SVTR neck.

        @see: U{Adapted from <https://github.com/PaddlePaddle/PaddleOCR/
            blob/main/ppocr/modeling/necks/rnn.py>}
        @see: U{Original code
            <https://github.com/PaddlePaddle/PaddleOCR>}
        @license: U{Apache License, Version 2.0
            <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE
            >}
        @see: U{Adapted from <https://github.com/PaddlePaddle/PaddleOCR/
            blob/main/ppocr/modeling/necks/rnn.py>}
        @see: U{Original code
            <https://github.com/PaddlePaddle/PaddleOCR>}
        @license: U{Apache License, Version 2.0
            <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE
            >}
        """
        super().__init__(**kwargs)
        self.encoder_reshape = Im2Seq(self.in_channels)
        self.out_channels = self.encoder_reshape.out_channels
        self.encoder = EncoderWithSVTR(
            self.encoder_reshape.out_channels,
        )
        self.out_channels = self.encoder.out_channels

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        return x
