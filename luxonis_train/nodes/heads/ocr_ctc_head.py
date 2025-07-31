import math
from collections.abc import Callable

from torch import Tensor, nn
from torch.nn import functional as F

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import OCRDecoder, OCREncoder


class OCRCTCHead(BaseHead[Tensor, Tensor]):
    in_channels: int
    task = Tasks.OCR

    parser: str = "ClassificationSequenceParser"

    def __init__(
        self,
        alphabet: list[str],
        ignore_unknown: bool = True,
        fc_decay: float = 0.0004,
        mid_channels: int | None = None,
        return_feats: bool = False,
        **kwargs,
    ):
        """OCR CTC head.

        @see: U{Adapted from <https://github.com/PaddlePaddle/PaddleOCR/
            blob/main/ppocr/modeling/heads/rec_ctc_head.py>}
        @see: U{Original code
            <https://github.com/PaddlePaddle/PaddleOCR>}
        @license: U{Apache License, Version 2.0
            <https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE
            >}
        @type alphabet: list[str]
        @param alphabet: List of characters.
        @type ignore_unknown: bool
        @param ignore_unknown: Whether to ignore unknown characters.
            Defaults to True.
        @type fc_decay: float
        @param fc_decay: L2 regularization factor. Defaults to 0.0004.
        @type mid_channels: int
        @param mid_channels: Number of middle channels. Defaults to
            None.
        @type return_feats: bool
        @param return_feats: Whether to return features. Defaults to
            False.
        """
        super().__init__(**kwargs)
        self.return_feats = return_feats

        self._encoder = OCREncoder(alphabet, ignore_unknown)
        self._decoder = OCRDecoder(self.encoder.char_to_int)

        self.out_channels = self._encoder.n_classes
        if mid_channels is None:
            weight_attr, bias_attr = get_para_bias_attr(
                fc_decay, self.in_channels
            )
            self.fc = nn.Linear(self.in_channels, self.out_channels)
            weight_attr(self.fc.weight)
            bias_attr(self.fc.bias)
        else:
            weight_attr1, bias_attr1 = get_para_bias_attr(
                fc_decay, self.in_channels
            )
            self.fc1 = nn.Linear(self.in_channels, mid_channels)
            weight_attr1(self.fc1.weight)
            bias_attr1(self.fc1.bias)

            weight_attr2, bias_attr2 = get_para_bias_attr(
                fc_decay, mid_channels
            )
            self.fc2 = nn.Linear(mid_channels, self.out_channels)
            weight_attr2(self.fc2.weight)
            bias_attr2(self.fc2.bias)

        self.mid_channels = mid_channels

        self._export_output_names = ["output_ocr_ctc"]

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        x = x.squeeze(2).permute(0, 2, 1)
        if self.mid_channels is None:
            predictions = self.fc(x)
        else:
            x = self.fc1(x)
            x = F.relu(x)
            predictions = self.fc2(x)

        if self.return_feats:
            result = (x, predictions)
        else:
            result = predictions

        if self.export:
            return F.softmax(predictions, dim=-1)

        return result

    def get_custom_head_config(self) -> dict:
        """Returns custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        return {
            "classes": self.encoder.alphabet,
            "n_classes": self.encoder.n_classes,
            "is_softmax": True,
            "concatenate_classes": True,
            "ignored_indexes": [0],
            "remove_duplicates": True,
        }

    @property
    def encoder(self) -> OCREncoder:
        return self._encoder

    @property
    def decoder(self) -> OCRDecoder:
        return self._decoder


def get_para_bias_attr(l2_decay: float, k: int) -> tuple[Callable, Callable]:
    stdv = 1.0 / math.sqrt(k * 1.0)

    def weight_attr(param: nn.Parameter) -> None:
        nn.init.uniform_(param, -stdv, stdv)
        param.regularizer = l2_decay  # type: ignore

    def bias_attr(param: nn.Parameter) -> None:
        nn.init.uniform_(param, -stdv, stdv)
        param.regularizer = l2_decay  # type: ignore

    return weight_attr, bias_attr
