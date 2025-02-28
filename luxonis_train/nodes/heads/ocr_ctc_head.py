import math

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
        self.fc_decay = fc_decay

        self._encoder = OCREncoder(alphabet, ignore_unknown)
        self._decoder = OCRDecoder(self.encoder.char_to_int)

        if mid_channels is None:
            self.block = self._construct_fc(
                self.in_channels, self.out_channels
            )
        else:
            self.block = nn.Sequential(
                self._construct_fc(self.in_channels, mid_channels),
                nn.ReLU(),
                self._construct_fc(mid_channels, self.out_channels),
            )

        self._export_output_names = ["output_ocr_ctc"]

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        x = x.squeeze(2).permute(0, 2, 1)
        predictions = self.block(x)

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

    @property
    def out_channels(self) -> int:
        return self._encoder.n_classes

    def _construct_fc(self, in_channels: int, out_channels: int) -> nn.Linear:
        fc = nn.Linear(in_channels, out_channels)

        # TODO: Why * 1.0?
        std = 1.0 / math.sqrt(in_channels * 1.0)
        nn.init.uniform_(fc.weight, -std, std)
        nn.init.uniform_(fc.bias, -std, std)

        # TODO: Doesn't seem to do anything, type checker
        # also hints that `regularizer` attribute doesn't exist
        fc.weight.regularizer = self.fc_decay  # type: ignore
        fc.bias.regularizer = self.fc_decay  # type: ignore

        return fc
