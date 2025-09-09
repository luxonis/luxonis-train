import math

from luxonis_ml.typing import Params
from torch import Tensor, nn
from torch.nn import functional as F
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils import OCRDecoder, OCREncoder


class OCRCTCHead(BaseHead):
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
        if len(set(alphabet)) != len(alphabet):  # pragma: no cover
            raise ValueError("Alphabet has duplicate characters.")

        self.return_feats = return_feats
        self.fc_decay = fc_decay

        self._encoder = OCREncoder(alphabet, ignore_unknown)
        self._decoder = OCRDecoder(self._encoder.char_to_int)

        if mid_channels is None:
            self.block = nn.Linear(self.in_channels, self.out_channels)
        else:
            self.block = nn.Sequential(
                nn.Linear(self.in_channels, mid_channels),
                nn.ReLU(),
                nn.Linear(mid_channels, self.out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        x = x.squeeze(2).permute(0, 2, 1)
        predictions = self.block(x)

        if self.export:
            predictions = F.softmax(predictions, dim=-1)

        return predictions

    @property
    def encoder(self) -> OCREncoder:
        """Returns the OCR encoder."""
        return self._encoder

    @property
    def decoder(self) -> OCRDecoder:
        """Returns the OCR decoder."""
        return self._decoder

    @property
    @override
    def export_output_names(self) -> list[str]:
        return ["output_ocr_ctc"]

    @override
    def get_custom_head_config(self) -> Params:
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
    def out_channels(self) -> int:
        return self.encoder.n_classes

    @override
    def initialize_weights(self, method: str | None = None) -> None:
        super().initialize_weights(method)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                std = 1.0 / math.sqrt(m.in_features)
                nn.init.uniform_(m.weight, -std, std)
                nn.init.uniform_(m.bias, -std, std)

                # TODO: This doesn't work in PyTorch. In PyTorch,
                # per-parameter regularizers are set by creating
                # multiple paremeter groups in the optimizer.
                # We need to first add support for this in
                # `LuxonisLightningModule`.
                m.weight.regularizer = self.fc_decay  # type: ignore
                m.bias.regularizer = self.fc_decay  # type: ignore
