import math

from loguru import logger
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
        mid_channels: int | None = None,
        return_feats: bool = False,
        fc_decay: float | None = None,
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
        @type mid_channels: int
        @param mid_channels: Number of middle channels. Defaults to
            None.
        @type return_feats: bool
        @param return_feats: Whether to return features. Defaults to
            False.
        @type fc_decay: float | None
        @param fc_decay: Deprecated and ignored. It never had any effect
            on training. Use a C{finetuning} entry with a
            C{weight_decay} optimizer parameter to regularize the fully
            connected layers instead.
        """
        super().__init__(**kwargs)
        if len(set(alphabet)) != len(alphabet):  # pragma: no cover
            raise ValueError("Alphabet has duplicate characters.")

        if fc_decay is not None:
            logger.warning(
                "The `fc_decay` parameter of 'OCRCTCHead' is deprecated "
                "and has no effect. It never applied any regularization. "
                "Use a `finetuning` entry on the node to set "
                "`weight_decay` for the fully connected layers instead:\n"
                "  finetuning:\n"
                "    - parameters:\n"
                "        module_type: Linear\n"
                "      optimizer:\n"
                "        params:\n"
                f"          weight_decay: {fc_decay}"
            )

        self.return_feats = return_feats

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
        """Get the OCR encoder."""
        return self._encoder

    @property
    def decoder(self) -> OCRDecoder:
        """Get the OCR decoder."""
        return self._decoder

    @property
    @override
    def export_output_names(self) -> list[str]:
        return ["output_ocr_ctc"]

    @override
    def get_custom_head_config(self) -> Params:
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
