from typing import Any

import pytest
import torch
from torch import Tensor

from luxonis_train.utils import ocr as ocr_module
from luxonis_train.utils.ocr import OCRDecoder, OCREncoder


def test_ocr_decoder_removes_ignored_tokens_and_duplicates():
    decoder = OCRDecoder({"": 0, "a": 1, "b": 2})
    logits = torch.full((1, 5, 3), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 1] = 10.0
    logits[0, 2, 0] = 10.0
    logits[0, 3, 2] = 10.0
    logits[0, 4, 2] = 10.0

    result = decoder(logits)

    assert result[0][0] == "ab"
    assert result[0][1] == pytest.approx(1.0)


def test_ocr_decoder_uses_default_confidence_when_probs_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_tensor = ocr_module.torch.tensor

    def tensor_float(data: Any) -> Tensor:
        return original_tensor(data, dtype=torch.float32)

    monkeypatch.setattr(
        ocr_module.torch,
        "max",
        lambda _preds, dim: (None, torch.tensor([[1]])),
    )
    monkeypatch.setattr(
        ocr_module.torch,
        "tensor",
        tensor_float,
    )
    decoder = OCRDecoder({"": 0, "a": 1})

    result = decoder.decode(torch.zeros((1, 1, 2)))

    assert result == [("a", 1.0)]


def test_ocr_encoder_ignores_unknown_and_preserves_padding():
    encoder = OCREncoder(["b", "a"])
    targets = torch.tensor([[ord("a"), ord("x"), 0]])

    encoded = encoder(targets)

    assert encoder.alphabet == ["", "a", "b"]
    assert encoder.n_classes == 3
    assert encoded.tolist() == [[1, 0, 0]]


def test_ocr_encoder_can_encode_unknown_token():
    encoder = OCREncoder(["a"], ignore_unknown=False)
    targets = torch.tensor([[ord("a"), ord("x")]])

    encoded = encoder.encode(targets)

    assert encoder.alphabet == ["", "a", "<UNK>"]
    assert encoded.tolist() == [[1, 2]]
