import pytest
import torch

from luxonis_train.utils import OCRDecoder


@pytest.mark.parametrize(
    ("ignored_tokens", "expected_text"),
    [([], "xa"), ([1], "x")],
)
def test_decoder_respects_ignored_tokens(
    ignored_tokens: list[int], expected_text: str
):
    decoder = OCRDecoder(
        {"x": 0, "a": 1},
        ignored_tokens=ignored_tokens,
        is_remove_duplicate=False,
    )
    logits = torch.tensor([[[10.0, 0.0], [0.0, 10.0]]])

    text, _ = decoder.decode(logits)[0]

    assert text == expected_text
