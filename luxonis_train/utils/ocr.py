import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class OCRDecoder:
    """OCR decoder for converting model predictions to text."""

    def __init__(
        self,
        char_to_int: dict,
        ignored_tokens: list[int] | None = None,
        is_remove_duplicate: bool = True,
    ):
        """Initializes the OCR decoder.

        @type char_to_int: dict
        @param char_to_int: A dictionary mapping characters to integers.
        @type ignored_tokens: list[int]
        @param ignored_tokens: A list of tokens to ignore when decoding.
            Defaults to [0].
        @type is_remove_duplicate: bool
        @param is_remove_duplicate: Whether to remove duplicate
            characters. Defaults to True.
        """
        if ignored_tokens is None:
            self.ignored_tokens = [0]

        self.int_to_char = {v: k for k, v in char_to_int.items()}
        self.is_remove_duplicate = is_remove_duplicate

    def decode(self, preds: Tensor) -> list[tuple[str, float]]:
        """Decodes the model predictions to text.

        @type preds: Tensor
        @param preds: A tensor containing the model predictions.
        @rtype: list[tuple[str, float]]
        @return: A list of tuples containing the decoded text and
            confidence score.
        """
        preds = F.softmax(preds, dim=-1)
        pred_probs, pred_ids = torch.max(preds, dim=-1)

        result_list = []
        batch_size = len(pred_ids)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(pred_ids[batch_idx])):
                if pred_ids[batch_idx][idx] in self.ignored_tokens:
                    continue
                if self.is_remove_duplicate and (
                    idx > 0
                    and pred_ids[batch_idx][idx - 1]
                    == pred_ids[batch_idx][idx]
                ):
                    continue
                char_list.append(
                    self.int_to_char[int(pred_ids[batch_idx][idx])]
                )
                if pred_probs is not None:
                    conf_list.append(pred_probs[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            result_list.append(
                (text, torch.mean(torch.tensor(conf_list)).item())
            )
        return result_list

    def __call__(self, preds: Tensor) -> list[tuple[str, float]]:
        return self.decode(preds)


class OCREncoder:
    """OCR encoder for converting text to model targets."""

    def __init__(self, alphabet: list[str], ignore_unknown: bool = True):
        """Initializes the OCR encoder.

        @type alphabet: list[str]
        @param alphabet: A list of characters in the alphabet.
        @type ignore_unknown: bool
        @param ignore_unknown: Whether to ignore unknown characters.
            Defaults to True.
        """
        self._alphabet = ["", *np.unique(alphabet)]
        self.char_to_int = {char: i for i, char in enumerate(self._alphabet)}

        self.ignore_unknown = ignore_unknown
        if not self.ignore_unknown:
            self._alphabet.append("<UNK>")
            self.char_to_int["<UNK>"] = len(self.char_to_int)

    def encode(self, targets: Tensor) -> Tensor:
        """Encodes the text targets to model targets.

        @type targets: list[int]
        @param targets: A list of text targets.
        @rtype: Tensor
        @return: A tensor containing the encoded targets.
        """
        encoded_targets = []
        for target in targets:
            encoded_target = []
            for char_code in target:
                if char_code == 0:
                    encoded_target.append(0)
                    continue
                char = chr(int(char_code.item()))
                if char in self.char_to_int:
                    encoded_target.append(self.char_to_int[char])
                elif not self.ignore_unknown:
                    encoded_target.append(self.char_to_int["<UNK>"])

            if len(encoded_target) != len(target):
                encoded_target += [0] * (len(target) - len(encoded_target))

            encoded_targets.append(encoded_target)

        return torch.tensor(encoded_targets)

    def __call__(self, targets: Tensor) -> Tensor:
        return self.encode(targets)

    @property
    def alphabet(self) -> list[str]:
        return self._alphabet

    @property
    def n_classes(self) -> int:
        return len(self._alphabet)
