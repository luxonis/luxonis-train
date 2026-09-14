"""The CTC encoder and decoder between text and the class indices that
the OCR head predicts.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


class OCRDecoder:
    """Greedy CTC decoder that turns class scores into text.

    The decoder takes the most probable class at each step of a
    sequence. It drops the ignored classes and, optionally, the steps
    that repeat the class of the previous step. A call of the decoder
    runs `decode`.

    """

    def __init__(
        self,
        char_to_int: dict,
        ignored_tokens: list[int] | None = None,
        is_remove_duplicate: bool = True,
    ):
        """Invert the character mapping and store the options.

        Args:
            char_to_int (dict): The class index of each character, as
                `OCREncoder` builds it.
            ignored_tokens (list[int] | None): The class indices to drop.
                ``None`` selects ``[0]``, the CTC blank. An empty list
                keeps every class.
            is_remove_duplicate (bool): Whether to drop a step whose class
                equals the class of the previous step.

        """
        self._ignored_tokens = (
            [0] if ignored_tokens is None else ignored_tokens
        )

        self._int_to_char = {v: k for k, v in char_to_int.items()}
        self._is_remove_duplicate = is_remove_duplicate

    def decode(self, preds: Tensor) -> list[tuple[str, float]]:
        """Decode the class scores of each sequence into text.

        The method applies a softmax over the classes and takes the most
        probable class at each step. It drops each step whose class is
        in ``ignored_tokens``. With ``is_remove_duplicate``, it also
        drops a step whose class equals the class of the previous step.
        The comparison uses the previous step also when the method
        dropped that step. Thus a blank between two equal characters
        keeps both characters.

        Args:
            preds (``Tensor``): The logits of shape ``[B, T, n_classes]``.

        Returns:
            list[tuple[str, float]]: One ``(text, confidence)`` pair for
            each sequence. The confidence is the mean probability of the
            kept steps, and ``nan`` for an empty text.

        Example:
            >>> import torch
            >>> from luxonis_train.utils import OCRDecoder
            >>> decoder = OCRDecoder({"": 0, "a": 1, "b": 2})
            >>> classes = torch.tensor([[1, 1, 0, 1, 2]])
            >>> logits = torch.nn.functional.one_hot(classes, 3) * 10.0
            >>> text, confidence = decoder.decode(logits)[0]
            >>> text, round(confidence, 3)
            ('aab', 1.0)

        """
        preds = F.softmax(preds, dim=-1)
        pred_probs, pred_ids = torch.max(preds, dim=-1)

        result_list = []
        batch_size = len(pred_ids)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(pred_ids[batch_idx])):
                if pred_ids[batch_idx][idx] in self._ignored_tokens:
                    continue
                if self._is_remove_duplicate and (
                    idx > 0
                    and pred_ids[batch_idx][idx - 1]
                    == pred_ids[batch_idx][idx]
                ):
                    continue
                char_list.append(
                    self._int_to_char[int(pred_ids[batch_idx][idx])]
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
        """Decode the class scores with `decode`.

        Args:
            preds (``Tensor``): The logits of shape ``[B, T, n_classes]``.

        Returns:
            list[tuple[str, float]]: The result of `decode`.

        """
        return self.decode(preds)


class OCREncoder:
    """CTC encoder that turns text labels into class indices.

    Class ``0`` is the CTC blank ``""``. The sorted unique characters of
    the alphabet follow. With ``ignore_unknown=False``, ``"<UNK>"`` is
    the last class. A call of the encoder runs `encode`.

    Attributes:
        char_to_int (dict): The class index of each character.

    Example:
        >>> import torch
        >>> from luxonis_train.utils import OCREncoder
        >>> encoder = OCREncoder(["b", "a"])
        >>> [str(char) for char in encoder.alphabet], encoder.n_classes
        (['', 'a', 'b'], 3)
        >>> codes = torch.tensor([[ord("b"), ord("x"), ord("a"), 0]])
        >>> encoder.encode(codes).tolist()
        [[2, 1, 0, 0]]
        >>> strict_encoder = OCREncoder(["b", "a"], ignore_unknown=False)
        >>> strict_encoder.encode(codes).tolist()
        [[2, 3, 1, 0]]

    """

    def __init__(self, alphabet: list[str], ignore_unknown: bool = True):
        """Build the alphabet and the class index of each character.

        Args:
            alphabet (list[str]): The characters of the labels. The
                encoder sorts them and drops the duplicates.
            ignore_unknown (bool): Whether `encode` drops a character that
                is not in ``alphabet``. With ``False``, the encoder adds
                the class ``"<UNK>"`` and maps such a character to it.

        """
        self._alphabet = ["", *np.unique(alphabet)]
        self.char_to_int = {char: i for i, char in enumerate(self._alphabet)}

        self._ignore_unknown = ignore_unknown
        if not self._ignore_unknown:
            self._alphabet.append("<UNK>")
            self.char_to_int["<UNK>"] = len(self.char_to_int)

    def encode(self, targets: Tensor) -> Tensor:
        """Convert the character codes of the labels into class indices.

        The value ``0`` is padding and gives the blank class ``0``. With
        ``ignore_unknown``, the method drops a character that is not in
        the alphabet. The later characters move to the left, and ``0``
        fills the end of the row. Otherwise the character gets the
        ``"<UNK>"`` class.

        Args:
            targets (``Tensor``): The Unicode code points of the labels,
                of shape ``[N, L]``.

        Returns:
            ``Tensor``: The class indices of shape ``[N, L]``, as
            ``int64``.

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
                elif not self._ignore_unknown:
                    encoded_target.append(self.char_to_int["<UNK>"])

            if len(encoded_target) != len(target):
                encoded_target += [0] * (len(target) - len(encoded_target))

            encoded_targets.append(encoded_target)

        return torch.tensor(encoded_targets)

    def __call__(self, targets: Tensor) -> Tensor:
        """Convert the character codes with `encode`.

        Args:
            targets (``Tensor``): The Unicode code points of the labels,
                of shape ``[N, L]``.

        Returns:
            ``Tensor``: The result of `encode`.

        """
        return self.encode(targets)

    @property
    def alphabet(self) -> list[str]:
        """The character of each class, in the order of the indices.

        The blank ``""`` comes first. The sorted unique characters
        follow, then ``"<UNK>"`` when ``ignore_unknown`` is ``False``.
        The sorted characters are ``np.str_`` values.

        """
        return self._alphabet

    @property
    def n_classes(self) -> int:
        """The number of classes, the length of `alphabet`."""
        return len(self._alphabet)
