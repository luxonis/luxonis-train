from typing import Literal, TypeAlias

from loguru import logger
from luxonis_ml.typing import Params
from typing_extensions import override

from .base_predefined_model import SimplePredefinedModel

AlphabetName: TypeAlias = Literal[
    "english",
    "english_lowercase",
    "numeric",
    "alphanumeric",
    "alphanumeric_lowercase",
    "punctuation",
    "ascii",
]


class OCRRecognitionModel(SimplePredefinedModel):
    def __init__(
        self,
        alphabet: list[str] | AlphabetName = "english",
        max_text_len: int = 40,
        ignore_unknown: bool = True,
        **kwargs,
    ):
        super().__init__(
            **{
                "backbone": "PPLCNetV3",
                "neck": "SVTRNeck",
                "head": "OCRCTCHead",
                "loss": "CTCLoss",
                "metrics": "OCRAccuracy",
                "confusion_matrix_available": False,
                "visualizer": "OCRVisualizer",
            }
            | kwargs
        )
        if "max_text_len" not in self._backbone_params:
            self._backbone_params["max_text_len"] = max_text_len
        if "alphabet" not in self._head_params:
            self._head_params["alphabet"] = self._generate_alphabet(alphabet)
        if "ignore_unknown" not in self._head_params:
            self._head_params["ignore_unknown"] = ignore_unknown

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Params]]:
        return "light", {
            "light": {
                "backbone_variant": "rec-light",
            }
        }

    @staticmethod
    def _generate_alphabet(alphabet: list[str] | AlphabetName) -> list[str]:
        alphabets = {
            "english": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "english_lowercase": "abcdefghijklmnopqrstuvwxyz",
            "numeric": "0123456789",
            "alphanumeric": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            "alphanumeric_lowercase": "abcdefghijklmnopqrstuvwxyz0123456789",
            "punctuation": " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
            "ascii": "".join(chr(i) for i in range(32, 127)),
        }

        if isinstance(alphabet, str):
            if alphabet not in alphabets:
                raise ValueError(
                    f"Invalid alphabet name '{alphabet}'. "
                    f"Available options are: {list(alphabets.keys())}. "
                    f"Alternatively, you can provide a custom alphabet as a list of characters."
                )
            logger.info(f"Using predefined alphabet '{alphabet}'.")
            alphabet = list(alphabets[alphabet])

        return alphabet
