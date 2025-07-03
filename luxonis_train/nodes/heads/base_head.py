from pathlib import Path
from typing import Any

from luxonis_ml.data import DatasetIterator
from torch import Tensor

from luxonis_train.config.config import PreprocessingConfig
from luxonis_train.nodes.base_node import (
    BaseNode,
    ForwardInputT,
    ForwardOutputT,
)
from luxonis_train.typing import Packet
from luxonis_train.utils.annotation import default_annotate


# TODO: We shouldn't skip the head completely
# if custom config is not defined.
class BaseHead(BaseNode[ForwardInputT, ForwardOutputT]):
    """Base class for all heads in the model.

    @type parser: str
    @ivar parser: Parser to use for the head.
    """

    parser: str = ""

    def get_head_config(self) -> dict[str, Any]:
        """Get head configuration.

        @rtype: dict
        @return: Head configuration.
        """
        config = self._get_base_head_config()
        config["metadata"] |= self.get_custom_head_config()
        return config

    def _get_base_head_config(self) -> dict[str, Any]:
        """Get base head configuration.

        @rtype: dict
        @return: Base head configuration.
        """
        return {
            "parser": self.parser,
            "metadata": {
                "classes": self.class_names,
                "n_classes": self.n_classes,
            },
        }

    def get_custom_head_config(self) -> dict[str, Any]:
        """Get custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        raise NotImplementedError(
            "get_custom_head_config method must be implemented."
        )

    def annotate(
        self,
        head_output: dict[str, Packet[Tensor]],
        image_paths: list[Path],
        config_preprocessing: PreprocessingConfig,
    ) -> DatasetIterator:
        """Convert head output to a DatasetIterator for dataset annotation. Data should be in standard
        U{luxonis-ml record format  <https://github.com/luxonis/luxonis-ml/blob/main/luxonis_ml/data/README.md>}.

        @type head_output: dict[str, Packet[Tensor]]
        @param head_output: Raw outputs from this head.
        @type image_paths: list[Path]
        @param image_paths: List of original image file paths to annotate.
        @type config_preprocessing: PreprocessingConfig
        @param config_preprocessing: Config containing train_image_size, keep_aspect_ratio, etc.
        @rtype: DatasetIterator
        @return: Iterator yielding annotation records in luxonis-ml format.
        """
        return default_annotate(
            self, head_output, image_paths, config_preprocessing
        )
