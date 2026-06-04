from pathlib import Path
from typing import Any

from luxonis_ml.data import DatasetIterator
from luxonis_ml.typing import Params
from torch import Tensor

from luxonis_train.config.config import PreprocessingConfig
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.tasks import Task
from luxonis_train.typing import Packet
from luxonis_train.utils.annotation import default_annotate


class BaseHead(BaseNode):
    """Base class for all heads in the model.

    Attributes:
        parser (str): Parser to use for the head.

    """

    parser: str = ""
    task: Task

    def get_head_config(self) -> dict[str, Any]:
        """Get head configuration.

        Returns:
            dict: Head configuration.

        """
        config = self._get_base_head_config()
        config["metadata"] |= self.get_custom_head_config()
        return config

    def _get_base_head_config(self) -> dict[str, Any]:
        """Get base head configuration.

        Returns:
            dict: Base head configuration.

        """
        return {
            "parser": self.parser,
            "metadata": {
                "classes": self.class_names,
                "n_classes": self.n_classes,
            },
        }

    def get_custom_head_config(self) -> Params:
        """Get a custom head configuration.

        Returns:
            dict: Custom head configuration.

        """
        return {}

    def annotate(
        self,
        head_output: Packet[Tensor],
        image_paths: list[Path],
        config_preprocessing: PreprocessingConfig,
    ) -> DatasetIterator:
        """Convert head output to a DatasetIterator for annotation.

        Data should be in standard `luxonis-ml record format <https://github.com/luxonis/luxonis-
        ml/blob/main/luxonis_ml/data/README.md>`_.

        Args:
            head_output (Packet[Tensor]): Raw outputs from this head.
            image_paths (list[Path]): List of original image file paths to annotate.
            config_preprocessing (PreprocessingConfig): Config containing train_image_size, keep_aspect_ratio, etc.

        Returns:
            DatasetIterator: Iterator yielding annotation records in luxonis-ml format.

        """
        return default_annotate(
            self, head_output, image_paths, config_preprocessing
        )
