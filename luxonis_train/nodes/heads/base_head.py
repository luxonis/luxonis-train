"""The base class every head inherits.

A head is a node with a task and an export parser. The task decides
which losses, metrics, and visualizers can attach to the head. The
parser name goes into the NN Archive, together with the class names of
the head.

"""

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

    A subclass sets the ``task`` class attribute. A subclass with an
    export parser also sets the ``parser`` class attribute. A subclass
    can override `get_custom_head_config` to give its parser more
    values, and `annotate` to support a task that the default annotation
    does not know.

    Attributes:
        parser (str): The name of the parser that reads the outputs of
            the head in the exported model. `get_head_config` puts it
            into the NN Archive entry of the head. It is ``""`` in the
            base class.
        task (Task): The task of the head. It gives the key of the main
            output and the labels that the head needs.

    """

    parser: str = ""
    task: Task

    def get_head_config(self) -> dict[str, Any]:
        """Return the entry of the head in the NN Archive config.

        The method starts from the ``parser`` class attribute, the
        `BaseNode.class_names`, and the `BaseNode.n_classes` of the
        head. Then it merges the result of `get_custom_head_config` into
        the ``"metadata"`` dictionary. A custom key replaces a base key
        of the same name. `LuxonisModel.archive` calls the method for
        each head whose ``remove_on_export`` is ``False``. Then it adds
        the ``"name"`` and the ``"outputs"`` keys.

        The class names always come from ``dataset_metadata``. Without
        it, `BaseNode.class_names` raises ``RuntimeError``. When the
        dataset has no task named ``task_name``, it raises
        ``ValueError``.

        Returns:
            ``dict[str, Any]``: A dictionary with the keys ``"parser"``
            and ``"metadata"``. The ``"metadata"`` dictionary holds
            ``"classes"``, ``"n_classes"``, and the custom keys.

        Example:
            >>> from torch import Size
            >>> from luxonis_train.nodes import ClassificationHead
            >>> from luxonis_train.utils import DatasetMetadata
            >>> head = ClassificationHead(
            ...     task_name="animals",
            ...     dataset_metadata=DatasetMetadata(
            ...         classes={"animals": {"cat": 0, "dog": 1}}
            ...     ),
            ...     input_shapes=[{"features": [Size([1, 8, 4, 4])]}],
            ... )
            >>> head.get_head_config()
            {'parser': 'ClassificationParser',
             'metadata': {'classes': ['cat', 'dog'], 'n_classes': 2,
                          'is_softmax': False}}

        """
        config = self._get_base_head_config()
        config["metadata"] |= self.get_custom_head_config()
        return config

    def _get_base_head_config(self) -> dict[str, Any]:
        """Return the part of the head config that every head shares.

        Returns:
            ``dict[str, Any]``: A dictionary with two keys. ``"parser"``
            holds the ``parser`` class attribute. ``"metadata"`` holds a
            dictionary with the ``"classes"`` and the ``"n_classes"`` of
            the head.

        """
        return {
            "parser": self.parser,
            "metadata": {
                "classes": self.class_names,
                "n_classes": self.n_classes,
            },
        }

    def get_custom_head_config(self) -> Params:
        """Return the head-specific metadata for the NN Archive.

        A subclass overrides the method to give its parser more values.
        `get_head_config` merges the result into the ``"metadata"``
        dictionary. The base implementation returns an empty dictionary.

        Returns:
            ``Params``: The additional metadata keys and their values.

        """
        return {}

    def annotate(
        self,
        head_output: Packet[Tensor],
        image_paths: list[Path],
        config_preprocessing: PreprocessingConfig,
    ) -> DatasetIterator:
        """Convert the outputs of the head into dataset records.

        This delegates to `luxonis_train.utils.annotation.default_annotate`.
        Override it for tasks that the default converter does not support.

        Args:
            head_output (``Packet[Tensor]``): The output packet of the
                head for one batch.
            image_paths (``list[Path]``): The paths of the original
                images, in the order of the batch.
            config_preprocessing (PreprocessingConfig): The preprocessing
                settings used to map predictions back to the images.

        Returns:
            ``DatasetIterator``: A generator of the annotation records.

        """
        return default_annotate(
            self, head_output, image_paths, config_preprocessing
        )
