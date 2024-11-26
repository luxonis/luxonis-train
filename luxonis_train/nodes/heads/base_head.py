
from abc import abstractmethod
from typing import Generic, Literal

from luxonis_train.nodes.base_node import BaseNode, ForwardInputT, ForwardOutputT
from luxonis_train.utils import deep_merge_dicts


class BaseHead(BaseNode[ForwardInputT, ForwardOutputT], Generic[ForwardInputT, ForwardOutputT]):
    """Base class for all heads in the model.

    @type parser: str | None
    @ivar parser: Parser to use for the head.

    @type is_softmaxed: bool | None
    @ivar is_softmaxed: Whether the head uses softmax or not.
    """

    parser: Literal[
        "ClassificationParser",
        "YOLO",
        "YoloDetectionNetwork",
        "SegmentationParser"
    ] | None = None
    is_softmaxed: bool | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __init_subclass__(cls, **kwargs):
        """Check if head has all required attributes."""
        super().__init_subclass__()
        if cls.parser is None:
            raise ValueError(f"Head `{cls.__name__}` must define a parser.")

    def get_head_config(self) -> dict:
        """Get head configuration.

        @rtype: dict
        @return: Head configuration.
        """
        config = self._get_base_head_config()
        custom_config = self._get_custom_head_config()
        deep_merge_dicts(config, custom_config)
        return config
        
    def _get_base_head_config(self) -> dict:
        """Get base head configuration.

        @rtype: dict
        @return: Base head configuration.
        """
        return {
            "parser": self.parser,
            "metadata": {
                "is_softmax": self.is_softmaxed,
                "classes": self.class_names,
                "n_classes": self.n_classes,
            }
        }

    @abstractmethod
    def _get_custom_head_config(self) -> dict:
        """Get custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        ...