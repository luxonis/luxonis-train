from typing import Any

from luxonis_train.nodes.base_node import (
    BaseNode,
    ForwardInputT,
    ForwardOutputT,
)


# TODO: We shouldn't skip the head completely
# if custom config is not defined.
class BaseHead(BaseNode[ForwardInputT, ForwardOutputT]):
    """Base class for all heads in the model.

    @type parser: str | None
    @ivar parser: Parser to use for the head.
    """

    parser: str | None = None

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
