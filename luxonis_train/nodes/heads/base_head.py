from luxonis_train.nodes.base_node import (
    BaseNode,
    ForwardInputT,
    ForwardOutputT,
)


class BaseHead(BaseNode[ForwardInputT, ForwardOutputT]):
    """Base class for all heads in the model.

    @type parser: str | None
    @ivar parser: Parser to use for the head.
    """

    parser: str | None = None

    def get_head_config(self) -> dict:
        """Get head configuration.

        @rtype: dict
        @return: Head configuration.
        """
        config = self._get_base_head_config()
        custom_config = self.get_custom_head_config()
        config["metadata"].update(custom_config)
        return config

    def _get_base_head_config(self) -> dict:
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

    def get_custom_head_config(self) -> dict:
        """Get custom head configuration.

        @rtype: dict
        @return: Custom head configuration.
        """
        raise NotImplementedError(
            "get_custom_head_config method must be implemented."
        )
