"""The embedding learning model."""

from luxonis_ml.typing import Params
from typing_extensions import override

from luxonis_train.config import (
    AttachedModuleConfig,
    LossModuleConfig,
    MetricModuleConfig,
    NodeConfig,
)
from luxonis_train.config.predefined_models.base_predefined_model import (
    BasePredefinedModel,
)


class EmbeddingsModel(BasePredefinedModel):
    """Embedding learning for face recognition or re-identification.

    The model maps an image to a vector. The loss pulls the vectors of
    one identity together and pushes the vectors of different identities
    apart. Compare two images by the distance between their vectors, so
    a new identity needs no retraining.

    Example:
        The ``model`` section of a config:

        .. code-block:: yaml

            model:
              predefined_model:
                name: EmbeddingsModel
                params:
                  variant: default

    Components:
        - Nodes: `GhostFaceNet` -> `GhostFaceNetHead`
        - Losses: ``SupConLoss``
        - Metrics:

          - `ClosestIsPositiveAccuracy`
          - `MedianDistances`

        - Visualizers: `EmbeddingsVisualizer`
        - Variants: ``default``

    """

    def __init__(
        self,
        embedding_size: int = 16,
        metadata_task_override: str = "color",
        alias: str | None = None,
    ):
        """Initialize the model.

        Args:
            embedding_size (int): The length of the embedding vector the
                head produces. It reaches the head as ``embedding_size``.
            metadata_task_override (str): The metadata field of the
                dataset that holds the identity of each sample. It
                renames the ``id`` metadata label that the embeddings
                task of the head requires. The example config
                ``embeddings_model.yaml`` uses ``"color"``.
            alias (str | None): The alias of the head node. ``None``
                gives ``"<metadata_task_override>-embeddings"``.

        """
        self._embedding_size = embedding_size
        self._metadata_task_override = metadata_task_override
        self._alias = alias or f"{metadata_task_override}-embeddings"

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Params]]:
        """Get the default variant name and the available variants.

        The model has one variant, ``default``, with no parameters.

        Returns:
            ``tuple[str, dict[str, Params]]``: ``"default"`` and the
            single empty variant.

        Example:
            >>> EmbeddingsModel.get_variants()
            ('default', {'default': {}})

        """
        return "default", {"default": {}}

    @property
    @override
    def nodes(self) -> list[NodeConfig]:
        """The `GhostFaceNet` backbone and the `GhostFaceNetHead`.

        The backbone has no inputs, so it reads from the loader. The
        head reads from the backbone. Both nodes keep the
        ``"default"`` variant. The head carries ``alias`` and
        ``metadata_task_override`` as node fields, and
        ``embedding_size`` in its ``params``. Its attached modules are:

        - the ``SupConLoss`` loss with a ``MultiSimilarityMiner``, a
          ``CosineSimilarity`` distance, a ``ThresholdReducer`` with
          ``high`` of ``0.3``, and an ``LpRegularizer``;
        - the `ClosestIsPositiveAccuracy` and `MedianDistances`
          metrics, neither marked as the main metric;
        - the `EmbeddingsVisualizer` visualizer.

        When no metric of the config is the main metric,
        `ModelConfig.check_main_metric` marks the first metric of the
        config. When the nodes that the config lists have no metrics,
        that is `ClosestIsPositiveAccuracy`.

        Example:
            >>> model = EmbeddingsModel(embedding_size=32)
            >>> head = model.nodes[-1]
            >>> head.alias, head.metadata_task_override, head.params
            ('color-embeddings', 'color', {'embedding_size': 32})

        """
        return [
            NodeConfig(name="GhostFaceNet"),
            NodeConfig(
                name="GhostFaceNetHead",
                inputs=["GhostFaceNet"],
                alias=self._alias,
                metadata_task_override=self._metadata_task_override,
                params={"embedding_size": self._embedding_size},
                losses=[
                    LossModuleConfig(
                        name="SupConLoss",
                        params={
                            "miner": "MultiSimilarityMiner",
                            "distance": "CosineSimilarity",
                            "reducer": "ThresholdReducer",
                            "reducer_params": {"high": 0.3},
                            "regularizer": "LpRegularizer",
                        },
                    )
                ],
                metrics=[
                    MetricModuleConfig(name="ClosestIsPositiveAccuracy"),
                    MetricModuleConfig(name="MedianDistances"),
                ],
                visualizers=[
                    AttachedModuleConfig(name="EmbeddingsVisualizer")
                ],
            ),
        ]
