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
    """GhostFaceNet embedding model for metric-learning tasks."""

    def __init__(
        self,
        embedding_size: int = 16,
        metadata_task_override: str = "color",
        alias: str | None = None,
    ):
        """Initialize the embeddings model.

        Args:
            embedding_size (int): Size of the produced embedding vector.
            metadata_task_override (str): Name of the dataset metadata field
                holding the identity to learn embeddings for. Defaults to
                ``"color"``, which suits the example re-ID dataset. Point it at
                whatever field your dataset actually provides.
            alias (str | None): Alias of the head node. Defaults to
                ``"<metadata_task_override>-embeddings"``.

        """
        self._embedding_size = embedding_size
        self._metadata_task_override = metadata_task_override
        self._alias = alias or f"{metadata_task_override}-embeddings"

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Params]]:
        return "default", {"default": {}}

    @property
    @override
    def nodes(self) -> list[NodeConfig]:
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
