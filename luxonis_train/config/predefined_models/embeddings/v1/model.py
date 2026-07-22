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

    def __init__(self, embedding_size: int = 16):
        self._embedding_size = embedding_size

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
                alias="color-embeddings",
                metadata_task_override="color",
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
