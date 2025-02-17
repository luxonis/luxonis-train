import pytorch_metric_learning.distances as pml_distances
import pytorch_metric_learning.losses as pml_losses
import pytorch_metric_learning.miners as pml_miners
import pytorch_metric_learning.reducers as pml_reducers
import pytorch_metric_learning.regularizers as pml_regularizers
from loguru import logger
from pytorch_metric_learning.losses import CrossBatchMemory
from torch import Tensor

from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.heads.ghostfacenet_head import GhostFaceNetHead
from luxonis_train.tasks import Tasks
from luxonis_train.utils.types import Kwargs

from .base_loss import BaseLoss

EMBEDDING_LOSSES = [
    "AngularLoss",
    "CircleLoss",
    "ContrastiveLoss",
    "DynamicSoftMarginLoss",
    "FastAPLoss",
    "HistogramLoss",
    "InstanceLoss",
    "IntraPairVarianceLoss",
    "GeneralizedLiftedStructureLoss",
    "LiftedStructureLoss",
    "MarginLoss",
    "MultiSimilarityLoss",
    "NPairsLoss",
    "NCALoss",
    "NTXentLoss",
    "PNPLoss",
    "RankedListLoss",
    "SignalToNoiseRatioContrastiveLoss",
    "SupConLoss",
    "ThresholdConsistentMarginLoss",
    "TripletMarginLoss",
    "TupletMarginLoss",
]

for _loss_name in EMBEDDING_LOSSES:

    class EmbeddingLossWrapper(BaseLoss, register_name=_loss_name):
        node: GhostFaceNetHead
        supported_tasks = [Tasks.EMBEDDINGS]
        miner: pml_miners.BaseMiner | None

        def __init__(
            self,
            *,
            miner: str | None = None,
            miner_params: Kwargs | None = None,
            distance: str | None = None,
            distance_params: Kwargs | None = None,
            reducer: str | None = None,
            reducer_params: Kwargs | None = None,
            regularizer: str | None = None,
            regularizer_params: Kwargs | None = None,
            node: BaseNode | None = None,
            **kwargs,
        ):
            super().__init__(node=node)
            loss_name = _loss_name  # noqa: B023

            if not hasattr(pml_losses, loss_name):
                raise ValueError(
                    f"Loss {loss_name} not found in pytorch-metric-learning"
                )
            Loss = getattr(pml_losses, loss_name)

            if reducer is not None:
                if not hasattr(pml_reducers, reducer):
                    raise ValueError(
                        f"Reducer {reducer} not found in pytorch-metric-learning"
                    )
                Reducer = getattr(pml_reducers, reducer)
                kwargs["reducer"] = Reducer(**(reducer_params or {}))
            if regularizer is not None:
                if not hasattr(pml_regularizers, regularizer):
                    raise ValueError(
                        f"Regularizer {regularizer} not found in pytorch-metric-learning"
                    )
                Regularizer = getattr(pml_regularizers, regularizer)
                kwargs["embedding_regularizer"] = Regularizer(
                    **(regularizer_params or {})
                )
            if distance is not None:
                if not hasattr(pml_distances, distance):
                    raise ValueError(
                        f"Distance {distance} not found in pytorch-metric-learning"
                    )
                Distance = getattr(pml_distances, distance)
                kwargs["distance"] = Distance(**(distance_params or {}))

            if miner is not None:
                if not hasattr(pml_miners, miner):
                    raise ValueError(
                        f"Miner {miner} not found in pytorch-metric-learning"
                    )
                Miner = getattr(pml_miners, miner)
                self.miner = Miner(**(miner_params or {}))
            else:
                self.miner = None

            self.loss = Loss(**kwargs)

            if self.node.cross_batch_memory_size is not None:
                if loss_name in CrossBatchMemory.supported_losses():
                    self.loss = CrossBatchMemory(
                        self.loss,
                        embedding_size=self.node.embedding_size,
                        miner=self.miner,
                    )
                else:
                    logger.warning(
                        f"'CrossBatchMemory' is not supported for {loss_name}. "
                        "Ignoring cross_batch_memory_size."
                    )

        def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
            if self.miner is not None:
                hard_pairs = self.miner(predictions, target)
                return self.loss(predictions, target, hard_pairs)
            return self.loss(predictions, target)

        @property
        def name(self) -> str:
            return _loss_name  # noqa: B023
