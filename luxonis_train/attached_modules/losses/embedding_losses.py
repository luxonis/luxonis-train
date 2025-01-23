import logging

import pytorch_metric_learning.losses as pml_losses
from pytorch_metric_learning.losses import CrossBatchMemory
from torch import Tensor

from luxonis_train.enums import Metadata
from luxonis_train.nodes.base_node import BaseNode
from luxonis_train.nodes.heads.ghostfacenet_head import GhostFaceNetHead

from .base_loss import BaseLoss

logger = logging.getLogger(__name__)

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

    class EmbeddingLossWrapper(
        BaseLoss[Tensor, Tensor], register_name=_loss_name
    ):
        node: GhostFaceNetHead
        supported_tasks = [Metadata("id")]

        def __init__(self, *, node: BaseNode | None = None, **kwargs):
            super().__init__(node=node)
            loss_name = _loss_name  # noqa: B023

            if not hasattr(pml_losses, loss_name):
                raise ValueError(
                    f"Loss {loss_name} not found in pytorch-metric-learning"
                )
            Loss = getattr(pml_losses, loss_name)
            self.loss_func = Loss(**kwargs)

            if self.node.cross_batch_memory_size is not None:
                if loss_name in CrossBatchMemory.supported_losses():
                    self.loss_func = CrossBatchMemory(
                        self.loss_func, embedding_size=self.node.embedding_size
                    )
                else:
                    logger.warning(
                        f"'CrossBatchMemory' is not supported for {loss_name}. "
                        "Ignoring cross_batch_memory_size."
                    )

        def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
            return self.loss_func(inputs, target)

        @property
        def name(self) -> str:
            return _loss_name  # noqa: B023
