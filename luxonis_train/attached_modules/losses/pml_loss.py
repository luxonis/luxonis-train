import logging

import pytorch_metric_learning.losses as pml_losses
from pytorch_metric_learning.losses import CrossBatchMemory
from torch import Tensor

from luxonis_train.enums import Metadata
from luxonis_train.nodes.backbones.ghostfacenet import GhostFaceNetV2
from luxonis_train.nodes.base_node import BaseNode

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

for loss_name in EMBEDDING_LOSSES:

    class EmbeddingLossWrapper(
        BaseLoss[Tensor, Tensor], register_name=loss_name
    ):
        node: GhostFaceNetV2
        supported_tasks = [Metadata("id")]

        def __init__(self, *, node: BaseNode | None = None, **kwargs):
            super().__init__(node=node)

            Loss = getattr(pml_losses, loss_name)  # noqa: B023
            self.loss_func = Loss(**kwargs)

            if self.node.embedding_size is not None:
                if loss_name in CrossBatchMemory.supported_losses():  # noqa: B023
                    self.loss_func = CrossBatchMemory(
                        self.loss_func, embedding_size=self.node.embedding_size
                    )
                else:
                    logger.warning(
                        f"CrossBatchMemory is not supported for {loss_name}. "  # noqa: B023
                        "Ignoring cross_batch_memory_size."
                    )

        def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
            return self.loss_func(inputs, target)
