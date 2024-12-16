import logging

import pytorch_metric_learning.losses as pml_losses
from pytorch_metric_learning.losses import CrossBatchMemory
from torch import Tensor

from .base_loss import BaseLoss

logger = logging.getLogger(__name__)

ALL_EMBEDDING_LOSSES = [
    "AngularLoss",
    "ArcFaceLoss",
    "CircleLoss",
    "ContrastiveLoss",
    "CosFaceLoss",
    "DynamicSoftMarginLoss",
    "FastAPLoss",
    "HistogramLoss",
    "InstanceLoss",
    "IntraPairVarianceLoss",
    "LargeMarginSoftmaxLoss",
    "GeneralizedLiftedStructureLoss",
    "LiftedStructureLoss",
    "MarginLoss",
    "MultiSimilarityLoss",
    "NPairsLoss",
    "NCALoss",
    "NormalizedSoftmaxLoss",
    "NTXentLoss",
    "PNPLoss",
    "ProxyAnchorLoss",
    "ProxyNCALoss",
    "RankedListLoss",
    "SignalToNoiseRatioContrastiveLoss",
    "SoftTripleLoss",
    "SphereFaceLoss",
    "SubCenterArcFaceLoss",
    "SupConLoss",
    "ThresholdConsistentMarginLoss",
    "TripletMarginLoss",
    "TupletMarginLoss",
]

CLASS_EMBEDDING_LOSSES = [
    "ArcFaceLoss",
    "CosFaceLoss",
    "LargeMarginSoftmaxLoss",
    "NormalizedSoftmaxLoss",
    "ProxyAnchorLoss",
    "ProxyNCALoss",
    "SoftTripleLoss",
    "SphereFaceLoss",
    "SubCenterArcFaceLoss",
]


class EmbeddingLossWrapper(BaseLoss):
    def __init__(
        self,
        loss_name: str,
        embedding_size: int = 512,
        cross_batch_memory_size=0,
        num_classes: int = 0,
        loss_kwargs: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if loss_kwargs is None:
            loss_kwargs = {}

        try:
            loss_cls = getattr(pml_losses, loss_name)
        except AttributeError as e:
            raise ValueError(
                f"Loss {loss_name} not found in pytorch_metric_learning"
            ) from e

        if loss_name in CLASS_EMBEDDING_LOSSES:
            if num_classes < 0:
                raise ValueError(
                    f"Loss {loss_name} requires num_classes to be set to a positive value"
                )
            loss_kwargs["num_classes"] = num_classes
            loss_kwargs["embedding_size"] = embedding_size

            # If we wanted to support these losses, we would need to add a separate optimizer for them.
            # They may be useful in some scenarios, so leaving this here for future reference.
            raise ValueError(
                f"Loss {loss_name} requires its own optimizer, and that is not currently supported."
            )

        self.loss_func = loss_cls(**loss_kwargs)

        if cross_batch_memory_size > 0:
            if loss_name in CrossBatchMemory.supported_losses():
                self.loss_func = CrossBatchMemory(
                    self.loss_func, embedding_size=embedding_size
                )
            else:
                logger.warning(
                    f"Cross batch memory is not supported for {loss_name}. Ignoring cross_batch_memory_size."
                )

    def prepare(
        self, inputs: dict[str, list[Tensor]], labels: dict[str, list[Tensor]]
    ) -> tuple[Tensor, Tensor]:
        embeddings = self.get_input_tensors(inputs, "features")[0]

        if labels is None or "id" not in labels:
            raise ValueError("Labels must contain 'id' key")

        ids = labels["id"][0][:, 0]
        return embeddings, ids

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        loss = self.loss_func(inputs, target)
        return loss
