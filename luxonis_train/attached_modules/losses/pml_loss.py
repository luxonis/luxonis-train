import warnings

from pytorch_metric_learning.losses import (
    AngularLoss,
    ArcFaceLoss,
    CircleLoss,
    ContrastiveLoss,
    CosFaceLoss,
    CrossBatchMemory,
    DynamicSoftMarginLoss,
    FastAPLoss,
    GeneralizedLiftedStructureLoss,
    HistogramLoss,
    InstanceLoss,
    IntraPairVarianceLoss,
    LargeMarginSoftmaxLoss,
    LiftedStructureLoss,
    ManifoldLoss,
    MarginLoss,
    MultiSimilarityLoss,
    NCALoss,
    NormalizedSoftmaxLoss,
    NPairsLoss,
    NTXentLoss,
    P2SGradLoss,
    PNPLoss,
    ProxyAnchorLoss,
    ProxyNCALoss,
    RankedListLoss,
    SignalToNoiseRatioContrastiveLoss,
    SoftTripleLoss,
    SphereFaceLoss,
    SubCenterArcFaceLoss,
    SupConLoss,
    TripletMarginLoss,
    TupletMarginLoss,
)
from torch import Tensor

from .base_loss import BaseLoss

# Dictionary mapping string keys to loss classes
loss_dict = {
    "AngularLoss": AngularLoss,
    "ArcFaceLoss": ArcFaceLoss,
    "CircleLoss": CircleLoss,
    "ContrastiveLoss": ContrastiveLoss,
    "CosFaceLoss": CosFaceLoss,
    "DynamicSoftMarginLoss": DynamicSoftMarginLoss,
    "FastAPLoss": FastAPLoss,
    "GeneralizedLiftedStructureLoss": GeneralizedLiftedStructureLoss,
    "InstanceLoss": InstanceLoss,
    "HistogramLoss": HistogramLoss,
    "IntraPairVarianceLoss": IntraPairVarianceLoss,
    "LargeMarginSoftmaxLoss": LargeMarginSoftmaxLoss,
    "LiftedStructureLoss": LiftedStructureLoss,
    "ManifoldLoss": ManifoldLoss,
    "MarginLoss": MarginLoss,
    "MultiSimilarityLoss": MultiSimilarityLoss,
    "NCALoss": NCALoss,
    "NormalizedSoftmaxLoss": NormalizedSoftmaxLoss,
    "NPairsLoss": NPairsLoss,
    "NTXentLoss": NTXentLoss,
    "P2SGradLoss": P2SGradLoss,
    "PNPLoss": PNPLoss,
    "ProxyAnchorLoss": ProxyAnchorLoss,
    "ProxyNCALoss": ProxyNCALoss,
    "RankedListLoss": RankedListLoss,
    "SignalToNoiseRatioContrastiveLoss": SignalToNoiseRatioContrastiveLoss,
    "SoftTripleLoss": SoftTripleLoss,
    "SphereFaceLoss": SphereFaceLoss,
    "SubCenterArcFaceLoss": SubCenterArcFaceLoss,
    "SupConLoss": SupConLoss,
    "TripletMarginLoss": TripletMarginLoss,
    "TupletMarginLoss": TupletMarginLoss,
}


class MetricLearningLoss(BaseLoss):
    def __init__(
        self,
        loss_name: str,
        embedding_size: int = 512,
        cross_batch_memory_size=0,
        loss_kwargs: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss_func = loss_dict[loss_name](
            **loss_kwargs
        )  # Instantiate the loss object
        if cross_batch_memory_size > 0:
            if loss_name in CrossBatchMemory.supported_losses():
                self.loss_func = CrossBatchMemory(
                    self.loss_func, embedding_size=embedding_size
                )
            else:
                # Warn that cross_batch_memory_size is ignored
                warnings.warn(
                    f"Cross batch memory is not supported for {loss_name}. Ignoring cross_batch_memory_size"
                )

        # self.miner_func = miner_func

    def prepare(self, inputs, labels):
        embeddings = inputs["features"][0]

        IDs = labels["id"][0][:, 0]
        return embeddings, IDs

    def forward(self, inputs: Tensor, target: Tensor):
        # miner_output = self.miner_func(inputs, target)

        loss = self.loss_func(inputs, target)

        return loss
