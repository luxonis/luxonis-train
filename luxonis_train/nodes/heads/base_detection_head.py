from typing import Any

import torch
from loguru import logger
from torch import Size, Tensor, nn
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead


# TODO: Maybe change `heads` to `branches`?
class BaseDetectionHead(BaseHead[list[Tensor], tuple[list[Tensor], ...]]):
    parser = "YOLO"

    in_channels: list[int]
    in_sizes: list[Size]

    def __init__(
        self,
        n_heads: int,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
        **kwargs,
    ):
        """Base class for YOLO-like multi-head instance detection heads.

        @type n_heads: int
        @param n_heads: Number of output heads.
        @type conf_thres: float
        @param conf_thres: Confidence threshold for NMS.
        @type iou_thres: float
        @param iou_thres: IoU threshold for NMS.
        @type max_det: int
        @param max_det: Maximum number of detections retained after NMS.
        """

        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        self.stride = self.fit_stride_to_heads()

    def fit_stride_to_heads(self) -> Tensor:
        return torch.tensor(
            [
                round(self.original_in_shape[1] / x[2])
                for x in self.in_sizes[: self.n_heads]
            ],
            dtype=torch.int,
        )

    @override
    def get_custom_head_config(self) -> dict[str, Any]:
        """Returns custom head configuration.

        @rtype: dict[str, Any]
        @return: Custom head configuration.
        """
        return {
            "iou_threshold": self.iou_thres,
            "conf_threshold": self.conf_thres,
            "max_det": self.max_det,
        }

    def get_output_names(self, default: list[str]) -> list[str]:
        export_names = super().export_output_names
        if export_names is not None:
            if len(export_names) == self.n_heads:
                return export_names

            logger.warning(
                f"Number of provided output names ({len(export_names)}) "
                f"does not match number of heads ({self.n_heads}). "
                f"Using default names."
            )
        else:
            logger.warning(
                "No output names provided. "
                "Using names compatible with DepthAI."
            )
        return default
