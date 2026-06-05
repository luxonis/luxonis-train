import torch
from loguru import logger
from luxonis_ml.typing import Params
from torch import Size, Tensor
from typing_extensions import override

from luxonis_train.nodes.heads import BaseHead


class BaseDetectionHead(BaseHead):
    """Base class for YOLO-like multi-head instance detection heads."""

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
        """Initialize the base detection head.

        Args:
            n_heads (int): Number of output heads.
            conf_thres (float): Confidence threshold for NMS.
            iou_thres (float): IoU threshold for NMS.
            max_det (int): Maximum number of detections retained after NMS.
            **kwargs (``Any``): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)

        self.n_heads = n_heads
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        if len(self.in_channels) < self.n_heads:
            logger.warning(
                f"Head '{self.name}' was set to use {self.n_heads} heads, "
                f"but received only {len(self.in_channels)} inputs. "
                f"Changing number of heads to {len(self.in_channels)}."
            )
            self.n_heads = len(self.in_channels)

        if "attach_index" not in kwargs:
            self.attach_index = (-self.n_heads - 1, -1)

        self.stride = self.fit_stride_to_heads()

    def _forward(
        self, inputs: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        features_list: list[Tensor] = []
        classes_list: list[Tensor] = []
        regressions_list: list[Tensor] = []

        for head, x in zip(self.heads, inputs, strict=True):  # type: ignore
            features, classes, regressions = head(x)
            features_list.append(features)
            classes_list.append(torch.sigmoid(classes))
            regressions_list.append(regressions)
        return features_list, classes_list, regressions_list

    @override
    def get_custom_head_config(self) -> Params:
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

    def fit_stride_to_heads(self) -> Tensor:
        return torch.tensor(
            [
                round(self.original_in_shape[1] / x[2])
                for x in self.in_sizes[: self.n_heads]
            ],
            dtype=torch.int,
        )
