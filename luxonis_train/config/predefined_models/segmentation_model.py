from luxonis_ml.typing import Params

from luxonis_train.config import LossModuleConfig, NodeConfig

from .base_predefined_model import SimplePredefinedModel


class SegmentationModel(SimplePredefinedModel):
    def __init__(self, aux_head_params: Params | None = None, **kwargs):
        kwargs = {
            "backbone": "DDRNet",
            "head": "DDRNetSegmentationHead",
            "loss": "OHEMLoss",
            "metrics": ["JaccardIndex", "F1Score"],
            "confusion_matrix_available": True,
            "main_metric": "JaccardIndex",
            "visualizer": "SegmentationVisualizer",
            "weights": "download",
        } | kwargs
        super().__init__(**kwargs)

        self._use_aux_heads = kwargs.get("use_aux_heads", True)

        self._aux_head_params = aux_head_params or {}
        if "attach_index" not in self._aux_head_params:
            self._aux_head_params["attach_index"] = -2
        if "aux_head" not in self._head_params:
            self._head_params["attach_index"] = -1

        remove_aux_on_export = self._aux_head_params.pop("use_aux_heads", True)
        if not isinstance(remove_aux_on_export, bool):
            raise TypeError(
                "The `aux_head_params.remove_on_export` parameter "
                f"must be a boolean. Got `{self._remove_aux_on_export}`."
            )
        self._remove_aux_on_export = remove_aux_on_export

    @staticmethod
    def get_variants() -> tuple[str, dict[str, Params]]:
        return "light", {
            "light": {
                "backbone": "DDRNet",
                "backbone_params": {
                    "variant": "23-slim",
                },
            },
            "heavy": {
                "backbone": "DDRNet",
                "backbone_params": {
                    "variant": "23",
                },
            },
        }

    @property
    def nodes(self) -> list[NodeConfig]:
        nodes = super().nodes
        if self._use_aux_heads:
            nodes.append(
                NodeConfig(
                    name=self._head,
                    alias=f"{self._head}_aux",
                    inputs=[self._backbone],
                    params=self._aux_head_params,
                    task_name=self._task_name,
                    remove_on_export=self._remove_aux_on_export,
                )
            )
        return nodes

    @property
    def losses(self) -> list[LossModuleConfig]:
        losses = super().losses
        if self._use_aux_heads:
            losses.append(
                LossModuleConfig(
                    name=self._loss,
                    attached_to=f"{self._head}_aux",
                    weight=0.4,
                )
            )
        return losses
