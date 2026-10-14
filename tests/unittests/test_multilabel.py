"""Multi-label heads and the modules that attach to them."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from bidict import bidict
from luxonis_ml.typing import Params
from torch import Size, Tensor

from luxonis_train.attached_modules.losses import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    OHEMLoss,
)
from luxonis_train.attached_modules.metrics import F1Score
from luxonis_train.attached_modules.visualizers import (
    ClassificationVisualizer,
)
from luxonis_train.config import Config
from luxonis_train.config.predefined_models import (
    ClassificationModel,
    SegmentationModel,
)
from luxonis_train.core import LuxonisModel
from luxonis_train.loaders import LuxonisLoaderTorch
from luxonis_train.nodes import (
    BaseHead,
    ClassificationHead,
    DDRNetSegmentationHead,
)
from luxonis_train.tasks import Tasks
from luxonis_train.utils.annotation import _annotate_classification


def _classification_head(
    multilabel: bool, n_classes: int = 3
) -> ClassificationHead:
    return ClassificationHead(
        n_classes=n_classes,
        multilabel=multilabel,
        input_shapes=[{"features": [Size([2, 16, 7, 7])]}],
    )


def test_a_box_head_cannot_be_multilabel():
    class BoxHead(BaseHead, register=False):
        task = Tasks.BOUNDINGBOX

        def forward(self, inputs: Tensor) -> Tensor:
            return inputs

    with pytest.raises(ValueError, match="cannot be multi-label"):
        BoxHead(multilabel=True)


def test_cross_entropy_is_binary_for_each_class():
    loss = CrossEntropyLoss(
        node=_classification_head(multilabel=True),
        weight=[1.0, 2.0, 3.0],
        label_smoothing=0.2,
    )
    logits = torch.tensor([[2.0, -1.0, 0.5]])

    value = loss(logits, torch.tensor([[1.0, 0.0, 1.0]]))

    expected = F.binary_cross_entropy_with_logits(
        logits,
        torch.tensor([[0.9, 0.1, 0.9]]),
        weight=torch.tensor([1.0, 2.0, 3.0]),
    )
    assert torch.isclose(value, expected)


@pytest.mark.parametrize(
    ("multilabel", "criterion"),
    [(False, CrossEntropyLoss), (True, BCEWithLogitsLoss)],
)
def test_ohem_picks_its_criterion(multilabel: bool, criterion: type):
    loss = OHEMLoss(node=_classification_head(multilabel))

    assert type(loss.criterion) is criterion


@pytest.mark.parametrize(
    ("multilabel", "n_classes", "task"),
    [
        (False, 3, "multiclass"),
        (True, 3, "multilabel"),
        (True, 1, "binary"),
    ],
)
def test_metric_task_follows_the_head(
    multilabel: bool, n_classes: int, task: str
):
    metric = F1Score(node=_classification_head(multilabel, n_classes))

    assert metric._torchmetric_task == task


@pytest.mark.parametrize("multilabel", [False, True])
def test_classification_visualizer_follows_the_head(multilabel: bool):
    visualizer = ClassificationVisualizer(
        node=_classification_head(multilabel)
    )

    assert visualizer._multilabel is multilabel


@pytest.mark.parametrize(
    ("multilabel", "shape"),
    [(False, (1, 32, 32)), (True, (1, 3, 32, 32))],
)
def test_ddrnet_export_shape(multilabel: bool, shape: tuple[int, ...]):
    head = DDRNetSegmentationHead(
        n_classes=3,
        multilabel=multilabel,
        input_shapes=[{"features": [Size([1, 8, 4, 4])]}],
        original_in_shape=Size([3, 32, 32]),
    )
    head.export = True

    output = head.eval()(torch.zeros(1, 8, 4, 4))

    assert output.shape == shape
    assert output.dtype == torch.int32


@pytest.mark.parametrize(
    ("multilabel", "logits", "annotations"),
    [
        (False, [2.0, -1.0, 3.0], [{"class": "dog"}]),
        (True, [2.0, -1.0, 3.0], [{"class": "cat"}, {"class": "dog"}]),
        (True, [-2.0, -1.0, -3.0], [None]),
    ],
)
def test_annotate_classification(
    multilabel: bool, logits: list[float], annotations: list[Any]
):
    head = SimpleNamespace(
        multilabel=multilabel,
        task_name="",
        classes=bidict({"cat": 0, "bird": 1, "dog": 2}),
    )

    records = _annotate_classification(
        cast(Any, head),
        Path("image.jpg"),
        {"pred_classes": np.array(logits)},
    )

    assert [
        cast(Params, record).get("annotation") for record in records
    ] == annotations


def test_classification_model_makes_its_head_multilabel():
    head = ClassificationModel(variant="light", multilabel=True).nodes[-1]

    assert head.params["multilabel"] is True


@pytest.mark.parametrize(
    "params",
    [{"multilabel": True}, {"head_params": {"multilabel": True}}],
)
def test_segmentation_model_makes_both_heads_multilabel(params: dict):
    nodes = SegmentationModel(variant="light", **params).nodes

    assert [node.params.get("multilabel") for node in nodes[1:]] == [
        True,
        True,
    ]


def test_loader_gets_the_tasks_of_multilabel_heads():
    cfg = Config.get_config(
        {
            "model": {
                "name": "multilabel",
                "predefined_model": {
                    "name": "SegmentationModel",
                    "params": {"multilabel": True, "task_name": "animals"},
                },
            },
            "loader": {"params": {"dataset_name": "unused"}},
        }
    )
    trainer = cast(LuxonisModel, SimpleNamespace(cfg=cfg))

    LuxonisModel._set_multilabel_tasks(trainer, LuxonisLoaderTorch)

    assert cfg.loader.params["multilabel_tasks"] == ["animals"]
