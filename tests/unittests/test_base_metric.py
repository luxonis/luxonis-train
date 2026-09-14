from typing import Annotated

import pytest
from torch import Tensor
from torchmetrics.utilities.data import (
    dim_zero_cat,
    dim_zero_max,
    dim_zero_sum,
)

from luxonis_train.attached_modules.metrics.base_metric import (
    BaseMetric,
    MetricState,
)


def test_explicit_reducer_wins_over_the_default():
    class Explicit(_DummyMetric, register=False):
        total: Annotated[Tensor, MetricState(default=0, dist_reduce_fx="max")]

    assert Explicit()._reductions["total"] is dim_zero_max


def test_reducer_defaults_follow_the_state_type():
    class Defaults(_DummyMetric, register=False):
        total: Annotated[Tensor, MetricState()]
        seen: Annotated[list[Tensor], MetricState()]

    metric = Defaults()
    assert metric._reductions["total"] is dim_zero_sum
    assert metric._reductions["seen"] is dim_zero_cat


def test_annotation_without_a_metric_state_is_ignored():
    class Annotated_(_DummyMetric, register=False):
        note: Annotated[str, "not a metric state"]

    assert Annotated_()._reductions == {}


def test_unsupported_state_type_is_rejected():
    class Unsupported(_DummyMetric, register=False):
        note: Annotated[str, MetricState()]

    with pytest.raises(ValueError, match="Unsupported type of a metric state"):
        Unsupported()


class _DummyMetric(BaseMetric, register=False):
    def update(self, *args: Tensor | list[Tensor]) -> None: ...

    def compute(self) -> Tensor: ...
