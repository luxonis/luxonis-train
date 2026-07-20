import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader

from luxonis_train.core.utils import aimet_utils


class _QATLossOnlyModel:
    def __init__(
        self, parameter: torch.nn.Parameter, automatic_optimization: bool
    ) -> None:
        self.parameter = parameter
        self.automatic_optimization = automatic_optimization
        self.loss_calls = 0
        self.automatic_optimization_states: list[bool] = []

    def train(self) -> "_QATLossOnlyModel":
        return self

    def cuda(self) -> "_QATLossOnlyModel":
        raise AssertionError("CUDA should be disabled in this test")

    def compute_training_loss(self, _batch: object) -> Tensor:
        self.loss_calls += 1
        self.automatic_optimization_states.append(self.automatic_optimization)
        return self.parameter * 2

    def training_step(self, _batch: object) -> Tensor:
        raise AssertionError("QAT should not enter Lightning training_step")


def _install_fake_aimet_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    aimet = ModuleType("aimet_torch")
    batch_norm_fold = ModuleType("aimet_torch.batch_norm_fold")
    bn_reestimation = ModuleType("aimet_torch.bn_reestimation")
    batch_norm_fold.__dict__["fold_all_batch_norms"] = lambda *_args: None
    bn_reestimation.__dict__["reestimate_bn_stats"] = lambda *_args: None
    monkeypatch.setitem(sys.modules, "aimet_torch", aimet)
    monkeypatch.setitem(
        sys.modules, "aimet_torch.batch_norm_fold", batch_norm_fold
    )
    monkeypatch.setitem(
        sys.modules, "aimet_torch.bn_reestimation", bn_reestimation
    )


@pytest.mark.parametrize("automatic_optimization", [True, False])
def test_quantization_aware_training_uses_loss_helper(
    monkeypatch: pytest.MonkeyPatch,
    automatic_optimization: bool,
) -> None:
    _install_fake_aimet_modules(monkeypatch)
    monkeypatch.setattr(aimet_utils, "check_aimet_available", lambda: None)
    monkeypatch.setattr(
        aimet_utils.CUDAAccelerator,
        "is_available",
        staticmethod(lambda: False),
    )

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    model = _QATLossOnlyModel(parameter, automatic_optimization)
    optimizer = SGD([parameter], lr=0.1)
    scheduler = ConstantLR(optimizer, factor=1.0)
    train_loader = cast(
        DataLoader[Any],
        [(torch.ones(1), {"target": torch.ones(1)})],
    )

    result = aimet_utils.quantization_aware_training(
        cast(Any, SimpleNamespace(model=model)),
        torch.ones(1),
        train_loader,
        optimizer,
        scheduler,
        epochs=1,
    )

    assert result is model
    assert model.automatic_optimization is automatic_optimization
    assert model.automatic_optimization_states == [False]
    assert model.loss_calls == 1
    assert parameter.item() == pytest.approx(0.8)


def test_quantization_aware_training_restores_automatic_optimization_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``quantization_aware_training`` toggles
    ``automatic_optimization`` to ``False`` so it can drive the QAT loop
    by hand.

    If any step inside the loop raises, the ``finally`` block must still
    restore the previous value — otherwise downstream Lightning phases
    (e.g. resuming standard training after a partially-failed QAT run)
    would silently run in manual-optimization mode. This regression
    would be invisible in metrics, hence a direct assertion here.
    """
    _install_fake_aimet_modules(monkeypatch)
    monkeypatch.setattr(aimet_utils, "check_aimet_available", lambda: None)
    monkeypatch.setattr(
        aimet_utils.CUDAAccelerator,
        "is_available",
        staticmethod(lambda: False),
    )

    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class _RaisingModel(_QATLossOnlyModel):
        def compute_training_loss(self, _batch: object) -> Tensor:
            self.loss_calls += 1
            raise RuntimeError("boom")

    model = _RaisingModel(parameter, automatic_optimization=True)
    optimizer = SGD([parameter], lr=0.1)
    scheduler = ConstantLR(optimizer, factor=1.0)
    train_loader = cast(
        DataLoader[Any],
        [(torch.ones(1), {"target": torch.ones(1)})],
    )

    with pytest.raises(RuntimeError, match="boom"):
        aimet_utils.quantization_aware_training(
            cast(Any, SimpleNamespace(model=model)),
            torch.ones(1),
            train_loader,
            optimizer,
            scheduler,
            epochs=1,
        )

    assert model.automatic_optimization is True


def test_quantization_aware_training_rejects_empty_train_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A training loader without a single batch is almost always a
    misconfiguration (e.g. an aggressive filter dropping every sample).

    The early ``assert`` surfaces this cleanly instead of silently
    returning an untouched model after zero optimization steps. The
    assertion fires inside the ``try`` block, so
    ``automatic_optimization`` must still be restored via the
    ``finally`` clause.
    """
    _install_fake_aimet_modules(monkeypatch)
    monkeypatch.setattr(aimet_utils, "check_aimet_available", lambda: None)
    monkeypatch.setattr(
        aimet_utils.CUDAAccelerator,
        "is_available",
        staticmethod(lambda: False),
    )

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    model = _QATLossOnlyModel(parameter, automatic_optimization=True)
    optimizer = SGD([parameter], lr=0.1)
    scheduler = ConstantLR(optimizer, factor=1.0)
    empty_loader = cast(DataLoader[Any], [])

    with pytest.raises(AssertionError, match="at least one batch"):
        aimet_utils.quantization_aware_training(
            cast(Any, SimpleNamespace(model=model)),
            torch.ones(1),
            empty_loader,
            optimizer,
            scheduler,
            epochs=1,
        )

    assert model.automatic_optimization is True
    assert model.loss_calls == 0
