import pytest
import torch
from torch import Size, nn

from luxonis_train.attached_modules.losses import (
    CWDDistillationLoss,
    LogitKDLoss,
)
from luxonis_train.typing import Packet


def test_logit_kd_is_zero_for_equal_logits():
    logits = torch.randn(4, 5)
    assert LogitKDLoss(temperature=4.0)(logits, logits).item() == 0


def test_logit_kd_scales_with_temperature_squared():
    student, teacher = torch.randn(4, 5), torch.randn(4, 5)
    kd = LogitKDLoss(temperature=1.0)(student * 3, teacher * 3)
    kd_t3 = LogitKDLoss(temperature=3.0)(student * 9, teacher * 9)
    # Dividing the logits by T gives the same distributions, so only the
    # T^2 factor remains.
    assert kd > 0
    assert torch.isclose(kd_t3, 9 * kd)


def test_logit_kd_uses_sigmoid_for_one_class():
    # A softmax over one class is constant, so it would give 0.
    student, teacher = torch.zeros(2, 1, 4, 4), torch.ones(2, 1, 4, 4)
    assert LogitKDLoss()(student, teacher).item() > 0
    assert LogitKDLoss(activation="softmax")(student, teacher).item() == 0


def test_logit_kd_reduces_dense_logits_over_classes():
    student, teacher = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    loss = LogitKDLoss()(student, teacher)
    per_pixel = LogitKDLoss()(
        student.permute(0, 2, 3, 1).reshape(-1, 3),
        teacher.permute(0, 2, 3, 1).reshape(-1, 3),
    )
    assert torch.isclose(loss, per_pixel)


def test_logit_kd_rejects_different_shapes():
    with pytest.raises(RuntimeError, match="different shapes"):
        LogitKDLoss()(torch.zeros(2, 3), torch.zeros(2, 4))


def test_logit_kd_gradient_reaches_only_the_student():
    student = torch.randn(4, 5, requires_grad=True)
    teacher = torch.randn(4, 5)
    LogitKDLoss(temperature=2.0)(student, teacher).backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_logit_kd_computes_in_float32_under_autocast():
    student, teacher = torch.randn(4, 5), torch.randn(4, 5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = LogitKDLoss()(student.bfloat16(), teacher.bfloat16())
    assert loss.dtype == torch.float32


def test_cwd_is_zero_for_equal_maps():
    maps = [torch.randn(2, 4, 8, 8), torch.randn(2, 8, 4, 4)]
    assert CWDDistillationLoss()(maps, maps).item() == 0


def test_cwd_builds_adapters_for_the_selected_levels():
    loss = CWDDistillationLoss(levels=[1])
    student: Packet[Size] = {
        "features": [Size([2, 4, 8, 8]), Size([2, 8, 4, 4])]
    }
    teacher: Packet[Size] = {
        "features": [Size([2, 4, 8, 8]), Size([2, 16, 4, 4])]
    }
    adapters = loss.build(student, teacher)
    assert isinstance(adapters, nn.ModuleList)
    assert len(adapters) == 1
    value = loss(
        [torch.randn(2, 4, 8, 8), torch.randn(2, 8, 4, 4)],
        [torch.randn(2, 4, 8, 8), torch.randn(2, 16, 4, 4)],
    )
    value.backward()
    assert all(p.grad is not None for p in adapters.parameters())


def test_cwd_keeps_identity_for_equal_channels():
    loss = CWDDistillationLoss()
    shapes: Packet[Size] = {"features": [Size([2, 4, 8, 8])]}
    adapters = loss.build(shapes, shapes)
    assert isinstance(adapters, nn.ModuleList)
    assert isinstance(adapters[0], nn.Identity)


def test_cwd_resizes_the_teacher_to_the_student_grid():
    student, teacher = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 16, 16)
    assert CWDDistillationLoss()(student, teacher).item() > 0


def test_cwd_needs_an_adapter_for_different_channels():
    with pytest.raises(RuntimeError, match="Call `build` first"):
        CWDDistillationLoss()(torch.randn(2, 4, 8, 8), torch.randn(2, 8, 8, 8))


def test_cwd_rejects_different_level_counts():
    student: Packet[Size] = {
        "features": [Size([2, 4, 8, 8]), Size([2, 8, 4, 4])]
    }
    teacher: Packet[Size] = {"features": [Size([2, 4, 8, 8])]}
    with pytest.raises(ValueError, match="2 student levels"):
        CWDDistillationLoss().build(student, teacher)
