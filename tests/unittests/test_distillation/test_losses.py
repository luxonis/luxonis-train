import pytest
import torch
from torch import Size, nn

from luxonis_train.attached_modules.losses import (
    BCEWithLogitsLoss,
    ChannelWiseDistillationLoss,
    CrossEntropyLoss,
    LogitDistillationLoss,
)
from luxonis_train.attached_modules.losses.distillation import (
    StudentContext,
)
from luxonis_train.tasks import Tasks
from luxonis_train.typing import Packet


class ClassificationLogitLoss(LogitDistillationLoss, register=False):
    # Without a node, the only supported task becomes the task.
    supported_tasks = [Tasks.CLASSIFICATION]


def test_logit_kd_is_zero_for_equal_logits():
    logits = torch.randn(4, 5)
    assert LogitDistillationLoss(temperature=4.0)(logits, logits).item() == 0


def test_logit_kd_scales_with_temperature_squared():
    student, teacher = torch.randn(4, 5), torch.randn(4, 5)
    kd = LogitDistillationLoss(temperature=1.0)(student * 3, teacher * 3)
    kd_t3 = LogitDistillationLoss(temperature=3.0)(student * 9, teacher * 9)
    # Dividing the logits by T gives the same distributions, so only the
    # T^2 factor remains.
    assert kd > 0
    assert torch.isclose(kd_t3, 9 * kd)


def test_logit_kd_uses_sigmoid_for_one_class():
    # A softmax over one class is constant, so it would give 0.
    student, teacher = torch.zeros(2, 1, 4, 4), torch.ones(2, 1, 4, 4)
    assert LogitDistillationLoss()(student, teacher).item() > 0
    softmax = LogitDistillationLoss(activation="softmax")
    assert softmax(student, teacher).item() == 0


def test_logit_kd_reduces_dense_logits_over_classes():
    student, teacher = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    loss = LogitDistillationLoss()(student, teacher)
    per_pixel = LogitDistillationLoss()(
        student.permute(0, 2, 3, 1).reshape(-1, 3),
        teacher.permute(0, 2, 3, 1).reshape(-1, 3),
    )
    assert torch.isclose(loss, per_pixel)


def test_logit_kd_rejects_different_shapes():
    with pytest.raises(RuntimeError, match="different shapes"):
        LogitDistillationLoss()(torch.zeros(2, 3), torch.zeros(2, 4))


def test_logit_kd_setup_rejects_different_shapes():
    student: Packet[Size] = {"classification": Size([2, 3])}
    teacher: Packet[Size] = {"classification": Size([2, 4])}
    with pytest.raises(ValueError, match="shapes differ"):
        ClassificationLogitLoss().setup(student, teacher)


def test_logit_kd_setup_needs_the_teacher_logits():
    student: Packet[Size] = {"classification": Size([2, 3])}
    teacher: Packet[Size] = {"features": Size([2, 8, 4, 4])}
    with pytest.raises(ValueError, match="teacher output 'classification'"):
        ClassificationLogitLoss().setup(student, teacher)


def test_logit_kd_follows_a_sigmoid_task_loss():
    sigmoid = StudentContext({"bce": BCEWithLogitsLoss()}, levels_read=None)
    softmax = StudentContext({"ce": CrossEntropyLoss()}, levels_read=None)
    assert LogitDistillationLoss.derive_params(sigmoid) == {
        "activation": "sigmoid"
    }
    assert LogitDistillationLoss.derive_params(softmax) == {}


def test_logit_kd_gradient_reaches_only_the_student():
    student = torch.randn(4, 5, requires_grad=True)
    teacher = torch.randn(4, 5)
    LogitDistillationLoss(temperature=2.0)(student, teacher).backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_logit_kd_computes_in_float32_under_autocast():
    student, teacher = torch.randn(4, 5), torch.randn(4, 5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = LogitDistillationLoss()(student.bfloat16(), teacher.bfloat16())
    assert loss.dtype == torch.float32


def test_cwd_is_zero_for_equal_maps():
    maps = [torch.randn(2, 4, 8, 8), torch.randn(2, 8, 4, 4)]
    assert ChannelWiseDistillationLoss()(maps, maps).item() == 0


def test_cwd_sets_up_adapters_for_the_selected_levels():
    loss = ChannelWiseDistillationLoss(levels=[-1])
    student: Packet[Size] = {
        "features": [Size([2, 4, 8, 8]), Size([2, 8, 4, 4])]
    }
    teacher: Packet[Size] = {
        "features": [
            Size([2, 4, 16, 16]),
            Size([2, 4, 8, 8]),
            Size([2, 16, 4, 4]),
        ]
    }
    adapters = loss.setup(student, teacher)
    assert isinstance(adapters, nn.ModuleList)
    assert len(adapters) == 1
    value = loss(
        [torch.randn(2, 4, 8, 8), torch.randn(2, 8, 4, 4)],
        [
            torch.randn(2, 4, 16, 16),
            torch.randn(2, 4, 8, 8),
            torch.randn(2, 16, 4, 4),
        ],
    )
    value.backward()
    assert all(p.grad is not None for p in adapters.parameters())


def test_cwd_keeps_identity_for_equal_channels():
    loss = ChannelWiseDistillationLoss()
    shapes: Packet[Size] = {"features": [Size([2, 4, 8, 8])]}
    adapters = loss.setup(shapes, shapes)
    assert isinstance(adapters, nn.ModuleList)
    assert isinstance(adapters[0], nn.Identity)


def test_cwd_resizes_the_teacher_to_the_student_grid():
    student, teacher = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 16, 16)
    assert ChannelWiseDistillationLoss()(student, teacher).item() > 0


def test_cwd_needs_an_adapter_for_different_channels():
    with pytest.raises(RuntimeError, match="Call `setup` first"):
        ChannelWiseDistillationLoss()(
            torch.randn(2, 4, 8, 8), torch.randn(2, 8, 8, 8)
        )


def test_cwd_rejects_different_level_counts():
    student: Packet[Size] = {
        "features": [Size([2, 4, 8, 8]), Size([2, 8, 4, 4])]
    }
    teacher: Packet[Size] = {"features": [Size([2, 4, 8, 8])]}
    with pytest.raises(ValueError, match="2 student levels"):
        ChannelWiseDistillationLoss().setup(student, teacher)


def test_cwd_rejects_levels_the_teacher_does_not_have():
    student: Packet[Size] = {
        "features": [Size([2, 4, 8, 8]), Size([2, 8, 4, 4])]
    }
    teacher: Packet[Size] = {"features": [Size([2, 4, 8, 8])]}
    with pytest.raises(ValueError, match="fewer levels"):
        ChannelWiseDistillationLoss(levels=[1]).setup(student, teacher)
