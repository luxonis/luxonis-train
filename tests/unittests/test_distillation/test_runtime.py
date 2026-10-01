import copy
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from luxonis_train.attached_modules.losses import (
    CWDDistillationLoss,
    LogitKDLoss,
)
from luxonis_train.config import Config, TeacherConfig
from luxonis_train.distillation.controller import build_controller
from luxonis_train.distillation.recipe import resolve_recipe
from luxonis_train.distillation.teacher import (
    load_teacher_checkpoint,
    teacher_node_configs,
)
from luxonis_train.lightning import LuxonisLightningModule
from luxonis_train.lightning.training_plan import resolve_training_plan
from luxonis_train.nodes.blocks import ConvBlock
from luxonis_train.utils import DatasetMetadata

from ._helpers import (
    CLASSES,
    INPUT_SHAPES,
    KDBackbone,
    make_config,
    save_teacher,
    student_nodes,
)


def recipe(
    cfg: Config, teacher_file: Path
) -> dict[str, tuple[str, str, object]]:
    ckpt = load_teacher_checkpoint(TeacherConfig(weights=str(teacher_file)))
    entries = resolve_recipe(
        cfg, student_nodes(cfg), teacher_node_configs(ckpt)
    )
    return {
        entry.student_node: (
            entry.loss.name,
            entry.teacher_node,
            entry.loss.params.get("levels"),
        )
        for entry in entries
    }


def test_recipe_distills_the_head_and_the_levels_it_reads(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher)
    assert recipe(cfg, teacher) == {
        "KDHead": ("LogitKDLoss", "KDHead", None),
        # The head reads only the last of the two backbone levels.
        "KDBackbone": ("CWDDistillationLoss", "KDBackbone", [1]),
    }


def test_recipe_respects_off_and_explicit_lists(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    cfg = make_config(
        4,
        teacher=teacher,
        backbone={"distillation": False},
        head={
            "distillation": [
                {"name": "LogitKDLoss", "alias": "kd", "params": {"dim": 1}}
            ]
        },
    )
    assert recipe(cfg, teacher) == {"KDHead": ("LogitKDLoss", "KDHead", None)}


def test_explicit_teacher_node_must_exist(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    cfg = make_config(
        4,
        teacher=teacher,
        head={
            "distillation": [
                {"name": "LogitKDLoss", "params": {"teacher_node": "Nope"}}
            ]
        },
    )
    with pytest.raises(ValueError, match="teacher has no such node"):
        recipe(cfg, teacher)


def test_controller_attaches_losses_and_connectors(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher)
    nodes = student_nodes(cfg)
    controller = build_controller(cfg, nodes, INPUT_SHAPES)
    assert controller is not None

    head_loss = nodes["KDHead"].distillation["LogitKDLoss"]
    backbone_loss = nodes["KDBackbone"].distillation["CWDDistillationLoss"]
    assert isinstance(head_loss, LogitKDLoss)
    assert head_loss.temperature == 4.0
    assert isinstance(backbone_loss, CWDDistillationLoss)
    # Student level 1 has 8 channels, teacher level 1 has 16.
    assert list(controller.connectors) == ["KDBackbone/CWDDistillationLoss"]
    adapters = controller.connectors["KDBackbone/CWDDistillationLoss"]
    assert isinstance(adapters, nn.ModuleList)
    block = adapters[0]
    assert isinstance(block, ConvBlock)
    assert block.conv.in_channels == 8
    assert block.conv.out_channels == 16


def test_teacher_is_frozen_unregistered_and_never_copied(tmp_path: Path):
    teacher_file, _ = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher_file)
    controller = build_controller(cfg, student_nodes(cfg), INPUT_SHAPES)
    assert controller is not None
    teacher = controller._teachers["teacher"]

    assert not any(p.requires_grad for p in teacher.parameters())
    teacher.train()
    assert not any(m.training for m in teacher.modules())
    assert all(".teacher" not in key for key in controller.state_dict())
    assert copy.deepcopy(controller)._teachers == {}


def test_teacher_keeps_batch_norm_statistics(tmp_path: Path):
    teacher_file, _ = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher_file)
    controller = build_controller(cfg, student_nodes(cfg), INPUT_SHAPES)
    assert controller is not None
    teacher = controller._teachers["teacher"]
    backbone = teacher.nodes["KDBackbone"].module
    assert isinstance(backbone, KDBackbone)
    bn = backbone.stem[1]
    assert isinstance(bn, nn.BatchNorm2d)
    assert bn.running_mean is not None
    before = bn.running_mean.clone()

    teacher.train()
    controller.run_teachers({"image": torch.randn(2, 3, 32, 32) * 5 + 3})

    assert torch.equal(bn.running_mean, before)


def test_teacher_loads_the_checkpoint_weights(tmp_path: Path):
    teacher_file, reference = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher_file)
    controller = build_controller(cfg, student_nodes(cfg), INPUT_SHAPES)
    assert controller is not None
    image = torch.randn(2, 3, 32, 32)

    outputs = controller.run_teachers({"image": image})

    for node in reference.values():
        node.eval()
    with torch.no_grad():
        features = reference["KDBackbone"].module.run([{"features": [image]}])
        logits = reference["KDHead"].module.run([features])
    assert outputs is not None
    teacher_logits = outputs["KDHead"]["classification"]
    reference_logits = logits["classification"]
    assert isinstance(teacher_logits, Tensor)
    assert isinstance(reference_logits, Tensor)
    assert torch.allclose(teacher_logits, reference_logits)


def test_teacher_builds_only_the_matched_nodes(tmp_path: Path):
    teacher_file, _ = save_teacher(tmp_path)
    cfg = make_config(
        4,
        teacher=teacher_file,
        head={"distillation": "off"},
        backbone={"distillation": [{"name": "CWDDistillationLoss"}]},
    )
    controller = build_controller(cfg, student_nodes(cfg), INPUT_SHAPES)
    assert controller is not None
    assert list(controller._teachers["teacher"].nodes) == ["KDBackbone"]


def test_classes_must_match(tmp_path: Path):
    teacher_file, _ = save_teacher(
        tmp_path, classes={"": {"dog": 0, "cat": 1}}
    )
    cfg = make_config(4, teacher=teacher_file)
    with pytest.raises(ValueError, match="same classes in the same order"):
        build_controller(cfg, student_nodes(cfg), INPUT_SHAPES)


def test_training_plan_claims_each_connector_once(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher)
    nodes = student_nodes(cfg)
    controller = build_controller(cfg, nodes, INPUT_SHAPES)
    assert controller is not None

    plan = resolve_training_plan(
        cfg, nodes, extra_modules={"distillation": controller.connectors}
    )

    claimed = [
        id(parameter)
        for inner in plan.inners
        for group in inner.groups
        for parameter in group.parameters
    ]
    connector_ids = {id(p) for p in controller.connectors.parameters()}
    assert connector_ids <= set(claimed)
    assert len(claimed) == len(set(claimed))


def test_lightning_module_runs_distillation_only_in_training(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher)
    module = LuxonisLightningModule(
        cfg, tmp_path, INPUT_SHAPES, DatasetMetadata(classes=CLASSES)
    )
    module.attach_distillation(INPUT_SHAPES)
    inputs = {"image": torch.randn(2, 3, 32, 32)}
    labels = {"/classification": torch.eye(2)}

    module.train()
    train_losses = module.full_forward(inputs, labels).losses
    module.eval()
    eval_losses = module.full_forward(inputs, labels).losses

    assert set(train_losses["KDHead"]) == {"CrossEntropyLoss", "LogitKDLoss"}
    assert set(train_losses["KDBackbone"]) == {"CWDDistillationLoss"}
    assert set(eval_losses["KDHead"]) == {"CrossEntropyLoss"}
    assert "KDBackbone" not in eval_losses
    keys = list(module.state_dict())
    assert any(key.startswith("distillation.connectors.") for key in keys)
    assert not any("teacher" in key for key in keys)


def test_lightning_module_without_teacher_attaches_nothing(tmp_path: Path):
    cfg = make_config(4)
    module = LuxonisLightningModule(
        cfg, tmp_path, INPUT_SHAPES, DatasetMetadata(classes=CLASSES)
    )
    module.attach_distillation(INPUT_SHAPES)
    assert module.distillation is None
