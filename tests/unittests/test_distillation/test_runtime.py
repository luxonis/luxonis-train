from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from luxonis_train.attached_modules.losses import (
    ChannelWiseDistillationLoss,
    LogitDistillationLoss,
)
from luxonis_train.config import Config, TeacherConfig
from luxonis_train.lightning import LuxonisLightningModule
from luxonis_train.lightning.distillation import (
    Distiller,
    ReleaseTeacherCallback,
)
from luxonis_train.lightning.distillation.recipe import (
    match_nodes,
    resolve_recipe,
)
from luxonis_train.lightning.distillation.teacher import (
    load_teacher_checkpoint,
    teacher_node_configs,
)
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

# A teacher whose nodes have other identifiers than the student nodes.
RENAMED = {
    "backbone": {"alias": "TeacherBackbone"},
    "head": {"alias": "TeacherHead", "inputs": ["TeacherBackbone"]},
}


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


def test_recipe_distills_the_head_and_its_feeder(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher)
    assert recipe(cfg, teacher) == {
        "KDHead": ("LogitDistillationLoss", "KDHead", None),
        "KDBackbone": ("ChannelWiseDistillationLoss", "KDBackbone", None),
    }


def test_recipe_matches_by_task_and_graph_position(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path, **RENAMED)
    cfg = make_config(4, teacher=teacher)
    ckpt = load_teacher_checkpoint(TeacherConfig(weights=str(teacher)))
    matches = match_nodes(student_nodes(cfg), teacher_node_configs(ckpt))
    assert {
        name: (m.teacher_node, m.reason) for name, m in matches.items()
    } == {
        "KDHead": ("TeacherHead", "same task"),
        "KDBackbone": ("TeacherBackbone", "same graph position"),
    }


def test_recipe_distills_the_feeder_of_a_head_without_default(
    tmp_path: Path,
):
    head = {"name": "KDPlainHead"}
    teacher, _ = save_teacher(tmp_path, head=head)
    cfg = make_config(4, teacher=teacher, head=head)
    assert recipe(cfg, teacher) == {
        "KDBackbone": ("ChannelWiseDistillationLoss", "KDBackbone", None),
    }


def test_recipe_respects_false_and_explicit_lists(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    cfg = make_config(
        4,
        teacher=teacher,
        backbone={"distillation": False},
        head={
            "distillation": [{"name": "LogitDistillationLoss", "alias": "kd"}]
        },
    )
    assert recipe(cfg, teacher) == {
        "KDHead": ("LogitDistillationLoss", "KDHead", None)
    }


def test_recipe_raises_when_nothing_is_distilled(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    cfg = make_config(
        4,
        teacher=teacher,
        backbone={"distillation": False},
        head={"distillation": False},
    )
    with pytest.raises(ValueError, match="no node gets a distillation loss"):
        recipe(cfg, teacher)


def test_explicit_teacher_node_must_exist(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    cfg = make_config(
        4,
        teacher=teacher,
        head={
            "distillation": [
                {"name": "LogitDistillationLoss", "teacher_node": "Nope"}
            ]
        },
    )
    with pytest.raises(ValueError, match="teacher has no such node"):
        recipe(cfg, teacher)


def test_distiller_attaches_losses_and_connectors(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher)
    distiller = Distiller.from_config(cfg, student_nodes(cfg))

    head_loss = distiller.losses("KDHead")["LogitDistillationLoss"]
    backbone_loss = distiller.losses("KDBackbone")[
        "ChannelWiseDistillationLoss"
    ]
    assert isinstance(head_loss, LogitDistillationLoss)
    assert head_loss.temperature == 4.0
    assert isinstance(backbone_loss, ChannelWiseDistillationLoss)
    # The head reads the last of the two backbone levels.
    assert backbone_loss.levels == [-1]
    # Student level -1 has 8 channels, teacher level -1 has 16.
    assert list(distiller.connectors) == [
        "KDBackbone/ChannelWiseDistillationLoss"
    ]
    adapters = distiller.connectors["KDBackbone/ChannelWiseDistillationLoss"]
    assert isinstance(adapters, nn.ModuleList)
    block = adapters[0]
    assert isinstance(block, ConvBlock)
    assert block.conv.in_channels == 8
    assert block.conv.out_channels == 16


def test_teacher_is_frozen_and_unregistered(tmp_path: Path):
    teacher_file, _ = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher_file)
    distiller = Distiller.from_config(cfg, student_nodes(cfg))
    teacher = distiller.teacher

    assert not any(p.requires_grad for p in teacher.parameters())
    teacher.train()
    assert not any(m.training for m in teacher.modules())
    assert all("teacher" not in key for key in distiller.state_dict())


def test_teacher_keeps_batch_norm_statistics(tmp_path: Path):
    teacher_file, _ = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher_file)
    distiller = Distiller.from_config(cfg, student_nodes(cfg))
    backbone = distiller.teacher.nodes["KDBackbone"].module
    assert isinstance(backbone, KDBackbone)
    bn = backbone.stem[1]
    assert isinstance(bn, nn.BatchNorm2d)
    assert bn.running_mean is not None
    before = bn.running_mean.clone()

    distiller.teacher.train()
    distiller.run_teacher({"image": torch.randn(2, 3, 32, 32) * 5 + 3})

    assert torch.equal(bn.running_mean, before)


def test_teacher_loads_the_checkpoint_weights(tmp_path: Path):
    teacher_file, reference = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher_file)
    distiller = Distiller.from_config(cfg, student_nodes(cfg))
    image = torch.randn(2, 3, 32, 32)

    outputs = distiller.run_teacher({"image": image})

    for node in reference.values():
        node.eval()
    with torch.no_grad():
        features = reference["KDBackbone"].module.run([{"features": [image]}])
        logits = reference["KDHead"].module.run([features])
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
        head={"distillation": False},
        backbone={"distillation": [{"name": "ChannelWiseDistillationLoss"}]},
    )
    distiller = Distiller.from_config(cfg, student_nodes(cfg))
    assert list(distiller.teacher.nodes) == ["KDBackbone"]


def test_classes_must_match(tmp_path: Path):
    teacher_file, _ = save_teacher(
        tmp_path, classes={"": {"dog": 0, "cat": 1}}
    )
    cfg = make_config(4, teacher=teacher_file)
    with pytest.raises(ValueError, match="same classes in the same order"):
        Distiller.from_config(cfg, student_nodes(cfg))


def test_loss_weight_scales_every_distillation_loss(tmp_path: Path):
    teacher_file, _ = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher_file)
    half = cfg.model_copy(deep=True)
    assert half.model.teacher is not None
    half.model.teacher.loss_weight = 0.5
    nodes = student_nodes(cfg)
    image = {"image": torch.randn(2, 3, 32, 32)}
    logits = nodes["KDHead"].module.run(
        [nodes["KDBackbone"].module.run([{"features": [image["image"]]}])]
    )

    values = []
    for config in (cfg, half):
        distiller = Distiller.from_config(config, nodes)
        teacher_outputs = distiller.run_teacher(image)
        value = distiller.compute_losses(
            "KDHead", logits, {}, teacher_outputs
        )["LogitDistillationLoss"]
        assert isinstance(value, Tensor)
        values.append(value)

    assert torch.isclose(values[1], 0.5 * values[0])


def test_training_plan_claims_each_connector_once(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    cfg = make_config(4, teacher=teacher)
    nodes = student_nodes(cfg)
    distiller = Distiller.from_config(cfg, nodes)

    plan = resolve_training_plan(
        cfg, nodes, extra_modules={"distiller": distiller.connectors}
    )

    claimed = [
        id(parameter)
        for inner in plan.inners
        for group in inner.groups
        for parameter in group.parameters
    ]
    connector_ids = {id(p) for p in distiller.connectors.parameters()}
    assert connector_ids <= set(claimed)
    assert len(claimed) == len(set(claimed))


def lightning_module(cfg: Config, tmp_path: Path) -> LuxonisLightningModule:
    return LuxonisLightningModule(
        cfg, tmp_path, INPUT_SHAPES, DatasetMetadata(classes=CLASSES)
    )


def test_fit_setup_builds_the_distiller_and_other_stages_do_not(
    tmp_path: Path,
):
    teacher, _ = save_teacher(tmp_path)
    module = lightning_module(make_config(4, teacher=teacher), tmp_path)

    module.setup("validate")
    assert module.distiller is None
    assert isinstance(module.configure_callbacks()[0], ReleaseTeacherCallback)

    module.setup("fit")
    distiller = module.distiller
    assert distiller is not None
    module.setup("fit")
    assert module.distiller is distiller


def test_lightning_module_runs_distillation_only_in_training(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    module = lightning_module(make_config(4, teacher=teacher), tmp_path)
    module.setup("fit")
    inputs = {"image": torch.randn(2, 3, 32, 32)}
    labels = {"/classification": torch.eye(2)}

    module.train()
    train_losses = module.full_forward(inputs, labels).losses
    module.eval()
    eval_losses = module.full_forward(inputs, labels).losses

    assert set(train_losses["KDHead"]) == {
        "CrossEntropyLoss",
        "LogitDistillationLoss",
    }
    assert set(train_losses["KDBackbone"]) == {"ChannelWiseDistillationLoss"}
    assert set(eval_losses["KDHead"]) == {"CrossEntropyLoss"}
    assert "KDBackbone" not in eval_losses
    keys = list(module.state_dict())
    assert any(key.startswith("distiller.connectors.") for key in keys)
    assert not any("teacher" in key for key in keys)


def test_released_distiller_stops_the_losses(tmp_path: Path):
    teacher, _ = save_teacher(tmp_path)
    module = lightning_module(make_config(4, teacher=teacher), tmp_path)
    module.setup("fit")
    distiller = module.distiller
    assert distiller is not None

    ReleaseTeacherCallback().on_train_end(cast(Any, None), module)
    module.train()
    losses = module.full_forward(
        {"image": torch.randn(2, 3, 32, 32)}, {"/classification": torch.eye(2)}
    ).losses

    assert not distiller.has_teacher
    assert set(losses) == {"KDHead"}
    assert set(losses["KDHead"]) == {"CrossEntropyLoss"}
    with pytest.raises(RuntimeError, match="released its teacher"):
        distiller.run_teacher({"image": torch.zeros(1, 3, 32, 32)})


def test_lightning_module_without_teacher_has_no_distiller(tmp_path: Path):
    module = lightning_module(make_config(4), tmp_path)
    module.setup("fit")
    assert module.distiller is None
    assert not any(
        isinstance(callback, ReleaseTeacherCallback)
        for callback in module.configure_callbacks()
    )
