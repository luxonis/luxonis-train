"""Attach a teacher to a model and run it during training."""

from typing import Any

import lightning.pytorch as pl
import torch
from loguru import logger
from torch import Size, Tensor, nn
from typing_extensions import override

from luxonis_train.attached_modules.losses import BaseDistillationLoss
from luxonis_train.config import Config
from luxonis_train.lightning.utils import Nodes
from luxonis_train.nodes import BaseNode
from luxonis_train.registry import LOSSES
from luxonis_train.typing import Packet

from .recipe import DistillationEntry, resolve_recipe
from .teacher import (
    Teacher,
    build_teacher,
    load_teacher_checkpoint,
    teacher_node_configs,
)


class DistillationController(nn.Module):
    """The distillation state of a `LuxonisLightningModule`.

    The module is a registered submodule of the Lightning module, so
    its ``connectors`` reach the state dict, the checkpoints, the
    optimizer, and the DDP wrapper. The teacher is held in a plain
    dictionary instead. No recursive method of ``torch.nn.Module``
    reaches it, a deep copy of the module leaves it out, and it never
    lands in a checkpoint.

    Attributes:
        connectors: The trainable parts of the distillation losses,
            keyed ``<node>/<loss>``. The exported model does not contain
            them.
        active: Whether `run_teachers` runs the teacher. The release
            callback clears it when training ends.

    """

    def __init__(self, teacher: Teacher):
        """Hold the teacher, without registering it.

        Args:
            teacher: The teacher.

        """
        super().__init__()
        self.connectors = nn.ModuleDict()
        self._teachers: dict[str, Teacher] = {"teacher": teacher}
        self._teacher_device = torch.device("cpu")
        self.active = True

    def run_teachers(
        self, inputs: dict[str, Tensor]
    ) -> dict[str, Packet[Tensor]] | None:
        """Run the teacher on a batch.

        The teacher moves to the device of the inputs on its first use,
        so it needs no ``to`` call from the trainer.

        Args:
            inputs: The loader inputs.

        Returns:
            The output packets of the teacher nodes, keyed by
            identifier, or ``None`` when the controller is not active or
            holds no teacher.

        """
        teacher = self._teachers.get("teacher")
        if not self.active or teacher is None:
            return None
        device = next(iter(inputs.values())).device
        if device != self._teacher_device:
            teacher.to(device)
            self._teacher_device = device
        return teacher(inputs)

    def release(self) -> None:
        """Stop running the teacher and move it off the accelerator."""
        self.active = False
        for teacher in self._teachers.values():
            teacher.to("cpu")
        self._teacher_device = torch.device("cpu")

    @override
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_teachers"] = {}
        return state


class ReleaseTeacherCallback(pl.Callback):
    """Release the teacher when training ends.

    The callback runs before the other callbacks, so the test, export
    and quantization callbacks that run at the end of training do not
    keep the teacher in the memory of the accelerator.

    """

    @override
    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        controller = getattr(pl_module, "distillation", None)
        if controller is not None:
            controller.release()


def build_controller(
    cfg: Config, nodes: Nodes, input_shapes: dict[str, Size]
) -> DistillationController | None:
    """Load the teacher and attach the distillation losses to the nodes.

    The function reads the teacher checkpoint of ``cfg.model.teacher``,
    resolves the losses with `resolve_recipe`, builds the teacher nodes
    that the losses read, and adds the losses to the
    `NodeWrapper.distillation` dictionaries of ``nodes``.

    Args:
        cfg: The config of the student. ``model.teacher`` must be set.
        nodes: The built student nodes.
        input_shapes: The shapes of the loader inputs, without the batch
            dimension.

    Returns:
        The controller, or ``None`` when no node gets a distillation
        loss.

    Raises:
        ValueError: When ``model.teacher`` is not set, or when a matched
            node has other classes than its teacher node.
        TypeError: When a ``distillation`` entry names a loss that is
            not a distillation loss.

    """
    teacher_cfg = cfg.model.teacher
    if teacher_cfg is None:
        raise ValueError("`model.teacher` is not set.")
    ckpt = load_teacher_checkpoint(teacher_cfg)
    teacher_nodes = teacher_node_configs(ckpt)
    entries = resolve_recipe(cfg, nodes, teacher_nodes)
    if not entries:
        logger.warning(
            "A teacher is set, but no node matches a teacher node. "
            "The model trains without distillation."
        )
        return None
    teacher = build_teacher(
        ckpt,
        teacher_nodes,
        {entry.teacher_node for entry in entries},
        cfg,
        input_shapes,
        strict=teacher_cfg.strict,
    )
    controller = DistillationController(teacher)
    for entry in entries:
        _attach_loss(controller, nodes, teacher, entry)
    _log_recipe(teacher_cfg.weights, entries)
    return controller


def _attach_loss(
    controller: DistillationController,
    nodes: Nodes,
    teacher: Teacher,
    entry: DistillationEntry,
) -> None:
    node = nodes[entry.student_node]
    teacher_node = teacher.nodes[entry.teacher_node]
    _check_classes(entry, node.module, teacher_node.module)
    Loss = LOSSES.get(entry.loss.name)
    if not issubclass(Loss, BaseDistillationLoss):
        raise TypeError(
            f"'{entry.loss.name}' in the `distillation` list of node "
            f"'{entry.student_node}' is not a distillation loss."
        )
    name = entry.loss.identifier
    if name in node.losses or name in node.distillation:
        raise ValueError(
            f"Node '{entry.student_node}' already has a loss named "
            f"'{name}'. Give the distillation loss another `alias`."
        )
    params = {**entry.loss.params, "teacher_node": entry.teacher_node}
    loss = Loss(
        **params, node=node.module, final_loss_weight=entry.loss.weight
    )
    connector = loss.build(
        nodes.output_shapes[entry.student_node],
        teacher.nodes.output_shapes[entry.teacher_node],
    )
    if connector is not None:
        controller.connectors[f"{entry.student_node}/{name}"] = connector
    node.distillation[name] = loss


def _check_classes(
    entry: DistillationEntry, student: BaseNode, teacher: BaseNode
) -> None:
    if student.task is None or teacher.task is None:
        return
    if student.classes != teacher.classes:
        raise ValueError(
            f"Node '{entry.student_node}' has the classes "
            f"{dict(student.classes)}, but teacher node "
            f"'{entry.teacher_node}' has {dict(teacher.classes)}. "
            "Distillation needs the same classes in the same order."
        )


def _log_recipe(weights: str, entries: list[DistillationEntry]) -> None:
    lines = [
        f"  {entry.student_node}: {entry.loss.identifier} <- "
        f"teacher {entry.teacher_node} ({entry.reason})"
        for entry in entries
    ]
    logger.info(f"Distillation from '{weights}':\n" + "\n".join(lines))
