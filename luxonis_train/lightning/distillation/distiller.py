"""Hold the teacher and the distillation losses of a training run."""

from typing import NamedTuple

import lightning.pytorch as pl
import torch
from loguru import logger
from torch import Tensor, nn
from typing_extensions import override

from luxonis_train.attached_modules.losses.distillation import (
    BaseDistillationLoss,
    StudentContext,
)
from luxonis_train.config import Config, TeacherConfig
from luxonis_train.lightning.utils import Nodes
from luxonis_train.nodes import BaseNode
from luxonis_train.registry import LOSSES
from luxonis_train.typing import Labels, Packet

from .recipe import DistillationEntry, levels_read, resolve_recipe
from .teacher import (
    Teacher,
    build_teacher,
    load_teacher_checkpoint,
    teacher_node_configs,
)


class _Binding(NamedTuple):
    """A loss of a student node and the teacher node it reads."""

    name: str
    loss: BaseDistillationLoss
    teacher_node: str


class Distiller(nn.Module):
    """The teacher and the distillation losses of a training run.

    `LuxonisLightningModule.setup` builds the distiller when ``fit``
    starts and ``model.teacher`` is set. The distiller is a registered
    submodule of the Lightning module, so its ``connectors`` reach the
    state dict, the checkpoints, the optimizer, and the DDP wrapper.

    The teacher is not registered. No recursive method of
    ``torch.nn.Module`` reaches it, and it never lands in a checkpoint.
    The losses live in a plain dictionary, like the losses of a
    `NodeWrapper`.

    Attributes:
        connectors: The trainable parts of the losses, keyed
            ``<node>/<loss>``. The exported model does not contain them.

    """

    _teacher: Teacher | None

    def __init__(self, teacher: Teacher):
        """Hold the teacher, without registering it.

        Args:
            teacher: The teacher.

        """
        super().__init__()
        self.connectors = nn.ModuleDict()
        self._bindings: dict[str, list[_Binding]] = {}
        self._teacher_device = torch.device("cpu")
        # `object.__setattr__` keeps the teacher out of `_modules`.
        object.__setattr__(self, "_teacher", teacher)

    @classmethod
    def from_config(cls, cfg: Config, nodes: Nodes) -> "Distiller":
        """Load the teacher of ``cfg.model.teacher`` and the losses.

        The method reads the teacher checkpoint, resolves the losses
        with `resolve_recipe`, builds the teacher nodes that the losses
        read, and builds and sets up each loss.

        Args:
            cfg: The config of the student. ``model.teacher`` must be
                set.
            nodes: The built student nodes.

        Returns:
            The distiller.

        Raises:
            ValueError: When ``model.teacher`` is not set, when no node
                gets a loss, or when a matched node has other classes
                than its teacher node.
            TypeError: When a ``distillation`` entry names a loss that
                is not a distillation loss.

        """
        teacher_cfg = cfg.model.teacher
        if teacher_cfg is None:
            raise ValueError("`model.teacher` is not set.")
        ckpt = load_teacher_checkpoint(teacher_cfg)
        teacher_nodes = teacher_node_configs(ckpt)
        entries = resolve_recipe(cfg, nodes, teacher_nodes)
        teacher = build_teacher(
            ckpt,
            teacher_nodes,
            {entry.teacher_node for entry in entries},
            cfg,
            nodes.input_shapes,
            strict=teacher_cfg.strict,
        )
        distiller = cls(teacher)
        for entry in entries:
            distiller._attach(nodes, entry, teacher_cfg)
        _log_recipe(teacher_cfg.weights, entries)
        return distiller

    @property
    def teacher(self) -> Teacher:
        """The teacher.

        Raises:
            RuntimeError: When `release` dropped the teacher.

        """
        if self._teacher is None:
            raise RuntimeError(
                "The distiller released its teacher when training ended."
            )
        return self._teacher

    @property
    def has_teacher(self) -> bool:
        """Whether the distiller holds a teacher, so the losses run."""
        return self._teacher is not None

    def losses(self, node_name: str) -> dict[str, BaseDistillationLoss]:
        """Give the distillation losses of a student node.

        Args:
            node_name: The identifier of the node.

        Returns:
            The losses, keyed by identifier.

        """
        return {
            binding.name: binding.loss
            for binding in self._bindings.get(node_name, [])
        }

    def run_teacher(
        self, inputs: dict[str, Tensor]
    ) -> dict[str, Packet[Tensor]]:
        """Run the teacher on a batch.

        The teacher moves to the device of the inputs on its first use,
        so it needs no ``to`` call from the trainer.

        Args:
            inputs: The loader inputs.

        Returns:
            The output packets of the teacher nodes, keyed by
            identifier.

        Raises:
            RuntimeError: When `release` dropped the teacher.

        """
        teacher = self.teacher
        device = next(iter(inputs.values())).device
        if device != self._teacher_device:
            teacher.to(device)
            self._teacher_device = device
        return teacher(inputs)

    def compute_losses(
        self,
        node_name: str,
        outputs: Packet[Tensor],
        labels: Labels,
        teacher_outputs: dict[str, Packet[Tensor]],
    ) -> dict[str, Tensor | tuple[Tensor, dict[str, Tensor]]]:
        """Run the distillation losses of one student node.

        Args:
            node_name: The identifier of the node.
            outputs: The output packet of the node.
            labels: The labels of the batch.
            teacher_outputs: The result of `run_teacher`.

        Returns:
            The value of each loss, keyed by identifier.

        """
        return {
            binding.name: binding.loss.run(
                outputs, labels, teacher_outputs[binding.teacher_node]
            )
            for binding in self._bindings.get(node_name, [])
        }

    def release(self) -> None:
        """Drop the teacher, so its memory is free.

        The losses stop running. The connectors stay, so the state dict
        keeps its keys.

        """
        self._teacher = None

    def _attach(
        self, nodes: Nodes, entry: DistillationEntry, cfg: TeacherConfig
    ) -> None:
        """Build one loss, set it up, and register its connector."""
        node = nodes[entry.student_node]
        teacher = self.teacher
        _check_classes(
            entry, node.module, teacher.nodes[entry.teacher_node].module
        )
        Loss = LOSSES.get(entry.loss.name)
        if not issubclass(Loss, BaseDistillationLoss):
            raise TypeError(
                f"'{entry.loss.name}' in the `distillation` list of node "
                f"'{entry.student_node}' is not a distillation loss."
            )
        name = entry.loss.identifier
        if name in node.losses or name in self.losses(entry.student_node):
            raise ValueError(
                f"Node '{entry.student_node}' already has a loss named "
                f"'{name}'. Give the distillation loss another `alias`."
            )
        context = StudentContext(
            losses=node.losses,
            levels_read=levels_read(nodes, entry.student_node),
        )
        loss = Loss(
            **(Loss.derive_params(context) | entry.loss.params),
            node=node.module,
            final_loss_weight=entry.loss.weight * cfg.loss_weight,
        )
        connector = loss.setup(
            nodes.output_shapes[entry.student_node],
            teacher.nodes.output_shapes[entry.teacher_node],
        )
        if connector is not None:
            self.connectors[f"{entry.student_node}/{name}"] = connector
        self._bindings.setdefault(entry.student_node, []).append(
            _Binding(name, loss, entry.teacher_node)
        )


class ReleaseTeacherCallback(pl.Callback):
    """Release the teacher when training ends.

    `LuxonisLightningModule.configure_callbacks` puts the callback
    before the other callbacks of the config. The test, export and
    quantization callbacks that run when training ends therefore find no
    teacher in memory, and they never run the distillation losses.

    """

    @override
    def on_train_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Release the teacher of the distiller of ``pl_module``."""
        distiller = getattr(pl_module, "distiller", None)
        if distiller is not None:
            distiller.release()


def _check_classes(
    entry: DistillationEntry, student: BaseNode, teacher: BaseNode
) -> None:
    """Require the same classes in a student and a teacher head."""
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
    """Log every loss with its teacher node and its reason."""
    lines = [
        f"  {entry.student_node}: {entry.loss.identifier} <- "
        f"teacher {entry.teacher_node} ({entry.reason})"
        for entry in entries
    ]
    logger.info(f"Distillation from '{weights}':\n" + "\n".join(lines))
