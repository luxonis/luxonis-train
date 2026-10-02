"""Match student nodes to teacher nodes and choose the distillation
losses.

The recipe runs when training starts. It never changes the config, so a
saved config stays the same when it is loaded again.

"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from loguru import logger
from torch import Size

from luxonis_train.config import Config, LossModuleConfig, NodeConfig
from luxonis_train.lightning.utils import Nodes
from luxonis_train.registry import NODES


@dataclass(frozen=True)
class DistillationEntry:
    """One distillation loss of one student node.

    Attributes:
        student_node: The identifier of the student node.
        teacher_node: The identifier of the matched teacher node.
        loss: The loss, as a config entry.
        reason: Why the loss is there, for the log.

    """

    student_node: str
    teacher_node: str
    loss: LossModuleConfig
    reason: str


def resolve_recipe(
    cfg: Config, nodes: Nodes, teacher_nodes: Mapping[str, NodeConfig]
) -> list[DistillationEntry]:
    """List the distillation losses of every student node.

    The explicit ``distillation`` lists of the nodes come first. Then
    the automatic recipe covers the nodes with ``distillation: auto``:

    - A head whose class sets ``distillation_loss`` gets that loss when
      the teacher has a node with the same identifier and the same task.
    - A node that feeds a head with a distillation loss gets
      `CWDDistillationLoss` on the ``features`` levels the head reads,
      when the teacher has a node with the same identifier.

    A loss reads the teacher node with the same identifier as its own
    node, unless its ``params`` set ``teacher_node``.

    Args:
        cfg: The config of the student.
        nodes: The built student nodes.
        teacher_nodes: The node configs of the teacher, keyed by
            identifier.

    Returns:
        The losses, in config order.

    Raises:
        ValueError: When an explicit loss names a teacher node that does
            not exist.

    """
    entries = [
        entry
        for node_cfg in cfg.model.nodes
        for entry in _explicit_entries(node_cfg, teacher_nodes)
    ]
    entries += _head_entries(cfg, teacher_nodes)
    distilled = {entry.student_node for entry in entries}
    entries += _feeder_entries(cfg, nodes, teacher_nodes, distilled)
    return entries


def _explicit_entries(
    node_cfg: NodeConfig, teacher_nodes: Mapping[str, NodeConfig]
) -> Iterator[DistillationEntry]:
    for loss_cfg in node_cfg.distillation_losses:
        teacher_node = loss_cfg.params.get("teacher_node", node_cfg.identifier)
        if teacher_node not in teacher_nodes:
            raise ValueError(
                f"Distillation loss '{loss_cfg.identifier}' of node "
                f"'{node_cfg.identifier}' reads teacher node "
                f"'{teacher_node}', but the teacher has no such node. "
                f"Teacher nodes: {list(teacher_nodes)}. Set "
                "`params.teacher_node` to one of them."
            )
        yield DistillationEntry(
            node_cfg.identifier, str(teacher_node), loss_cfg, "config"
        )


def _head_entries(
    cfg: Config, teacher_nodes: Mapping[str, NodeConfig]
) -> Iterator[DistillationEntry]:
    for node_cfg in cfg.model.nodes:
        default = getattr(NODES.get(node_cfg.name), "distillation_loss", None)
        if node_cfg.distillation != "auto" or default is None:
            continue
        teacher_cfg = teacher_nodes.get(node_cfg.identifier)
        if teacher_cfg is None or not _same_task(node_cfg, teacher_cfg):
            logger.info(
                f"Distillation: the teacher has no head "
                f"'{node_cfg.identifier}' with the same task, so the head "
                "is not distilled."
            )
            continue
        yield DistillationEntry(
            node_cfg.identifier,
            node_cfg.identifier,
            LossModuleConfig(**default),
            "default of the head",
        )


def _feeder_entries(
    cfg: Config,
    nodes: Nodes,
    teacher_nodes: Mapping[str, NodeConfig],
    distilled: set[str],
) -> Iterator[DistillationEntry]:
    configs = {node_cfg.identifier: node_cfg for node_cfg in cfg.model.nodes}
    levels: dict[str, set[int]] = {}
    for head in sorted(distilled):
        for feeder in nodes.graph[head]:
            if (
                configs[feeder].distillation == "auto"
                and feeder in teacher_nodes
                and _has_feature_maps(nodes, feeder)
            ):
                read = _levels_read(nodes, head, feeder)
                levels.setdefault(feeder, set()).update(read)
    for feeder, read in levels.items():
        yield DistillationEntry(
            feeder,
            feeder,
            LossModuleConfig(
                name="CWDDistillationLoss", params={"levels": sorted(read)}
            ),
            "feeds a distilled head",
        )


def _same_task(student: NodeConfig, teacher: NodeConfig) -> bool:
    student_task = getattr(NODES.get(student.name), "task", None)
    teacher_task = getattr(NODES.get(teacher.name), "task", None)
    if student_task is None or teacher_task is None:
        return False
    return student_task.name == teacher_task.name


def _has_feature_maps(nodes: Nodes, node_name: str) -> bool:
    shapes = nodes.output_shapes[node_name].get("features")
    if shapes is None:
        return False
    shapes = shapes if isinstance(shapes, list) else [shapes]
    return all(len(shape) == 4 for shape in shapes)


def _levels_read(nodes: Nodes, head: str, feeder: str) -> list[int]:
    shapes = nodes.output_shapes[feeder]["features"]
    n_levels = len(shapes) if isinstance(shapes, list) else 1
    # One-element sizes stand for the level indices, so `get_attached`
    # applies the attach index of the head to them.
    markers = [Size([index]) for index in range(n_levels)]
    read = nodes[head].module.get_attached(markers)
    return [
        marker[0] for marker in (read if isinstance(read, list) else [read])
    ]
